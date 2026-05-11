// SPDX-License-Identifier: Apache-2.0
//
// VecFlow multi-label (2-label AND) example on SIFT1M.
//
// Each query carries exactly two labels; a candidate must contain BOTH.
// Build the index with multi_label=true so the CSR label arrays land on
// the index struct, then route queries through `vecflow::search_multi_labels`.

#include <cuvs/neighbors/shared_resources.hpp>
#include <cuvs/neighbors/vecflow.hpp>

#include <raft/core/device_mdarray.hpp>

#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>

#include "common.cuh"

using namespace cuvs::neighbors;
using json = nlohmann::json;

static std::string default_config_path(const char* argv0) {
	try {
		auto dir = std::filesystem::canonical(argv0).parent_path();
		return (dir / ".." / "src" / "config_multi.json").lexically_normal().string();
	} catch (...) {
		return "../src/config_multi.json";
	}
}

int main(int argc, char** argv) {
	std::string config_file;
	if (argc < 3 || std::string(argv[1]) != "--config") {
		config_file = default_config_path(argv[0]);
		printf("No --config provided. Using bundled default: %s\n", config_file.c_str());
	} else {
		config_file = argv[2];
	}

	std::ifstream f(config_file);
	if (!f.is_open()) {
		fprintf(stderr, "Unable to open config: %s\n", config_file.c_str());
		return 1;
	}
	json cfg; f >> cfg;

	// Anchor data_dir to the config file's directory if it's relative.
	std::string data_dir = cfg["data_dir"];
	{
		std::filesystem::path dd(data_dir);
		if (dd.is_relative()) {
			auto cdir = std::filesystem::canonical(config_file).parent_path();
			data_dir  = (cdir / dd).lexically_normal().string();
			if (!data_dir.empty() && data_dir.back() != '/') data_dir += '/';
		}
	}

	std::vector<int> itopk_sizes;
	if (cfg["itopk_size"].is_array()) {
		for (auto& v : cfg["itopk_size"]) itopk_sizes.push_back(v.get<int>());
	} else {
		itopk_sizes.push_back(cfg["itopk_size"].get<int>());
	}
	const int specificity_threshold = cfg["spec_threshold"];
	const int graph_degree          = cfg["graph_degree"];
	const int topk                  = cfg["topk"];
	const int num_runs              = cfg["num_runs"];
	const int warmup_runs           = cfg["warmup_runs"];
	const bool force_rebuild        = cfg["force_rebuild"];

	const std::string full_data        = data_dir + std::string(cfg["data_fname"]);
	const std::string full_query       = data_dir + std::string(cfg["query_fname"]);
	const std::string full_data_lbl    = data_dir + std::string(cfg["data_label_fname"]);
	const std::string full_query_lbl   = data_dir + std::string(cfg["query_label_fname"]);
	const std::string full_graph_fname = data_dir + std::string(cfg["ivf_graph_fname"]);
	const std::string full_bfs_fname   = data_dir + std::string(cfg["ivf_bfs_fname"]);
	const std::string full_gt_fname    = data_dir + std::string(cfg["ground_truth_fname"]);

	printf("\n=== Configuration ===\n");
	printf("  itopk sizes: [");
	for (size_t i = 0; i < itopk_sizes.size(); i++)
		printf("%d%s", itopk_sizes[i], i + 1 < itopk_sizes.size() ? ", " : "");
	printf("]\n");
	printf("  specificity threshold: %d\n", specificity_threshold);
	printf("  graph degree:          %d\n", graph_degree);
	printf("  topk:                  %d\n", topk);
	printf("  runs / warmup:         %d / %d\n", num_runs, warmup_runs);

	// ── Load data + labels ────────────────────────────────────────────────
	std::vector<float> h_data, h_queries;
	std::vector<std::vector<int>> label_data_vecs, data_label_vecs, query_label_vecs;
	uint32_t N, Nq, dim;
	read_labeled_data<float, int64_t>(full_data, full_data_lbl,
	                                  full_query, full_query_lbl,
	                                  &h_data, &h_queries,
	                                  &label_data_vecs, &data_label_vecs, &query_label_vecs,
	                                  &N, &Nq, &dim);

	printf("\n=== Dataset Information ===\n");
	printf("  base:  N=%u, dim=%u\n", N, dim);
	printf("  query: N=%u, dim=%u\n", Nq, dim);

	// Filter to queries that carry exactly 2 labels. The generator (`generate_multi_query.py`)
	// emits an empty row for queries whose AND result set is too small; those are
	// silently dropped here so every retained query has the guaranteed-large
	// intersection from the generator's --min-and-size threshold.
	std::vector<uint32_t> keep_idx;
	keep_idx.reserve(Nq);
	for (uint32_t i = 0; i < Nq; i++) {
		if (query_label_vecs[i].size() >= 2) keep_idx.push_back(i);
	}
	const uint32_t Nq_valid = static_cast<uint32_t>(keep_idx.size());
	if (Nq_valid == 0) {
		fprintf(stderr, "ERROR: 0 queries with ≥2 labels — regenerate query_multi.txt "
		                "with a lower --min-and-size.\n");
		return 1;
	}
	if (Nq_valid != Nq) {
		printf("  WARNING: %u/%u queries were dropped (have <2 labels); using %u valid queries\n",
		       Nq - Nq_valid, Nq, Nq_valid);
	}

	// Compact the host queries + labels down to the valid subset.
	std::vector<float> h_queries_valid(static_cast<size_t>(Nq_valid) * dim);
	std::vector<int>   q_label_a(Nq_valid), q_label_b(Nq_valid);
	for (uint32_t i = 0; i < Nq_valid; i++) {
		const uint32_t src = keep_idx[i];
		std::copy(h_queries.begin() + static_cast<size_t>(src) * dim,
		          h_queries.begin() + static_cast<size_t>(src + 1) * dim,
		          h_queries_valid.begin() + static_cast<size_t>(i) * dim);
		q_label_a[i] = query_label_vecs[src][0];
		q_label_b[i] = query_label_vecs[src][1];
	}

	// ── Build VecFlow index with multi_label=true ─────────────────────────
	shared_resources::configured_raft_resources res;
	auto dataset = raft::make_device_matrix<float, int64_t>(res, N, dim);
	auto queries = raft::make_device_matrix<float, int64_t>(res, Nq_valid, dim);
	raft::copy(dataset.data_handle(), h_data.data(),         h_data.size(),         raft::resource::get_cuda_stream(res));
	raft::copy(queries.data_handle(), h_queries_valid.data(), h_queries_valid.size(), raft::resource::get_cuda_stream(res));

	// Move both label arrays to device.
	auto q_lbl_a = raft::make_device_vector<uint32_t, int64_t>(res, Nq_valid);
	auto q_lbl_b = raft::make_device_vector<uint32_t, int64_t>(res, Nq_valid);
	{
		std::vector<uint32_t> ha(Nq_valid), hb(Nq_valid);
		for (uint32_t i = 0; i < Nq_valid; i++) {
			ha[i] = (uint32_t)q_label_a[i];
			hb[i] = (uint32_t)q_label_b[i];
		}
		raft::update_device(q_lbl_a.data_handle(), ha.data(), Nq_valid, raft::resource::get_cuda_stream(res));
		raft::update_device(q_lbl_b.data_handle(), hb.data(), Nq_valid, raft::resource::get_cuda_stream(res));
	}

	printf("\n=== Building VecFlow Index (multi_label=true) ===\n");
	auto idx = vecflow::build(res,
	                          raft::make_const_mdspan(dataset.view()),
	                          data_label_vecs,
	                          graph_degree,
	                          specificity_threshold,
	                          full_graph_fname,
	                          full_bfs_fname,
	                          force_rebuild,
	                          /*multi_label=*/true);

	auto neighbors    = raft::make_device_matrix<uint32_t, int64_t>(res, Nq_valid, topk);
	auto distances    = raft::make_device_matrix<float,    int64_t>(res, Nq_valid, topk);
	auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, Nq_valid, topk);

	printf("\n=== Generating Ground Truth (AND, brute force, cached) ===\n");
	generate_ground_truth_multi(res,
	                            raft::make_const_mdspan(dataset.view()),
	                            raft::make_const_mdspan(queries.view()),
	                            label_data_vecs,
	                            q_label_a, q_label_b,
	                            gt_neighbors.view(),
	                            full_gt_fname);

	// ── Search sweep ──────────────────────────────────────────────────────
	printf("\n=== Performing Search Sweep ===\n");
	for (int itopk : itopk_sizes) {
		for (int w = 0; w < warmup_runs; w++) {
			vecflow::search_multi_labels(res, idx,
			                             raft::make_const_mdspan(queries.view()),
			                             q_lbl_a.view(), q_lbl_b.view(),
			                             itopk,
			                             neighbors.view(), distances.view());
			raft::resource::sync_stream(res);
		}

		auto t0 = std::chrono::high_resolution_clock::now();
		for (int r = 0; r < num_runs; r++) {
			vecflow::search_multi_labels(res, idx,
			                             raft::make_const_mdspan(queries.view()),
			                             q_lbl_a.view(), q_lbl_b.view(),
			                             itopk,
			                             neighbors.view(), distances.view());
			raft::resource::sync_stream(res);
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		const double total  = std::chrono::duration<double>(t1 - t0).count();
		const double avg_ms = total * 1000.0 / num_runs;
		const double qps    = num_runs * (double)queries.extent(0) / total;
		const double recall = compute_recall(res, neighbors.view(), gt_neighbors.view());
		printf("  itopk=%4d  qps=%10.1f  avg=%6.3f ms  recall=%.4f\n",
		       itopk, qps, avg_ms, recall);
	}
	return 0;
}
