#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <chrono>
#include <nlohmann/json.hpp>

#include "common.cuh"

using namespace cuvs::neighbors;
using json = nlohmann::json;

int main(int argc, char** argv) {
	// Check if config file is provided
	std::string config_file;
	if (argc < 3 || std::string(argv[1]) != "--config") {
		printf("No config file provided. Using default configuration file './config/default_config.json'.\n");
		config_file = "../src/config.json";
	} else {
		config_file = argv[2];
	}

	// Variables to store configuration
	std::string data_dir;
	std::string data_fname;
	std::string query_fname;
	std::string data_label_fname;
	std::string query_label_fname;
	std::string ivf_graph_fname;
	std::string ivf_bfs_fname;
	std::string ground_truth_fname;
	int itopk_size;
	int specificity_threshold;
	int graph_degree;
	int topk;
	int num_runs;
	int warmup_runs;
	bool force_rebuild;

	// Load configuration from file
	std::ifstream file(config_file);
	if (!file.is_open()) {
		fprintf(stderr, "Unable to open config file: %s\n", config_file.c_str());
		return 1;
	}

	try {
		printf("Loading configuration from %s\n", config_file.c_str());
		json config;
		file >> config;

		// Load all parameters directly from config
		data_dir = config["data_dir"];
		data_fname = config["data_fname"];
		query_fname = config["query_fname"];
		data_label_fname = config["data_label_fname"];
		query_label_fname = config["query_label_fname"];
		specificity_threshold = config["spec_threshold"];
		graph_degree = config["graph_degree"];
		force_rebuild = config["force_rebuild"];
		itopk_size = config["itopk_size"];
		topk = config["topk"];
		num_runs = config["num_runs"];
		warmup_runs = config["warmup_runs"];
		ivf_graph_fname = config["ivf_graph_fname"];
		ivf_bfs_fname = config["ivf_bfs_fname"];
		ground_truth_fname = config["ground_truth_fname"];

	} catch (const std::exception& e) {
		fprintf(stderr, "Error parsing JSON config file: %s\n", e.what());
		return 1;
	}

	// Construct full file paths
	std::string full_data_fname = data_dir + data_fname;
	std::string full_query_fname = data_dir + query_fname;
	std::string full_data_label_fname = data_dir + data_label_fname;
	std::string full_query_label_fname = data_dir + query_label_fname;
	std::string full_ivf_graph_fname = data_dir + ivf_graph_fname;
	std::string full_ivf_bfs_fname = data_dir + ivf_bfs_fname;
	std::string full_ground_truth_fname = data_dir + ground_truth_fname;

	// Print configuration
	printf("\n=== Configuration ===\n");
	printf("iTopK size: %d\n", itopk_size);
	printf("Specificity threshold: %d\n", specificity_threshold);
	printf("Graph degree: %d\n", graph_degree);
	printf("TopK: %d\n", topk);
	printf("Number of runs: %d\n", num_runs);
	printf("Warmup runs: %d\n", warmup_runs);

	std::vector<float> h_data;
	std::vector<float> h_queries;
	std::vector<std::vector<int>> label_data_vecs;
	std::vector<std::vector<int>> data_label_vecs;
	std::vector<std::vector<int>> query_label_vecs;
	uint32_t N, Nq, dim;
	read_labeled_data<float, int64_t>(full_data_fname, full_data_label_fname, full_query_fname, full_query_label_fname,
																	  &h_data, &h_queries,
																	  &label_data_vecs, &data_label_vecs, &query_label_vecs,
																	  &N, &Nq, &dim);

	printf("\n=== Dataset Information ===\n");
	printf("Base dataset size: N=%u, dim=%u\n", N, dim);
	printf("Query dataset size: N=%u, dim=%u\n", Nq, dim);

	shared_resources::configured_raft_resources res;

	auto dataset = raft::make_device_matrix<float, int64_t>(res, N, dim);
	auto queries = raft::make_device_matrix<float, int64_t>(res, Nq, dim);
	raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), raft::resource::get_cuda_stream(res));
	raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(res));

	std::vector<uint32_t> h_query_labels(Nq);
	for (int i = 0; i < Nq; i++) h_query_labels[i] = query_label_vecs[i][0];
	auto query_labels = raft::make_device_vector<uint32_t, int64_t>(res, Nq);
	raft::copy(query_labels.data_handle(),
						 h_query_labels.data(),
						 Nq,
						 raft::resource::get_cuda_stream(res));

	// Build VecFlow index
	auto idx = vecflow::build(res,
														raft::make_const_mdspan(dataset.view()),
														data_label_vecs,
														graph_degree,
														specificity_threshold,
														full_ivf_graph_fname,
														full_ivf_bfs_fname,
														force_rebuild);

	auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries.extent(0), topk);
	auto distances = raft::make_device_matrix<float, int64_t>(res, queries.extent(0), topk);

	// Warmup runs
	printf("\n=== Performing Search ===\n");
	for (int i = 0; i < warmup_runs; i++) {
		vecflow::search(res, 
		                idx,
										raft::make_const_mdspan(queries.view()),
										query_labels.view(),
										itopk_size,
										neighbors.view(),
										distances.view());
		raft::resource::sync_stream(res);
	}

	// Timed runs
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_runs; i++) {
		vecflow::search(res,
										idx,
										raft::make_const_mdspan(queries.view()),
										query_labels.view(),
										itopk_size,
										neighbors.view(),
										distances.view());
		raft::resource::sync_stream(res);
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration<double>(end_time - start_time).count();
	double avg_ms = (total_time * 1000.0) / num_runs;
	double qps = num_runs * queries.extent(0) / total_time;
	printf("Search timing (%d runs):\n", num_runs);
	printf("- Total time: %.2f ms\n", total_time * 1000.0);
	printf("- Average per search: %.2f ms\n", avg_ms);
	printf("- QPS: %.2f\n", qps);

	auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries.extent(0), topk);
	generate_ground_truth(res,
												raft::make_const_mdspan(dataset.view()),
												raft::make_const_mdspan(queries.view()),
												label_data_vecs,
												query_label_vecs,
												gt_neighbors.view(),
												full_ground_truth_fname);
	double recall = compute_recall(res, neighbors.view(), gt_neighbors.view());
	printf("Recall: %.4f\n", recall);
}