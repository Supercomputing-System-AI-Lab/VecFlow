// End-to-end vecflow-chamfer demo driven by a JSON config.
//
// Usage:
//   VECFLOW_CHAMFER [config.json]
//   (defaults to ../config/config.json relative to the binary)
//
// Pipeline: load_dataset -> load_queries -> build -> warmup ->
//           search (with stats) for each (search_itopk, search_topk) pair ->
//           print per-stage timings, optional recall@K vs. ground truth,
//           top-K samples for the last configuration.
//
// Ground truth (optional):
//   Tries to load `<prefix>.ground.truth.ibin` first (multi-vector convention:
//   2x int32 header `[num_queries, top_k]` then a stream of
//   `[query_id:int32, doc_id:int32, score:float]` triplets). If the file is
//   missing or its top_k is smaller than recall_k, computes exact GT on the
//   GPU via vecflow_chamfer::kernels::chamfer_score over all documents in
//   doc-batches of `gt_doc_batch_size`. When `cache_gt` is true and the run
//   scored the full query set, the kernel result is written back to the file
//   so the next run skips the kernel pass.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <nlohmann/json.hpp>

#include "common.cuh"

struct PipelineConfig {
  std::string dataset_dir;
  std::string prefix;
  int tokens_per_query;

  // index.* (build-time)
  int n_anchors;
  int graph_degree;
  int itopk_size;     // build-time CAGRA itopk
  int n_iter;
  bool force_rebuild;
  // index.vpq.* — when vpq_build_enabled, the build pass also encodes each
  // doc token's residual via PQ and stores it on the index. pq_bits/pq_dim
  // are read only when vpq_build_enabled.
  bool     vpq_build_enabled;
  uint32_t pq_bits;
  uint32_t pq_dim;

  // search.* (runtime)
  std::vector<int> search_itopk;  // multi-config: itopk[i] paired with search_topk[i]
  std::vector<int> search_topk;
  uint32_t final_topk;
  int n_queries;                  // -1 means all queries
  float    refine_rate_anchor;    // selectivity after the anchor proxy stage

  // search.vpq.* — when vpq_search_enabled, slot a VPQ rerank between anchor
  // proxy and full-chamfer. Pipeline:
  //   anchor-proxy -> top-K(refine_rate_anchor * final_topk)
  //                   -> [VPQ -> top-K(refine_rate_vpq * final_topk) ->]
  //                   full-chamfer -> top-K(final_topk)
  bool  vpq_search_enabled;
  float refine_rate_vpq;

  // output.*
  uint32_t max_queries_to_print;
  bool show_performance;
  bool evaluate_recall;
  uint32_t recall_k;          // 0 -> use final_topk
  int64_t gt_doc_batch_size;  // batch over docs for kernel GT
  bool cache_gt;              // write kernel-computed GT to disk for reuse
};

static PipelineConfig load_config(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open config file: " + path);
  }
  nlohmann::json j;
  f >> j;

  PipelineConfig c;
  c.dataset_dir          = j["dataset"]["dir"];
  // Resolve a relative dataset_dir against the config file's directory, so the
  // bundled config can ship with a portable `../datasets/lifestyle` path that
  // works regardless of the caller's CWD.
  if (std::filesystem::path p{c.dataset_dir}; p.is_relative()) {
    auto cfg_dir = std::filesystem::absolute(path).parent_path();
    c.dataset_dir = std::filesystem::weakly_canonical(cfg_dir / p).string();
  }
  c.prefix               = j["dataset"]["prefix"];
  c.tokens_per_query     = (j.contains("queries") && j["queries"].contains("tokens_per_query"))
                             ? j["queries"]["tokens_per_query"].get<int>() : 32;

  c.n_anchors            = j["index"]["n_anchors"];
  c.graph_degree         = j["index"]["graph_degree"];
  c.itopk_size           = j["index"].value("itopk_size", j["index"].value("build_itopk", 256));
  c.n_iter               = j["index"]["n_iter"];
  c.force_rebuild        = j["index"].value("force_rebuild", false);
  {
    const auto& v = j["index"].contains("vpq")
                      ? j["index"]["vpq"]
                      : nlohmann::json::object();
    c.vpq_build_enabled = v.value("enabled", false);
    c.pq_bits           = c.vpq_build_enabled ? v.value("pq_bits", 8u)  : 0u;
    c.pq_dim            = c.vpq_build_enabled ? v.value("pq_dim",  32u) : 0u;
  }

  c.search_itopk         = j["search"]["search_itopk"].get<std::vector<int>>();
  c.search_topk          = j["search"]["search_topk"].get<std::vector<int>>();
  if (c.search_itopk.size() != c.search_topk.size() || c.search_itopk.empty()) {
    throw std::runtime_error("search_itopk and search_topk must be same-length non-empty arrays");
  }
  c.final_topk           = j["search"]["final_topk"];
  c.n_queries            = j["search"].value("n_queries", -1);
  c.refine_rate_anchor   = j["search"].value("refine_rate_anchor", 10.0f);
  {
    const auto& v = j["search"].contains("vpq")
                      ? j["search"]["vpq"]
                      : nlohmann::json::object();
    c.vpq_search_enabled = v.value("enabled",     false);
    c.refine_rate_vpq    = v.value("refine_rate",  2.0f);
  }
  if (c.vpq_search_enabled && !c.vpq_build_enabled) {
    throw std::runtime_error(
      "search.vpq.enabled=true requires index.vpq.enabled=true — the index must carry VPQ codes");
  }

  c.max_queries_to_print = j["output"].value("max_queries_to_print", 1);
  c.show_performance     = j["output"].value("show_performance", true);
  c.evaluate_recall      = j["output"].value("evaluate_recall", true);
  c.recall_k             = j["output"].value("recall_k", 0);
  c.gt_doc_batch_size    = j["output"].value("gt_doc_batch_size", int64_t{1 << 20});
  c.cache_gt             = j["output"].value("cache_gt", true);
  return c;
}

static std::string default_config_path(const char* relative) {
  // Resolve relative to the binary's directory so the example works regardless
  // of the caller's CWD.
  try {
    auto exe = std::filesystem::canonical("/proc/self/exe");
    return (exe.parent_path() / relative).lexically_normal().string();
  } catch (...) {
    return relative;
  }
}

int main(int argc, char** argv) {
  try {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      std::cout << "Usage: " << argv[0] << " [config.json]" << std::endl;
      return 0;
    }

    const std::string config_path = (argc > 1)
      ? std::string(argv[1])
      : default_config_path("../config/config.json");

    PipelineConfig cfg = load_config(config_path);

    std::cout << "============================================================\n"
              << "  vecflow-chamfer multi-vector search demo\n"
              << "============================================================\n"
              << "  config:   " << config_path << '\n'
              << "  dataset:  " << cfg.dataset_dir << "/" << cfg.prefix << '\n';

    // Install a 2 GiB RMM pool BEFORE constructing raft::device_resources so
    // per-query allocations (incl. CAGRA's internal workspaces) hit the pool
    // instead of cudaMallocAsync. rmm 26.06+ takes ownership of an any_resource
    // and returns the previous resource to restore on exit.
    std::cout << "\n[1/4] Initialize GPU\n";
    auto prev_device_mr = rmm::mr::set_current_device_resource(
      rmm::mr::pool_memory_resource{rmm::mr::get_current_device_resource_ref(),
                                    2ull * 1024 * 1024 * 1024});  // 2 GiB
    raft::device_resources res;
    std::cout << "      rmm pool:        2 GiB\n";

    std::cout << "\n[2/4] Load dataset\n";
    auto load_t0 = std::chrono::high_resolution_clock::now();
    auto dataset = vecflow_chamfer::load_dataset(cfg.dataset_dir, cfg.prefix);
    auto queries = vecflow_chamfer::load_queries(cfg.dataset_dir, cfg.prefix,
                                                 dataset.embedding_dim, cfg.tokens_per_query);
    auto load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - load_t0).count();
    {
      int min_tokens = INT_MAX, max_tokens = 0;
      int64_t total_tokens = 0;
      for (int32_t i = 0; i < dataset.num_docs; ++i) {
        int t = dataset.doc_offsets[i + 1] - dataset.doc_offsets[i];
        min_tokens = std::min(min_tokens, t);
        max_tokens = std::max(max_tokens, t);
        total_tokens += t;
      }
      const double avg_tokens = static_cast<double>(total_tokens) / dataset.num_docs;
      auto kv = [](const char* k, auto&& v) {
        std::cout << "      " << std::left << std::setw(17) << k << v << '\n';
      };
      kv("docs:",         dataset.num_docs);
      kv("dim:",          dataset.embedding_dim);
      {
        std::ostringstream s;
        s << "avg " << std::fixed << std::setprecision(1) << avg_tokens
          << "   min " << min_tokens << "   max " << max_tokens;
        kv("tokens/doc:",  s.str());
      }
      kv("queries:",      queries.num_queries);
      kv("tokens/query:", queries.num_tokens_per_query);
      {
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << (load_ms / 1000.0) << " s";
        kv("elapsed:",    s.str());
      }
    }

    std::cout << "\n[3/4] Build index\n";
    auto kv_idx = [](const char* k, auto&& v) {
      std::cout << "      " << std::left << std::setw(17) << k << v << '\n';
    };
    {
      std::ostringstream s;
      s << "n_anchors=" << cfg.n_anchors
        << "  graph_degree=" << cfg.graph_degree
        << "  itopk_size=" << cfg.itopk_size
        << "  n_iter=" << cfg.n_iter;
      kv_idx("params:", s.str());
    }
    if (cfg.vpq_build_enabled) {
      std::ostringstream s;
      s << "pq_bits=" << cfg.pq_bits << "  pq_dim=" << cfg.pq_dim;
      kv_idx("vpq:", s.str());
    }

    // Cache lives flat next to the dataset, two files sharing a stem:
    //   <stem>.maxivf   CAGRA index over anchors
    //   <stem>.meta     vecflow-chamfer sidecar (anchor labels, CSRs, VPQ tail)
    //
    // The VPQ shape is baked into the stem so toggling vpq.enabled or changing
    // pq_bits/pq_dim doesn't silently load a cache built with a different schema.
    const std::string vpq_tag = cfg.vpq_build_enabled
      ? (".pq" + std::to_string(cfg.pq_bits) + "x" + std::to_string(cfg.pq_dim))
      : std::string(".novpq");
    const std::string cache_stem = cfg.dataset_dir + "/" + cfg.prefix
                                 + "." + std::to_string(cfg.n_anchors)
                                 + vpq_tag;
    kv_idx("cache stem:", cache_stem);

    vecflow_chamfer::index_params build_params{
      .n_anchors    = cfg.n_anchors,
      .graph_degree = cfg.graph_degree,
      .itopk_size   = cfg.itopk_size,
      .n_iter       = cfg.n_iter,
      .pq_bits      = cfg.pq_bits,    // 0 -> VPQ disabled
      .pq_dim       = cfg.pq_dim,
    };
    // The library never prints; this callback drives a single-line progress
    // bar in-place via \r so build output stays on one row.
    build_params.on_progress = [](int iter, int n_iter_total, double frac) {
      constexpr int bar_w = 20;
      const int filled = std::min(bar_w, static_cast<int>(frac * bar_w));
      std::cout << "\r      iter " << (iter + 1) << "/" << n_iter_total << "  ["
                << std::string(filled, '=')
                << std::string(bar_w - filled, ' ') << "] "
                << std::setw(3) << static_cast<int>(frac * 100) << "%";
      if (frac >= 1.0 && iter == n_iter_total - 1) std::cout << '\n';
      std::cout.flush();
    };

    // Existence check is the load/build decision — no try/catch fallback.
    // If both files exist we trust them; deserialize failures propagate.
    const bool cache_present =
      std::filesystem::exists(cache_stem + ".maxivf") &&
      std::filesystem::exists(cache_stem + ".meta");

    const auto build_t0 = std::chrono::high_resolution_clock::now();
    const bool loaded_from_cache = !cfg.force_rebuild && cache_present;
    vecflow_chamfer::index index = [&] {
      if (loaded_from_cache) {
        return vecflow_chamfer::deserialize(res, cache_stem);
      }
      auto idx = vecflow_chamfer::build(res, dataset, build_params);
      vecflow_chamfer::serialize(res, cache_stem, idx);
      return idx;
    }();
    const double build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - build_t0).count();

    // Inverted-list stats — pull offsets to host once for reporting.
    {
      const auto& d_offsets = index.anchor_to_doc_ids_offsets;
      std::vector<uint32_t> h_offsets(d_offsets.size());
      raft::copy(h_offsets.data(), d_offsets.data_handle(),
                 h_offsets.size(), raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);

      const int n_anchors = static_cast<int>(h_offsets.size()) - 1;
      int min_docs = INT_MAX, max_docs = 0;
      int64_t total_docs = 0;
      for (int i = 0; i < n_anchors; ++i) {
        const int d = static_cast<int>(h_offsets[i + 1] - h_offsets[i]);
        min_docs = std::min(min_docs, d);
        max_docs = std::max(max_docs, d);
        total_docs += d;
      }
      const double avg_docs = static_cast<double>(total_docs) / n_anchors;

      std::ostringstream dpa, dur;
      dpa << "avg " << std::fixed << std::setprecision(1) << avg_docs
          << "   min " << min_docs << "   max " << max_docs;
      dur << std::fixed << std::setprecision(2) << (build_ms / 1000.0) << " s"
          << (loaded_from_cache ? "   (loaded from cache)" : "");
      kv_idx("docs/anchor:", dpa.str());
      kv_idx("elapsed:",     dur.str());
    }

    auto sub_queries = vecflow_chamfer_demo::slice_queries(queries, cfg.n_queries);
    const uint32_t actual_n_queries = sub_queries.num_queries;

    std::cout << "\n[4/4] Search\n"
              << "      queries:         " << actual_n_queries << '\n'
              << "      configurations:  " << cfg.search_itopk.size() << '\n';

    // Warmup with 10 queries on the first configuration.
    auto warmup = vecflow_chamfer_demo::slice_queries(sub_queries, std::min<int>(10, actual_n_queries));
    {
      vecflow_chamfer::search_params warmup_sp{
        .itopk           = static_cast<uint32_t>(cfg.search_itopk[0]),
        .search_topk     = static_cast<uint32_t>(cfg.search_topk[0]),
        .refine_rate     = cfg.refine_rate_anchor,
        .final_topk      = cfg.final_topk,
        .use_vpq         = cfg.vpq_search_enabled,
        .refine_rate_vpq = cfg.refine_rate_vpq,
      };
      auto wn = raft::make_device_matrix<uint32_t, int64_t>(res, warmup.num_queries, warmup_sp.final_topk);
      auto wd = raft::make_device_matrix<float,    int64_t>(res, warmup.num_queries, warmup_sp.final_topk);
      vecflow_chamfer::search(res, warmup_sp, index, dataset, warmup, wn.view(), wd.view());
    }

    vecflow_chamfer_demo::GroundTruth gt;
    bool has_gt = false;
    if (cfg.evaluate_recall) {
      const uint32_t k_for_gt = cfg.recall_k ? cfg.recall_k : cfg.final_topk;
      std::cout << "\n      ground truth:\n";

      try {
        gt = vecflow_chamfer_demo::load_ground_truth(
          cfg.dataset_dir, cfg.prefix, actual_n_queries);
        has_gt = true;
        std::cout << "        loaded from file  (" << gt.per_query.size()
                  << " queries x top-" << gt.top_k << ")\n";
      } catch (const std::exception&) {
        // File missing or unreadable — fall through to the kernel path.
      }

      const bool gt_too_small = has_gt && gt.top_k < static_cast<int32_t>(k_for_gt);
      if (!has_gt || gt_too_small) {
        if (gt_too_small) {
          std::cout << "        file top-" << gt.top_k << " < recall_k " << k_for_gt
                    << ", regenerating exactly\n";
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "        computing exact GT via chamfer_score kernel  "
                  << "(top-" << k_for_gt
                  << ", doc batch=" << cfg.gt_doc_batch_size << ")\n";
        gt = vecflow_chamfer_demo::generate_ground_truth_exact(
          res, dataset, sub_queries, static_cast<int>(k_for_gt), cfg.gt_doc_batch_size);
        const double secs = std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "        exact GT ready in " << std::fixed << std::setprecision(2)
                  << secs << " s\n";
        has_gt = true;

        // Cache to disk so subsequent runs hit the file path. Only safe when
        // we scored the full query set — a partial GT under the canonical
        // filename would silently corrupt later runs that expect all queries.
        if (cfg.cache_gt && actual_n_queries == static_cast<uint32_t>(queries.num_queries)) {
          try {
            vecflow_chamfer_demo::save_ground_truth(cfg.dataset_dir, cfg.prefix, gt);
            std::cout << "        cached GT to "
                      << cfg.prefix << ".ground.truth.ibin\n";
          } catch (const std::exception& e) {
            std::cerr << "        warning: failed to cache GT (" << e.what() << ")\n";
          }
        } else if (cfg.cache_gt) {
          std::cout << "        skipping GT cache (scored " << actual_n_queries
                    << "/" << queries.num_queries << " queries — partial)\n";
        }
      }
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> last_results;
    vecflow_chamfer::search_stats last_stats;
    for (size_t i = 0; i < cfg.search_itopk.size(); ++i) {
      const uint32_t s_itopk = static_cast<uint32_t>(cfg.search_itopk[i]);
      const uint32_t s_topk  = static_cast<uint32_t>(cfg.search_topk[i]);

      std::cout << "\n  --- config " << (i + 1) << "/" << cfg.search_itopk.size() << " ---\n";
      {
        std::ostringstream s;
        s << "itopk=" << s_itopk
          << "  search_topk=" << s_topk
          << "  refine_rate_anchor=" << cfg.refine_rate_anchor;
        if (cfg.vpq_search_enabled) s << "  refine_rate_vpq=" << cfg.refine_rate_vpq;
        s << "  final_topk=" << cfg.final_topk
          << "  rerank=" << (cfg.vpq_search_enabled ? "anchor->vpq->full" : "anchor->full");
        std::cout << "      " << std::left << std::setw(17) << "params:" << s.str() << '\n';
      }

      vecflow_chamfer::search_params sp{
        .itopk           = s_itopk,
        .search_topk     = s_topk,
        .refine_rate     = cfg.refine_rate_anchor,
        .final_topk      = cfg.final_topk,
        .use_vpq         = cfg.vpq_search_enabled,
        .refine_rate_vpq = cfg.refine_rate_vpq,
      };
      auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(
        res, sub_queries.num_queries, sp.final_topk);
      auto d_distances = raft::make_device_matrix<float, int64_t>(
        res, sub_queries.num_queries, sp.final_topk);
      vecflow_chamfer::search(res, sp, index, dataset, sub_queries,
                              d_neighbors.view(), d_distances.view(), last_stats);
      last_results = vecflow_chamfer_demo::copy_topk_to_host(
        res,
        raft::make_device_matrix_view<const uint32_t, int64_t>(
          d_neighbors.data_handle(), sub_queries.num_queries, sp.final_topk),
        raft::make_device_matrix_view<const float, int64_t>(
          d_distances.data_handle(), sub_queries.num_queries, sp.final_topk));
      const auto& stats = last_stats;

      if (cfg.show_performance) {
        // Stage breakdown: 8-space indent, fixed label column, right-aligned
        // 7-wide millisecond value with three decimal places.
        auto stage_row = [](const std::string& label, double ms) {
          std::cout << "        " << std::left << std::setw(46) << label
                    << std::right << std::fixed << std::setprecision(3)
                    << std::setw(7) << ms << " ms\n";
        };
        std::cout << "      per-query latency:\n";
        stage_row("stage 1   CAGRA candidate search",        stats.avg_candidate_generation_ms);
        stage_row("stage 2   hash-table dedup",              stats.avg_unique_id_finding_ms);
        stage_row("stage 3   anchor-chamfer proxy rerank",   stats.avg_anchor_chamfer_reranking_ms);
        {
          std::ostringstream lbl;
          lbl << "stage 4   top-K  (" << static_cast<int>(stats.avg_stage4_input_count)
              << " -> "               << static_cast<int>(stats.avg_stage4_output_count) << ")";
          stage_row(lbl.str(),                               stats.avg_first_sorting_ms);
        }
        const int stage_full   = cfg.vpq_search_enabled ? 7 : 5;
        const int stage_finalk = cfg.vpq_search_enabled ? 8 : 6;
        if (cfg.vpq_search_enabled) {
          stage_row("stage 5   vpq-chamfer rerank",          stats.avg_vpq_reranking_ms);
          std::ostringstream lbl;
          lbl << "stage 6   top-K  (" << static_cast<int>(stats.avg_stage4_output_count)
              << " -> "               << static_cast<int>(stats.avg_stage_vpq_output_count) << ")";
          stage_row(lbl.str(),                               stats.avg_vpq_sorting_ms);
        }
        {
          std::ostringstream lbl;
          lbl << "stage " << stage_full << "   full-chamfer rerank";
          stage_row(lbl.str(),                               stats.avg_full_chamfer_reranking_ms);
        }
        {
          const int input_count = cfg.vpq_search_enabled
            ? static_cast<int>(stats.avg_stage_vpq_output_count)
            : static_cast<int>(stats.avg_stage4_output_count);
          std::ostringstream lbl;
          lbl << "stage " << stage_finalk << "   top-K  (" << input_count
              << " -> " << static_cast<int>(stats.avg_stage6_output_count) << ")";
          stage_row(lbl.str(),                               stats.avg_second_sorting_ms);
        }
        std::cout << "        " << std::string(56, '-') << '\n';
        stage_row("total",                                   stats.avg_per_query_ms);
        std::cout << '\n';

        auto kv_search = [](const char* k, auto&& v) {
          std::cout << "      " << std::left << std::setw(20) << k << v << '\n';
        };
        std::ostringstream tput, cand;
        tput << std::fixed << std::setprecision(1) << stats.throughput_qps << " q/s";
        cand << std::fixed << std::setprecision(1) << stats.avg_candidates_per_query << " (avg)";
        kv_search("throughput:",      tput.str());
        kv_search("candidates/query:", cand.str());
      }

      if (has_gt) {
        auto kv_search = [](const char* k, auto&& v) {
          std::cout << "      " << std::left << std::setw(20) << k << v << '\n';
        };
        const uint32_t k = cfg.recall_k ? cfg.recall_k : cfg.final_topk;
        const double r   = vecflow_chamfer_demo::recall_at_k(last_results, gt, k);
        const double cov = vecflow_chamfer_demo::first_stage_coverage(
                             stats.first_stage_unique_doc_ids, gt, k);
        std::ostringstream rs, cs, rkey, ckey;
        rkey << "recall@" << k << ":";
        ckey << "coverage@" << k << ":";
        rs << std::fixed << std::setprecision(2) << (r   * 100.0) << " %";
        cs << std::fixed << std::setprecision(2) << (cov * 100.0) << " %   (1st-stage)";
        kv_search(rkey.str().c_str(), rs.str());
        kv_search(ckey.str().c_str(), cs.str());
      }
    }

    // Print top-K samples for the LAST configuration's results.
    if (cfg.max_queries_to_print > 0) {
      const auto& results = last_results;
      const uint32_t show = std::min<uint32_t>(cfg.max_queries_to_print,
                                               static_cast<uint32_t>(results.size()));
      std::cout << "\n  --- top-K samples (config " << cfg.search_itopk.size() << ") ---\n";
      for (uint32_t q = 0; q < show; ++q) {
        const auto& [ids, scores] = results[q];
        std::cout << "\n      query " << q << "  (top-" << ids.size() << ")\n";
        std::cout << "        " << std::left << std::setw(6) << "rank"
                  << std::right << std::setw(10) << "doc id"
                  << std::setw(14) << "score" << '\n';
        for (size_t i = 0; i < ids.size(); ++i) {
          std::cout << "        " << std::left << std::setw(6) << (i + 1)
                    << std::right << std::setw(10) << ids[i]
                    << std::setw(14) << std::fixed << std::setprecision(6) << scores[i]
                    << '\n';
        }
      }
      if (results.size() > show) {
        std::cout << "\n      ... (" << (results.size() - show) << " more queries)\n";
      }
    }

    vecflow_chamfer::release_dataset(dataset);
    rmm::mr::set_current_device_resource(std::move(prev_device_mr));
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nerror: " << e.what() << '\n';
    return 1;
  }
}
