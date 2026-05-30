// Shared helpers for the vecflow-chamfer examples.
//
//   - Ground truth (file-based and kernel-generated)
//   - Recall / first-stage-coverage metrics
//   - Query slicing
//   - Device top-K -> host vec-of-pairs conversion (strips padding sentinels)
//
// Header-only / inline so each .cu translation unit can include it without
// pulling in extra build steps.
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <thrust/sequence.h>

#include <vecflow_chamfer/vecflow_chamfer.hpp>

namespace vecflow_chamfer_demo {

// --- Ground truth -----------------------------------------------------------

struct GroundTruth {
  // per_query[q] = vector of (doc_id, score) in descending-score order.
  std::vector<std::vector<std::pair<int32_t, float>>> per_query;
  int32_t top_k;
};

// Read a `<prefix>.ground.truth.ibin` file laid out as the multi-vector
// convention: 2x int32 header `[num_queries, top_k]`, then a stream of triplets
// `[query_id:int32, doc_id:int32, score:float]`.
inline GroundTruth load_ground_truth(const std::string& dataset_dir,
                                     const std::string& prefix,
                                     int max_queries) {
  std::string path = dataset_dir + "/" + prefix + ".ground.truth.ibin";
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open ground truth file: " + path);
  }

  GroundTruth gt;
  int32_t num_queries = 0;
  f.read(reinterpret_cast<char*>(&num_queries), sizeof(int32_t));
  f.read(reinterpret_cast<char*>(&gt.top_k),    sizeof(int32_t));
  gt.per_query.resize(num_queries);

  int32_t query_id = 0, doc_id = 0;
  float score = 0.0f;
  while (f.read(reinterpret_cast<char*>(&query_id), sizeof(int32_t))) {
    if (!f.read(reinterpret_cast<char*>(&doc_id), sizeof(int32_t))) break;
    if (!f.read(reinterpret_cast<char*>(&score),  sizeof(float)))   break;
    if (query_id >= 0 && query_id < num_queries) {
      gt.per_query[query_id].emplace_back(doc_id, score);
    }
  }
  if (max_queries > 0 && max_queries < num_queries) {
    gt.per_query.resize(max_queries);
  }
  return gt;
}

// Write a GroundTruth to `<prefix>.ground.truth.ibin` in the exact format
// load_ground_truth() reads (header [num_queries, top_k] then [qid, did, score]
// triplets). Triplets are emitted in descending-score order, matching the
// in-memory layout produced by generate_ground_truth_exact.
inline void save_ground_truth(const std::string& dataset_dir,
                              const std::string& prefix,
                              const GroundTruth& gt) {
  std::string path = dataset_dir + "/" + prefix + ".ground.truth.ibin";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open ground truth file for write: " + path);
  }
  int32_t num_queries = static_cast<int32_t>(gt.per_query.size());
  f.write(reinterpret_cast<const char*>(&num_queries), sizeof(int32_t));
  f.write(reinterpret_cast<const char*>(&gt.top_k),    sizeof(int32_t));
  for (int32_t q = 0; q < num_queries; ++q) {
    for (const auto& [doc_id, score] : gt.per_query[q]) {
      f.write(reinterpret_cast<const char*>(&q),      sizeof(int32_t));
      f.write(reinterpret_cast<const char*>(&doc_id), sizeof(int32_t));
      f.write(reinterpret_cast<const char*>(&score),  sizeof(float));
    }
  }
}

// Exact ground truth computed via the library's own chamfer_score kernel.
//
// For each query we score *all* documents in slices of `doc_batch_size` (so the
// per-query score buffer stays small enough to fit in HBM) and keep a running
// top-K via a min-heap. The doc embeddings stay in host memory and are read
// from the GPU through HMM / C2C on GH200 — no H2D staging copy.
inline GroundTruth generate_ground_truth_exact(
    raft::device_resources const& res,
    const vecflow_chamfer::dataset& ds,
    const vecflow_chamfer::query_set& qs,
    int top_k,
    int64_t doc_batch_size = 1 << 20) {
  if (top_k <= 0) throw std::invalid_argument("generate_ground_truth_exact: top_k must be > 0");

  doc_batch_size = std::min<int64_t>(doc_batch_size, ds.num_docs);
  doc_batch_size = std::max<int64_t>(doc_batch_size, 1);

  GroundTruth gt;
  gt.top_k = top_k;
  gt.per_query.assign(qs.num_queries, {});

  auto stream = raft::resource::get_cuda_stream(res);

  auto d_queries = raft::make_device_matrix<__half, int64_t>(
    res, static_cast<int64_t>(qs.num_queries) * qs.num_tokens_per_query, qs.embedding_dim);
  raft::copy(d_queries.data_handle(), qs.query_embeddings.data(),
             qs.query_embeddings.size(), stream);

  auto d_doc_offsets = raft::make_device_vector<uint32_t, int64_t>(res, ds.doc_offsets.size());
  raft::copy(d_doc_offsets.data_handle(), ds.doc_offsets.data(),
             ds.doc_offsets.size(), stream);

  auto d_batch_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, doc_batch_size);
  auto d_batch_scores  = raft::make_device_vector<float,    int64_t>(res, doc_batch_size);
  std::vector<float> h_scores(doc_batch_size);

  // Score-major min-heap: smallest score at the top, so a new score that
  // exceeds the top is a candidate for the top-K set.
  using ScoreId = std::pair<float, int32_t>;
  std::priority_queue<ScoreId, std::vector<ScoreId>, std::greater<ScoreId>> heap;

  constexpr int bar_w = 20;
  auto draw_progress = [&](int32_t q_done) {
    const double frac = static_cast<double>(q_done) / qs.num_queries;
    const int filled  = std::min(bar_w, static_cast<int>(frac * bar_w));
    std::cout << "\r      [" << std::string(filled, '=')
              << std::string(bar_w - filled, ' ') << "] "
              << std::setw(3) << static_cast<int>(frac * 100) << "%   ("
              << q_done << "/" << qs.num_queries << " queries)";
    std::cout.flush();
  };

  for (int32_t q = 0; q < qs.num_queries; ++q) {
    const __half* qptr = d_queries.data_handle()
                       + static_cast<int64_t>(q) * qs.num_tokens_per_query * qs.embedding_dim;

    while (!heap.empty()) heap.pop();

    for (int64_t doc_start = 0; doc_start < ds.num_docs; doc_start += doc_batch_size) {
      const int64_t batch_n = std::min<int64_t>(doc_batch_size, ds.num_docs - doc_start);

      thrust::sequence(raft::resource::get_thrust_policy(res),
                       d_batch_doc_ids.data_handle(),
                       d_batch_doc_ids.data_handle() + batch_n,
                       static_cast<uint32_t>(doc_start));

      vecflow_chamfer::kernels::chamfer_score(
        static_cast<uint32_t>(batch_n),
        static_cast<uint32_t>(qs.num_tokens_per_query),
        static_cast<uint32_t>(qs.embedding_dim),
        qptr, ds.doc_embeddings,
        d_batch_doc_ids.data_handle(),
        d_doc_offsets.data_handle(),
        d_batch_scores.data_handle(),
        stream);

      raft::copy(h_scores.data(), d_batch_scores.data_handle(), batch_n, stream);
      raft::resource::sync_stream(res);

      for (int64_t i = 0; i < batch_n; ++i) {
        const int32_t did = static_cast<int32_t>(doc_start + i);
        const float   s   = h_scores[i];
        if (static_cast<int>(heap.size()) < top_k) {
          heap.emplace(s, did);
        } else if (s > heap.top().first) {
          heap.pop();
          heap.emplace(s, did);
        }
      }
    }

    // Drain heap (ascending by score) and reverse so per_query[q] is descending.
    std::vector<std::pair<int32_t, float>> top;
    top.reserve(heap.size());
    while (!heap.empty()) {
      top.emplace_back(heap.top().second, heap.top().first);
      heap.pop();
    }
    std::reverse(top.begin(), top.end());
    gt.per_query[q] = std::move(top);

    // Update bar every ~1% of queries, plus the final iteration.
    const int32_t step = std::max(1, qs.num_queries / 100);
    if ((q + 1) % step == 0 || q + 1 == qs.num_queries) {
      draw_progress(q + 1);
    }
  }
  std::cout << '\n';
  return gt;
}

// --- Recall metrics ---------------------------------------------------------

inline double recall_at_k(
  const std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>>& results,
  const GroundTruth& gt,
  uint32_t k) {

  const size_t n = std::min(results.size(), gt.per_query.size());
  if (n == 0) return 0.0;

  uint64_t hit = 0, denom = 0;
  for (size_t q = 0; q < n; ++q) {
    const auto& pred_ids   = results[q].first;
    const auto& gt_entries = gt.per_query[q];
    const size_t k_gt   = std::min<size_t>(k, gt_entries.size());
    const size_t k_pred = std::min<size_t>(k, pred_ids.size());

    std::set<int32_t> gt_set;
    for (size_t i = 0; i < k_gt; ++i) gt_set.insert(gt_entries[i].first);

    for (size_t i = 0; i < k_pred; ++i) {
      if (gt_set.count(static_cast<int32_t>(pred_ids[i]))) ++hit;
    }
    denom += k_gt;
  }
  return denom ? static_cast<double>(hit) / static_cast<double>(denom) : 0.0;
}

inline double first_stage_coverage(
  const std::vector<std::vector<uint32_t>>& first_stage_ids,
  const GroundTruth& gt,
  uint32_t k) {

  const size_t n = std::min(first_stage_ids.size(), gt.per_query.size());
  if (n == 0) return 0.0;

  uint64_t hit = 0;
  for (size_t q = 0; q < n; ++q) {
    const auto& ids = first_stage_ids[q];
    const auto& gt_entries = gt.per_query[q];
    const size_t k_gt = std::min<size_t>(k, gt_entries.size());

    std::set<uint32_t> stage_set(ids.begin(), ids.end());
    for (size_t i = 0; i < k_gt; ++i) {
      if (stage_set.count(static_cast<uint32_t>(gt_entries[i].first))) ++hit;
    }
  }
  return static_cast<double>(hit) / static_cast<double>(n * k);
}

// --- Misc -------------------------------------------------------------------

inline vecflow_chamfer::query_set slice_queries(
  const vecflow_chamfer::query_set& src, int n_queries) {
  if (n_queries <= 0 || n_queries >= src.num_queries) return src;
  vecflow_chamfer::query_set sub;
  sub.embedding_dim = src.embedding_dim;
  sub.num_tokens_per_query = src.num_tokens_per_query;
  sub.num_queries = n_queries;
  size_t per_query = static_cast<size_t>(src.num_tokens_per_query) * src.embedding_dim;
  sub.query_embeddings.assign(src.query_embeddings.begin(),
                              src.query_embeddings.begin() + n_queries * per_query);
  return sub;
}

// Copy device top-K matrices back to the host-side vec-of-pairs shape the
// example's printing / recall helpers consume. Strips UINT32_MAX sentinel
// entries that pad short rows.
inline std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>>
copy_topk_to_host(raft::device_resources const& res,
                  raft::device_matrix_view<const uint32_t, int64_t> neighbors,
                  raft::device_matrix_view<const float,    int64_t> distances) {
  const int64_t n_queries = neighbors.extent(0);
  const int64_t topk      = neighbors.extent(1);

  std::vector<uint32_t> h_ids(static_cast<size_t>(n_queries) * topk);
  std::vector<float>    h_scr(static_cast<size_t>(n_queries) * topk);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(h_ids.data(), neighbors.data_handle(), h_ids.size(), stream);
  raft::copy(h_scr.data(), distances.data_handle(), h_scr.size(), stream);
  raft::resource::sync_stream(res);

  std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>> out(n_queries);
  for (int64_t q = 0; q < n_queries; ++q) {
    std::vector<uint32_t> ids;    ids.reserve(topk);
    std::vector<float>    scores; scores.reserve(topk);
    for (int64_t k = 0; k < topk; ++k) {
      uint32_t id = h_ids[q * topk + k];
      if (id == 0xFFFFFFFFu) break;
      ids.push_back(id);
      scores.push_back(h_scr[q * topk + k]);
    }
    out[q] = {std::move(ids), std::move(scores)};
  }
  return out;
}

} // namespace vecflow_chamfer_demo
