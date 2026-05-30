#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include <raft/core/device_mdarray.hpp>
#include <cuvs/neighbors/cagra.hpp>

namespace vecflow_chamfer {

// Document set used by both indexing and search.
//
// doc_embeddings backing depends on the active GPU:
//   * Grace-Hopper / any device reporting cudaDevAttrPageableMemoryAccess==1:
//     plain malloc + cudaMemAdvise_v2(PreferredLocation=host). The GPU reads
//     the buffer through the C2C link / HMM / ATS — no HBM copy.
//   * PCIe-attached GPU (no pageable access): cudaMallocManaged with
//     PreferredLocation=host and AccessedBy=device, so pages stay resident on
//     host RAM and the GPU maps them directly (zero-copy over PCIe).
// Use release_dataset() to free — the right deallocator is picked from the
// internal `managed` flag below.
struct dataset {
  __half* doc_embeddings = nullptr;  // host memory, GPU-accessible (C2C or PCIe UM)
  std::vector<uint32_t> doc_offsets;
  std::vector<uint32_t> doc_token_to_doc;
  int embedding_dim = 0;
  int32_t num_docs = 0;
  int32_t num_doc_tokens = 0;
  bool doc_embeddings_managed = false;  // true => cudaFree, false => free()
};

// Query set: search-only. Embeddings live on host; search() uploads once.
struct query_set {
  std::vector<__half> query_embeddings;
  int32_t num_queries = 0;
  int32_t num_tokens_per_query = 0;
  int embedding_dim = 0;
};

// MaxIVF build-time parameters (Chamfer).
struct index_params {
  int n_anchors;
  int graph_degree;
  int itopk_size;
  int n_iter;

  // Optional VPQ rerank — when enabled, build() trains a PQ codebook on
  // per-token *residuals* (doc_token - anchor_centroid[anchor_labels[t]]) and
  // encodes every doc token. The MaxIVF anchors themselves play the VQ role,
  // so no separate VQ k-means runs and no vq_codebook is stored.
  //
  //   pq_bits == 0   -> VPQ disabled; idx.vpq stays std::nullopt
  //   pq_bits >  0   -> in [4..16] (cuVS constraint, 8 is the common pick)
  //   pq_dim         -> number of PQ subspaces; must divide embedding_dim
  uint32_t pq_bits = 0;
  uint32_t pq_dim  = 0;

  // Optional progress callback. If set, build() invokes it during the
  // per-iteration batch sweep with:
  //   iter:          0-indexed iteration in [0, n_iter)
  //   n_iter_total:  echoes index_params.n_iter for caller convenience
  //   fraction:      [0.0, 1.0] progress within the current iteration. The
  //                  library guarantees one final call per iteration with
  //                  fraction == 1.0 once anchor refinement is complete.
  //
  // The library itself never writes to stdout or stderr; route this to a
  // logger, progress bar, etc. if you want feedback during long builds.
  std::function<void(int iter, int n_iter_total, double fraction)> on_progress = nullptr;
};

// VPQ rerank side-data. Present iff index_params.pq_bits > 0 was used at build.
//
// The VQ side of the encoding reuses the MaxIVF anchors:
//   anchor centroids = idx.anchor_cagra_index.dataset()  (fp16, [n_anchors, dim])
//   per-token VQ idx = idx.anchor_labels                 (uint32, [n_tokens])
//
// Stage-5 of search reconstructs each doc token as:
//   token ≈ anchor_centroid[anchor_labels[t]]
//         + pq_decode(codes[t], pq_codebook)
struct vpq_data {
  raft::device_matrix<float, uint32_t> pq_codebook;     // [pq_dim * (1<<pq_bits), pq_len]
  raft::device_matrix<uint8_t, int64_t> codes;          // [n_tokens, quantized_dim]
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t pq_len;          // = dim / pq_dim
};

// MaxIVF search-time parameters.
//
// Pipeline keeps decreasing the candidate set per query:
//   stage 1   CAGRA over anchors                -> ~tokens_per_query * search_topk hits
//   stage 2   hash-table dedup to unique docs   -> unique candidates
//   stage 3   anchor-Chamfer proxy rerank       -> score per candidate
//   stage 4   top-K (refine_rate * final_topk)  -> ~refine_rate * final_topk
//   stage 5   [optional] VPQ chamfer rerank     -> requires use_vpq && idx.vpq
//   stage 6   [optional] top-K (refine_rate_vpq * final_topk)
//   stage 7   full-precision Chamfer rerank
//   stage 8   final top-K  (final_topk)
//
// When use_vpq is false, the VPQ pair (stages 5+6) is skipped and the top-K
// from stage 4 feeds straight into the full rerank.
struct search_params {
  uint32_t itopk;             // CAGRA itopk for stage 1
  uint32_t search_topk;       // CAGRA k for stage 1 (per query token)
  float    refine_rate;       // stage 4 keeps refine_rate * final_topk
  uint32_t final_topk;        // output top-K per query

  // VPQ rerank slot. Effective only if idx.vpq has a value (i.e. the index was
  // built with index_params.pq_bits > 0). refine_rate_vpq must be in
  // [1, refine_rate] — the VPQ stage can only narrow further before full rerank.
  bool  use_vpq         = false;
  float refine_rate_vpq = 1.5f;
};

// MaxIVF index: CAGRA over anchors + anchor->doc inverted lists.
// Full-precision doc embeddings are not held here; the search path reads them
// from dataset.doc_embeddings (system memory via C2C) during rerank.
struct index {
  cuvs::neighbors::cagra::index<half, uint32_t> anchor_cagra_index;
  raft::device_vector<uint32_t, int64_t> anchor_labels;
  raft::device_vector<uint32_t, int64_t> anchor_to_doc_ids;
  raft::device_vector<uint32_t, int64_t> anchor_to_doc_ids_offsets;
  raft::device_vector<uint32_t, int64_t> doc_offsets;
  std::optional<vpq_data> vpq;       // populated iff index_params.pq_bits > 0
};

// Per-query timing breakdown. All `avg_*_ms` fields are the per-query mean
// over `n_queries` queries (measured with std::chrono + stream sync per stage).
// `total_wall_ms` is the wall clock of the entire search call; `avg_per_query_ms`
// is `total_wall_ms / n_queries`. `throughput_qps` is `n_queries * 1000 / total_wall_ms`.
struct search_stats {
  double avg_candidate_generation_ms;     // Stage 1: CAGRA anchor search
  double avg_unique_id_finding_ms;        // Stage 2: hash-table dedup
  double avg_anchor_chamfer_reranking_ms; // Stage 3: anchor-Chamfer proxy
  double avg_first_sorting_ms;            // Stage 4: top-K of proxy scores
  double avg_vpq_reranking_ms;            // Stage 5: VPQ rerank (0 if !use_vpq)
  double avg_vpq_sorting_ms;              // Stage 6: top-K of VPQ scores (0 if !use_vpq)
  double avg_full_chamfer_reranking_ms;   // Stage 7: full-precision Chamfer rerank
  double avg_second_sorting_ms;           // Stage 8: final top-K
  double avg_candidates_per_query;

  double avg_per_query_ms;                // wall time per query (incl. unmeasured gaps)
  double total_wall_ms;                   // wall time of the entire search call
  double throughput_qps;                  // queries per second

  double avg_stage4_input_count;
  double avg_stage4_output_count;
  double avg_stage_vpq_output_count;      // post-stage-6 candidate count (0 if !use_vpq)
  double avg_stage6_output_count;         // final-topk count actually written

  // Unique doc IDs from the first candidate-generation stage (one vector per query).
  std::vector<std::vector<uint32_t>> first_stage_unique_doc_ids;
};

} // namespace vecflow_chamfer
