#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Public chamfer kernels (libvecflow_chamfer_kernels).
//
// Two ways to invoke them:
//
//   1. High-level (recommended): call vecflow_chamfer::kernels::chamfer_score or
//      vecflow_chamfer::kernels::chamfer_score_anchor. The launcher handles
//      grid/block sizing, stream binding, and shape validation. Consumers never
//      need to know the kernel's internal launch configuration or shape
//      constraints.
//
//   2. Low-level: launch the kernel template directly from
//      vecflow_chamfer::kernels::detail. You take responsibility for picking a
//      block size that satisfies the kernel's `__launch_bounds__` and for shape
//      compatibility. Internal constants are in src/search/chamfer_core_impl.cuh.
//
// Both kernels compute, per document, the Chamfer score:
//   sum_{i in [0, tokens_per_query)} max_{j in doc} <q_i, d_j>
// using fp16 tensor-core WMMA (m16n16k16).
//
// chamfer_score_anchor: scores documents by their *anchor representative tokens*
//   (indirect lookup via doc_token_to_rep). Used as a proxy filter pre-rerank.
//
// chamfer_score: scores documents by their *raw token embeddings* (direct read
//   from `dataset` in device memory). Used as the final full-precision rerank.
//
// chamfer_score_vpq: scores documents by reconstructing each doc token from
//   VPQ side-data (anchor centroid + PQ residual). Cheaper than chamfer_score
//   because the codes are 1 byte per subspace; used as a mid-pipeline filter
//   between anchor proxy and full rerank. Requires the cuVS PQ codebook layout
//   produced with use_subspaces=true (shape [pq_dim*(1<<pq_bits), pq_len], fp32)
//   and use_vq=false (the MaxIVF anchors play the VQ role).

namespace vecflow_chamfer::kernels {

namespace detail {

template <typename Index_t>
__global__ void chamfer_score_anchor(const __half* __restrict__ query,
                                     const __half* __restrict__ dataset,
                                     const uint32_t* __restrict__ doc_ids,
                                     const uint32_t* __restrict__ doc_offsets,
                                     const uint32_t* __restrict__ doc_token_to_rep,
                                     float* __restrict__ chamfer_distances);

template <typename Index_t>
__global__ void chamfer_score(const __half* __restrict__ query,
                              const __half* __restrict__ dataset,
                              const uint32_t* __restrict__ doc_ids,
                              const uint32_t* __restrict__ doc_offsets,
                              float* __restrict__ chamfer_distances);

template <typename Index_t>
__global__ void chamfer_score_vpq(const __half* __restrict__ query,
                                  const __half* __restrict__ vq_codebook,
                                  const float* __restrict__ pq_codebook,
                                  const uint8_t* __restrict__ pq_codes,
                                  const uint32_t* __restrict__ anchor_labels,
                                  const uint32_t* __restrict__ doc_ids,
                                  const uint32_t* __restrict__ doc_offsets,
                                  uint32_t encoded_row_length,
                                  uint32_t pq_bits,
                                  uint32_t pq_dim,
                                  uint32_t pq_len,
                                  float* __restrict__ chamfer_distances);

extern template __global__ void chamfer_score_anchor<int>(
  const __half*, const __half*, const uint32_t*, const uint32_t*, const uint32_t*, float*);
extern template __global__ void chamfer_score<int>(
  const __half*, const __half*, const uint32_t*, const uint32_t*, float*);
extern template __global__ void chamfer_score_vpq<int>(
  const __half*, const __half*, const float*, const uint8_t*,
  const uint32_t*, const uint32_t*, const uint32_t*,
  uint32_t, uint32_t, uint32_t, uint32_t,
  float*);

} // namespace detail

// ---- High-level launchers ---------------------------------------------------
//
// `tokens_per_query` and `dim` are passed for input-shape validation against
// the kernel's compiled-in tile dimensions. A mismatch throws std::invalid_argument.

void chamfer_score(uint32_t n_docs,
                   uint32_t tokens_per_query,
                   uint32_t dim,
                   const __half* query,
                   const __half* dataset,
                   const uint32_t* doc_ids,
                   const uint32_t* doc_offsets,
                   float* chamfer_distances,
                   cudaStream_t stream = 0);

void chamfer_score_anchor(uint32_t n_docs,
                          uint32_t tokens_per_query,
                          uint32_t dim,
                          const __half* query,
                          const __half* dataset,
                          const uint32_t* doc_ids,
                          const uint32_t* doc_offsets,
                          const uint32_t* doc_token_to_rep,
                          float* chamfer_distances,
                          cudaStream_t stream = 0);

// VPQ chamfer rerank. Reconstructs each doc token as
//   anchor_centroid[anchor_labels[token]] + pq_decode(pq_codes[token])
// using cuVS use_subspaces=true layout. pq_bits must be in [4, 8].
void chamfer_score_vpq(uint32_t n_docs,
                       uint32_t tokens_per_query,
                       uint32_t dim,
                       const __half* query,
                       const __half* vq_codebook,
                       const float* pq_codebook,
                       const uint8_t* pq_codes,
                       const uint32_t* anchor_labels,
                       const uint32_t* doc_ids,
                       const uint32_t* doc_offsets,
                       uint32_t encoded_row_length,
                       uint32_t pq_bits,
                       uint32_t pq_dim,
                       uint32_t pq_len,
                       float* chamfer_distances,
                       cudaStream_t stream = 0);

} // namespace vecflow_chamfer::kernels
