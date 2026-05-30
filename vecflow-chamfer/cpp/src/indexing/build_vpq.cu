#include "build_vpq.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

namespace vecflow_chamfer::detail {

namespace {

// Compute fp32 residuals (token - anchor_centroid[anchor_labels[t]]) for a
// contiguous or strided slice of tokens. One block per output token; threads
// within a block stride over `dim`.
__global__ void compute_residuals_kernel(
    const __half* __restrict__ doc_embeddings,    // [n_tokens, dim], host (C2C-accessible)
    const __half* __restrict__ anchor_centroids,  // [n_anchors, dim], device fp16
    const uint32_t* __restrict__ anchor_labels,   // [n_tokens], device
    float* __restrict__ residuals_out,             // [batch_n, dim]
    int64_t token_start,
    int64_t stride,
    int batch_n,
    int dim) {
  const int t = blockIdx.x;
  if (t >= batch_n) return;
  const int64_t global_t = token_start + static_cast<int64_t>(t) * stride;
  const uint32_t a       = anchor_labels[global_t];
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float emb = __half2float(doc_embeddings[global_t * dim + d]);
    const float ctr = __half2float(anchor_centroids[a * dim + d]);
    residuals_out[static_cast<int64_t>(t) * dim + d] = emb - ctr;
  }
}

} // namespace

vpq_data build_vpq(raft::device_resources const& res,
                   dataset const& ds,
                   uint32_t* d_anchor_labels,
                   const __half* d_anchor_centroids,
                   uint32_t pq_bits,
                   uint32_t pq_dim) {
  const int64_t n_tokens = ds.num_doc_tokens;
  const int     dim      = ds.embedding_dim;
  auto stream            = raft::resource::get_cuda_stream(res);

  if (pq_dim == 0 || dim % static_cast<int>(pq_dim) != 0) {
    throw std::invalid_argument(
      "build_vpq: pq_dim (" + std::to_string(pq_dim) + ") must divide embedding_dim (" +
      std::to_string(dim) + ")");
  }
  if (pq_bits < 4 || pq_bits > 16) {
    throw std::invalid_argument("build_vpq: pq_bits must be in [4,16]");
  }

  const uint32_t pq_len = static_cast<uint32_t>(dim) / pq_dim;

  cuvs::preprocessing::quantize::pq::params pq_params{
    pq_bits, pq_dim,
    /*use_subspaces=*/true,
    /*use_vq=*/false,        // anchors already supply the VQ partition
    /*vq_n_centers=*/0,
    /*kmeans_n_iters=*/20};
  const int64_t quantized_dim =
    cuvs::preprocessing::quantize::pq::get_quantized_dim(pq_params);

  // --- Training sample ----------------------------------------------------
  // Cap training-sample tokens to bound the fp32 buffer (4M tokens × 128 dim
  // × 4 B ≈ 2 GiB at our default).
  constexpr int64_t kMaxTrainTokens = 1 << 22;
  const int64_t sample_size = std::min(n_tokens, kMaxTrainTokens);
  const int64_t stride      = std::max<int64_t>(1, n_tokens / sample_size);

  auto train_residuals = raft::make_device_matrix<float, int64_t>(res, sample_size, dim);
  {
    dim3 block(std::min(dim, 256));
    dim3 grid(sample_size);
    compute_residuals_kernel<<<grid, block, 0, stream>>>(
      ds.doc_embeddings, d_anchor_centroids, d_anchor_labels,
      train_residuals.data_handle(),
      /*token_start=*/0, stride, static_cast<int>(sample_size), dim);
    raft::resource::sync_stream(res);
  }

  auto quantizer = cuvs::preprocessing::quantize::pq::build(
    res, pq_params,
    raft::make_device_matrix_view<const float, int64_t>(
      train_residuals.data_handle(), sample_size, dim));

  // Free the training sample before allocating the (potentially much larger)
  // codes matrix.
  train_residuals = raft::make_device_matrix<float, int64_t>(res, 0, dim);

  // --- Encode every token -------------------------------------------------
  auto codes = raft::make_device_matrix<uint8_t, int64_t>(res, n_tokens, quantized_dim);

  constexpr int64_t kEncodeBatchTokens = 1 << 18;  // 256K tokens × dim × 4 B
  auto batch_residuals =
    raft::make_device_matrix<float, int64_t>(res, kEncodeBatchTokens, dim);

  for (int64_t batch_start = 0; batch_start < n_tokens; batch_start += kEncodeBatchTokens) {
    const int64_t batch_n =
      std::min<int64_t>(kEncodeBatchTokens, n_tokens - batch_start);

    {
      dim3 block(std::min(dim, 256));
      dim3 grid(batch_n);
      compute_residuals_kernel<<<grid, block, 0, stream>>>(
        ds.doc_embeddings, d_anchor_centroids, d_anchor_labels,
        batch_residuals.data_handle(),
        batch_start, /*stride=*/1, static_cast<int>(batch_n), dim);
    }

    auto residuals_view = raft::make_device_matrix_view<const float, int64_t>(
      batch_residuals.data_handle(), batch_n, dim);
    auto codes_slice = raft::make_device_matrix_view<uint8_t, int64_t>(
      codes.data_handle() + batch_start * quantized_dim, batch_n, quantized_dim);

    cuvs::preprocessing::quantize::pq::transform(
      res, quantizer, residuals_view, codes_slice, std::nullopt);
  }
  raft::resource::sync_stream(res);

  return vpq_data{
    std::move(quantizer.vpq_codebooks.pq_code_book),
    std::move(codes),
    pq_bits,
    pq_dim,
    pq_len,
  };
}

} // namespace vecflow_chamfer::detail
