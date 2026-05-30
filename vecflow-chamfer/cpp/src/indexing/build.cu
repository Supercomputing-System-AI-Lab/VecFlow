#include <vecflow_chamfer/build.hpp>
#include "recompute_anchors.hpp"
#include "doc_id_mapping.hpp"
#include "build_vpq.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/neighbors/cagra.hpp>

namespace vecflow_chamfer {

namespace {

// GPU gather kernel: collect n_anchors rows from `src` into `dst`. One coalesced
// kernel launch in place of n_anchors host-issued cudaMemcpyAsync calls.
__global__ void gather_anchor_rows_kernel(const __half* __restrict__ src,
                                          const uint32_t* __restrict__ ids,
                                          __half* __restrict__ dst,
                                          int n_anchors,
                                          int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_anchors) return;
  int64_t src_row = static_cast<int64_t>(ids[i]) * dim;
  int64_t dst_row = static_cast<int64_t>(i) * dim;
  for (int d = 0; d < dim; ++d) {
    dst[dst_row + d] = src[src_row + d];
  }
}

} // namespace

index build(raft::device_resources const& res,
            dataset const& ds,
            index_params const& params) {
  const int64_t n_tokens = ds.num_doc_tokens;
  const int64_t dim      = ds.embedding_dim;
  auto stream            = raft::resource::get_cuda_stream(res);

  std::vector<uint32_t> all_indices(n_tokens);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::mt19937 g(42);
  std::shuffle(all_indices.begin(), all_indices.end(), g);

  std::vector<uint32_t> anchor_ids(params.n_anchors);
  std::copy(all_indices.begin(), all_indices.begin() + params.n_anchors, anchor_ids.begin());
  std::sort(anchor_ids.begin(), anchor_ids.end());

  auto anchor_embeddings = raft::make_device_matrix<half, int64_t>(res, params.n_anchors, dim);
  {
    auto d_anchor_ids = raft::make_device_vector<uint32_t, int64_t>(res, params.n_anchors);
    raft::copy(d_anchor_ids.data_handle(), anchor_ids.data(), params.n_anchors, stream);
    raft::resource::sync_stream(res);

    dim3 block(256);
    dim3 grid((params.n_anchors + block.x - 1) / block.x);
    gather_anchor_rows_kernel<<<grid, block, 0, stream>>>(
      ds.doc_embeddings,
      d_anchor_ids.data_handle(),
      anchor_embeddings.data_handle(),
      params.n_anchors,
      static_cast<int>(dim));
    raft::resource::sync_stream(res);
  }

  cuvs::neighbors::cagra::index_params cagra_index_params;
  cagra_index_params.intermediate_graph_degree = params.graph_degree * 2;
  cagra_index_params.graph_degree              = params.graph_degree;
  cagra_index_params.metric                    = cuvs::distance::DistanceType::InnerProduct;

  auto anchor_cagra_index = cuvs::neighbors::cagra::build(
    res, cagra_index_params, raft::make_const_mdspan(anchor_embeddings.view()));
  auto anchor_labels = raft::make_device_vector<uint32_t, int64_t>(res, n_tokens);

  cuvs::neighbors::cagra::search_params cagra_search_params;
  cagra_search_params.itopk_size = params.itopk_size;
  cagra_search_params.algo       = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;

  const int64_t batch_size = 10000;
  const int64_t n_batches  = (n_tokens + batch_size - 1) / batch_size;

  for (int iter = 0; iter < params.n_iter; iter++) {
    for (int64_t batch = 0; batch < n_batches; batch++) {
      const int64_t start_idx          = batch * batch_size;
      const int64_t end_idx            = std::min(start_idx + batch_size, n_tokens);
      const int64_t current_batch_size = end_idx - start_idx;

      auto query_batch = raft::make_device_matrix<half, int64_t>(res, current_batch_size, dim);
      raft::copy(query_batch.data_handle(),
                 ds.doc_embeddings + start_idx * dim,
                 current_batch_size * dim, stream);

      auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, current_batch_size, 1);
      auto distances = raft::make_device_matrix<float,    int64_t>(res, current_batch_size, 1);

      cuvs::neighbors::cagra::search(res, cagra_search_params, anchor_cagra_index,
                                     raft::make_const_mdspan(query_batch.view()),
                                     neighbors.view(), distances.view());
      raft::resource::sync_stream(res);

      raft::copy(anchor_labels.data_handle() + start_idx,
                 neighbors.data_handle(),
                 current_batch_size, stream);

      if (params.on_progress) {
        // Reserve the upper half of the progress range for the batch sweep;
        // anchor refinement + CAGRA rebuild fill the remaining half once the
        // sweep completes. Avoids reporting 100% before the iter is actually
        // done.
        const double sweep_frac =
          static_cast<double>(batch + 1) / static_cast<double>(n_batches);
        params.on_progress(iter, params.n_iter, 0.5 * sweep_frac);
      }
    }

    detail::recompute_anchors<half>(res, ds.doc_embeddings, anchor_labels.view(),
                                    anchor_embeddings.view(), n_tokens, dim, params.n_anchors);
    anchor_cagra_index = cuvs::neighbors::cagra::build(
      res, cagra_index_params, raft::make_const_mdspan(anchor_embeddings.view()));
    raft::resource::sync_stream(res);

    if (params.on_progress) params.on_progress(iter, params.n_iter, 1.0);
  }

  std::vector<uint32_t> h_anchor_to_doc_ids;
  std::vector<uint32_t> h_anchor_to_doc_ids_offsets;

  auto doc_token_to_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, n_tokens);
  raft::copy(doc_token_to_doc_ids.data_handle(),
             ds.doc_token_to_doc.data(),
             n_tokens, stream);

  detail::anchor_to_doc_id_mapping(res,
                                   anchor_labels.view(),
                                   doc_token_to_doc_ids.view(),
                                   n_tokens, params.n_anchors,
                                   h_anchor_to_doc_ids,
                                   h_anchor_to_doc_ids_offsets);

  auto d_anchor_to_doc_ids =
    raft::make_device_vector<uint32_t, int64_t>(res, h_anchor_to_doc_ids.size());
  raft::copy(d_anchor_to_doc_ids.data_handle(), h_anchor_to_doc_ids.data(),
             h_anchor_to_doc_ids.size(), stream);

  auto d_anchor_to_doc_ids_offsets =
    raft::make_device_vector<uint32_t, int64_t>(res, h_anchor_to_doc_ids_offsets.size());
  raft::copy(d_anchor_to_doc_ids_offsets.data_handle(), h_anchor_to_doc_ids_offsets.data(),
             h_anchor_to_doc_ids_offsets.size(), stream);

  auto d_doc_offsets = raft::make_device_vector<uint32_t, int64_t>(res, ds.doc_offsets.size());
  raft::copy(d_doc_offsets.data_handle(), ds.doc_offsets.data(),
             ds.doc_offsets.size(), stream);
  raft::resource::sync_stream(res);

  // Optional VPQ side-encode. Train PQ on residuals against the *final*
  // anchor centroids; reuse anchor_labels as the VQ partition. anchor_labels
  // are slightly stale (last assignment was prior to the final recompute +
  // rebuild) but the PQ codebook absorbs that error, and the kernel reads
  // both back together at search time.
  std::optional<vpq_data> vpq;
  if (params.pq_bits > 0) {
    vpq = detail::build_vpq(res, ds,
                    anchor_labels.data_handle(),
                    anchor_cagra_index.dataset().data_handle(),
                    params.pq_bits, params.pq_dim);
  }

  return index{
    std::move(anchor_cagra_index),
    std::move(anchor_labels),
    std::move(d_anchor_to_doc_ids),
    std::move(d_anchor_to_doc_ids_offsets),
    std::move(d_doc_offsets),
    std::move(vpq),
  };
}

} // namespace vecflow_chamfer
