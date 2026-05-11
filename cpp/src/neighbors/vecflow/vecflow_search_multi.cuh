/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <cuvs/neighbors/vecflow.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

#include "vecflow_common.cuh"

namespace cuvs::neighbors::vecflow {

namespace detail {

template <typename data_t>
void search_multi_labels(shared_resources::configured_raft_resources& res,
                         cuvs::neighbors::vecflow::index<data_t>&     index,
                         raft::device_matrix_view<const data_t, int64_t> queries,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_a,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_b,
                         int itopk_size,
                         raft::device_matrix_view<uint32_t, int64_t> neighbors,
                         raft::device_matrix_view<float, int64_t>    distances)
{
  // Contract: the index must have been built with `multi_label=true`.
  RAFT_EXPECTS(index.dataset_labels.size() > 0 && index.dataset_label_offsets.size() > 0,
               "vecflow::search_multi_labels requires an index built with multi_label=true; "
               "the dataset_labels CSR is empty. Rebuild with `multi_label=true`.");

  const int64_t n_queries = queries.extent(0);
  const int64_t topk      = neighbors.extent(1);
  auto stream             = raft::resource::get_cuda_stream(res);

  // ── 1. Pick primary/secondary per query by label_freq ───────────────────────
  // Primary = the rarer label (smaller label_freq). Goes into `classify_queries`
  // to pick the per-label sub-index, so traversal happens over the smaller
  // candidate set. The more common SECONDARY is passed into the kernel as the
  // inline AND post-filter.
  auto primary_labels   = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto secondary_labels = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  {
    constexpr int kBlock = 256;
    const int kGrid      = (n_queries + kBlock - 1) / kBlock;
    pick_primary_by_label_freq_kernel<<<kGrid, kBlock, 0, stream>>>(
      query_labels_a.data_handle(),
      query_labels_b.data_handle(),
      index.label_freq.data_handle(),
      primary_labels.data_handle(),
      secondary_labels.data_handle(),
      static_cast<int>(n_queries));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // ── 2. Configure CAGRA search params (SINGLE_CTA_FILTERED path) ───────────
  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo       = cagra::search_algo::SINGLE_CTA_FILTERED;
  search_params.itopk_size = itopk_size;

  // ── 3. Partition queries by primary specificity (reuse classify_queries) ──
  auto query_info = classify_queries<data_t>(
    res,
    queries,
    raft::make_device_vector_view<uint32_t, int64_t>(primary_labels.data_handle(), n_queries),
    index.label_freq.view(),
    index.specificity_threshold);

  const int n_cagra_queries = query_info.cagra_query_map.size();
  const int n_bfs_queries   = query_info.bfs_query_map.size();

  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float,    int64_t>(res, n_cagra_queries, topk);
  auto bfs_neighbors   = raft::make_device_matrix<int64_t,  int64_t>(res, n_bfs_queries,   topk);
  auto bfs_distances   = raft::make_device_matrix<float,    int64_t>(res, n_bfs_queries,   topk);
  raft::resource::sync_stream(res);

  // ── 4. Gather secondary labels in each lane's partition order ─────────────
  auto cagra_secondary_labels =
    raft::make_device_vector<uint32_t, int64_t>(res, n_cagra_queries);
  auto bfs_secondary_labels =
    raft::make_device_vector<uint32_t, int64_t>(res, n_bfs_queries);

  if (n_cagra_queries > 0) {
    constexpr int kBlock = 256;
    const int kGrid      = (n_cagra_queries + kBlock - 1) / kBlock;
    gather_by_map_kernel<<<kGrid, kBlock, 0, stream>>>(
      secondary_labels.data_handle(),
      query_info.cagra_query_map.data_handle(),
      cagra_secondary_labels.data_handle(),
      n_cagra_queries);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  if (n_bfs_queries > 0) {
    constexpr int kBlock = 256;
    const int kGrid      = (n_bfs_queries + kBlock - 1) / kBlock;
    gather_by_map_kernel<<<kGrid, kBlock, 0, stream>>>(
      secondary_labels.data_handle(),
      query_info.bfs_query_map.data_handle(),
      bfs_secondary_labels.data_handle(),
      n_bfs_queries);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // ── 5. CAGRA lane: pass dataset_labels / offsets / secondary to inline AND ─
  if (n_cagra_queries > 0) {
    cagra::filtered_search(res,
                           search_params,
                           index.ivf_graph_index,
                           query_info.cagra_queries.view(),
                           cagra_neighbors.view(),
                           cagra_distances.view(),
                           query_info.cagra_query_labels.view(),
                           index.graph_index_map.view(),
                           index.graph_label_size.view(),
                           index.graph_label_offset.view(),
                           cuvs::neighbors::filtering::none_sample_filter{},
                           index.dataset_labels.data_handle(),
                           index.dataset_label_offsets.data_handle(),
                           cagra_secondary_labels.data_handle());
    raft::resource::sync_stream(res);
  }

  // ── 6. BFS lane: same set of optional buffers ─────────────────────────────
  if (n_bfs_queries > 0) {
    search_filtered_bfs(res,
                        index.ivf_bfs_index,
                        raft::make_const_mdspan(query_info.bfs_queries.view()),
                        query_info.bfs_query_labels.view(),
                        index.bfs_label_size.view(),
                        bfs_neighbors.view(),
                        bfs_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded,
                        cuvs::neighbors::filtering::none_sample_filter{},
                        index.dataset_labels.data_handle(),
                        index.dataset_label_offsets.data_handle(),
                        bfs_secondary_labels.data_handle());
    raft::resource::sync_stream(res);
  }

  // ── 7. Merge results back into the caller's output buffers ────────────────
  merge_search_results<data_t>(res,
                               neighbors,
                               distances,
                               query_info,
                               bfs_neighbors.view(),
                               bfs_distances.view(),
                               cagra_neighbors.view(),
                               cagra_distances.view(),
                               topk);
}

}  // namespace detail

template <typename data_t>
void search_multi_labels(shared_resources::configured_raft_resources& res,
                         cuvs::neighbors::vecflow::index<data_t>&     index,
                         raft::device_matrix_view<const data_t, int64_t> queries,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_a,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_b,
                         int itopk_size,
                         raft::device_matrix_view<uint32_t, int64_t> neighbors,
                         raft::device_matrix_view<float, int64_t>    distances)
{
  cuvs::neighbors::vecflow::detail::search_multi_labels<data_t>(
    res, index, queries, query_labels_a, query_labels_b, itopk_size, neighbors, distances);
}

}  // namespace cuvs::neighbors::vecflow
