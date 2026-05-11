/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <cstdint>
#include <vector>

#include "vecflow_common.cuh"

namespace cuvs::neighbors::vecflow {

namespace detail {
template <typename data_t>
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<data_t>& index,
            raft::device_matrix_view<const data_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances) {

  // Configure standard search parameters
  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;
  search_params.itopk_size = itopk_size;

  auto query_info = classify_queries<data_t>(res,
                                             queries,
                                             query_labels,
                                             index.label_freq.view(),
                                             index.specificity_threshold);

  int64_t topk = neighbors.extent(1);
  int n_cagra_queries = query_info.cagra_query_map.size();
  int n_bfs_queries = query_info.bfs_query_map.size();
  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(res, n_cagra_queries, topk);
  auto bfs_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_bfs_queries, topk);
  auto bfs_distances = raft::make_device_matrix<float, int64_t>(res, n_bfs_queries, topk);
  raft::resource::sync_stream(res);

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
                           index.graph_label_offset.view());
    raft::resource::sync_stream(res);
  }

  if (n_bfs_queries > 0) {
    search_filtered_bfs(res,
                        index.ivf_bfs_index,
                        raft::make_const_mdspan(query_info.bfs_queries.view()),
                        query_info.bfs_query_labels.view(),
                        index.bfs_label_size.view(),
                        bfs_neighbors.view(),
                        bfs_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(res);
  }

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
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<data_t>& index,
            raft::device_matrix_view<const data_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances)
{
  cuvs::neighbors::vecflow::detail::search<data_t>(
    res, index, queries, query_labels, itopk_size, neighbors, distances);
}

}  // namespace cuvs::neighbors::vecflow
