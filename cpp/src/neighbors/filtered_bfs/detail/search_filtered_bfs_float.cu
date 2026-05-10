/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/common.hpp>

#include "../search_filtered_bfs.cuh"

#define instantiate_search_filtered_bfs_d(idx_t, data_t) \
  void cuvs::neighbors::search_filtered_bfs( \
    raft::resources const& res, \
    cuvs::neighbors::ivf_flat::index<data_t, idx_t>& idx, \
    raft::device_matrix_view<const data_t, int64_t, raft::row_major> queries, \
    raft::device_vector_view<uint32_t, int64_t> query_labels, \
    raft::device_vector_view<uint32_t, int64_t> label_size, \
    raft::device_matrix_view<idx_t, int64_t, raft::row_major> neighbors, \
    raft::device_matrix_view<float, int64_t, raft::row_major> distances, \
    cuvs::distance::DistanceType metric, \
    const cuvs::neighbors::filtering::base_filter& sample_filter_ref) \
  { \
    search_filtered_bfs_impl<idx_t, data_t>( \
      res, idx, queries, query_labels, label_size, neighbors, distances, metric, sample_filter_ref); \
  }

instantiate_search_filtered_bfs_d(int64_t, float);

#undef instantiate_search_filtered_bfs_d
