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
    const cuvs::neighbors::filtering::base_filter& sample_filter_ref, \
    const uint32_t* dataset_labels_ptr, \
    const int64_t*  dataset_label_offsets_ptr, \
    const uint32_t* query_labels_second_ptr) \
  { \
    search_filtered_bfs_impl<idx_t, data_t>( \
      res, idx, queries, query_labels, label_size, neighbors, distances, metric, sample_filter_ref, \
      dataset_labels_ptr, dataset_label_offsets_ptr, query_labels_second_ptr); \
  }

instantiate_search_filtered_bfs_d(int64_t, int8_t);

#undef instantiate_search_filtered_bfs_d
