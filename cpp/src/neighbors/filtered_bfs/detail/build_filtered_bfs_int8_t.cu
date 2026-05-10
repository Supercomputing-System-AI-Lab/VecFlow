/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/filtered_bfs.hpp>
    
#include "../build_filtered_bfs.cuh"

#define instantiate_build_filtered_bfs_d(idx_t, data_t) \
  void cuvs::neighbors::build_filtered_bfs( \
    raft::resources const& res, \
    cuvs::neighbors::ivf_flat::index<data_t, idx_t>* idx, \
    raft::device_matrix_view<const data_t, int64_t, raft::row_major> dataset, \
    raft::device_vector_view<uint32_t, int64_t> index_map, \
    raft::device_vector_view<uint32_t, int64_t> label_size, \
    raft::device_vector_view<uint32_t, int64_t> label_offset) \
  { \
    build_filtered_bfs_impl<idx_t, data_t>( \
      res, idx, dataset, index_map, label_size, label_offset); \
  }

instantiate_build_filtered_bfs_d(int64_t, int8_t);

#undef instantiate_build_filtered_bfs_d
