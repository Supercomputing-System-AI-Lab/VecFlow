/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_FILTERED_SEARCH(T, IdxT)                                    \
  void filtered_search(raft::resources const& handle,                                       \
              cuvs::neighbors::cagra::search_params const& params,                 \
              const cuvs::neighbors::cagra::index<T, IdxT>& index,                 \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries, \
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,  \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances, \
              raft::device_vector_view<uint32_t, int64_t> query_labels,            \
              raft::device_vector_view<uint32_t, int64_t> index_map,               \
              raft::device_vector_view<uint32_t, int64_t> label_size,              \
              raft::device_vector_view<uint32_t, int64_t> label_offset,            \
              const cuvs::neighbors::filtering::base_filter& sample_filter,        \
              const uint32_t* dataset_labels_ptr,                                  \
              const int64_t*  dataset_label_offsets_ptr,                           \
              const uint32_t* query_labels_second_ptr)                             \
  {                                                                                \
    cuvs::neighbors::cagra::filtered_search<T, IdxT>(                                       \
      handle, params, index, queries, neighbors, distances, query_labels, index_map, label_size, label_offset, sample_filter, \
      dataset_labels_ptr, dataset_label_offsets_ptr, query_labels_second_ptr);     \
  }

CUVS_INST_CAGRA_FILTERED_SEARCH(float, uint32_t);

#undef CUVS_INST_CAGRA_FILTERED_SEARCH

}  // namespace cuvs::neighbors::cagra
