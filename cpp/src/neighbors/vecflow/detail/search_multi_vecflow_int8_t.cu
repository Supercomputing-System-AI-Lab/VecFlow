/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/shared_resources.hpp>
#include <cuvs/neighbors/vecflow.hpp>

#include "../vecflow_search_multi.cuh"

namespace cuvs::neighbors::vecflow {

#define instantiate_search_multi_vecflow_d(data_t)                                       \
  void search_multi_labels(                                                              \
    shared_resources::configured_raft_resources& res,                                    \
    cuvs::neighbors::vecflow::index<data_t>& index,                                      \
    raft::device_matrix_view<const data_t, int64_t> queries,                             \
    raft::device_vector_view<const uint32_t, int64_t> query_labels_a,                    \
    raft::device_vector_view<const uint32_t, int64_t> query_labels_b,                    \
    int itopk_size,                                                                      \
    raft::device_matrix_view<uint32_t, int64_t> neighbors,                               \
    raft::device_matrix_view<float, int64_t> distances)                                  \
  {                                                                                      \
    cuvs::neighbors::vecflow::search_multi_labels<data_t>(                               \
      res, index, queries, query_labels_a, query_labels_b, itopk_size, neighbors, distances); \
  }

instantiate_search_multi_vecflow_d(int8_t);
#undef instantiate_search_multi_vecflow_d

}  // namespace cuvs::neighbors::vecflow
