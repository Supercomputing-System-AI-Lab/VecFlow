/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include "../vecflow_search.cuh"

namespace cuvs::neighbors::vecflow {

#define instantiate_search_vecflow_d(data_t) \
 void search( \
	 shared_resources::configured_raft_resources& res, \
	 cuvs::neighbors::vecflow::index<data_t>& index, \
	 raft::device_matrix_view<const data_t, int64_t> queries, \
	 raft::device_vector_view<uint32_t, int64_t> query_labels, \
	 int itopk_size, \
	 raft::device_matrix_view<uint32_t, int64_t> neighbors, \
	 raft::device_matrix_view<float, int64_t> distances) \
 { \
    cuvs::neighbors::vecflow::search<data_t>( \
      res, index, queries, query_labels, itopk_size, neighbors, distances); \
  }

instantiate_search_vecflow_d(float);
#undef instantiate_search_vecflow_d

}  // namespace cuvs::neighbors::vecflow