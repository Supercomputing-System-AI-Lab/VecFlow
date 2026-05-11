/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include "../vecflow_build.cuh"

namespace cuvs::neighbors::vecflow {

#define instantiate_build_vecflow_d(data_t) \
 auto build( \
    shared_resources::configured_raft_resources& res, \
    raft::device_matrix_view<const data_t, int64_t> dataset, \
    const std::vector<std::vector<int>>& data_label_vecs, \
    int graph_degree, \
    int specificity_threshold, \
    const std::string& graph_fname, \
    const std::string& bfs_fname, \
    bool force_rebuild, \
    bool multi_label) -> cuvs::neighbors::vecflow::index<data_t> \
  { \
    return cuvs::neighbors::vecflow::build<data_t>( \
      res, dataset, data_label_vecs, graph_degree, specificity_threshold, \
      graph_fname, bfs_fname, force_rebuild, multi_label); \
  }

instantiate_build_vecflow_d(int8_t);
#undef instantiate_build_vecflow_d

}  // namespace cuvs::neighbors::vecflow