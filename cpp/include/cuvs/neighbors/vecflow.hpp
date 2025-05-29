/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::vecflow {

/**
 * @brief The vecflow index holds all internal information required for search.
 *
 * This includes the IVF-graph index, the IVF-BFS index, and metadata such as label sizes,
 * offsets, and the mapping of data points for each label.
 *
 */
template <typename data_t>
struct index {
  cuvs::neighbors::cagra::index<data_t, uint32_t> ivf_graph_index;
  cuvs::neighbors::ivf_flat::index<data_t, int64_t> ivf_bfs_index;

  int specificity_threshold = 2000;
  
  raft::device_vector<uint32_t, int64_t> cagra_index_map;
  raft::device_vector<uint32_t, int64_t> cagra_label_size;
  raft::device_vector<uint32_t, int64_t> cagra_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  raft::device_vector<uint32_t, int64_t> cat_freq;
};

/**
 * @brief Builds (or loads) the vecflow index.
 *
 * This function builds (or loads from file) both the IVF-graph index and the IVF-BFS index.
 *
 * @param res                   RAFT shared resources.
 * @param d_dataset             Device matrix view of the dataset.
 * @param data_label_vecs       Vector of vectors of data labels.
 * @param graph_degree          Desired graph degree.
 * @param specificity_threshold Threshold to decide which labels go to CAGRA vs. BFS.
 * @param graph_fname           (Optional) File name to load/save the IVF-graph index.
 * @param bfs_fname             (Optional) File name to load/save the BFS index.
 * @param force_rebuild         (Optional) Whether to force rebuild the index.
 */
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const float, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false) -> cuvs::neighbors::vecflow::index<float>;

auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const int8_t, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false) -> cuvs::neighbors::vecflow::index<int8_t>;
/**
 * @brief Performs a vecflow search.
 *
 * Given a set of queries and query labels, this function searches for the top-k nearest
 * neighbors using the vecflow index.
 *
 * @param res           RAFT shared resources.
 * @param index         The vecflow index to use.
 * @param queries       Device matrix view of query vectors.
 * @param query_labels  Device vector view of query labels.
 * @param itopk_size    Number of top results to return.
 * @param neighbors     [out] Device matrix view to hold neighbor indices.
 * @param distances     [out] Device matrix view to hold distances.
 */
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<float>& index,
            raft::device_matrix_view<const float, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<int8_t>& index,
            raft::device_matrix_view<const int8_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

} // namespace cuvs::neighbors::vecflow
