/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
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
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::vecflow {

/**
 * Structure to hold metadata for the indexes.
 */
struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> cagra_index_map;
  raft::device_vector<uint32_t, int64_t> cagra_label_size;
  raft::device_vector<uint32_t, int64_t> cagra_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  raft::device_vector<uint32_t, int64_t> cat_freq;

  CombinedIndices(raft::resources& res)
    : cagra_index_map(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cagra_label_size(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cagra_label_offset(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      bfs_label_size(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cat_freq(raft::make_device_vector<uint32_t, int64_t>(res, 0))
  {}
  // Overload constructor for aggregate return.
  CombinedIndices(raft::device_vector<uint32_t, int64_t>&& c_idx_map,
                  raft::device_vector<uint32_t, int64_t>&& c_label_size,
                  raft::device_vector<uint32_t, int64_t>&& c_label_offset,
                  raft::device_vector<uint32_t, int64_t>&& b_label_size,
                  raft::device_vector<uint32_t, int64_t>&& cat_f)
    : cagra_index_map(std::move(c_idx_map)),
      cagra_label_size(std::move(c_label_size)),
      cagra_label_offset(std::move(c_label_offset)),
      bfs_label_size(std::move(b_label_size)),
      cat_freq(std::move(cat_f))
  {}
};

/**
 * @brief The vecflow index holds all internal information required for search.
 *
 * This includes the CAGRA index, the IVF-BFS index, and metadata such as label sizes,
 * offsets, and the mapping of data points for each label.
 *
 * The index also contains a shared resource handle that is initialized when the index is created.
 */

template <typename data_t>
struct index {
  // Shared RAFT resources provided by our shared_resources module.
  shared_resources::configured_raft_resources res;
  // The CAGRA graph index.
  cuvs::neighbors::cagra::index<data_t, uint32_t> cagra_index;
  // The IVF-BFS index.
  cuvs::neighbors::ivf_flat::index<data_t, int64_t> bfs_index;
  // Combined metadata information.
  CombinedIndices metadata;
  // Specificity threshold (used later during search).
  int specificity_threshold;

  // Constructor: Initialize the shared resource and the contained indexes.
  index()
    : res{}, 
      cagra_index(res),
      bfs_index(res, cuvs::distance::DistanceType::L2Unexpanded, 0, false, true, 0),
      metadata(res),
      specificity_threshold(0)
  {}
};

/**
 * @brief Builds (or loads) the vecflow index.
 *
 * This function builds (or loads from file) both the CAGRA graph index and the IVF-BFS index.
 *
 * @param idx                   Vecflow index to update.
 * @param d_dataset             Device matrix view of the dataset.
 * @param data_label_fname      File name to load data labels.
 * @param graph_degree          Desired graph degree.
 * @param specificity_threshold Threshold to decide which labels go to CAGRA vs. BFS.
 * @param graph_fname           (Optional) File name to load/save the CAGRA index.
 * @param bfs_fname             (Optional) File name to load/save the BFS index.
 */
void build(cuvs::neighbors::vecflow::index<float>& idx,
           raft::device_matrix_view<const float, int64_t> d_dataset,
           const std::string& data_label_fname,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "");

/**
 * @brief Performs a vecflow search.
 *
 * Given a set of queries and query labels, this function searches for the top-k nearest
 * neighbors using the vecflow index.
 *
 * @param idx           The vecflow index to use.
 * @param queries       Device matrix view of query vectors.
 * @param query_labels  Device vector view of query labels.
 * @param itopk_size    Number of top results to return.
 * @param neighbors     [out] Device matrix view to hold neighbor indices.
 * @param distances     [out] Device matrix view to hold distances.
 */
void search(cuvs::neighbors::vecflow::index<float>& idx,
            raft::device_matrix_view<const float, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

} // namespace cuvs::neighbors::vecflow
