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

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/device_vector.hpp>

#include <omp.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <fstream>
#include <vector>

#include "vecflow_common.cuh"


namespace cuvs::neighbors::vecflow {

namespace detail {
template<typename data_t>
void search(
	cuvs::neighbors::vecflow::index<data_t>& idx,
	raft::device_matrix_view<const float, int64_t> queries,
	raft::device_vector_view<uint32_t, int64_t> query_labels,
	int itopk_size,
	raft::device_matrix_view<uint32_t, int64_t> neighbors,
	raft::device_matrix_view<float, int64_t> distances) 
{
	
	// Configure standard search parameters
  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;
	search_params.itopk_size = itopk_size;

  auto query_info = classify_queries<data_t>(idx.res,
																						queries,
																						query_labels,
																						idx.metadata.cat_freq.view(),
																						idx.specificity_threshold);

  int64_t topk = neighbors.extent(1);
  int n_cagra_queries = query_info.cagra_query_map.size();
  int n_bfs_queries = query_info.bfs_query_map.size();
  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(idx.res, n_cagra_queries, topk);
  auto bfs_neighbors = raft::make_device_matrix<int64_t, int64_t>(idx.res, n_bfs_queries, topk);
  auto bfs_distances = raft::make_device_matrix<float, int64_t>(idx.res, n_bfs_queries, topk);
  raft::resource::sync_stream(idx.res);

  if (n_cagra_queries > 0) {
    cagra::filtered_search(idx.res, 
                          search_params,
                          idx.cagra_index,
                          query_info.cagra_queries.view(),
                          cagra_neighbors.view(),
                          cagra_distances.view(),
                          query_info.cagra_query_labels.view(),
                          idx.metadata.cagra_index_map.view(),
                          idx.metadata.cagra_label_size.view(),
                          idx.metadata.cagra_label_offset.view());
    raft::resource::sync_stream(idx.res);
  }

  if (n_bfs_queries > 0) {
    search_filtered_ivf(idx.res,
												idx.bfs_index,
												raft::make_const_mdspan(query_info.bfs_queries.view()),
												query_info.bfs_query_labels.view(),
												idx.metadata.bfs_label_size.view(),
												bfs_neighbors.view(),
												bfs_distances.view(),
												cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(idx.res);
  }
  
  merge_search_results<data_t>(idx.res,
															neighbors,
															distances,
															query_info,
															bfs_neighbors.view(),
															bfs_distances.view(),
															cagra_neighbors.view(),
															cagra_distances.view(),
															topk);
}

} // namespace detail

template<typename data_t>
void search(
  cuvs::neighbors::vecflow::index<data_t>& idx,
  raft::device_matrix_view<const data_t, int64_t> queries,
  raft::device_vector_view<uint32_t, int64_t> query_labels,
  int itopk_size,
  raft::device_matrix_view<uint32_t, int64_t> neighbors,
  raft::device_matrix_view<float, int64_t> distances)
{
  cuvs::neighbors::vecflow::detail::search<data_t>(
    idx, queries, query_labels, itopk_size, neighbors, distances);
}

} // namespace cuvs::neighbors::vecflow
