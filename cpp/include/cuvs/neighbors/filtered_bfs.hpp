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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors {

void build_filtered_bfs(raft::resources const& res,
                        cuvs::neighbors::ivf_flat::index<float, int64_t>* idx,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                        raft::device_vector_view<uint32_t, int64_t> index_map,
                        raft::device_vector_view<uint32_t, int64_t> label_size,
                        raft::device_vector_view<uint32_t, int64_t> label_offset);

void build_filtered_bfs(raft::resources const& res,
                        cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx,
                        raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                        raft::device_vector_view<uint32_t, int64_t> index_map,
                        raft::device_vector_view<uint32_t, int64_t> label_size,
                        raft::device_vector_view<uint32_t, int64_t> label_offset);

void build_filtered_bfs(raft::resources const& res,
                        cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx,
                        raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                        raft::device_vector_view<uint32_t, int64_t> index_map,
                        raft::device_vector_view<uint32_t, int64_t> label_size,
                        raft::device_vector_view<uint32_t, int64_t> label_offset);

void build_filtered_bfs(raft::resources const& res,
                        cuvs::neighbors::ivf_flat::index<half, int64_t>* idx,
                        raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                        raft::device_vector_view<uint32_t, int64_t> index_map,
                        raft::device_vector_view<uint32_t, int64_t> label_size,
                        raft::device_vector_view<uint32_t, int64_t> label_offset);

void search_filtered_bfs(raft::resources const& res,
                         cuvs::neighbors::ivf_flat::index<float, int64_t>& idx,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                         raft::device_vector_view<uint32_t, int64_t> query_labels,
                         raft::device_vector_view<uint32_t, int64_t> label_size,
                         raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                         cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
                         const cuvs::neighbors::filtering::base_filter& sample_filter =
                            cuvs::neighbors::filtering::none_sample_filter{});

void search_filtered_bfs(raft::resources const& res,
                         cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx,
                         raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
                         raft::device_vector_view<uint32_t, int64_t> query_labels,
                         raft::device_vector_view<uint32_t, int64_t> label_size,
                         raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                         cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
                         const cuvs::neighbors::filtering::base_filter& sample_filter =
                          cuvs::neighbors::filtering::none_sample_filter{});

void search_filtered_bfs(raft::resources const& res,
                         cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx,
                         raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
                         raft::device_vector_view<uint32_t, int64_t> query_labels,
                         raft::device_vector_view<uint32_t, int64_t> label_size,
                         raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                         cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
                         const cuvs::neighbors::filtering::base_filter& sample_filter =
                          cuvs::neighbors::filtering::none_sample_filter{});

void search_filtered_bfs(raft::resources const& res,
                         cuvs::neighbors::ivf_flat::index<half, int64_t>& idx,
                         raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
                         raft::device_vector_view<uint32_t, int64_t> query_labels,
                         raft::device_vector_view<uint32_t, int64_t> label_size,
                         raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                         cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
                         const cuvs::neighbors::filtering::base_filter& sample_filter =
                          cuvs::neighbors::filtering::none_sample_filter{});

}  // namespace cuvs::neighbors