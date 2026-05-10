/*
 * SPDX-License-Identifier: Apache-2.0
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

}  // namespace cuvs::neighbors