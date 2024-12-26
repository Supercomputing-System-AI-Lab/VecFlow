/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../core/nvtx.hpp"
#include "../ivf_flat/ivf_flat_interleaved_scan.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <thrust/sequence.h>

namespace cuvs::neighbors {

namespace detail {

template <typename idx_t, typename data_t, typename FilterT>
void search_filtered_ivf_core(
  raft::resources const& handle,
  cuvs::neighbors::ivf_flat::index<data_t, idx_t>& idx,
  raft::device_matrix_view<const data_t, int64_t, raft::row_major> queries,
  raft::device_vector_view<uint32_t, int64_t> query_labels,
  raft::device_vector_view<uint32_t, int64_t> label_size,
  raft::device_matrix_view<idx_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::distance::DistanceType metric,
  FilterT sample_filter)
{
  int64_t  n_queries = queries.extent(0);
  uint32_t k         = static_cast<uint32_t>(neighbors.extent(1));

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<uint32_t> chunk_index(n_queries, raft::resource::get_cuda_stream(handle));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(neighbors.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                raft::resource::get_cuda_stream(handle));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  // we know that each cluster has exactly n_candidates entries
  auto chunk_fill = [label_size = label_size.data_handle(), 
                    query_labels = query_labels.data_handle(),
                    chunk_data = chunk_index.data()] __device__(int i) {
    uint32_t label = query_labels[i];
    chunk_data[i] = label_size[label];
  };

  thrust::for_each(raft::resource::get_thrust_policy(handle),
                  thrust::counting_iterator<int>(0),
                  thrust::counting_iterator<int>(n_queries),
                  chunk_fill);

  uint32_t grid_dim_x = 1;

  cuvs::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<
    data_t,
    typename cuvs::spatial::knn::detail::utils::config<data_t>::value_t,
    idx_t>(idx,
           queries.data_handle(),
           query_labels.data_handle(),
           static_cast<uint32_t>(n_queries),
           0,
           cuvs::distance::DistanceType(idx.metric()),
           1,
           k,
           0,
           chunk_index.data(),
           cuvs::distance::is_min_close(cuvs::distance::DistanceType(metric)),
           sample_filter,
           neighbors_uint32,
           distances.data_handle(),
           grid_dim_x,
           raft::resource::get_cuda_stream(handle));
 
  cuvs::neighbors::ivf::detail::postprocess_neighbors(neighbors.data_handle(),
                                                      neighbors_uint32,
                                                      idx.inds_ptrs().data_handle(),
                                                      query_labels.data_handle(),
                                                      chunk_index.data(),
                                                      n_queries,
                                                      1,
                                                      k,
                                                      raft::resource::get_cuda_stream(handle));
}
}  // namespace detail

template <typename idx_t, typename data_t>
void search_filtered_ivf_impl(
  raft::resources const& handle,
  cuvs::neighbors::ivf_flat::index<data_t, idx_t>& idx,
  raft::device_matrix_view<const data_t, int64_t, raft::row_major> queries,
  raft::device_vector_view<uint32_t, int64_t> query_labels,
  raft::device_vector_view<uint32_t, int64_t> label_size,
  raft::device_matrix_view<idx_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::distance::DistanceType metric,
  const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    using none_filter_type = cuvs::neighbors::filtering::none_sample_filter;
    auto& sample_filter = dynamic_cast<const none_filter_type&>(sample_filter_ref);
    auto sample_filter_copy = sample_filter;
    detail::search_filtered_ivf_core(
      handle, idx, queries, query_labels, label_size, neighbors, distances, metric, sample_filter_copy);
    return;
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter = 
      dynamic_cast<const cuvs::neighbors::filtering::cagra_filter&>(
        sample_filter_ref);
    auto sample_filter_copy = sample_filter;
    detail::search_filtered_ivf_core(
      handle, idx, queries, query_labels, label_size, neighbors, distances, metric, sample_filter_copy);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}
}  // namespace cuvs::neighbors