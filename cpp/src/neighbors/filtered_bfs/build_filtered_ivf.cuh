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
#include "../ivf_flat/ivf_flat_build.cuh"
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

template <typename idx_t, typename data_t>
void build_filtered_IVF_index_core(
  raft::resources const& handle,
  cuvs::neighbors::ivf_flat::index<data_t, idx_t>* idx,
  raft::device_matrix_view<const data_t, int64_t, raft::row_major> dataset,
  raft::device_vector_view<uint32_t, int64_t> index_map,
  raft::device_vector_view<uint32_t, int64_t> label_size,
  raft::device_vector_view<uint32_t, int64_t> label_offset)
{
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded;
  
  uint32_t n_lists = label_size.size();
  uint32_t dim = dataset.extent(1);
  
  *idx = cuvs::neighbors::ivf_flat::index<data_t, idx_t>(
    handle, cuvs::distance::DistanceType(metric), n_lists, false, true, dim);

  cuvs::neighbors::ivf_flat::detail::fill_variable_size_index(handle,
                                                              idx,  // Take address of reference
                                                              dataset,
                                                              index_map,
                                                              label_size,
                                                              label_offset);
}

}  // namespace detail


template <typename idx_t, typename data_t>
void build_filtered_IVF_index_impl(
  raft::resources const& handle,
  cuvs::neighbors::ivf_flat::index<data_t, idx_t>* idx,
  raft::device_matrix_view<const data_t, int64_t, raft::row_major> dataset,
  raft::device_vector_view<uint32_t, int64_t> index_map,
  raft::device_vector_view<uint32_t, int64_t> label_size,
  raft::device_vector_view<uint32_t, int64_t> label_offset)
{
  detail::build_filtered_IVF_index_core(handle, idx, dataset, index_map, label_size, label_offset);
}
}  // namespace cuvs::neighbors