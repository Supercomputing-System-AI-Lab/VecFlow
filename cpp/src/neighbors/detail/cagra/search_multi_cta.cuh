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

#include "bitonic.hpp"
#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_multi_cta_kernel.cuh"
#include "search_plan.cuh"
#include "topk_for_cagra/topk.h"  // TODO replace with raft topk if possible
#include "utils.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <raft/linalg/map.cuh>

// TODO: This shouldn't be invoking anything in spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace cuvs::neighbors::cagra::detail {
namespace multi_cta_search {

template <typename DataT, typename IndexT, typename DistanceT, typename SAMPLE_FILTER_T>
struct search : public search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T> {
  using base_type  = search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type ::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  using base_type::algo;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::itopk_size;
  using base_type::max_iterations;
  using base_type::max_queries;
  using base_type::min_iterations;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  using base_type::search_width;
  using base_type::team_size;
  using base_type::thread_block_size;

  using base_type::dim;
  using base_type::graph_degree;
  using base_type::topk;

  using base_type::hash_bitlen;

  using base_type::dataset_size;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;

  using base_type::smem_size;

  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;

  uint32_t num_cta_per_query;
  lightweight_uvector<INDEX_T> intermediate_indices;
  lightweight_uvector<float> intermediate_distances;
  size_t topk_workspace_size;
  lightweight_uvector<uint32_t> topk_workspace;

  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, graph_degree, topk),
      intermediate_indices(res),
      intermediate_distances(res),
      topk_workspace(res)

  {
    set_params(res, params);
  }

  void set_params(raft::resources const& res, const search_params& params)
  {
    constexpr unsigned muti_cta_itopk_size = 32;
    this->itopk_size                       = muti_cta_itopk_size;
    search_width                           = 1;
    num_cta_per_query =
      max(params.search_width, raft::ceildiv(params.itopk_size, (size_t)muti_cta_itopk_size));
    result_buffer_size = itopk_size + search_width * graph_degree;
    typedef raft::Pow2<32> AlignBytes;
    unsigned result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);
    // constexpr unsigned max_result_buffer_size = 256;
    RAFT_EXPECTS(result_buffer_size_32 <= 256, "Result buffer size cannot exceed 256");

    smem_size = dataset_desc.smem_ws_size_in_bytes +
                (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
                sizeof(uint32_t) * search_width + sizeof(uint32_t);
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);

    //
    // Determine the thread block size
    //
    constexpr unsigned min_block_size = 64;
    constexpr unsigned max_block_size = 1024;
    uint32_t block_size               = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      // Increase block size according to shared memory requirements.
      // If block size is 32, upper limit of shared memory size per
      // thread block is set to 4096. This is GPU generation dependent.
      constexpr unsigned ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when total number of
      // CTAs (= num_cta_per_query * max_queries) is small.
      cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
      RAFT_LOG_DEBUG("# multiProcessorCount: %d", deviceProp.multiProcessorCount);
      while ((block_size < max_block_size) &&
             (graph_degree * search_width * team_size >= block_size * 2) &&
             (num_cta_per_query * max_queries <=
              (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    //
    // Allocate memory for intermediate buffer and workspace.
    //
    uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    intermediate_indices.resize(num_intermediate_results * max_queries,
                                raft::resource::get_cuda_stream(res));
    intermediate_distances.resize(num_intermediate_results * max_queries,
                                  raft::resource::get_cuda_stream(res));

    hashmap.resize(hashmap_size, raft::resource::get_cuda_stream(res));

    topk_workspace_size = _cuann_find_topk_bufferSize(
      topk, max_queries, num_intermediate_results, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, raft::resource::get_cuda_stream(res));
  }

  void check(const uint32_t topk) override
  {
    RAFT_EXPECTS(num_cta_per_query * 32 >= topk,
                 "`num_cta_per_query` (%u) * 32 must be equal to or greater than "
                 "`topk` (%u) when 'search_mode' is \"multi-cta\". "
                 "(`num_cta_per_query`=max(`search_width`, ceildiv(`itopk_size`, 32)))",
                 num_cta_per_query,
                 topk);
  }

  ~search() {}
  using base_type::operator();
  void operator()(raft::resources const& res,
                  raft::device_matrix_view<const INDEX_T, int64_t, raft::row_major> graph,
                  INDEX_T* const topk_indices_ptr,       // [num_queries, topk]
                  DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,       // [num_queries, dataset_dim]
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
                  uint32_t* const num_executed_iterations,  // [num_queries,]
                  uint32_t topk,
                  SAMPLE_FILTER_T sample_filter)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(res);
    select_and_run(dataset_desc,
                   graph,
                   intermediate_indices.data(),
                   intermediate_distances.data(),
                   queries_ptr,
                   num_queries,
                   dev_seed_ptr,
                   num_executed_iterations,
                   *this,
                   topk,
                   thread_block_size,
                   result_buffer_size,
                   smem_size,
                   hash_bitlen,
                   hashmap.data(),
                   num_cta_per_query,
                   num_seeds,
                   sample_filter,
                   stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Select the top-k results from the intermediate results
    const uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    _cuann_find_topk(topk,
                     num_queries,
                     num_intermediate_results,
                     intermediate_distances.data(),
                     num_intermediate_results,
                     intermediate_indices.data(),
                     num_intermediate_results,
                     topk_distances_ptr,
                     topk,
                     topk_indices_ptr,
                     topk,
                     topk_workspace.data(),
                     true,
                     NULL,
                     stream);
  }
};

}  // namespace multi_cta_search
}  // namespace cuvs::neighbors::cagra::detail
