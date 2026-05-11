/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <neighbors/detail/cagra/compute_distance-ext.cuh>

#include <cuvs/neighbors/cagra.hpp>

#include <optional>

namespace cuvs::neighbors::cagra::detail::filtered_single_cta_search {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
  uintptr_t topk_indices_ptr,     // [num_queries, topk]
  DistanceT* topk_distances_ptr,  // [num_queries, topk]
  const DataT* queries_ptr,       // [num_queries, dataset_dim]
  uint32_t num_queries,
  const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  // Per-label graph-slice routing (filtered search).
  const uint32_t* query_labels_ptr,    // [num_queries]
  const uint32_t* index_map_ptr,       // [graph size]
  const uint32_t* label_size_ptr,      // [num_labels]
  const uint32_t* label_offset_ptr,    // [num_labels]
  // Optional multi-label AND inputs — nullptr disables AND mode.
  const uint32_t* dataset_labels_ptr,
  const int64_t*  dataset_label_offsets_ptr,
  const uint32_t* query_labels_second_ptr,
  cudaStream_t stream);

}
