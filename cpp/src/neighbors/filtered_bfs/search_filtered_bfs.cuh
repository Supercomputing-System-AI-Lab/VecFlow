/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../core/nvtx.hpp"
#include "../ivf_common.cuh"                              // ivf::detail::postprocess_neighbors
#include "../ivf_flat/ivf_flat_interleaved_scan_ext.cuh"  // interleaved_scan
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

#include <optional>

namespace cuvs::neighbors {

namespace detail {

template <typename idx_t, typename data_t, typename FilterT>
void search_filtered_bfs_core(raft::resources const& res,
                              cuvs::neighbors::ivf_flat::index<data_t, idx_t>& idx,
                              raft::device_matrix_view<const data_t, int64_t, raft::row_major> queries,
                              raft::device_vector_view<uint32_t, int64_t> query_labels,
                              raft::device_vector_view<uint32_t, int64_t> label_size,
                              raft::device_matrix_view<idx_t, int64_t, raft::row_major> neighbors,
                              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                              cuvs::distance::DistanceType metric,
                              FilterT sample_filter,
                              // Optional multi-label AND inline filter buffers.
                              // All-null leaves the kernel in single-label mode.
                              const uint32_t* dataset_labels_ptr        = nullptr,
                              const int64_t*  dataset_label_offsets_ptr = nullptr,
                              const uint32_t* query_labels_second_ptr   = nullptr) {

  int64_t  n_queries = queries.extent(0);
  uint32_t k         = static_cast<uint32_t>(neighbors.extent(1));

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, raft::resource::get_cuda_stream(res));
  rmm::device_uvector<uint32_t> chunk_index(n_queries, raft::resource::get_cuda_stream(res));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(neighbors.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                raft::resource::get_cuda_stream(res));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  // we know that each cluster has exactly n_candidates entries
  auto chunk_fill = [label_size = label_size.data_handle(), 
                    query_labels = query_labels.data_handle(),
                    chunk_data = chunk_index.data()] __device__(int i) {
    uint32_t label = query_labels[i];
    chunk_data[i] = label_size[label];
  };

  thrust::for_each(raft::resource::get_thrust_policy(res),
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
           raft::resource::get_cuda_stream(res),
           std::nullopt,
           // VecFlow multi-label AND inline filter buffers (nullptr ⇒ off).
           dataset_labels_ptr,
           dataset_label_offsets_ptr,
           query_labels_second_ptr);
 
  cuvs::neighbors::ivf::detail::postprocess_neighbors(neighbors.data_handle(),
                                                      neighbors_uint32,
                                                      idx.inds_ptrs().data_handle(),
                                                      query_labels.data_handle(),
                                                      chunk_index.data(),
                                                      n_queries,
                                                      1,
                                                      k,
                                                      raft::resource::get_cuda_stream(res));
}
}  // namespace detail

template <typename idx_t, typename data_t>
void search_filtered_bfs_impl(raft::resources const& res,
                              cuvs::neighbors::ivf_flat::index<data_t, idx_t>& idx,
                              raft::device_matrix_view<const data_t, int64_t, raft::row_major> queries,
                              raft::device_vector_view<uint32_t, int64_t> query_labels,
                              raft::device_vector_view<uint32_t, int64_t> label_size,
                              raft::device_matrix_view<idx_t, int64_t, raft::row_major> neighbors,
                              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                              cuvs::distance::DistanceType metric,
                              const cuvs::neighbors::filtering::base_filter& sample_filter_ref,
                              // Optional multi-label AND inputs — all-null disables AND mode.
                              // The buffers are passed straight into cuVS's
                              // `ivfflat_interleaved_scan`, which now embeds the
                              // per-thread AND-check inline at the kernel body.
                              const uint32_t* dataset_labels_ptr        = nullptr,
                              const int64_t*  dataset_label_offsets_ptr = nullptr,
                              const uint32_t* query_labels_second_ptr   = nullptr) {
  try {
    using none_filter_type = cuvs::neighbors::filtering::none_sample_filter;
    auto& sample_filter = dynamic_cast<const none_filter_type&>(sample_filter_ref);
    auto sample_filter_copy = sample_filter;
    detail::search_filtered_bfs_core(
      res, idx, queries, query_labels, label_size, neighbors, distances, metric, sample_filter_copy,
      dataset_labels_ptr, dataset_label_offsets_ptr, query_labels_second_ptr);
    return;
  } catch (const std::bad_cast&) {
  }

  RAFT_FAIL("Unsupported sample filter type for filtered_bfs");
}
}  // namespace cuvs::neighbors