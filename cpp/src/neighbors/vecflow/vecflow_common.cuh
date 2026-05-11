/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_KERNEL, RAFT_CUDA_TRY

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuvs::neighbors::vecflow {

inline void save_matrix_to_ibin(const std::string& filename,
                                raft::host_matrix_view<uint32_t, int64_t> matrix) {

  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  std::ofstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot create file: " + filename);

  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
  std::cout << "Saving graph to " << filename << std::endl;
}

inline void load_matrix_from_ibin(const std::string& filename,
                                  raft::host_matrix_view<uint32_t, int64_t> matrix) {

  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

  if (rows != matrix.extent(0) || cols != matrix.extent(1))
    throw std::runtime_error("File dimensions do not match pre-allocated graph dimensions");

  file.read(reinterpret_cast<char*>(matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
  std::cout << "Loading graph from " << filename << std::endl;
}

template <typename T>
struct QueryInfo {
  raft::device_vector<uint32_t, int64_t> cagra_query_map;
  raft::device_matrix<T, int64_t> cagra_queries;
  raft::device_vector<uint32_t, int64_t> cagra_query_labels;
  raft::device_vector<uint32_t, int64_t> bfs_query_map;
  raft::device_matrix<T, int64_t> bfs_queries;
  raft::device_vector<uint32_t, int64_t> bfs_query_labels;
};

template <typename T>
RAFT_KERNEL classify_queries_kernel(const T* queries,
                                        uint32_t* query_labels,
                                        uint32_t* label_freq,
                                        uint32_t* temp_cagra_map,
                                        uint32_t* temp_bfs_map,
                                        T* temp_cagra_queries,
                                        T* temp_bfs_queries,
                                        uint32_t* temp_cagra_labels,
                                        uint32_t* temp_bfs_labels,
                                        int n_queries,
                                        int dim,
                                        int specificity_threshold,
                                        int* cagra_count,
                                        int* bfs_count) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_queries) return;

  uint32_t label = query_labels[tid];
  uint32_t freq = label_freq[label];
  bool is_cagra = freq > specificity_threshold;

  int pos;
  if (is_cagra) {
    pos = atomicAdd(cagra_count, 1);
    temp_cagra_map[pos] = tid;
    temp_cagra_labels[pos] = label;

    for (int j = 0; j < dim; j++) {
      temp_cagra_queries[pos * dim + j] = queries[tid * dim + j];
    }
  } else {
    pos = atomicAdd(bfs_count, 1);
    temp_bfs_map[pos] = tid;
    temp_bfs_labels[pos] = label;

    for (int j = 0; j < dim; j++) {
      temp_bfs_queries[pos * dim + j] = queries[tid * dim + j];
    }
  }
}

template <typename T>
inline auto classify_queries(raft::resources const& res,
                             raft::device_matrix_view<const T, int64_t> queries,
                             raft::device_vector_view<uint32_t, int64_t> query_labels,
                             raft::device_vector_view<uint32_t, int64_t> label_freq,
                             int specificity_threshold) -> QueryInfo<T> {

  int n_queries = queries.extent(0);
  int dim = queries.extent(1);

  auto stream = raft::resource::get_cuda_stream(res);

  // Create temporary device memory
  rmm::device_uvector<uint32_t> temp_cagra_map(n_queries, stream);
  rmm::device_uvector<uint32_t> temp_bfs_map(n_queries, stream);
  rmm::device_uvector<T> temp_cagra_queries(n_queries * dim, stream);
  rmm::device_uvector<T> temp_bfs_queries(n_queries * dim, stream);
  rmm::device_uvector<uint32_t> temp_cagra_labels(n_queries, stream);
  rmm::device_uvector<uint32_t> temp_bfs_labels(n_queries, stream);

  // Counters
  rmm::device_scalar<int> d_cagra_count(0, stream);
  rmm::device_scalar<int> d_bfs_count(0, stream);

  // Launch kernel
  int block_size = 256;
  int grid_size = (n_queries + block_size - 1) / block_size;

  classify_queries_kernel<<<grid_size, block_size, 0, stream>>>(
    queries.data_handle(),
    query_labels.data_handle(),
    label_freq.data_handle(),
    temp_cagra_map.data(),
    temp_bfs_map.data(),
    temp_cagra_queries.data(),
    temp_bfs_queries.data(),
    temp_cagra_labels.data(),
    temp_bfs_labels.data(),
    n_queries,
    dim,
    specificity_threshold,
    d_cagra_count.data(),
    d_bfs_count.data()
  );
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Get final counts
  int h_cagra_count = d_cagra_count.value(stream);
  int h_bfs_count = d_bfs_count.value(stream);

  // Initialize raft structures with correct sizes
  auto cagra_query_map = raft::make_device_vector<uint32_t, int64_t>(res, h_cagra_count);
  auto cagra_queries = raft::make_device_matrix<T, int64_t>(res, h_cagra_count, dim);
  auto cagra_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, h_cagra_count);
  auto bfs_query_map = raft::make_device_vector<uint32_t, int64_t>(res, h_bfs_count);
  auto bfs_queries = raft::make_device_matrix<T, int64_t>(res, h_bfs_count, dim);
  auto bfs_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, h_bfs_count);

  // Copy from temporary buffers to final raft structures
  raft::copy(cagra_query_map.data_handle(),
             temp_cagra_map.data(),
             h_cagra_count,
             stream);

  raft::copy(cagra_queries.data_handle(),
             temp_cagra_queries.data(),
             h_cagra_count * dim,
             stream);

  raft::copy(cagra_query_labels.data_handle(),
             temp_cagra_labels.data(),
             h_cagra_count,
             stream);

  raft::copy(bfs_query_map.data_handle(),
             temp_bfs_map.data(),
             h_bfs_count,
             stream);

  raft::copy(bfs_queries.data_handle(),
             temp_bfs_queries.data(),
             h_bfs_count * dim,
             stream);

  raft::copy(bfs_query_labels.data_handle(),
             temp_bfs_labels.data(),
             h_bfs_count,
             stream);

  return QueryInfo<T> {
    std::move(cagra_query_map),
    std::move(cagra_queries),
    std::move(cagra_query_labels),
    std::move(bfs_query_map),
    std::move(bfs_queries),
    std::move(bfs_query_labels)
  };
}

template <typename T, typename IdxT>
RAFT_KERNEL merge_neighbors_kernel(uint32_t* neighbors,
                                       float* distances,
                                       const IdxT* neighbor_src,
                                       const float* distance_src,
                                       const uint32_t* indices,
                                       int n_queries,
                                       int topk) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elements = n_queries * topk;

  if (tid >= n_elements) return;

  int query_idx = tid / topk;
  int k_idx = tid % topk;

  int src_offset = query_idx * topk + k_idx;
  int dst_offset = indices[query_idx] * topk + k_idx;

  neighbors[dst_offset] = static_cast<uint32_t>(neighbor_src[src_offset]);
  distances[dst_offset] = distance_src[src_offset];
}

template <typename T>
inline void merge_search_results(raft::resources const& res,
                                 raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                 raft::device_matrix_view<float, int64_t> distances,
                                 QueryInfo<T>& query_info,
                                 raft::device_matrix_view<int64_t, int64_t> bfs_neighbors,
                                 raft::device_matrix_view<float, int64_t> bfs_distances,
                                 raft::device_matrix_view<uint32_t, int64_t> cagra_neighbors,
                                 raft::device_matrix_view<float, int64_t> cagra_distances,
                                 int topk) {

  auto stream = raft::resource::get_cuda_stream(res);

  // Launch kernels for both BFS and CAGRA results
  int block_size = 256;

  if (query_info.bfs_query_map.size() > 0) {
    // BFS kernel - handles conversion and merging in one step
    int n_bfs_elements = query_info.bfs_query_map.size() * topk;
    int grid_size_bfs = (n_bfs_elements + block_size - 1) / block_size;

    merge_neighbors_kernel<T, int64_t><<<grid_size_bfs, block_size, 0, stream>>>(
      neighbors.data_handle(),
      distances.data_handle(),
      bfs_neighbors.data_handle(),  // Direct use of int64_t input
      bfs_distances.data_handle(),
      query_info.bfs_query_map.data_handle(),
      query_info.bfs_query_map.size(),
      topk);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  if (query_info.cagra_query_map.size() > 0) {
    // CAGRA kernel
    int n_cagra_elements = query_info.cagra_query_map.size() * topk;
    int grid_size_cagra = (n_cagra_elements + block_size - 1) / block_size;

    merge_neighbors_kernel<T, uint32_t><<<grid_size_cagra, block_size, 0, stream>>>(
      neighbors.data_handle(),
      distances.data_handle(),
      cagra_neighbors.data_handle(),
      cagra_distances.data_handle(),
      query_info.cagra_query_map.data_handle(),
      query_info.cagra_query_map.size(),
      topk);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Multi-label AND helpers
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build the CSR-flat dataset-label arrays used by `search_multi_labels`.
 * Each per-point label slice is sorted ascending — the kernel-side inline
 * AND check (binary search) relies on this invariant.
 *
 * @param[in]  data_label_vecs        Host list of per-point label lists.
 * @param[out] out_dataset_labels     Flat device array of all labels.
 * @param[out] out_dataset_offsets    CSR offsets, length n_points+1.
 */
inline void prepare_dataset_label_csr(
  shared_resources::configured_raft_resources&          res,
  const std::vector<std::vector<int>>&                  data_label_vecs,
  raft::device_vector<uint32_t, int64_t>&               out_dataset_labels,
  raft::device_vector<int64_t,  int64_t>&               out_dataset_label_offsets)
{
  const int64_t n_points = static_cast<int64_t>(data_label_vecs.size());

  // Build sorted host slices + CSR offsets.
  std::vector<int64_t> h_offsets(n_points + 1, 0);
  for (int64_t i = 0; i < n_points; i++) {
    h_offsets[i + 1] = h_offsets[i] + static_cast<int64_t>(data_label_vecs[i].size());
  }
  const int64_t nnz = h_offsets[n_points];

  std::vector<uint32_t> h_labels(nnz);
  for (int64_t i = 0; i < n_points; i++) {
    auto sorted = data_label_vecs[i];
    std::sort(sorted.begin(), sorted.end());
    for (size_t j = 0; j < sorted.size(); j++) {
      h_labels[h_offsets[i] + static_cast<int64_t>(j)] = static_cast<uint32_t>(sorted[j]);
    }
  }

  // Materialize on device.
  out_dataset_labels        = raft::make_device_vector<uint32_t, int64_t>(res, nnz);
  out_dataset_label_offsets = raft::make_device_vector<int64_t,  int64_t>(res, n_points + 1);
  auto stream               = raft::resource::get_cuda_stream(res);

  raft::update_device(out_dataset_labels.data_handle(),        h_labels.data(),  nnz,             stream);
  raft::update_device(out_dataset_label_offsets.data_handle(), h_offsets.data(), n_points + 1,    stream);
}

/**
 * For each query, pick the rarer label (smaller `label_freq`) as the primary
 * and the other as the secondary. Searching the smaller candidate set first
 * and post-filtering by the more common label minimizes wasted work and
 * keeps itopk from being dominated by hits that only match the common label.
 *
 * Out arrays must be pre-sized to n_queries.
 */
RAFT_KERNEL pick_primary_by_label_freq_kernel(const uint32_t* __restrict__ labels_a,
                                            const uint32_t* __restrict__ labels_b,
                                            const uint32_t* __restrict__ label_freq,
                                            uint32_t*       __restrict__ primary_out,
                                            uint32_t*       __restrict__ secondary_out,
                                            int                          n_queries)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_queries) return;
  const uint32_t a = labels_a[i];
  const uint32_t b = labels_b[i];
  // Tie-break: when equal frequencies, pick `a` as primary deterministically.
  if (label_freq[a] <= label_freq[b]) {
    primary_out[i]   = a;
    secondary_out[i] = b;
  } else {
    primary_out[i]   = b;
    secondary_out[i] = a;
  }
}

/**
 * Gather one element per output position from `src`, indexed by `map`.
 * Used to permute the secondary-label array along the same partition order
 * that `classify_queries` produced for primary labels.
 */
RAFT_KERNEL gather_by_map_kernel(const uint32_t* __restrict__ src,
                                 const uint32_t* __restrict__ map,
                                 uint32_t*       __restrict__ dst,
                                 int                          n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[map[i]];
}

}  // namespace cuvs::neighbors::vecflow