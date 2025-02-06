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

namespace cuvs::neighbors::vecflow {

template<typename T, typename idxT>
void read_data_labels(std::string data_label_fname, 
                      std::vector<std::vector<int>>* labels_data, 
                      std::vector<int>* cat_freq) {

	std::ifstream labelfile(data_label_fname, std::ios::binary);
	if (!labelfile) {
		throw std::runtime_error("Unable to open data label file: " + data_label_fname);
	}
	std::vector<int64_t> sizes(3);
	labelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
	int64_t nrow = sizes[0], ncol = sizes[1], nnz = sizes[2];
	std::vector<int64_t> indptr(nrow + 1);
	labelfile.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
	if (nnz != indptr.back()) {
		throw std::runtime_error("Inconsistent nnz in data labels");
	}
	std::vector<int> indices(nnz);
	labelfile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
	if (!std::all_of(indices.begin(), indices.end(), [ncol](int i) { return i >= 0 && i < ncol; })) {
		throw std::runtime_error("Invalid indices in data labels");
	}
	labelfile.close();
	labels_data->reserve(ncol);
	labels_data->resize(ncol);
	cat_freq->resize(ncol, 0);
	for (uint32_t i = 0; i < nrow; ++i) {
		std::vector<int> label_list;
		for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
			label_list.push_back(indices[j]);
			(*cat_freq)[indices[j]]++;
			(*labels_data)[indices[j]].push_back(i);
		}
	}
}

inline void save_matrix_to_ibin(const std::string& filename,
                         const raft::host_matrix_view<uint32_t, int64_t>& matrix) {
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

inline void load_matrix_from_ibin(raft::resources const& res,
                           const std::string& filename,
                           const raft::host_matrix_view<uint32_t, int64_t>& graph) {
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

  if (rows != graph.extent(0) || cols != graph.extent(1))
    throw std::runtime_error("File dimensions do not match pre-allocated graph dimensions");

  file.read(reinterpret_cast<char*>(graph.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
}

inline void read_ground_truth_file(const std::string& fname, std::vector<std::vector<uint32_t>>& gt_indices) {

	std::ifstream file(fname, std::ios::binary);
	if (!file) {
		std::cout << "Warning: Ground truth file not found: " << fname << std::endl;
		return;
	}

	// Read dimensions
	int64_t rows, cols;
	file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
	file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

	// std::cout << "\n=== Reading Ground Truth File ===" << std::endl;
	// std::cout << "Dimensions: [" << rows << " x " << cols << "]" << std::endl;

	// Read data to temporary host buffer
	std::vector<uint32_t> h_matrix(rows * cols);
	file.read(reinterpret_cast<char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
	file.close();

	// Reshape into vector of vectors
	gt_indices.resize(rows);
	for (int64_t i = 0; i < rows; i++) {
		gt_indices[i].resize(cols);
		for (int64_t j = 0; j < cols; j++) {
			gt_indices[i][j] = h_matrix[i * cols + j];
		}
	}
}


template<typename T>
struct QueryInfo {
  raft::device_vector<uint32_t, int64_t> cagra_query_map;
  raft::device_matrix<T, int64_t> cagra_queries;
  raft::device_vector<uint32_t, int64_t> cagra_query_labels;
  raft::device_vector<uint32_t, int64_t> bfs_query_map;
  raft::device_matrix<T, int64_t> bfs_queries;
  raft::device_vector<uint32_t, int64_t> bfs_query_labels;
};

template<typename T>
__global__ void classify_queries_kernel(
  const T* queries,
  uint32_t* query_labels,
  uint32_t* cat_freq,
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
  uint32_t freq = cat_freq[label];
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

template<typename T>
inline auto classify_queries(raft::resources const& res,
                     raft::device_matrix_view<const T, int64_t> queries,
                     raft::device_vector_view<uint32_t, int64_t> query_labels,
                     raft::device_vector_view<uint32_t, int64_t> cat_freq,
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
    cat_freq.data_handle(),
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

template<typename T, typename IdxT>
__global__ void merge_neighbors_kernel(
  uint32_t* neighbors,
  T* distances,
  const IdxT* neighbor_src,
  const T* distance_src,
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

template<typename T>
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
}

} // namespace cuvs::neighbors::vecflow