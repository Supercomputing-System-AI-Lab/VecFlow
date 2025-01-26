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

#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <rmm/device_vector.hpp>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <ctime> 

template<typename T, typename idxT>
void read_labeled_data(std::string data_fname, 
											 std::string data_label_fname, 
											 std::string query_fname, 
											 std::string query_label_fname, 
											 std::vector<T>* data, 
											 std::vector<std::vector<int>>* data_labels, 
											 std::vector<std::vector<int>>* labels_data, 
											 std::vector<T>* queries, 
											 std::vector<std::vector<int>>* query_labels, 
											 std::vector<int>* cat_freq, 
											 std::vector<int>* query_freq, 
											 int max_N) {

	// Read datafile in
	std::ifstream datafile(data_fname, std::ifstream::binary);
	if (!datafile) {
		throw std::runtime_error("Unable to open data file: " + data_fname);
	}
	uint32_t N, dim;
	datafile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
	datafile.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
	if (N > static_cast<uint32_t>(max_N)) N = max_N;
	data->resize(N*dim);
	datafile.read(reinterpret_cast<char*>(data->data()), N*dim*sizeof(T));
	datafile.close();

	// read query data in
	std::ifstream queryfile(query_fname, std::ifstream::binary);
	if (!queryfile) {
		throw std::runtime_error("Unable to open query file: " + query_fname);
	}
	uint32_t q_N, q_dim;
	queryfile.read(reinterpret_cast<char*>(&q_N), sizeof(uint32_t));
	queryfile.read(reinterpret_cast<char*>(&q_dim), sizeof(uint32_t));
	if (q_dim != dim) {
		throw std::runtime_error("Query dim and data dim don't match!");
	}
	queries->resize(q_N*dim);
	queryfile.read(reinterpret_cast<char*>(queries->data()), q_N*dim*sizeof(T));
	queryfile.close();

	// read labels for data vectors
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
	data_labels->reserve(N);
	labels_data->reserve(ncol);
	labels_data->resize(ncol);
	cat_freq->resize(ncol, 0);
	for (uint32_t i = 0; i < N; ++i) {
		std::vector<int> label_list;
		for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
			label_list.push_back(indices[j]);
			(*cat_freq)[indices[j]]++;
			(*labels_data)[indices[j]].push_back(i);
		}
		data_labels->push_back(std::move(label_list));
	}

	// read labels for queries
	std::ifstream qlabelfile(query_label_fname, std::ios::binary);
	if (!qlabelfile) {
		throw std::runtime_error("Unable to open query label file: " + query_label_fname);
	}
	qlabelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
	nrow = sizes[0], ncol = sizes[1], nnz = sizes[2];
	indptr.resize(nrow + 1);
	qlabelfile.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
	if (nnz != indptr.back()) {
		throw std::runtime_error("Inconsistent nnz in query labels");
	}
	indices.resize(nnz);
	qlabelfile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
	if (!std::all_of(indices.begin(), indices.end(), [ncol](int i) { return i >= 0 && i < ncol; })) {
		throw std::runtime_error("Invalid indices in query labels");
	}
	qlabelfile.close();
	query_labels->reserve(q_N);
	query_freq->resize(ncol, 0);
	for (uint32_t i = 0; i < q_N; ++i) {
		std::vector<int> label_list;
		for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
			label_list.push_back(indices[j]);
			(*query_freq)[indices[j]]++;
		}
		query_labels->push_back(std::move(label_list));
	}
}


__global__ void convert_int64_to_uint32(const int64_t* input, uint32_t* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		output[idx] = static_cast<uint32_t>(input[idx]);
	}
}

void convert_neighbors_to_uint32(raft::resources const& res,
                               const int64_t* input,
                               uint32_t* output,
                               int n_queries,
                               int k) {
  int total_elements = n_queries * k;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  
  convert_int64_to_uint32<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
      input, output, total_elements);
}


void read_ground_truth_file(const std::string& fname, std::vector<std::vector<uint32_t>>& gt_indices) {

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

void compute_recall(const raft::host_matrix_view<uint32_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<int64_t>& valid_query_indices) {
	
  int n_queries = neighbors.extent(0);
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;

  for (int i = 0; i < n_queries; i++) {
    int matches = 0;
    auto gt_idx = valid_query_indices[i];
    
    // Count matches between found and ground truth neighbors
    for (int j = 0; j < topk; j++) {
      uint32_t neighbor_idx = neighbors(i, j);
      if (std::find(gt_indices[gt_idx].begin(), 
                  gt_indices[gt_idx].begin() + topk, 
                  neighbor_idx) != 
        gt_indices[gt_idx].begin() + topk) {
        matches++;
      }
    }
    
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;
  }

  // Print summary statistics
  std::cout << "\n=== Recall Analysis ===" << std::endl;
  std::cout << "Overall recall (" << n_queries << " queries): " 
            << std::fixed << total_recall / n_queries << std::endl;
}