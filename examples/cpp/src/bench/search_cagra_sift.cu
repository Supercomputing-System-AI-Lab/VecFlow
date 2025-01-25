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

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <fstream>
#include <vector>

#include <omp.h>
#include "../common.cuh"
#include "../utils.h"

using namespace cuvs::neighbors;

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
};

template<typename T>
void save_matrix_to_ibin(raft::resources const& res,
                         const std::string& filename, 
                         const raft::device_matrix_view<T, int64_t>& matrix) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filename);
  }

  // Write dimensions
  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

  // Copy data to host
  std::vector<T> h_matrix(rows * cols);
  raft::copy(h_matrix.data(), matrix.data_handle(), rows * cols, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  
  // Write data
  file.write(reinterpret_cast<const char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
  file.close();
}

void load_matrix_from_ibin(raft::resources const& res, 
                          const std::string& filename,
                          const raft::device_matrix_view<uint32_t, int64_t>& graph) {  // Pass graph as reference
  
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // Read dimensions
  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

  // Verify dimensions match
  if (rows != graph.extent(0) || cols != graph.extent(1)) {
    throw std::runtime_error("File dimensions do not match pre-allocated graph dimensions");
  }

  // Read data to host
  std::vector<uint32_t> h_matrix(rows * cols);
  file.read(reinterpret_cast<char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
  file.close();

  // Copy data to pre-allocated device matrix
  raft::update_device(graph.data_handle(), h_matrix.data(), rows * cols, 
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
}

void read_ground_true_file(const std::string& fname, std::vector<std::vector<uint32_t>>& gt_indices) {

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

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<std::vector<int>>& data_label_vecs,
                              const std::vector<std::vector<int>>& query_label_vecs,
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq) -> CombinedIndices {
  
  int label_number = query_freq.size();
    
  // First pass: calculate total size and get graph width
  int64_t total_rows = 0;
  int64_t total_data_labels = 0;
  for (uint32_t i = 0; i < label_number; i++) {
    if (query_freq[i] > 0) {
      total_rows += label_data_vecs[i].size();
    }
  }
  for (uint32_t i = 0; i < data_label_vecs.size(); i++) {
    total_data_labels += data_label_vecs[i].size();
  }
  std::cout << "\n=== Index Combination ===" << std::endl;
  std::cout << "Number of labels: " << label_number << std::endl;
  std::cout << "Total rows: " << total_rows << std::endl;
  std::cout << "Total data labels: " << total_data_labels << std::endl;

  // Create matrices and vectors
  auto index_map = raft::make_device_vector<uint32_t, int64_t>(res, total_rows);
  auto label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
    
  // Combine indices
  int64_t current_offset = 0;
  std::vector<uint32_t> host_label_size(label_number);
  std::vector<uint32_t> host_label_offset(label_number);

  for (uint32_t i = 0; i < label_number; i++) {
    if (query_freq[i] == 0) {
      host_label_size[i] = 0;
      host_label_offset[i] = current_offset;
      continue;
    }

    auto n_rows = label_data_vecs[i].size();
    auto d_indices = raft::make_device_vector<uint32_t>(res, n_rows);
    raft::copy(d_indices.data_handle(),
              reinterpret_cast<const uint32_t*>(label_data_vecs[i].data()),
              n_rows,
              raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    raft::copy(index_map.data_handle() + current_offset,
              d_indices.data_handle(),
              n_rows,
              raft::resource::get_cuda_stream(res)); 
    raft::resource::sync_stream(res);

    host_label_size[i] = n_rows;
    host_label_offset[i] = current_offset;
    current_offset += n_rows;
  }
  
  // Copy label information and set up filter components
  raft::copy(label_size.data_handle(), host_label_size.data(), label_number, raft::resource::get_cuda_stream(res));
  raft::copy(label_offset.data_handle(), host_label_offset.data(), label_number, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);


  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset)
  };
}

void compute_recall(const raft::host_matrix_view<uint32_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<int64_t>& valid_query_indices,
                    const std::vector<std::vector<int>>& query_label_vecs) {
  int n_queries = neighbors.extent(0);
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;

  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/sitft1M_recall_results.csv");
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open output file");
  }

  // Write CSV header
  outfile << "query_id,labels,recall\n";

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

    // Write to CSV
    outfile << gt_idx << ",";
    outfile << query_label_vecs[gt_idx][0];
    if (query_label_vecs[gt_idx].size() > 1) {
      outfile << "," << query_label_vecs[gt_idx][1];
    } else {
      outfile << ",";
    }
    outfile << "," << recall << "\n";
  }
  outfile.close();

  // Print summary statistics
  std::cout << "\n=== Recall Analysis ===" << std::endl;
  std::cout << "Overall recall (" << n_queries << " queries): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to sift1M_recall_results.csv" << std::endl;
}

void cagra_build_search_variants(shared_resources::configured_raft_resources& dev_resources,
                                raft::device_matrix_view<const float, int64_t> dataset,
                                raft::device_matrix_view<const float, int64_t> queries,
                                const std::vector<std::vector<int>>& data_label_vecs,
                                const std::vector<std::vector<int>>& label_data_vecs,
                                const std::vector<std::vector<int>>& query_label_vecs,
                                const std::vector<int>& cat_freq,
                                const std::vector<int>& query_freq,
                                int itopk_size,
                                int topk,
                                int specificity_threshold,
                                int graph_degree) {

  // Build index
  int64_t total_rows = 0;
  for (uint32_t i = 0; i < label_data_vecs.size(); i++) {
    if (query_freq[i] > 0) {
      total_rows += label_data_vecs[i].size();
    }
  }
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, total_rows, graph_degree);
  raft::resource::sync_stream(dev_resources);

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  std::string index_file = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/index_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string graph_file = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::ifstream test_file(index_file);
  auto index = cagra::index<float, uint32_t>(dev_resources);
  if (test_file.good()) {  
    cagra::deserialize(dev_resources, index_file, &index);
    index.update_dataset(dev_resources, raft::make_const_mdspan(dataset));
    raft::resource::sync_stream(dev_resources);
    load_matrix_from_ibin(dev_resources, graph_file, graph.view());
    index.update_graph(dev_resources, raft::make_const_mdspan(graph.view()));
    test_file.close();
  } else {
    return;
  }
  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;
  std::cout << "Loading index from: " << index_file << std::endl;
  std::cout << "CAGRA index size: " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph: [" << index.graph().extent(0) << ", " 
            << index.graph().extent(1) << "] (degree " << index.graph_degree() << ")" << std::endl;

  // Configure standard search parameters
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_label_vecs, 
                                                  query_freq, 
                                                  cat_freq);
  raft::resource::sync_stream(dev_resources);

  // Find valid queries
  int n_queries = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) continue;
    n_queries ++;
  }

  std::vector<int64_t> query_indices(n_queries);
  std::vector<uint32_t> h_filtered_labels(n_queries);
  int current = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) continue;
    h_filtered_labels[current] = query_label_vecs[i][0];
    query_indices[current] = i;
    current++;
  }

  // Create filtered query labels vector
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);

  // Copy mapped labels to device
  raft::update_device(query_labels.data_handle(),
                      h_filtered_labels.data(),
                      n_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Copy valid queries to new matrix
  auto filtered_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, queries.extent(1));
  for (int64_t i = 0; i < n_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);


  // Resize result arrays for filtered queries
  // int64_t topk = 10;
  auto query_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, topk);
  auto query_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);

  // warm-up
  for (int i = 0; i < 10; i ++) {
   cagra::filtered_search(dev_resources, 
                          search_params,
                          index,
                          filtered_queries.view(),
                          query_neighbors.view(),
                          query_distances.view(),
                          query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
  }

  // For single label search:
  int num_runs = 100;
  double total_time_single = 0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    cagra::filtered_search(dev_resources, 
                          search_params,
                          index,
                          filtered_queries.view(),
                          query_neighbors.view(),
                          query_distances.view(),
                          query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_single += duration.count() / 1000.0; // Convert to milliseconds
  }

  double avg_time_single_ms = total_time_single / num_runs;
  double qps_single = (n_queries * 1000.0) / avg_time_single_ms;
  std::cout << "\n=== SIFT1M Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_queries << std::endl;
  std::cout << "itopk_size: " << itopk_size << std::endl;
  std::cout << "topk: " << topk << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time_single_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_single << std::endl;

  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), query_neighbors.data_handle(), query_neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  // Read ground truth neighbors
  std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.groundtruth.neighbors.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_ground_true_file(gt_fname, gt_indices);
  compute_recall(h_neighbors.view(), gt_indices, query_indices, query_label_vecs);
}


int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 32;  // Default value
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int topk = 10;  // Default value
  if (argc > 2) {
    topk = std::stoi(argv[2]);
  }
  int specificity_threshold = 2000;
  if (argc > 3) {
    specificity_threshold = std::stoi(argv[3]);
  }
  int graph_degree = 16;
  if (argc > 4) {
    graph_degree = std::stoi(argv[4]);
  }
    
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.fbin";
  std::string data_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.spmat";
  std::string query_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.fbin";
  std::string query_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.spmat";

  std::vector<float> h_data;
  std::vector<float> h_queries;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);
  int max_N = 1000000;

  read_labeled_data<float, int64_t>(data_fname, data_label_fname, query_fname, query_label_fname,
                  &h_data, &data_label_vecs, &label_data_vecs,
                  &h_queries, &query_label_vecs, &cat_freq, &query_freq, max_N);

  size_t N = data_label_vecs.size();
  size_t Nq = query_label_vecs.size();
  size_t dim = h_data.size() / N;
  printf("\n=== Dataset Information ===\n");
  printf("Base dataset size: N=%lld, dim=%lld\n", N, dim);
  printf("Query dataset size: N=%lld, dim=%lld\n", Nq, dim);
  auto dataset = raft::make_device_matrix<float, int64_t>(dev_resources, N, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(dev_resources, Nq, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(dev_resources));

  // run the interesting part of the program
  cagra_build_search_variants(dev_resources, 
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              data_label_vecs,
                              label_data_vecs,
                              query_label_vecs,
                              cat_freq,
                              query_freq,
                              itopk_size,
                              topk,
                              specificity_threshold,
                              graph_degree);
}
