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
#include "../merge_topk.cuh"

using namespace cuvs::neighbors;

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;

  raft::device_vector<uint32_t, int64_t> data_labels;
  raft::device_vector<uint32_t, int64_t> data_label_size;
  raft::device_vector<uint32_t, int64_t> data_label_offset;
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
    total_rows += label_data_vecs[i].size();
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
  raft::resource::sync_stream(res);
    
  // Combine indices
  int64_t current_offset = 0;
  std::vector<uint32_t> host_label_size(label_number);
  std::vector<uint32_t> host_label_offset(label_number);

  for (uint32_t i = 0; i < label_number; i++) {

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

  auto data_labels = raft::make_device_vector<uint32_t, int64_t>(res, total_data_labels);
  auto data_label_size = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());
  auto data_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());
  raft::resource::sync_stream(res);

  std::vector<uint32_t> h_data_labels(total_data_labels);
  std::vector<uint32_t> h_data_label_size(data_label_vecs.size());
  std::vector<uint32_t> h_data_label_offset(data_label_vecs.size());

  int64_t label_offset_ = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    const auto& labels = data_label_vecs[i];
    h_data_label_size[i] = labels.size();
    h_data_label_offset[i] = label_offset_;
    
    for (size_t j = 0; j < labels.size(); j++) {
      h_data_labels[label_offset_ + j] = labels[j];
    }
    label_offset_ += labels.size();
  }

  // Copy filter data to device
  raft::update_device(data_labels.data_handle(), h_data_labels.data(), total_data_labels, raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_size.data_handle(), h_data_label_size.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_offset.data_handle(), h_data_label_offset.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset),
    std::move(data_labels),
    std::move(data_label_size),
    std::move(data_label_offset)
  };
}

void compute_recall(const raft::host_matrix_view<uint32_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<int64_t>& valid_query_indices,
                    int n_double_queries,
                    const std::vector<std::vector<int>>& query_label_vecs) {
  
  int n_queries = neighbors.extent(0);
  int n_ivf_queries = n_queries - n_double_queries;
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;
  float double_recall = 0.0f;
  float ivf_recall = 0.0f;

  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/wiki1M_recall_results.csv");
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open output file");
  }

  // std::cout << "\n=== IVF Query Debug Information ===\n";
  // // Print first 5 IVF queries or less if there are fewer
  // int debug_count = std::min(100, n_queries - n_double_queries);
  
  // for (int i = n_double_queries; i < n_double_queries + debug_count; i++) {
  //   int gt_idx = valid_query_indices[i];
  //   std::cout << "\nQuery " << i << " (original index: " << gt_idx << "):\n";
  //   std::cout << "Labels: " << query_label_vecs[gt_idx][0] << ", " 
  //             << query_label_vecs[gt_idx][1] << "\n";
    
  //   std::cout << "Ground Truth (first " << topk << "): ";
  //   for (int j = 0; j < topk; j++) {
  //     std::cout << gt_indices[gt_idx][j] << " ";
  //   }
  //   std::cout << "\n";
    
  //   std::cout << "IVF Results   (first " << topk << "): ";
  //   for (int j = 0; j < topk; j++) {
  //     std::cout << neighbors(i, j) << " ";
  //   }
  //   std::cout << "\n";
    
  //   // Count and print matches for this query
  //   int matches = 0;
  //   for (int j = 0; j < topk; j++) {
  //     uint32_t neighbor_idx = neighbors(i, j);
  //     if (std::find(gt_indices[gt_idx].begin(), 
  //                 gt_indices[gt_idx].begin() + topk, 
  //                 neighbor_idx) != gt_indices[gt_idx].begin() + topk) {
  //       matches++;
  //     }
  //   }
  //   float recall = static_cast<float>(matches) / topk;
  //   std::cout << "Matches: " << matches << "/" << topk << " (recall: " 
  //             << recall << ")\n";
  // }

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
    
    if (i < n_double_queries) {
      double_recall += recall;
      total_recall += recall;
    } else {
      ivf_recall += recall;
      total_recall += 1.0;
    }

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
  std::cout << "Double label queries (" << n_double_queries << "): " 
            << std::fixed << double_recall / n_double_queries << std::endl;
  std::cout << "Filtered IVF queries (" << n_ivf_queries << "): " 
            << std::fixed << ivf_recall / n_ivf_queries << std::endl;
  std::cout << "Overall recall (" << n_queries << "): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to wiki1M_recall_results.csv" << std::endl;
}

__global__ void create_bitmap_filter_kernel(const int64_t* __restrict__ row_offsets,
                                          const int64_t* __restrict__ indices,
                                          const uint32_t* __restrict__ query_labels,
                                          uint32_t* __restrict__ bitmap,
                                          const int num_queries,
                                          const int num_cols,
                                          const int words_per_row) {
    
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;
  
  // Get start and end indices for this query's label
  const int start = row_offsets[query_labels[query_idx]];
  const int end = row_offsets[query_labels[query_idx] + 1];
  
  // Each thread handles one index
  for (int i = threadIdx.x; i < (end - start); i += blockDim.x) {
    const int idx = indices[start + i];
    if (idx >= num_cols) continue;  // Add bounds check
    
    // Calculate position in bitmap
    const int word_idx = query_idx * words_per_row + (idx / (sizeof(uint32_t) * 8));
    const unsigned bit_offset = idx % (sizeof(uint32_t) * 8);
    
    // Set bit using atomic operation
    atomicOr(&bitmap[word_idx], 1u << bit_offset);
  }
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
                                int graph_degree,
                                int specificity_threshold) {

  // Build index
  int64_t total_rows = 0;
  for (uint32_t i = 0; i < label_data_vecs.size(); i++) {
    total_rows += label_data_vecs[i].size();
  }
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, total_rows, graph_degree);
  raft::resource::sync_stream(dev_resources);

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  std::string cagra_index_file = "/projects/bdes/cmo1/CAGRA/indices_wiki1M/index_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string graph_file = "/projects/bdes/cmo1/CAGRA/indices_wiki1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string ivf_index_file = "/scratch/bdes/cmo1/CAGRA/indices_wiki1M/spec_" + 
                            std::to_string(specificity_threshold) + "_index.bin";
  std::ifstream test_cagra_file(cagra_index_file);
  std::ifstream test_cagra_graph_file(graph_file);
  std::ifstream test_ivf_file(ivf_index_file);

  auto cagra_index = cagra::index<float, uint32_t>(dev_resources);
  auto filtered_ivf_index = ivf_flat::index<float, int64_t>(dev_resources,
                                                            cuvs::distance::DistanceType::L2Unexpanded,
                                                            label_data_vecs.size(),
                                                            false,
                                                            true,
                                                            queries.extent(1));
   if (test_cagra_file.good() && test_cagra_graph_file.good() && test_ivf_file.good()) {  
    cagra::deserialize(dev_resources, cagra_index_file, &cagra_index);
    cagra_index.update_dataset(dev_resources, raft::make_const_mdspan(dataset));
    raft::resource::sync_stream(dev_resources);
    load_matrix_from_ibin(dev_resources, graph_file, graph.view());
    raft::resource::sync_stream(dev_resources);
    cagra_index.update_graph(dev_resources, raft::make_const_mdspan(graph.view()));
    raft::resource::sync_stream(dev_resources);
    ivf_flat::deserialize(dev_resources, ivf_index_file, &filtered_ivf_index);
    raft::resource::sync_stream(dev_resources);
    test_cagra_file.close();
    test_cagra_graph_file.close();
    test_ivf_file.close();
  } else {
    std::cout << "\n=== Missing Index Files ===\n";
    return;
  }
  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;
  // Index file locations
  std::cout << "Index Locations:" << std::endl;
  std::cout << "  CAGRA index: " << cagra_index_file << std::endl;
  std::cout << "  IVF index:   " << ivf_index_file << std::endl;
  // CAGRA statistics
  std::cout << "\nCAGRA Index Stats:" << std::endl;
  std::cout << "  Total vectors:  " << cagra_index.size() << std::endl;
  std::cout << "  Graph size:     [" << cagra_index.graph().extent(0) << " × " 
            << cagra_index.graph().extent(1) << "]" << std::endl;
  std::cout << "  Graph degree:   " << cagra_index.graph_degree() << std::endl;
  // IVF statistics
  std::cout << "\nIVF Index Stats:" << std::endl;
  std::cout << "  Number of lists: " << filtered_ivf_index.n_lists() << std::endl;

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
  int n_double_queries = 0;
  int n_ivf_queries = 0;
  int total_ivf_points = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    int label1 = query_label_vecs[i][0];
    int label2 = query_label_vecs[i][1];
    if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) {
      n_ivf_queries++;
      int label = (cat_freq[label1] <= cat_freq[label2]) ? label1 : label2;
      total_ivf_points += cat_freq[label];
    } else {
      n_double_queries++;
    }
  }

  std::vector<int64_t> double_query_indices(n_double_queries);
  std::vector<int64_t> ivf_query_indices(n_ivf_queries);
  std::vector<uint32_t> h_double_query_labels(n_double_queries*2);
  std::vector<uint32_t> h_ivf_query_labels(n_ivf_queries);
  std::vector<int64_t> h_double_filter_labels(n_double_queries*2);
  std::vector<uint32_t> h_double_bitmap_labels(n_double_queries*2);
  std::vector<int64_t> h_ivf_filter_labels(n_ivf_queries);
  int current_double = 0;
  int current_ivf = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    int label1 = query_label_vecs[i][0];
    int label2 = query_label_vecs[i][1];
    if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) {
      h_ivf_query_labels[current_ivf] = (cat_freq[label1] <= cat_freq[label2]) ? label1 : label2;
      h_ivf_filter_labels[current_ivf] = (cat_freq[label1] > cat_freq[label2]) ? label1 : label2;
      ivf_query_indices[current_ivf] = i;
      current_ivf++;
    } else {
      h_double_query_labels[current_double * 2] = label1;
      h_double_filter_labels[current_double * 2] = label2;
      h_double_bitmap_labels[current_double * 2] = label2;
      h_double_query_labels[current_double * 2 + 1] = label2;
      h_double_filter_labels[current_double * 2 + 1] = label1;
      h_double_bitmap_labels[current_double * 2 + 1] = label1;
      double_query_indices[current_double] = i;
      current_double++;
    }
  }
  // Update valid n_queries
  int n_queries = n_double_queries + n_ivf_queries;
 
  // Create filtered query labels vector
  auto double_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_double_queries*2);
  auto double_filter_labels = raft::make_device_vector<int64_t, int64_t>(dev_resources, n_double_queries*2);
  auto double_bitmap_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_double_queries*2);
  auto ivf_query_labels    = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_ivf_queries);
  auto ivf_filter_labels    = raft::make_device_vector<int64_t, int64_t>(dev_resources, n_ivf_queries);

  raft::update_device(double_query_labels.data_handle(),
                      h_double_query_labels.data(),
                      n_double_queries * 2,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  raft::update_device(double_filter_labels.data_handle(),
                      h_double_filter_labels.data(),
                      n_double_queries * 2,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  raft::update_device(double_bitmap_labels.data_handle(),
                      h_double_bitmap_labels.data(),
                      n_double_queries * 2,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  raft::update_device(ivf_query_labels.data_handle(),
                      h_ivf_query_labels.data(),
                      n_ivf_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  raft::update_device(ivf_filter_labels.data_handle(),
                      h_ivf_filter_labels.data(),
                      n_ivf_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  

  // Copy valid queries to new matrix
  auto double_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_double_queries*2, queries.extent(1));
  auto ivf_queries    = raft::make_device_matrix<float, int64_t>(dev_resources, n_ivf_queries, queries.extent(1));
  raft::resource::sync_stream(dev_resources);
  for (int64_t i = 0; i < n_double_queries*2; i++) {
    raft::copy_async(double_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + double_query_indices[i/2] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);
  for (int64_t i = 0; i < n_ivf_queries; i++) {
    raft::copy_async(ivf_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + ivf_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);

  // Create data labels vector
  auto double_filter = filtering::cagra_filter(combined_indices.data_labels.view(),
                                              combined_indices.data_label_size.view(), 
                                              combined_indices.data_label_offset.view(),
                                              double_filter_labels.view());
  auto ivf_filter = filtering::cagra_filter(combined_indices.data_labels.view(),
                                            combined_indices.data_label_size.view(), 
                                            combined_indices.data_label_offset.view(),
                                            ivf_filter_labels.view());
  raft::resource::sync_stream(dev_resources);

  const int64_t bits_per_uint32 = sizeof(uint32_t) * 8;
  const int64_t words_per_row = (dataset.extent(0) + bits_per_uint32 - 1) / bits_per_uint32;
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(
      dev_resources, n_double_queries*2, words_per_row);
                    
  int64_t total_indices = 0;
  for (const auto& vec : label_data_vecs) {
    total_indices += vec.size();
  }
  // Create and fill row offsets
  std::vector<int64_t> h_row_offsets(label_data_vecs.size() + 1, 0);
  int64_t current_offset = 0;
  for (size_t i = 0; i < label_data_vecs.size(); i++) {
    h_row_offsets[i] = current_offset;
    current_offset += label_data_vecs[i].size();
  }
  h_row_offsets[label_data_vecs.size()] = current_offset;
  // Create indices array
  std::vector<int64_t> h_indices(total_indices);
  current_offset = 0;
  for (const auto& vec : label_data_vecs) {
    std::copy(vec.begin(), vec.end(), h_indices.begin() + current_offset);
    current_offset += vec.size();
  }
  // Copy data to GPU
  auto d_row_offsets = raft::make_device_vector<int64_t, int64_t>(dev_resources, h_row_offsets.size());
  auto d_indices = raft::make_device_vector<int64_t, int64_t>(dev_resources, h_indices.size());
  raft::update_device(d_row_offsets.data_handle(),
                      h_row_offsets.data(),
                      h_row_offsets.size(),
                      raft::resource::get_cuda_stream(dev_resources));
  raft::update_device(d_indices.data_handle(),
                      h_indices.data(),
                      h_indices.size(),
                      raft::resource::get_cuda_stream(dev_resources));
  // Zero initialize bitmap
  // RAFT_CUDA_TRY(cudaMemsetAsync(
  //     bitmap.data_handle(),
  //     0,
  //     n_double_queries*2 * words_per_row * sizeof(uint32_t),
  //     raft::resource::get_cuda_stream(dev_resources)));
  
  int block_size = 256;
  // create_bitmap_filter_kernel<<<n_double_queries*2, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
  //     d_row_offsets.data_handle(),
  //     d_indices.data_handle(),
  //     double_bitmap_labels.data_handle(),
  //     bitmap.data_handle(),
  //     n_double_queries*2,
  //     dataset.extent(0),
  //     words_per_row);
  // raft::resource::sync_stream(dev_resources);
  // auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
  //     bitmap.data_handle(), n_double_queries*2, dataset.extent(0));
  // auto bitmap_filter = filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);

  // Resize result arrays for filtered queries
  auto double_query_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_double_queries*2, topk);
  auto double_query_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_double_queries*2, topk);
  auto double_query_final_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_double_queries, topk);
  auto double_query_final_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_double_queries, topk);
  auto ivf_query_neighbors    = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_ivf_queries, topk);
  auto ivf_query_distances    = raft::make_device_matrix<float, int64_t>(dev_resources, n_ivf_queries, topk);
  raft::resource::sync_stream(dev_resources);

  // warm-up
  for (int i = 0; i < 10; i ++) {
   cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view(),
                          double_filter);
    raft::resource::sync_stream(dev_resources);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_double_queries*2 * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(dev_resources)));
    create_bitmap_filter_kernel<<<n_double_queries*2, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
      d_row_offsets.data_handle(),
      d_indices.data_handle(),
      double_bitmap_labels.data_handle(),
      bitmap.data_handle(),
      n_double_queries*2,
      dataset.extent(0),
      words_per_row);
    raft::resource::sync_stream(dev_resources);
    auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
        bitmap.data_handle(), n_double_queries*2, dataset.extent(0));
    auto bitmap_filter = filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view(),
                          bitmap_filter);
    merge_search_results(topk,
                        n_double_queries,
                        double_query_distances.data_handle(),
                        double_query_neighbors.data_handle(),
                        double_query_final_distances.data_handle(),
                        double_query_final_neighbors.data_handle(),
                        raft::resource::get_cuda_stream(dev_resources));
    search_filtered_ivf(dev_resources,
                      filtered_ivf_index,
                      ivf_queries.view(),
                      ivf_query_labels.view(),
                      combined_indices.label_size.view(),
                      ivf_query_neighbors.view(),
                      ivf_query_distances.view(),
                      cuvs::distance::DistanceType::L2Unexpanded,
                      ivf_filter);
    raft::resource::sync_stream(dev_resources);
  }

  // For double label search:
  double total_time_double = 0;
  int num_runs = 100;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    // cagra::filtered_search(dev_resources, 
    //                       search_params,
    //                       cagra_index,
    //                       double_queries.view(),
    //                       double_query_neighbors.view(),
    //                       double_query_distances.view(),
    //                       double_query_labels.view(),
    //                       combined_indices.index_map.view(),
    //                       combined_indices.label_size.view(),
    //                       combined_indices.label_offset.view(),
    //                       double_filter);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_double_queries*2 * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(dev_resources)));
    create_bitmap_filter_kernel<<<n_double_queries*2, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
      d_row_offsets.data_handle(),
      d_indices.data_handle(),
      double_bitmap_labels.data_handle(),
      bitmap.data_handle(),
      n_double_queries*2,
      dataset.extent(0),
      words_per_row);
    raft::resource::sync_stream(dev_resources);
    auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
        bitmap.data_handle(), n_double_queries*2, dataset.extent(0));
    auto bitmap_filter = filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view(),
                          bitmap_filter);
    merge_search_results(topk,
                        n_double_queries,
                        double_query_distances.data_handle(),
                        double_query_neighbors.data_handle(),
                        double_query_final_distances.data_handle(),
                        double_query_final_neighbors.data_handle(),
                        raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_double += duration.count() / 1000.0;
  }

  double avg_time_double_ms = total_time_double / num_runs;
  double qps_double = (n_double_queries * 1000.0) / avg_time_double_ms;
  std::cout << "\n=== CAGRA Double Label Bitmap Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_double_queries << std::endl;
  std::cout << "iTopK: " << itopk_size << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << avg_time_double_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_double << std::endl;


  // For bitmap construction timing
  double total_bitmap_time = 0;
  for (int run = 0; run < num_runs; run++) {
    auto bitmap_start = std::chrono::system_clock::now();
    // Reset bitmap
    RAFT_CUDA_TRY(cudaMemsetAsync(
        bitmap.data_handle(),
        0,
        n_double_queries*2 * words_per_row * sizeof(uint32_t),
        raft::resource::get_cuda_stream(dev_resources)));
        
    // Create bitmap
    create_bitmap_filter_kernel<<<n_double_queries*2, block_size, 0, 
        raft::resource::get_cuda_stream(dev_resources)>>>(
        d_row_offsets.data_handle(),
        d_indices.data_handle(),
        double_bitmap_labels.data_handle(),
        bitmap.data_handle(),
        n_double_queries*2,
        dataset.extent(0),
        words_per_row);
    raft::resource::sync_stream(dev_resources);
    
    auto bitmap_end = std::chrono::system_clock::now();
    auto bitmap_duration = std::chrono::duration_cast<std::chrono::microseconds>(bitmap_end - bitmap_start);
    total_bitmap_time += bitmap_duration.count() / 1000.0;  // Convert to milliseconds
  }
  double avg_bitmap_time_ms = total_bitmap_time / num_runs;
  // Print bitmap construction statistics
  std::cout << "\n=== Bitmap Construction Performance ===" << std::endl;
  std::cout << "Number of queries: " << n_double_queries*2 << std::endl;
  std::cout << "Words per row: " << words_per_row << std::endl;
  std::cout << "Bitmap size: " << (n_double_queries*2 * words_per_row * sizeof(uint32_t) / (1024.0 * 1024.0)) << " MB" << std::endl;
  std::cout << "Average construction time: " << std::fixed << std::setprecision(3) << avg_bitmap_time_ms << " ms" << std::endl;

   // For double label search:
  total_time_double = 0;
  num_runs = 100;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view(),
                          double_filter);
    merge_search_results(topk,
                        n_double_queries,
                        double_query_distances.data_handle(),
                        double_query_neighbors.data_handle(),
                        double_query_final_distances.data_handle(),
                        double_query_final_neighbors.data_handle(),
                        raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_double += duration.count() / 1000.0;
  }

  avg_time_double_ms = total_time_double / num_runs;
  qps_double = (n_double_queries * 1000.0) / avg_time_double_ms;
  std::cout << "\n=== CAGRA Double Label Predicate Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_double_queries << std::endl;
  std::cout << "iTopK: " << itopk_size << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << avg_time_double_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_double << std::endl;

  // For filtered ivf search:
  double total_ivf_time = 0.0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        ivf_queries.view(),
                        ivf_query_labels.view(),
                        combined_indices.label_size.view(),
                        ivf_query_neighbors.view(),
                        ivf_query_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded,
                        ivf_filter);
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_ivf_time += duration.count() / 1000.0; // Convert to milliseconds
  }

  double avg_ivf_time_ms = total_ivf_time / num_runs;
  double qps_ivf = (n_ivf_queries * 1000.0) / avg_ivf_time_ms;
  std::cout << "\n=== Filtered IVF Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_ivf_queries << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average points per query label: " << std::fixed << std::setprecision(2) << 
    static_cast<double>(total_ivf_points) / n_ivf_queries << std::endl;
  std::cout << "Average time: " << std::fixed << avg_ivf_time_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_ivf << std::endl;


  // Common benchmark settings
  std::cout << "\n=== Running Combined Search Benchmark (" << num_runs << " iterations) ===" << std::endl;
  double total_time = 0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view(),
                          double_filter);
    merge_search_results(topk,
                        n_double_queries,
                        double_query_distances.data_handle(),
                        double_query_neighbors.data_handle(),
                        double_query_final_distances.data_handle(),
                        double_query_final_neighbors.data_handle(),
                        raft::resource::get_cuda_stream(dev_resources));
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        ivf_queries.view(),
                        ivf_query_labels.view(),
                        combined_indices.label_size.view(),
                        ivf_query_neighbors.view(),
                        ivf_query_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded,
                        ivf_filter);
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time += duration.count() / 1000.0;
  }

  // Calculate combined metrics
  double avg_time_ms = total_time / num_runs;
  double total_queries = n_double_queries + n_ivf_queries;
  double qps = (total_queries * 1000.0) / avg_time_ms;
  std::cout << "\n=== Combined Search Performance ===" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "  CAGRA iTopK (double):  " << itopk_size << std::endl;
  std::cout << "\nQuery Statistics:" << std::endl;
  std::cout << "  CAGRA Double Label:  " << std::setw(8) << n_double_queries << " queries" << std::endl;
  std::cout << "  IVF:                 " << std::setw(8) << n_ivf_queries << " queries" << std::endl;
  std::cout << "  Total:               " << std::setw(8) << total_queries << " queries" << std::endl;
  std::cout << "\nPerformance Metrics:" << std::endl;
  std::cout << "  Average time:        " << std::fixed << std::setprecision(2) << std::setw(8) << avg_time_ms << " ms" << std::endl;
  std::cout << "  Queries per second:  " << std::scientific << std::setprecision(2) << qps << std::endl;


  // Gather neighbors
  auto temp_ivf_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_ivf_queries, topk);
  convert_neighbors_to_uint32(dev_resources,
                              ivf_query_neighbors.data_handle(),
                              temp_ivf_neighbors.data_handle(),
                              n_ivf_queries,
                              topk);
  auto final_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, topk);
  raft::resource::sync_stream(dev_resources);
  // Copy single query results first (they come first in valid_query_indices)
  raft::copy(final_neighbors.data_handle(),
            double_query_final_neighbors.data_handle(),
            topk * n_double_queries,
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  raft::copy(final_neighbors.data_handle() + n_double_queries * topk,
            temp_ivf_neighbors.data_handle(),
            topk * n_ivf_queries,
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Add all single query indices first
  std::vector<int64_t> valid_query_indices;
  valid_query_indices.reserve(n_queries);
  std::copy(double_query_indices.begin(), double_query_indices.end(), std::back_inserter(valid_query_indices));
  std::copy(ivf_query_indices.begin(), ivf_query_indices.end(), std::back_inserter(valid_query_indices));

  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), final_neighbors.data_handle(), final_neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  // Read ground truth neighbors
  std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/dataset/wiki1M/wiki.GT.filtered.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_ground_true_file(gt_fname, gt_indices);
  compute_recall(h_neighbors.view(), gt_indices, valid_query_indices, n_double_queries, query_label_vecs);
}


int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 32;  // Default value
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int specificity_threshold =2500;
  if (argc > 2) {
    specificity_threshold = std::stoi(argv[2]);
  }
  int graph_degree = 16;
  if (argc > 3) {
    graph_degree = std::stoi(argv[3]);
  }
  int topk = 10;  // Default value
  if (argc > 4) {
    topk = std::stoi(argv[4]);
  }
    
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/scratch/bdes/cmo1/CAGRA/dataset/wiki1M/wiki.base.bin";
  std::string data_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/wiki1M/wiki.base.spmat";
  std::string query_fname = "/scratch/bdes/cmo1/CAGRA/dataset/wiki1M/wiki.query.filtered.bin";
  std::string query_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/wiki1M/wiki.query.filtered.spmat";

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
                              graph_degree,
                              specificity_threshold);
}
