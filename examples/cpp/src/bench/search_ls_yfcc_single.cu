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

#include <omp.h>
#include "../common.cuh"
#include "../utils.h"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/brute_force.hpp>
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

using namespace cuvs::neighbors;

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
};

void save_matrix_to_ibin(raft::resources const& res,
                         const std::string& filename, 
                         const raft::device_matrix_view<uint32_t, int64_t>& matrix) {
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
  std::vector<uint32_t> h_matrix(rows * cols);
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

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<std::vector<int>>& data_label_vecs,
                              const std::vector<std::vector<int>>& query_label_vecs,
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq,
                              int specificity_threshold) -> CombinedIndices {
  
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
  raft::resource::sync_stream(res);
    
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

  // Read specificities
  std::vector<double> specificities = read_specificities("/u/cmo1/Filtered/cuvs/examples/cpp/src/query_specificities.txt", 
                                                        100000);
  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/hybrid_recall_results.csv");
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open output file");
  }
  // Write CSV header
  outfile << "query_id,labels,specificity,recall\n";

  for (int i = 0; i < neighbors.extent(0); i++) {
    int matches = 0;
    auto gt_idx = valid_query_indices[i];

    for (int j = 0; j < topk; j++) {
      uint32_t neighbor_idx = neighbors(i, j);
      if (std::find(gt_indices[gt_idx].begin(), gt_indices[gt_idx].end(), neighbor_idx) != gt_indices[gt_idx].end()) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;

    // Write to CSV: query_id, number of labels, specificity, recall
    outfile << gt_idx << ",";
    outfile << query_label_vecs[gt_idx][0];
    if (query_label_vecs[gt_idx].size() > 1) {
      outfile << "," << query_label_vecs[gt_idx][1];
    } else {
      outfile << ",";  // Empty second label field if there isn't one
    }
    outfile << "," << specificities[gt_idx] << "," 
            << recall << "\n";
  }
  // Close the file
  outfile.close();

  // Print summary to console
  std::cout << "Overall recall (" << n_queries << "): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to hybrid_recall_results.csv" << std::endl;
}


void cagra_ivf_build_search_variants(shared_resources::configured_raft_resources& dev_resources,
                                    raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                    raft::device_matrix_view<const uint8_t, int64_t> queries,
                                    const std::vector<std::vector<int>>& data_label_vecs,
                                    const std::vector<std::vector<int>>& label_data_vecs,
                                    const std::vector<std::vector<int>>& query_label_vecs,
                                    const std::vector<int>& cat_freq,
                                    const std::vector<int>& query_freq,
                                    int itopk_size,
                                    int specificity_threshold = 2000,
                                    int graph_degree = 16) {

  // Build index
  std::cout << "Building CAGRA and IVF index (search graph)" << std::endl;
  std::string cagra_index_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/index_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string graph_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string ivf_index_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/IVF-BFS_indices/spec_" + 
                            std::to_string(specificity_threshold) + "_index.bin";

  std::ifstream test_cagra_file(cagra_index_file);
  std::ifstream test_cagra_graph_file(graph_file);
  std::ifstream test_ivf_file(ivf_index_file);

  auto cagra_index = cagra::index<uint8_t, uint32_t>(dev_resources);
  auto filtered_ivf_index = ivf_flat::index<uint8_t, int64_t>(dev_resources,
                                                              cuvs::distance::DistanceType::L2Unexpanded,
                                                              label_data_vecs.size(),
                                                              false,
                                                              true,
                                                              queries.extent(1));
  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_label_vecs, 
                                                  query_freq, 
                                                  cat_freq,
                                                  0);
  raft::resource::sync_stream(dev_resources);

  int64_t total_rows = 0;
  for (uint32_t i = 0; i < label_data_vecs.size(); i++) {
    if (query_freq[i] > 0) {
      total_rows += label_data_vecs[i].size();
    }
  }
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, total_rows, graph_degree);
  raft::resource::sync_stream(dev_resources);
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
  std::cout << "  Indexed points:  " << combined_indices.index_map.size() << std::endl;

  // Configure standard search parameters
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED; 

  // Find valid queries
  int n_ivf_queries = 0;
  int total_ivf_points = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
        n_ivf_queries++;
        total_ivf_points += cat_freq[query_label_vecs[i][0]];
      }
    }
  }

  std::vector<int64_t> ivf_query_indices(n_ivf_queries);
  std::vector<uint32_t> h_ivf_filtered_labels(n_ivf_queries);
  int current_ivf = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
        ivf_query_indices[current_ivf] = i;
        h_ivf_filtered_labels[current_ivf] = query_label_vecs[i][0];
        current_ivf++;
      }
    }
  }
 
  // Create filtered query labels vector
  auto ivf_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_ivf_queries);

  raft::update_device(ivf_query_labels.data_handle(),
                      h_ivf_filtered_labels.data(),
                      n_ivf_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Copy valid queries to new matrix
  auto ivf_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_ivf_queries, queries.extent(1));
  raft::resource::sync_stream(dev_resources);
  for (int64_t i = 0; i < n_ivf_queries; i++) {
    raft::copy_async(ivf_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + ivf_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);

  // Resize result arrays for filtered queries
  int64_t topk = 10;
  auto cagra_query_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_ivf_queries, topk);
  auto cagra_query_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_ivf_queries, topk);
  auto ivf_query_neighbors   = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_ivf_queries, topk);
  auto ivf_query_distances   = raft::make_device_matrix<float, int64_t>(dev_resources, n_ivf_queries, topk);
  raft::resource::sync_stream(dev_resources);

  // warm-up
  for (int i = 0; i < 10; i++) {
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          ivf_queries.view(),
                          cagra_query_neighbors.view(),
                          cagra_query_distances.view(),
                          ivf_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        ivf_queries.view(),
                        ivf_query_labels.view(),
                        combined_indices.label_size.view(),
                        ivf_query_neighbors.view(),
                        ivf_query_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(dev_resources);
  }
  
  // For single label search:
  int num_runs = 100;
  double total_time_single = 0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    cagra::filtered_search(dev_resources, 
                          search_params,
                          cagra_index,
                          ivf_queries.view(),
                          cagra_query_neighbors.view(),
                          cagra_query_distances.view(),
                          ivf_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double run_time_ms = duration.count() / 1000.0;
    total_time_single += run_time_ms;
  }


  double avg_time_single_ms = total_time_single / num_runs;
  double qps_cagra = (n_ivf_queries * 1000.0) / avg_time_single_ms;
  std::cout << "\n=== CAGRA LS Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_ivf_queries << std::endl;
  std::cout << "iTopK: " << itopk_size << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time_single_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_cagra << std::endl;


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
                        cuvs::distance::DistanceType::L2Unexpanded);
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


  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_ivf_queries, topk);
  raft::copy(h_neighbors.data_handle(), cagra_query_neighbors.data_handle(), cagra_query_neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_gt_file(gt_fname, gt_indices);
  std::cout << "\n=== CAGRA LS Recall Analysis ===" << std::endl;
  compute_recall(h_neighbors.view(), gt_indices, ivf_query_indices, query_label_vecs);
}

template<typename index_t = int, typename bitmap_t = uint32_t>
__global__ void create_bitmap_filter_kernel(const index_t* __restrict__ row_offsets,
                                            const index_t* __restrict__ indices,
                                            const uint32_t* __restrict__ query_labels,
                                            bitmap_t* __restrict__ bitmap,
                                            const index_t num_queries,
                                            const index_t num_cols,
                                            const index_t words_per_row) {
    
  const index_t query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;
  
  // Get start and end indices for this query's label
  const index_t start = row_offsets[query_labels[query_idx]];
  const index_t end = row_offsets[query_labels[query_idx] + 1];
  
  // Each thread handles one index
  for (index_t i = threadIdx.x; i < (end - start); i += blockDim.x) {
    const index_t idx = indices[start + i];
    if (idx >= num_cols) continue;  // Add bounds check
    
    // Calculate position in bitmap
    const index_t word_idx = query_idx * words_per_row + (idx / (sizeof(bitmap_t) * 8));
    const unsigned bit_offset = idx % (sizeof(bitmap_t) * 8);
    
    // Set bit using atomic operation
    atomicOr(&bitmap[word_idx], bitmap_t(1) << bit_offset);
  }
}

__global__ void convert_indices_kernel(const int64_t* input, uint32_t* output, size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<uint32_t>(input[idx]);
  }
}

void create_bitmap_filter_fast(raft::resources const& handle,
                             const std::vector<std::vector<int>>& label_data_vecs,
                             const std::vector<std::vector<int>>& query_label_vecs,
                             const raft::device_vector<uint32_t, int64_t>& query_labels,
                             int64_t n_queries,
                             int64_t n_database,
                             const raft::device_matrix_view<uint32_t, int64_t>& bitmap) {
    
  // Calculate total size needed for indices
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
  auto d_row_offsets = raft::make_device_vector<int64_t, int64_t>(handle, h_row_offsets.size());
  auto d_indices = raft::make_device_vector<int64_t, int64_t>(handle, h_indices.size());
  
  raft::update_device(d_row_offsets.data_handle(),
                      h_row_offsets.data(),
                      h_row_offsets.size(),
                      raft::resource::get_cuda_stream(handle));
  
  raft::update_device(d_indices.data_handle(),
                      h_indices.data(),
                      h_indices.size(),
                      raft::resource::get_cuda_stream(handle));
  
  // Get the actual words_per_row from the bitmap view
  const int64_t words_per_row = bitmap.extent(1);
  
  // Zero initialize bitmap
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_queries * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(handle)));
  
  // Launch kernel
  const int block_size = 256;
  create_bitmap_filter_kernel<<<n_queries, block_size, 0, raft::resource::get_cuda_stream(handle)>>>(
    d_row_offsets.data_handle(),
    d_indices.data_handle(),
    query_labels.data_handle(),
    bitmap.data_handle(),
    n_queries,
    n_database,
    words_per_row
  );
  
  raft::resource::sync_stream(handle);
}

__global__ void convert_dataset_kernel(const uint8_t* input, float* output, size_t size, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<float>(input[idx]) * scale;
  }
}

void benchmark_brute_force_search(shared_resources::configured_raft_resources& dev_resources,
                                raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                raft::device_matrix_view<const uint8_t, int64_t> queries,
                                const std::vector<std::vector<int>>& data_label_vecs,
                                const std::vector<std::vector<int>>& label_data_vecs,
                                const std::vector<std::vector<int>>& query_label_vecs,
                                const std::vector<int>& cat_freq,
                                const std::vector<int>& query_freq,
                                int specificity_threshold = 2000) {
  // Parameters
  const int64_t topk = 10;
  const int num_runs = 10;
  
  std::cout << "\n=== Setting up BFS Search ===" << std::endl;
  // 1. Find valid queries based on specificity threshold
  int64_t n_valid_queries = 0;
  int64_t total_valid_points = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
        n_valid_queries++;
        total_valid_points += cat_freq[query_label_vecs[i][0]];
      }
    }
  }
  
  // 2. Create filtered query arrays
  std::vector<int64_t> valid_query_indices(n_valid_queries);
  std::vector<uint32_t> h_query_labels(n_valid_queries);
  int current_idx = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
        valid_query_indices[current_idx] = i;
        h_query_labels[current_idx] = query_label_vecs[i][0];
        current_idx++;
      }
    }
  }

  // 3. Create filtered query matrix
  auto filtered_queries = raft::make_device_matrix<uint8_t, int64_t>(
      dev_resources, n_valid_queries, queries.extent(1));
  
  for (int64_t i = 0; i < n_valid_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + valid_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  
  // 4. Convert uint8_t data to float
  auto float_dataset = raft::make_device_matrix<float, int64_t>(
      dev_resources, dataset.extent(0), dataset.extent(1));
  auto float_queries = raft::make_device_matrix<float, int64_t>(
      dev_resources, n_valid_queries, queries.extent(1));
  
  // Convert data (0-255 -> 0-1)
  const float scale = 1.0f / 255.0f;
  const int block_size = 256;
  const int grid_size = (dataset.size() + block_size - 1) / block_size;
  
  convert_dataset_kernel<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
    dataset.data_handle(),
    float_dataset.data_handle(),
    dataset.size(),
    scale
  );
  
  const int query_grid_size = (filtered_queries.size() + block_size - 1) / block_size;
  convert_dataset_kernel<<<query_grid_size, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
    filtered_queries.data_handle(),
    float_queries.data_handle(),
    filtered_queries.size(),
    scale
  );
  
  raft::resource::sync_stream(dev_resources);

  // 5. Setup bitmap filtering
  const int64_t bits_per_uint32 = sizeof(uint32_t) * 8;
  const int64_t words_per_row = (dataset.extent(0) + bits_per_uint32 - 1) / bits_per_uint32;
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(
      dev_resources, n_valid_queries, words_per_row);
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_valid_queries);
  raft::update_device(query_labels.data_handle(),
                    h_query_labels.data(),
                    n_valid_queries,
                    raft::resource::get_cuda_stream(dev_resources));
                    
  
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
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_valid_queries * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(dev_resources)));
  
  // Warm-up runs
  for (int i = 0; i < 10; i++) {
    create_bitmap_filter_kernel<<<n_valid_queries, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
      d_row_offsets.data_handle(),
      d_indices.data_handle(),
      query_labels.data_handle(),
      bitmap.data_handle(),
      n_valid_queries,
      dataset.extent(0),
      words_per_row
    );
    raft::resource::sync_stream(dev_resources);
  }
  // Timed runs
  double bitmap_total_time = 0.0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    create_bitmap_filter_kernel<<<n_valid_queries, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
      d_row_offsets.data_handle(),
      d_indices.data_handle(),
      query_labels.data_handle(),
      bitmap.data_handle(),
      n_valid_queries,
      dataset.extent(0),
      words_per_row
    );
    raft::resource::sync_stream(dev_resources);  // Ensure operation is complete
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    bitmap_total_time += duration;
  }
  // Calculate and print statistics
  double bitmap_avg_time = bitmap_total_time / num_runs;
  double bitmap_creation_rate = (n_valid_queries * 1000.0) / bitmap_avg_time;  // bitmaps per second
  std::cout << "\n=== Bitmap Creation Performance ===" << std::endl;
  std::cout << "Number of queries: " << n_valid_queries << std::endl;
  std::cout << "Database size: " << dataset.extent(0) << std::endl;
  std::cout << "Average creation time: " << std::fixed << std::setprecision(2) 
            << bitmap_avg_time << " ms" << std::endl;
  std::cout << "Bitmap creation rate: " << std::scientific 
            << bitmap_creation_rate << " bitmaps/second" << std::endl;

  // create_bitmap_filter_fast(dev_resources,
  //                         label_data_vecs,
  //                         query_label_vecs,
  //                         query_labels,
  //                         n_valid_queries,
  //                         dataset.extent(0),
  //                         bitmap.view());
  
  auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
      bitmap.data_handle(), n_valid_queries, dataset.extent(0));
  auto filter = filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);

  // 6. Build and search with BFS index
  std::cout << "Building BFS index..." << std::endl;
  auto bfs_index = cuvs::neighbors::brute_force::build(dev_resources,
                                                      float_dataset.view(),
                                                      cuvs::distance::DistanceType::L2Expanded);
  
  // Create results arrays
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_valid_queries, topk);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_valid_queries, topk);

  // 7. Warm-up runs
  // std::cout << "Performing warm-up..." << std::endl;
  for (int i = 0; i < 10; i++) {
    cuvs::neighbors::brute_force::search(dev_resources,
                                        bfs_index,
                                        float_queries.view(),
                                        neighbors.view(),
                                        distances.view(),
                                        filter);
    raft::resource::sync_stream(dev_resources);
  }

  // 8. Benchmark
  // std::cout << "\nStarting benchmark..." << std::endl;
  double total_time = 0.0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    cuvs::neighbors::brute_force::search(dev_resources,
                                        bfs_index,
                                        float_queries.view(),
                                        neighbors.view(),
                                        distances.view(),
                                        filter);
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    total_time += duration;
  }

  // 9. Report results
  double avg_time = total_time / num_runs;
  double qps = (n_valid_queries * 1000.0) / avg_time;
  
  std::cout << "\n=== BFS Search Results ===" << std::endl;
  std::cout << "Number of queries: " << n_valid_queries << std::endl;
  std::cout << "Average points per query: " << 
      static_cast<double>(total_valid_points) / n_valid_queries << std::endl;
  std::cout << "Average search time: " << std::fixed << avg_time << " ms" << std::endl;
  std::cout << "Queries per second: " << std::scientific << qps << std::endl;

  // // 10. Optional: Compute recall
  auto uint32_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_valid_queries, topk);
  const int neighbor_block_size = 256;
  const int neighbor_grid_size = (neighbors.size() + neighbor_block_size - 1) / neighbor_block_size;
  convert_indices_kernel<<<neighbor_grid_size, neighbor_block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
    neighbors.data_handle(),
    uint32_neighbors.data_handle(),
    neighbors.size()
  );
  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_valid_queries, topk);
  raft::copy(h_neighbors.data_handle(), 
            uint32_neighbors.data_handle(), 
            uint32_neighbors.size(), 
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_gt_file(gt_fname, gt_indices);
  std::cout << "\n=== BFS Recall Analysis ===" << std::endl;
  compute_recall(h_neighbors.view(), gt_indices, valid_query_indices, query_label_vecs);
}

int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 4;  // Default value
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int specificity_threshold = 2000;
  if (argc > 2) {
    specificity_threshold = std::stoi(argv[2]);
  }
  int graph_degree = 16;
  if (argc > 3) {
    graph_degree = std::stoi(argv[3]);
  }

  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.10M.u8bin";
  std::string data_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.sorted.metadata.10M.spmat";
  std::string query_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.public.100K.u8bin";
  std::string query_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.metadata.public.100K.spmat";

  std::vector<uint8_t> h_data;
  std::vector<uint8_t> h_queries;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);
  int max_N = 10000000;

  read_labeled_data<uint8_t, int64_t>(data_fname, data_label_fname, query_fname, query_label_fname,
                  &h_data, &data_label_vecs, &label_data_vecs,
                  &h_queries, &query_label_vecs, &cat_freq, &query_freq, max_N);

  size_t N = data_label_vecs.size();
  size_t Nq = query_label_vecs.size();
  size_t dim = h_data.size() / N;
  printf("\n=== Dataset Information ===\n");
  printf("Base dataset size: N=%lld, dim=%lld\n", N, dim);
  printf("Query dataset size: N=%lld, dim=%lld\n", Nq, dim);
  auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, N, dim);
  auto queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, Nq, dim);

  // Copy datasets to GPU
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), stream);

  // run the interesting part of the program
  cagra_ivf_build_search_variants(dev_resources, 
                                  raft::make_const_mdspan(dataset.view()),
                                  raft::make_const_mdspan(queries.view()),
                                  data_label_vecs,
                                  label_data_vecs,
                                  query_label_vecs,
                                  cat_freq,
                                  query_freq,
                                  itopk_size,
                                  specificity_threshold,
                                  graph_degree);
  
  benchmark_brute_force_search(dev_resources,
                              dataset.view(),
                              queries.view(),
                              data_label_vecs,
                              label_data_vecs,
                              query_label_vecs,
                              cat_freq,
                              query_freq,
                              specificity_threshold);
}
