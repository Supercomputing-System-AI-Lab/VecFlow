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
#include "common.cuh"
#include "utils.h"

#include <cuvs/neighbors/cagra.hpp>
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

  raft::device_vector<uint32_t, int64_t> data_labels;
  raft::device_vector<uint32_t, int64_t> data_label_size;
  raft::device_vector<uint32_t, int64_t> data_label_offset;
  raft::device_vector<int64_t, int64_t> query_labels;
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
  
  auto data_labels = raft::make_device_vector<uint32_t, int64_t>(res, total_data_labels);
  auto data_label_size = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());
  auto data_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());

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

  std::vector<int64_t> h_query_labels;
  h_query_labels.reserve(query_label_vecs.size());
  int n_double_queries = 0;
  for (size_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) continue;
    int label1 = query_label_vecs[i][0];
    int label2 = query_label_vecs[i][1];
    if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) continue;
    h_query_labels.push_back((cat_freq[label1] > cat_freq[label2]) ? label1 : label2);
    n_double_queries++;
  }

  auto query_labels = raft::make_device_vector<int64_t, int64_t>(res, n_double_queries);
  // Copy filter data to device
  raft::update_device(data_labels.data_handle(), h_data_labels.data(), total_data_labels, raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_size.data_handle(), h_data_label_size.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_offset.data_handle(), h_data_label_offset.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(query_labels.data_handle(), h_query_labels.data(), n_double_queries, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset),
    std::move(data_labels),
    std::move(data_label_size),
    std::move(data_label_offset),
    std::move(query_labels)
  };
}

void compute_recall(const raft::host_matrix_view<uint32_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<int64_t>& valid_query_indices,
                    int n_single_queries,
                    const std::vector<std::vector<int>>& query_label_vecs) {
  int n_queries = neighbors.extent(0);
  int n_double_queries = n_queries - n_single_queries;
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;
  float single_recall = 0.0f;
  float double_recall = 0.0f;

  // Read specificities
  std::vector<double> specificities = read_specificities("/u/cmo1/Filtered/cuvs/examples/cpp/src/query_specificities.txt", 
                                                        100000);
  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/recall_results.csv");
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

    if (i < n_single_queries) {
      single_recall += recall;
    } else {
      double_recall += recall;
    }

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
  std::cout << "\n=== Recall Analysis ===" << std::endl;
  std::cout << "Single label queries (" << n_single_queries << "): " 
            << std::fixed << std::setprecision(4) << single_recall / n_single_queries << std::endl;
  std::cout << "Double label queries (" << n_double_queries << "): " 
            << std::fixed << double_recall / n_double_queries << std::endl;
  std::cout << "Overall recall (" << n_queries << "): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to recall_results.csv" << std::endl;
}

void cagra_build_search_variants(shared_resources::configured_raft_resources& dev_resources,
                                raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                raft::device_matrix_view<const uint8_t, int64_t> queries,
                                const std::vector<std::vector<int>>& data_label_vecs,
                                const std::vector<std::vector<int>>& label_data_vecs,
                                const std::vector<std::vector<int>>& query_label_vecs,
                                const std::vector<int>& cat_freq,
                                const std::vector<int>& query_freq,
                                int single_query_itopk_size = 32,
                                int double_query_itopk_size = 32,
                                int specificity_threshold = 0) {

  int64_t topk = 10;
  int64_t n_queries = queries.extent(0);

  // Create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Build index
  int64_t total_rows = 0;
  int64_t graph_width = 16;
  for (uint32_t i = 0; i < label_data_vecs.size(); i++) {
    if (query_freq[i] > 0) {
      total_rows += label_data_vecs[i].size();
    }
  }
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, total_rows, graph_width);
  raft::resource::sync_stream(dev_resources);

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  std::string index_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/index_32_16.bin";
  std::string graph_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/graph_32_16.bin";
  std::ifstream test_file(index_file);
  auto index = cagra::index<uint8_t, uint32_t>(dev_resources);
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
  search_params.itopk_size = 32;
  cagra::search_params filtered_single_search_params = search_params;
  filtered_single_search_params.itopk_size = single_query_itopk_size;
  filtered_single_search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;
  cagra::search_params filtered_double_search_params = search_params;
  filtered_double_search_params.itopk_size = double_query_itopk_size;
  filtered_double_search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_label_vecs, 
                                                  query_freq, 
                                                  cat_freq,
                                                  specificity_threshold);
  // index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));
  // save_matrix_to_ibin(dev_resources, graph_file, combined_indices.final_matrix.view());
  // std::cout << "Filtered index has " << index.size() << " vectors" << std::endl;
  // std::cout << "Filtered graph has degree " << index.graph_degree() 
  //           << ", graph size [" << index.graph().extent(0) << ", "
  //           << index.graph().extent(1) << "]" << std::endl;
  raft::resource::sync_stream(dev_resources);

  // Find valid queries
  int n_single_queries = 0;
  int n_double_queries = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) continue;
      n_single_queries ++;
    } else {
      int label1 = query_label_vecs[i][0];
      int label2 = query_label_vecs[i][1];
      if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) continue;
      n_double_queries ++;
    }
  }

  std::vector<int64_t> single_query_indices(n_single_queries);
  std::vector<int64_t> double_query_indices(n_double_queries);
  std::vector<uint32_t> h_single_filtered_labels(n_single_queries);
  std::vector<uint32_t> h_double_filtered_labels(n_double_queries);
  int current_single = 0;
  int current_double = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1) {
      if (cat_freq[query_label_vecs[i][0]] <= specificity_threshold) continue;
      h_single_filtered_labels[current_single] = query_label_vecs[i][0];
      single_query_indices[current_single] = i;
      current_single++;
    } else {
      int label1 = query_label_vecs[i][0];
      int label2 = query_label_vecs[i][1];
      if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) continue;
      int label = (cat_freq[label1] <= cat_freq[label2]) ? label1 : label2;
      h_double_filtered_labels[current_double] = label;
      double_query_indices[current_double] = i;
      current_double++;
    }
  }
  // Update valid n_queries
  n_queries = n_single_queries + n_double_queries;

  // Create filtered query labels vector
  auto single_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_single_queries);
  auto double_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_double_queries);

  // Copy mapped labels to device
  raft::update_device(single_query_labels.data_handle(),
                      h_single_filtered_labels.data(),
                      n_single_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::update_device(double_query_labels.data_handle(),
                      h_double_filtered_labels.data(),
                      n_double_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Copy valid queries to new matrix
  auto single_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_single_queries, queries.extent(1));
  auto double_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_double_queries, queries.extent(1));
  for (int64_t i = 0; i < n_single_queries; i++) {
    raft::copy_async(single_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + single_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);
  for (int64_t i = 0; i < n_double_queries; i++) {
    raft::copy_async(double_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + double_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);

  // Create data labels vector
  auto filter = filtering::cagra_filter(combined_indices.data_labels.view(),
                                        combined_indices.data_label_size.view(), 
                                        combined_indices.data_label_offset.view(),
                                        combined_indices.query_labels.view());
  raft::resource::sync_stream(dev_resources);

  // Resize result arrays for filtered queries
  auto single_query_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_single_queries, topk);
  auto single_query_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_single_queries, topk);
  auto double_query_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_double_queries, topk);
  auto double_query_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_double_queries, topk);

  // warm-up
  for (int i = 0; i < 10; i ++) {
    cagra::filtered_search(dev_resources, 
                          filtered_single_search_params,
                          index,
                          single_queries.view(),
                          single_query_neighbors.view(),
                          single_query_distances.view(),
                          single_query_labels.view(),
                          combined_indices.index_map.view(),
                          combined_indices.label_size.view(),
                          combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    cagra::filtered_search(dev_resources, 
                          filtered_double_search_params,
                          index,
                          double_queries.view(),
                          double_query_neighbors.view(),
                          double_query_distances.view(),
                          double_query_labels.view(),
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
                        filtered_single_search_params,
                        index,
                        single_queries.view(),
                        single_query_neighbors.view(),
                        single_query_distances.view(),
                        single_query_labels.view(),
                        combined_indices.index_map.view(),
                        combined_indices.label_size.view(),
                        combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_single += duration.count() / 1000.0; // Convert to milliseconds
  }

  double avg_time_single_ms = total_time_single / num_runs;
  double qps_single = (n_single_queries * 1000.0) / avg_time_single_ms;
  std::cout << "\n=== Single Label Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_single_queries << std::endl;
  std::cout << "iTopK: " << single_query_itopk_size << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time_single_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_single << std::endl;

  // For double label search:
  double total_time_double = 0;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    cagra::filtered_search(dev_resources, 
                        filtered_double_search_params,
                        index,
                        double_queries.view(),
                        double_query_neighbors.view(),
                        double_query_distances.view(),
                        double_query_labels.view(),
                        combined_indices.index_map.view(),
                        combined_indices.label_size.view(),
                        combined_indices.label_offset.view(),
                        filter);
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_double += duration.count() / 1000.0;
  }

  double avg_time_double_ms = total_time_double / num_runs;
  double qps_double = (n_double_queries * 1000.0) / avg_time_double_ms;
  std::cout << "\n=== Double Label Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_double_queries << std::endl;
  std::cout << "iTopK: " << double_query_itopk_size << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Average time: " << std::fixed << avg_time_double_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps_double << std::endl;

  // Calculate total QPS
  int total_queries = n_single_queries + n_double_queries;
  double total_time_ms = avg_time_single_ms + avg_time_double_ms;
  double total_qps = (total_queries * 1000.0) / total_time_ms;
  std::cout << "\n=== Overall Performance ===" << std::endl;
  std::cout << "Total queries: " << total_queries << std::endl;
  std::cout << "Total time: " << std::fixed << total_time_ms << " ms" << std::endl;
  std::cout << "Total QPS: " << std::scientific << total_qps << std::endl;

  // Resize neighbors
  auto final_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, topk);
  // Copy single query results first (they come first in valid_query_indices)
  raft::copy(final_neighbors.data_handle(),
            single_query_neighbors.data_handle(),
            topk * n_single_queries,
            raft::resource::get_cuda_stream(dev_resources));
  raft::copy(final_neighbors.data_handle() + n_single_queries * topk,
            double_query_neighbors.data_handle(),
            topk * n_double_queries,
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Add all single query indices first
  std::vector<int64_t> valid_query_indices;
  valid_query_indices.reserve(n_queries);
  std::copy(single_query_indices.begin(), single_query_indices.end(), std::back_inserter(valid_query_indices));
  std::copy(double_query_indices.begin(), double_query_indices.end(), std::back_inserter(valid_query_indices));

  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), final_neighbors.data_handle(), final_neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  // Read ground truth neighbors
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_gt_file(gt_fname, gt_indices);
  compute_recall(h_neighbors.view(), gt_indices, valid_query_indices, n_single_queries, query_label_vecs);
}


int main(int argc, char** argv) {
  // Parse command line argument
  int single_query_itopk_size = 64;  // Default value
  if (argc > 1) {
    single_query_itopk_size = std::stoi(argv[1]);
  }
  int double_query_itopk_size = 64;  // Default value
  if (argc > 2) {
    double_query_itopk_size = std::stoi(argv[2]);
  }
  int specificity_threshold = 2000;
  if (argc > 3) {
    specificity_threshold = std::stoi(argv[3]);
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
  try {
    cagra_build_search_variants(dev_resources, 
                                raft::make_const_mdspan(dataset.view()),
                                raft::make_const_mdspan(queries.view()),
                                data_label_vecs,
                                label_data_vecs,
                                query_label_vecs,
                                cat_freq,
                                query_freq,
                                single_query_itopk_size,
                                double_query_itopk_size,
                                specificity_threshold);
  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
}
