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

#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <array>
#include <cstdint>
#include <fstream>

// Structure to hold both graph and filter data
struct FilteredCombinedIndices {
  // Original CombinedIndices members
  raft::device_matrix<uint32_t, int64_t> final_matrix;
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
  
  // Additional filter-specific members
  raft::device_vector<uint32_t, int64_t> data_labels;
  raft::device_vector<uint32_t, int64_t> data_label_size;
  raft::device_vector<uint32_t, int64_t> data_label_offset;
  raft::device_vector<int64_t, int64_t> query_labels;
};


// Helper function to generate random uint8_t data using thrust
void generate_random_dataset(std::vector<uint8_t>& data, int n_rows, int dim) {
  thrust::default_random_engine rng(12345);
  thrust::uniform_int_distribution<int> dist(0, 255);
  
  for(int i = 0; i < n_rows * dim; i++) {
    data[i] = static_cast<uint8_t>(dist(rng));
  }
}

// Helper function to validate configuration parameters
void validate_configuration(int n_rows, int points_per_label, int batch_size) {
  if (n_rows <= 0 || points_per_label <= 0 || batch_size <= 0) {
    throw std::invalid_argument("All size parameters must be positive");
  }
  
  if (points_per_label > n_rows) {
    throw std::invalid_argument("Points per label cannot exceed total number of rows");
  }
  
  if (batch_size > n_rows) {
    throw std::invalid_argument("Batch size cannot exceed dataset size");
  }
}

// Function to build label-specific graphs and combine them
auto build_or_load_filtered_label_graphs(raft::device_resources& dev_resources,
                                        const raft::device_matrix_view<const uint8_t, int64_t>& dataset,
                                        int n_labels,
                                        const std::vector<uint32_t>& h_label_sizes,
                                        const std::vector<uint32_t>& h_label_offsets,
                                        int graph_degree,
                                        int points_per_label,
                                        int batch_size) -> FilteredCombinedIndices {
  
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  int64_t total_rows = dataset.extent(0);
  
  // Create directory if it doesn't exist
  std::string base_path = "/scratch/bdes/cmo1/CAGRA/indices_test/points_per_label_" + std::to_string(points_per_label);
  std::filesystem::create_directories(base_path);
  
  // Create matrices and vectors for graph
  auto final_matrix = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, total_rows, graph_degree);
  auto index_map = raft::make_device_vector<uint32_t, int64_t>(dev_resources, total_rows);
  auto label_size = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_labels);
  auto label_offset = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_labels);

  // Build or load graph for each label
  for (int i = 0; i < n_labels; i++) {
    int64_t label_rows = h_label_sizes[i];
    if (label_rows == 0) continue;

    std::string index_path = base_path + "/label_" + std::to_string(i) + "_index.bin";
    bool index_exists = std::filesystem::exists(index_path);

    if (index_exists) {
      auto temp_index = cuvs::neighbors::cagra::index<uint8_t, uint32_t>(dev_resources);
      cuvs::neighbors::cagra::deserialize(dev_resources, index_path, &temp_index);
      
      auto label_graph = temp_index.graph();
      raft::copy_async(final_matrix.data_handle() + h_label_offsets[i] * graph_degree,
                      label_graph.data_handle(),
                      label_rows * graph_degree,
                      stream);
    } else {
      std::cout << "Building index for label " << i << std::endl;
      auto label_data = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, label_rows, dataset.extent(1));
      raft::copy_async(label_data.data_handle(),
                      dataset.data_handle() + h_label_offsets[i] * dataset.extent(1),
                      label_rows * dataset.extent(1),
                      stream);

      cuvs::neighbors::cagra::index_params index_params;
      index_params.intermediate_graph_degree = graph_degree * 2;
      index_params.graph_degree = graph_degree;
      
      auto label_index = cuvs::neighbors::cagra::build(
          dev_resources,
          index_params,
          raft::make_const_mdspan(label_data.view())
      );

      cuvs::neighbors::cagra::serialize(dev_resources, index_path, label_index);

      auto label_graph = label_index.graph();
      raft::copy_async(final_matrix.data_handle() + h_label_offsets[i] * graph_degree,
                      label_graph.data_handle(),
                      label_rows * graph_degree,
                      stream);
    }
  }

  // Initialize index map
  std::vector<uint32_t> h_index_map(total_rows);
  std::iota(h_index_map.begin(), h_index_map.end(), 0);
  raft::update_device(index_map.data_handle(), h_index_map.data(), total_rows, stream);

  // Copy label information to device
  raft::update_device(label_size.data_handle(), h_label_sizes.data(), n_labels, stream);
  raft::update_device(label_offset.data_handle(), h_label_offsets.data(), n_labels, stream);

  // Prepare filter-specific data
  // Create data_labels: assign label index to each point in its group
  std::vector<uint32_t> h_data_labels(total_rows);
  for (int i = 0; i < n_labels; i++) {
    std::fill(h_data_labels.begin() + h_label_offsets[i], 
              h_data_labels.begin() + h_label_offsets[i] + h_label_sizes[i], 
              i);
  }
  
  // Create point-wise label size (1 label per point) and offset arrays
  std::vector<uint32_t> h_point_label_sizes(total_rows, 1);
  std::vector<uint32_t> h_point_label_offsets(total_rows);
  std::iota(h_point_label_offsets.begin(), h_point_label_offsets.end(), 0);
  
  // Generate random query labels
  std::vector<int64_t> h_query_labels(batch_size);
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int64_t> query_label_dist(-1, n_labels - 1);
  for (int i = 0; i < batch_size; i++) {
    h_query_labels[i] = query_label_dist(gen);
    // h_query_labels[i] = -1;
  }
  
  // Create and populate filter device vectors
  auto data_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, total_rows);
  auto data_label_size = raft::make_device_vector<uint32_t, int64_t>(dev_resources, total_rows);
  auto data_label_offset = raft::make_device_vector<uint32_t, int64_t>(dev_resources, total_rows);
  auto query_labels = raft::make_device_vector<int64_t, int64_t>(dev_resources, batch_size);
  
  // Update filter device vectors
  raft::update_device(data_labels.data_handle(), h_data_labels.data(), total_rows, stream);
  raft::update_device(data_label_size.data_handle(), h_point_label_sizes.data(), total_rows, stream);
  raft::update_device(data_label_offset.data_handle(), h_point_label_offsets.data(), total_rows, stream);
  raft::update_device(query_labels.data_handle(), h_query_labels.data(), batch_size, stream);

  // Return combined structure
  return FilteredCombinedIndices{
    std::move(final_matrix),
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset),
    std::move(data_labels),
    std::move(data_label_size),
    std::move(data_label_offset),
    std::move(query_labels)
  };
}

void run_benchmark(raft::device_resources& dev_resources,
                  int n_rows,
                  int dim,
                  int points_per_label,
                  int batch_size,
                  int k,
                  int graph_degree = 32,
                  int n_warmup = 10,    
                  int n_runs = 100) {
  try {
    validate_configuration(n_rows, points_per_label, batch_size);
    
    int n_labels = (n_rows + points_per_label - 1) / points_per_label;
    auto stream = raft::resource::get_cuda_stream(dev_resources);

    // Generate and copy dataset
    std::vector<uint8_t> h_data(n_rows * dim);
    generate_random_dataset(h_data, n_rows, dim);
    auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_rows, dim);
    raft::update_device(dataset.data_handle(), h_data.data(), n_rows * dim, stream);

    // Generate and copy queries
    std::vector<uint8_t> h_queries(batch_size * dim);
    generate_random_dataset(h_queries, batch_size, dim);
    auto queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, batch_size, dim);
    raft::update_device(queries.data_handle(), h_queries.data(), batch_size * dim, stream);

    // Setup label sizes and offsets
    std::vector<uint32_t> h_label_sizes(n_labels, points_per_label);
    h_label_sizes[n_labels - 1] = n_rows - (n_labels - 1) * points_per_label;
    
    std::vector<uint32_t> h_label_offsets(n_labels);
    for(int i = 0; i < n_labels; i++) {
      h_label_offsets[i] = i * points_per_label;
    }

    // Generate random query labels
    auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, batch_size);
    std::vector<uint32_t> h_query_labels(batch_size);
    std::uniform_int_distribution<uint32_t> dist(0, n_labels - 1);
    std::mt19937 gen(12345);
    for(int i = 0; i < batch_size; i++) {
      h_query_labels[i] = dist(gen);
    }
    raft::update_device(query_labels.data_handle(), h_query_labels.data(), batch_size, stream);

    // Build combined indices
    std::cout << "Building/loading label-specific graphs..." << std::endl;
    auto combined_indices = build_or_load_filtered_label_graphs(dev_resources, 
                                                                dataset.view(),
                                                                n_labels,
                                                                h_label_sizes,
                                                                h_label_offsets,
                                                                graph_degree,
                                                                points_per_label,
                                                                batch_size);

    // Check if main index exists
    std::string main_index_path = "/scratch/bdes/cmo1/CAGRA/indices_test/points_per_label_" + 
                                 std::to_string(points_per_label) + "/main_index.bin";
    
    auto index = cuvs::neighbors::cagra::index<uint8_t, uint32_t>(dev_resources);
    if (std::filesystem::exists(main_index_path)) {
      std::cout << "Loading main index from " << main_index_path << std::endl;
      cuvs::neighbors::cagra::deserialize(dev_resources, main_index_path, &index);
      index.update_dataset(dev_resources, raft::make_const_mdspan(dataset.view()));
    } else {
      std::cout << "Building main index" << std::endl;
      cuvs::neighbors::cagra::index_params index_params;
      index_params.intermediate_graph_degree = graph_degree * 2;
      index_params.graph_degree = graph_degree;
      
      index = cuvs::neighbors::cagra::build(
        dev_resources,
        index_params,
        raft::make_const_mdspan(dataset.view())
      );
      cuvs::neighbors::cagra::serialize(dev_resources, main_index_path, index);
    }

    // Update index with combined graph
    index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));

    // Prepare search results storage
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, batch_size, k);
    auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);

    // Configure search parameters
    cuvs::neighbors::cagra::search_params search_params;
    search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA_FILTERED;
    search_params.itopk_size = graph_degree * 2;

    auto filter = cuvs::neighbors::filtering::cagra_filter(combined_indices.data_labels.view(),
                                                          combined_indices.data_label_size.view(), 
                                                          combined_indices.data_label_offset.view(),
                                                          combined_indices.query_labels.view());

    // Warmup runs
    std::cout << "Warming up..." << std::endl;
    for(int i = 0; i < n_warmup; i++) {
      cuvs::neighbors::cagra::filtered_search(
        dev_resources,
        search_params,
        index,
        queries.view(),
        neighbors.view(),
        distances.view(),
        query_labels.view(),
        combined_indices.index_map.view(),
        combined_indices.label_size.view(),
        combined_indices.label_offset.view(),
        filter);
    }
    cudaDeviceSynchronize();

    // Timed runs
    std::vector<double> timings;
    timings.reserve(n_runs);
    
    std::cout << "Running timed iterations..." << std::endl;
    for(int i = 0; i < n_runs; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      
      cuvs::neighbors::cagra::filtered_search(
        dev_resources,
        search_params,
        index,
        queries.view(),
        neighbors.view(),
        distances.view(),
        query_labels.view(),
        combined_indices.index_map.view(),
        combined_indices.label_size.view(),
        combined_indices.label_offset.view(),
        filter);
      
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      timings.push_back(duration.count() / 1000.0);
    }

    // Calculate and print statistics
    double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / n_runs;
    double min_time = *std::min_element(timings.begin(), timings.end());
    double max_time = *std::max_element(timings.begin(), timings.end());

    std::cout << std::left 
              << std::setw(20) << "Batch Size: " << batch_size << "\n"
              << std::setw(20) << "Points per Label: " << points_per_label << "\n"
              << std::setw(20) << "Graph Degree: " << graph_degree << "\n"
              << std::setw(20) << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n"
              << std::setw(20) << "Min Time: " << min_time << " ms\n"
              << std::setw(20) << "Max Time: " << max_time << " ms\n"
              << std::setw(20) << "Avg QPS: " << (batch_size * 1000.0) / avg_time << "\n"
              << std::setw(20) << "Peak QPS: " << (batch_size * 1000.0) / min_time << "\n"
              << std::string(60, '-') << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error in benchmark configuration (n_rows=" << n_rows 
              << ", points_per_label=" << points_per_label 
              << ", batch_size=" << batch_size 
              << "): " << e.what() << std::endl;
  }
}

int main() {
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  const int max_labels = 100;   // Maximum number of labels
  const int dim = 192;          // 192 dimensions
  const int k = 10;             // top-10 neighbors
  const int graph_degree = 16;  // graph degree for CAGRA index
  
  std::vector<int> batch_sizes = {10000};
  std::vector<int> points_per_label = {20000};

  std::cout << "\nPerformance Evaluation (CAGRA Filtered Search)\n"
            << "Dimensions: " << dim << "\n"
            << "Graph Degree: " << graph_degree << "\n"
            << std::string(60, '-') << "\n";

  for(int batch_size : batch_sizes) {
    for(int ppl : points_per_label) {
      int n_rows = max_labels * ppl;
      run_benchmark(dev_resources, n_rows, dim, ppl, batch_size, k, graph_degree);
    }
  }

  return 0;
}