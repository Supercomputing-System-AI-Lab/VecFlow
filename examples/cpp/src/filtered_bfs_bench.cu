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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <stdexcept>

// Helper function to generate random uint8_t data using thrust
void generate_random_dataset(std::vector<uint8_t>& data, int n_rows, int dim) {
  thrust::default_random_engine rng(12345);  // Fixed seed for reproducibility
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

// Helper function to setup and run the benchmark
void run_benchmark(raft::device_resources& dev_resources,
                  int n_rows,
                  int dim,
                  int points_per_label,
                  int batch_size,
                  int k,
                  int n_warmup = 10,    
                  int n_runs = 100) {
  try {
    validate_configuration(n_rows, points_per_label, batch_size);
    
    // Calculate number of labels based on points per label
    int n_labels = (n_rows + points_per_label - 1) / points_per_label;
    auto stream = raft::resource::get_cuda_stream(dev_resources);

    // Generate random dataset
    std::vector<uint8_t> h_data(n_rows * dim);
    generate_random_dataset(h_data, n_rows, dim);

    // Create dataset on device
    auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_rows, dim);
    raft::update_device(dataset.data_handle(), h_data.data(), n_rows * dim, stream);

    // Generate random queries
    std::vector<uint8_t> h_queries(batch_size * dim);
    generate_random_dataset(h_queries, batch_size, dim);
    auto queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, batch_size, dim);
    raft::update_device(queries.data_handle(), h_queries.data(), batch_size * dim, stream);

    // Setup label sizes and offsets
    std::vector<uint32_t> h_label_sizes(n_labels, points_per_label);
    h_label_sizes[n_labels - 1] = n_rows - (n_labels - 1) * points_per_label; // Adjust last label if needed
    
    std::vector<uint32_t> h_label_offsets(n_labels);
    for(int i = 0; i < n_labels; i++) {
      h_label_offsets[i] = i * points_per_label;
    }

    // Create device vectors for labels
    auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, batch_size);
    auto label_sizes = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_labels);
    auto label_offsets = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_labels);

    // Generate random query labels
    std::vector<uint32_t> h_query_labels(batch_size);
    std::uniform_int_distribution<uint32_t> dist(0, n_labels - 1);
    std::mt19937 gen(12345);
    for(int i = 0; i < batch_size; i++) {
      h_query_labels[i] = dist(gen);
    }

    // Update device memory
    raft::update_device(label_sizes.data_handle(), h_label_sizes.data(), n_labels, stream);
    raft::update_device(label_offsets.data_handle(), h_label_offsets.data(), n_labels, stream);
    raft::update_device(query_labels.data_handle(), h_query_labels.data(), batch_size, stream);

    // Build IVF index
    cuvs::neighbors::ivf_flat::index<uint8_t, int64_t> ivf_index(
      dev_resources, 
      cuvs::distance::DistanceType::L2Unexpanded,
      n_labels,
      false, 
      false, 
      dim);

    // Create and initialize index map
    auto index_map = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_rows);
    std::vector<uint32_t> h_index_map(n_rows);
    std::iota(h_index_map.begin(), h_index_map.end(), 0);
    raft::update_device(index_map.data_handle(), h_index_map.data(), n_rows, stream);

    // Build filtered IVF index
    cuvs::neighbors::build_filtered_IVF_index(
      dev_resources,
      &ivf_index,
      dataset.view(),
      index_map.view(),
      label_sizes.view(),
      label_offsets.view());

    // Prepare search results storage
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, k);
    auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);

    // Warmup runs
    std::cout << "Warming up..." << std::endl;
    for(int i = 0; i < n_warmup; i++) {
      cuvs::neighbors::search_filtered_ivf(
        dev_resources,
        ivf_index,
        queries.view(),
        query_labels.view(),
        label_sizes.view(),
        neighbors.view(),
        distances.view(),
        cuvs::distance::DistanceType::L2Unexpanded);
    }
    cudaDeviceSynchronize();

    // Timed runs
    std::vector<double> timings;
    timings.reserve(n_runs);
    
    std::cout << "Running timed iterations..." << std::endl;
    for(int i = 0; i < n_runs; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      
      cuvs::neighbors::search_filtered_ivf(
        dev_resources,
        ivf_index,
        queries.view(),
        query_labels.view(),
        label_sizes.view(),
        neighbors.view(),
        distances.view(),
        cuvs::distance::DistanceType::L2Unexpanded);
      
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      timings.push_back(duration.count() / 1000.0); // Convert to milliseconds
    }

    // Calculate and print statistics
    double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / n_runs;
    double min_time = *std::min_element(timings.begin(), timings.end());
    double max_time = *std::max_element(timings.begin(), timings.end());

    std::cout << std::left 
              << std::setw(20) << "Batch Size: " << batch_size << "\n"
              << std::setw(20) << "Points per Label: " << points_per_label << "\n"
              << std::setw(20) << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n"
              << std::setw(20) << "Min Time: " << min_time << " ms\n"
              << std::setw(20) << "Max Time: " << max_time << " ms\n"
              << std::setw(20) << "Avg QPS: " << (batch_size * 1000.0) / avg_time << "\n"
              << std::setw(20) << "Peak QPS: " << (batch_size * 1000.0) / min_time << "\n"
              << std::string(60, '-') << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error in configuration (n_rows=" << n_rows 
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

  // Test configurations
  const int n_rows = 10000000;   // 1M dataset size
  const int dim = 192;          // 192 dimensions
  const int k = 10;             // top-10 neighbors
  
  // Test matrix: batch_size vs points_per_label
  std::vector<int> batch_sizes = {1, 100, 1000, 10000};
  std::vector<int> points_per_label = {100, 200, 400, 1000, 2000, 5000};

  std::cout << "\nPerformance Evaluation (uint8_t)\n"
            << "Dataset Size: " << n_rows << "\n"
            << "Dimensions: " << dim << "\n"
            << std::string(60, '-') << "\n";

  // Run benchmark for each configuration
  for(int batch_size : batch_sizes) {
    for(int ppl : points_per_label) {
      run_benchmark(dev_resources, n_rows, dim, ppl, batch_size, k);
    }
  }

  return 0;
}