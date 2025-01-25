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

#include "common.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <fstream>
#include <vector>

using namespace cuvs::neighbors;

struct CombinedIndices {
    raft::device_matrix<uint32_t, int64_t> final_matrix;
    raft::device_vector<uint32_t, int64_t> index_map;
    raft::device_vector<uint32_t, int64_t> label_size;
    raft::device_vector<uint32_t, int64_t> label_offset;
};

auto load_and_combine_indices(raft::resources const& res, int label_number) -> CombinedIndices {
    
    std::cout << "Starting load_and_combine_indices with label_number: " << label_number << std::endl;
    
    // First pass: calculate total size and get graph width
    int64_t total_rows = 0;
    int64_t graph_width = 0;
    
    std::cout << "First pass: calculating sizes..." << std::endl;
    for (uint32_t i = 1; i <= label_number; i++) {
        std::string index_path = "indices/label_" + std::to_string(i) + "_index_16_8.bin";
        cagra::index<float, uint32_t> temp_index(res);
        cagra::deserialize(res, index_path, &temp_index);
        int64_t n_rows = temp_index.graph().extent(0);
        total_rows += n_rows;
        graph_width = temp_index.graph().extent(1);
    }
    
    std::cout << "Total rows: " << total_rows << ", Graph width: " << graph_width << std::endl;

    // Create final matrices and vectors
    auto final_matrix = raft::make_device_matrix<uint32_t, int64_t>(res, total_rows, graph_width);
    auto index_map = raft::make_device_vector<uint32_t, int64_t>(res, total_rows);
    auto label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
    auto label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
    
    // Second pass: combine indices
    std::cout << "\nSecond pass: combining indices..." << std::endl;
    int64_t current_offset = 0;
    std::vector<uint32_t> host_label_size(label_number);
    std::vector<uint32_t> host_label_offset(label_number);

    for (uint32_t i = 1; i <= label_number; i++) {
        
        auto temp_index = cagra::index<float, uint32_t>(res);
        
        std::string index_path = "indices/label_" + std::to_string(i) + "_index_16_8.bin";
        cagra::deserialize(res, index_path, &temp_index);
        
        auto graph = temp_index.graph();
        int64_t n_rows = graph.extent(0);

        // Copy operations
        try {
            raft::copy_async(final_matrix.data_handle() + current_offset * graph_width,
                           graph.data_handle(),
                           n_rows * graph_width,
                           raft::resource::get_cuda_stream(res));
            
            // Read indices from file into host memory
            std::vector<uint32_t> h_indices(n_rows);
            std::string indices_path = "indices/label_" + std::to_string(i) + "_indices.bin";
            std::ifstream file(indices_path, std::ios::binary);
            if (!file.read(reinterpret_cast<char*>(h_indices.data()), n_rows * sizeof(uint32_t))) {
                throw std::runtime_error("Failed to read indices file: " + indices_path);
            }
            file.close();
            
            // Create device memory and copy in two steps
            auto d_indices = raft::make_device_vector<uint32_t>(res, n_rows);
            raft::update_device(d_indices.data_handle(),
                              h_indices.data(),
                              n_rows,
                              raft::resource::get_cuda_stream(res));
            raft::resource::sync_stream(res);
            
            raft::copy_async(index_map.data_handle() + current_offset,
                           d_indices.data_handle(),
                           n_rows,
                           raft::resource::get_cuda_stream(res));
            
        } catch (const std::exception& e) {
            std::cerr << "Error during copy operations: " << e.what() << std::endl;
            throw;
        }

        // Update label size and offset
        host_label_size[i-1] = n_rows;
        host_label_offset[i-1] = current_offset;
        
        current_offset += n_rows;
    }

    std::cout << "Copying label information to device..." << std::endl;
    
    // Copy label size and offset to device
    raft::copy(label_size.data_handle(), 
               host_label_size.data(), 
               label_number,
               raft::resource::get_cuda_stream(res));
    raft::copy(label_offset.data_handle(),
               host_label_offset.data(),
               label_number,
               raft::resource::get_cuda_stream(res));

    std::cout << "Successfully completed load_and_combine_indices" << std::endl;
    return CombinedIndices{
        std::move(final_matrix),
        std::move(index_map),
        std::move(label_size),
        std::move(label_offset)
    };
}

// A helper to measure the execution time of a function
template <typename F, typename... Args>
void time_it(std::string label, F f, Args &&...xs) {
  auto start = std::chrono::system_clock::now();
  f(std::forward<Args>(xs)...);
  auto end = std::chrono::system_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto t_ms = double(t.count()) / 1000.0;
  std::cout << "[" << label << "] execution time: " << t_ms << " ms"
            << std::endl;
}
void cagra_build_search_variants(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<const float, int64_t> dataset,
                               raft::device_matrix_view<const float, int64_t> queries,
                               int itopk_size = 32,
                               int uniform_label = -1) {

  int64_t topk = 10;
  int64_t n_queries = queries.extent(0);

  // Create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Build index
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_degree = 32;
  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() 
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;

  // Configure standard search parameters
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  cagra::search_params filtered_search_params = search_params;
  filtered_search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;
  // Configure persistent search parameters
  cagra::search_params search_params_persistent = search_params;
  search_params_persistent.persistent = true;
  search_params_persistent.algo = cagra::search_algo::SINGLE_CTA;
  search_params_persistent.persistent_device_usage = 0.95;

  cagra::search_params filtered_search_params_persistent = filtered_search_params;
  filtered_search_params_persistent.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  raft::resource::sync_stream(dev_resources);
  // warmup
  for (int i = 0; i < 5; i++) {
    cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  }
  raft::resource::sync_stream(dev_resources);
  // 1. Batch mode search
  time_it("standard/batch", [&]() {
    cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
    raft::resource::sync_stream(dev_resources);
  });
  print_results(dev_resources, neighbors.view(), distances.view());
  // Reset arrays
  std::vector<uint32_t> h_neighbors(n_queries * topk, 0);
  std::vector<float> h_distances(n_queries * topk, 0);
  // raft::copy(neighbors.data_handle(), h_neighbors.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  // raft::copy(distances.data_handle(), h_distances.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  
  raft::resource::sync_stream(dev_resources);
  std::cout << "Calculating recall@10..." << std::endl;
  auto neighbors_host =
      raft::make_host_matrix<uint32_t, int64_t>(neighbors.extent(0), topk);
  raft::copy(neighbors_host.data_handle(), neighbors.data_handle(), neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  // Load ground truth data
  std::ifstream file("sift/sift_groundtruth.ivecs", std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open ground truth file");
  }

  // Read ground truth data
  int32_t k;
  file.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
  
  std::vector<uint32_t> ground_truth(n_queries * topk);
  std::vector<int32_t> temp_data(n_queries * (k+1));
  
  // Read the full data but only keep first topk columns
  file.read(reinterpret_cast<char*>(temp_data.data()), temp_data.size() * sizeof(int32_t));
  
  // Copy only the first topk columns
  for (int32_t row = 0; row < n_queries; row++) {
    for (int32_t col = 0; col < topk; col++) {
      ground_truth[row * topk + col] = static_cast<uint32_t>(temp_data[row * (k+1) + col + 1]);
    }
  }

  // Calculate recall
  int total_correct = 0;
  
  for (int64_t i = 0; i < n_queries; i++) {
    // Get top k results for this query
    std::set<uint32_t> result_set;
    for (int k = 0; k < topk; k++) {
      result_set.insert(neighbors_host(i, k));

    }
    
    // Compare with ground truth
    int correct = 0;
    for (size_t j = 0; j < topk; j++) {
      if (result_set.count(ground_truth[i * topk + j]) > 0) {
        correct++;
      }
    }
    total_correct += correct;
  }

  float recall = static_cast<float>(total_correct) / (n_queries * topk);
  std::cout << "Recall@10: " << recall << std::endl;

  // 2. Async persistent search (one query per job)
  time_it("persistent/async", [&]() {
    constexpr int64_t kMaxJobs = 1000;
    std::array<std::future<void>, kMaxJobs> futures;
    
    for (int64_t i = 0; i < n_queries + kMaxJobs; i++) {
      if (i >= kMaxJobs) futures[i % kMaxJobs].wait();
      if (i < n_queries) {
        futures[i % kMaxJobs] = std::async(std::launch::async, [&, i]() {
          auto query = raft::make_device_matrix_view<const float, int64_t>(
            queries.data_handle() + i * queries.extent(1), 1, queries.extent(1));
          auto neighbor = raft::make_device_matrix_view<uint32_t, int64_t>(
            neighbors.data_handle() + i * topk, 1, topk);
          auto distance = raft::make_device_matrix_view<float, int64_t>(
            distances.data_handle() + i * topk, 1, topk);
          cagra::search(dev_resources, search_params_persistent, index, query, neighbor, distance);
        });
      }
    }
  });
  print_results(dev_resources, neighbors.view(), distances.view());
  raft::copy(neighbors.data_handle(), h_neighbors.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(), h_distances.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));

  uint32_t label_number = 50;
  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, label_number);
  index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));
  std::cout << "Filtered index has " << index.size() << " vectors" << std::endl;
  std::cout << "Filtered graph has degree " << index.graph_degree() 
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;
  raft::resource::sync_stream(dev_resources);
  auto query_labels = raft::make_device_vector<uint32_t>(dev_resources, n_queries);
  // Initialize query_labels with random integers between 0 and label_number-1
  raft::random::RngState r(1234ULL);
  raft::random::uniformInt(
      dev_resources,
      r,
      query_labels.view(),
      uint32_t{0},
      uint32_t{label_number - 1}
  );
  if (uniform_label != -1) {
    std::vector<uint32_t> h_labels(n_queries, uniform_label);
    raft::copy(query_labels.data_handle(), h_labels.data(), n_queries, raft::resource::get_cuda_stream(dev_resources));
  }
  // 3. Batch filtered search
  time_it("filtered/batch", [&]() {
    cagra::filtered_search(dev_resources, 
                         filtered_search_params,
                         index,
                         queries,
                         neighbors.view(),
                         distances.view(),
                         query_labels.view(),
                         combined_indices.index_map.view(),
                         combined_indices.label_size.view(),
                         combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
  });
  print_results(dev_resources, neighbors.view(), distances.view());
  // raft::copy(neighbors.data_handle(), h_neighbors.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  // raft::copy(distances.data_handle(), h_distances.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  // 4. Async filtered search (one query per job)
  // time_it("filtered/async", [&]() {
  //   constexpr int64_t kMaxJobs = 1000;
  //   std::array<std::future<void>, kMaxJobs> futures;
    
  //   for (int64_t i = 0; i < n_queries + kMaxJobs; i++) {
  //     if (i >= kMaxJobs) futures[i % kMaxJobs].wait();
  //     if (i < n_queries) {
  //       futures[i % kMaxJobs] = std::async(std::launch::async, [&, i]() {
  //         auto query = raft::make_device_matrix_view<const float, int64_t>(
  //           queries.data_handle() + i * queries.extent(1), 1, queries.extent(1));
  //         auto neighbor = raft::make_device_matrix_view<uint32_t, int64_t>(
  //           neighbors.data_handle() + i * topk, 1, topk);
  //         auto distance = raft::make_device_matrix_view<float, int64_t>(
  //           distances.data_handle() + i * topk, 1, topk);
  //         auto query_label = raft::make_device_vector_view<uint32_t, int64_t>(
  //           query_labels.data_handle() + i, 1);

  //         cagra::filtered_search(dev_resources,
  //                              filtered_search_params_persistent,
  //                              index,
  //                              query,
  //                              neighbor,
  //                              distance,
  //                              query_label,
  //                              combined_indices.index_map.view(),
  //                              combined_indices.label_size.view(),
  //                              combined_indices.label_offset.view());
  //       });
  //     }
  //   }
  // });
  // print_results(dev_resources, neighbors.view(), distances.view());

  // Calculate recall@10
  std::cout << "Calculating recall@10..." << std::endl;
  neighbors_host =
      raft::make_host_matrix<uint32_t, int64_t>(neighbors.extent(0), topk);
  raft::copy(neighbors_host.data_handle(), neighbors.data_handle(), neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  // Load ground truth files for each label
  std::vector<std::vector<uint32_t>> ground_truths(label_number);
  for (int i = 0; i < label_number; i++) {
    std::string gt_file = "sift/sift_gt_" + std::to_string(i+1) + ".bin";
    std::ifstream file(gt_file, std::ios::binary);
    if (!file) {
      std::cout << "File doesn't exist: " << gt_file << std::endl;
      continue; // Skip if file doesn't exist
    }
    
    // Read metadata first (n and k)
    int32_t n, k;
    file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&k), sizeof(int32_t));
    
    // Only store top-k results
    ground_truths[i].resize(n * topk);
    
    // Read the full data but only keep first topk columns
    std::vector<int32_t> temp_data((2 * n * k));
    file.read(reinterpret_cast<char*>(temp_data.data()), temp_data.size() * sizeof(int32_t));
    
    // Copy only the first topk columns of IDs
    for (int32_t row = 0; row < n; row++) {
      for (int32_t col = 0; col < topk; col++) {
        ground_truths[i][row * topk + col] = static_cast<uint32_t>(temp_data[row * k + col]);
      }
    }
  }

  // Calculate recall
  total_correct = 0;
  int total_queries = 0;

  std::vector<uint32_t> labels(n_queries, 0);
  raft::copy(labels.data(), query_labels.data_handle(), n_queries, raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  for (int64_t i = 0; i < n_queries; i++) {
    uint32_t label = labels[i];
    if (label >= ground_truths.size() || ground_truths[label].empty()) {
      continue; // Skip if no ground truth exists
    }
    
    // Get top k results for this query
    std::set<uint32_t> result_set;
    for (int k = 0; k < topk; k++) {
      result_set.insert(neighbors_host(i, k));
    }
    // Compare with ground truth
    int correct = 0;
    for (size_t j = 0; j < topk; j++) {
      if (result_set.count(ground_truths[label][j + i * topk]) > 0) {
        correct++;
      }
    }
    
    total_correct += correct;
    total_queries++;
  }

  if (total_queries > 0) {
    float recall = static_cast<float>(total_correct) / (total_queries * topk);
    std::cout << "Recall@10: " << recall << std::endl;
  } else {
    std::cout << "No valid queries found for recall calculation" << std::endl;
  }
}

auto load_fvecs(const std::string& file_path) -> std::pair<std::vector<float>, int64_t> {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    int32_t dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    
    // Get file size to calculate number of vectors
    file.seekg(0, std::ios::end);
    int64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    int64_t vector_size = (dim + 1) * sizeof(float);  // +1 for dimension header
    int64_t n_vectors = file_size / vector_size;
    
    std::vector<float> data(n_vectors * dim);
    
    // Read vectors one by one, skipping dimension headers
    std::vector<float> temp_vec(dim + 1);
    for (int64_t i = 0; i < n_vectors; i++) {
        file.read(reinterpret_cast<char*>(temp_vec.data()), vector_size);
        std::copy(temp_vec.begin() + 1, temp_vec.end(), data.begin() + i * dim);
    }
    
    return {data, dim};
}

int main(int argc, char** argv) {
    // Parse command line argument
    int itopk_size = 32;  // Default value
    if (argc > 1) {
        itopk_size = std::stoi(argv[1]);
    }
    int uniform_label = -1;
    if (argc > 2) {
        uniform_label = std::stoi(argv[2]);
    }
    
    raft::device_resources res;

    // Set pool memory resource with 1 GiB initial pool size
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);

    // Load SIFT base dataset
    std::cout << "Loading SIFT base dataset..." << std::endl;
    auto [sift_base_data, base_dim] = load_fvecs("sift/sift_base.fvecs");
    int64_t n_samples = sift_base_data.size() / base_dim;

    // Load SIFT query dataset
    std::cout << "Loading SIFT query dataset..." << std::endl;
    auto [sift_query_data, query_dim] = load_fvecs("sift/sift_query.fvecs");
    int64_t n_queries = sift_query_data.size() / query_dim;

    // Verify dimensions match
    if (base_dim != query_dim) {
        throw std::runtime_error("Base and query datasets have different dimensions!");
    }

    std::cout << "Dataset loaded: " << n_samples << " base vectors, " 
              << n_queries << " query vectors, " 
              << base_dim << " dimensions" << std::endl;

    // Create and populate GPU matrices
    auto dataset = raft::make_device_matrix<float, int64_t>(res, n_samples, base_dim);
    auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, query_dim);
    
    // Copy datasets to GPU
    raft::copy(dataset.data_handle(), sift_base_data.data(), n_samples * base_dim, 
               raft::resource::get_cuda_stream(res));
    raft::copy(queries.data_handle(), sift_query_data.data(), n_queries * query_dim,
               raft::resource::get_cuda_stream(res));

    // run the interesting part of the program
    cagra_build_search_variants(res, 
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              itopk_size, 
                              uniform_label);
}
