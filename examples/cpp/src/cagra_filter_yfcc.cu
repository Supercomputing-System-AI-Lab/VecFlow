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
        std::string index_path = "indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
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
        
        std::string index_path = "indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
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
            std::string indices_path = "indices_yfcc/label_" + std::to_string(i) + "_indices.bin";
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

auto read_query_labels(raft::device_resources const& dev_resources) -> std::vector<std::vector<int>> {
    std::string query_labels_path = "yfcc100M/query.metadata.private.2727415019.100K.spmat";
    std::ifstream file(query_labels_path, std::ios::binary);
    
    // Read header information
    std::vector<int64_t> sizes(3);
    file.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
    int64_t nrow = sizes[0], ncol = sizes[1], nnz = sizes[2];
    
    // Read row pointers and indices in one go
    std::vector<int64_t> indptr(nrow + 1);
    std::vector<int> indices(nnz);
    file.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
    file.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
    
    // Pre-allocate the vector of vectors with known size
    std::vector<std::vector<int>> labels(nrow);
    
    // Reserve space for each inner vector based on number of labels
    for (uint32_t i = 0; i < nrow; ++i) {
        int64_t num_labels = indptr[i + 1] - indptr[i];
        labels[i].reserve(num_labels);
        for (int64_t j = indptr[i]; j < indptr[i + 1]; ++j) {
            labels[i].push_back(indices[j]);
        }
    }
    
    return labels;
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
  index_params.intermediate_graph_degree = 32;
  index_params.graph_degree = 16;
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
  raft::copy(neighbors.data_handle(), h_neighbors.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(), h_distances.data(), n_queries * topk, raft::resource::get_cuda_stream(dev_resources));
  
  raft::resource::sync_stream(dev_resources);
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

  std::string valid_labels_path = "indices_yfcc/valid_labels.bin";
  std::ifstream valid_labels_file(valid_labels_path, std::ios::binary);
  // Read size first
  int size;
  valid_labels_file.read(reinterpret_cast<char*>(&size), sizeof(int));
  // Resize and read the actual data
  std::vector<int> valid_labels(size);
  valid_labels_file.read(reinterpret_cast<char*>(valid_labels.data()), size * sizeof(int));
  valid_labels_file.close();
  std::unordered_map<int, int> label_map;
  for (int i = 0; i < size; i++) {
    label_map[valid_labels[i]] = i;
  }
  uint32_t label_number = size;
  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, label_number);
  index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));
  std::cout << "Filtered index has " << index.size() << " vectors" << std::endl;
  std::cout << "Filtered graph has degree " << index.graph_degree() 
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;
  raft::resource::sync_stream(dev_resources);
  std::cout << "Reading query labels..." << std::endl;
  auto original_query_labels = read_query_labels(dev_resources);
  std::cout << "Query labels read" << std::endl;
  // Count queries with exactly one label and in valid_labels
  std::cout << "Counting queries with exactly one label and in valid_labels..." << std::endl;
  std::vector<int64_t> valid_query_indices;
  for (int64_t i = 0; i < n_queries; i++) {
    if (original_query_labels[i].size() == 1 && label_map.find(original_query_labels[i][0]) != label_map.end()) {
      valid_query_indices.push_back(i);
    }
  }
  std::cout << "Counted " << valid_query_indices.size() << " queries with exactly one label and in valid_labels" << std::endl;
  n_queries = valid_query_indices.size();
  // Create new filtered queries matrix
  auto filtered_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, queries.extent(1));
  
  // Copy valid queries to new matrix
  for (int64_t i = 0; i < n_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + valid_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  std::cout << "Copied valid queries to new matrix" << std::endl;
  // Create filtered query labels vector
  auto filtered_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  std::vector<uint32_t> h_filtered_labels(n_queries);
  
  // Map original labels to new indices using label_map
  for (int64_t i = 0; i < n_queries; i++) {
    h_filtered_labels[i] = label_map[original_query_labels[valid_query_indices[i]][0]];
  }
  std::cout << "Mapped original labels to new indices" << std::endl;
  // Copy mapped labels to device
  raft::update_device(filtered_query_labels.data_handle(),
                     h_filtered_labels.data(),
                     n_queries,
                     raft::resource::get_cuda_stream(dev_resources));
  std::cout << "Copied mapped labels to device" << std::endl;
  // delete original queries and query_labels

  // Resize result arrays for filtered queries
  neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, topk);
  distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);
  // 3. Batch filtered search
  time_it("filtered/batch", [&]() {
    cagra::filtered_search(dev_resources, 
                         filtered_search_params,
                         index,
                         filtered_queries.view(),
                         neighbors.view(),
                         distances.view(),
                         filtered_query_labels.view(),
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
  auto neighbors_host =
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
  int total_correct = 0;
  int total_queries = 0;

  std::vector<uint32_t> labels(n_queries, 0);
  raft::copy(labels.data(), filtered_query_labels.data_handle(), n_queries, raft::resource::get_cuda_stream(dev_resources));
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

auto load_data_bin(const std::string& file_path) -> std::tuple<std::vector<float>, int64_t, int64_t> {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    int32_t N, dim;
    file.read(reinterpret_cast<char*>(&N), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    
    std::vector<float> data(N * dim);
    for (int64_t i = 0; i < N; i++) {
        file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }
    
    return {data, N, dim};
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

    // Load YFCC base dataset
    std::cout << "Loading YFCC base dataset..." << std::endl;
    auto [yfcc_base_data, n_samples, base_dim] = load_data_bin("yfcc100M/base.10M.u8bin");

    // Load YFCC query dataset
    std::cout << "Loading YFCC query dataset..." << std::endl;
    auto [yfcc_query_data, n_queries, query_dim] = load_data_bin("yfcc100M/query.private.2727415019.100K.u8bin");

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
    raft::copy(dataset.data_handle(), yfcc_base_data.data(), n_samples * base_dim, 
               raft::resource::get_cuda_stream(res));
    raft::copy(queries.data_handle(), yfcc_query_data.data(), n_queries * query_dim,
               raft::resource::get_cuda_stream(res));

    // run the interesting part of the program
    cagra_build_search_variants(res, 
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              itopk_size, 
                              uniform_label);
}
