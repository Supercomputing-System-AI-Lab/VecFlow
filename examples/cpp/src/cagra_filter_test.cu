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
  raft::device_matrix<uint32_t, int64_t> final_matrix;
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
};

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq,
                              int specificity_threshould = 0) -> CombinedIndices {
  
  int label_number = query_freq.size();
  std::cout << "Starting load_and_combine_indices with label_number: " << label_number << std::endl;
    
  // First pass: calculate total size and get graph width
  int64_t total_rows = 0;
  int64_t graph_width = 0;
    
  std::cout << "First pass: calculating sizes..." << std::endl;
  for (uint32_t i = 0; i < label_number; i++) {
    if (query_freq[i] == 0 || cat_freq[i] < specificity_threshould) continue;
    std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
    cagra::index<uint8_t, uint32_t> temp_index(res);
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

  for (uint32_t i = 0; i < label_number; i++) {
    int64_t n_rows = 0;
    if (query_freq[i] > 0 && cat_freq[i] >= specificity_threshould) {
      auto temp_index = cagra::index<uint8_t, uint32_t>(res);
      std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
      cagra::deserialize(res, index_path, &temp_index);
  
      auto graph = temp_index.graph();
      n_rows = graph.extent(0);
      raft::copy_async(final_matrix.data_handle() + current_offset * graph_width,
                      graph.data_handle(),
                      n_rows * graph_width,
                      raft::resource::get_cuda_stream(res));
      
      // Read indices from file into host memory
      std::vector<uint32_t> h_indices(n_rows);
      auto d_indices = raft::make_device_vector<uint32_t>(res, n_rows);
      raft::copy(d_indices.data_handle(),
                reinterpret_cast<const uint32_t*>(label_data_vecs[i].data()),
                n_rows,
                raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
          
      raft::copy_async(index_map.data_handle() + current_offset,
                      d_indices.data_handle(),
                      n_rows,
                      raft::resource::get_cuda_stream(res)); 
    }

    // Update label size and offset
    host_label_size[i] = n_rows;
    host_label_offset[i] = current_offset;

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


void cagra_build_search_variants(shared_resources::configured_raft_resources& dev_resources,
                                raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                raft::device_matrix_view<const uint8_t, int64_t> queries,
                                const std::vector<std::vector<int>>& data_label_vecs,
                                const std::vector<std::vector<int>>& label_data_vecs,
                                const std::vector<std::vector<int>>& query_label_vecs,
                                const std::vector<int>& cat_freq,
                                const std::vector<int>& query_freq,
                                int itopk_size = 32,
                                int specificity_threshould = 0) {

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
  std::string index_file = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/index_32_16.bin";
  std::ifstream test_file(index_file);
  auto index = cagra::index<uint8_t, uint32_t>(dev_resources);
  if (test_file.good()) {  
    std::cout << "Loading index from file: " << index_file << std::endl;
    cagra::deserialize(dev_resources, index_file, &index);
    index.update_dataset(dev_resources, raft::make_const_mdspan(dataset));
    test_file.close();
  } else {
    std::cout << "Building index from scratch" << std::endl;
    index = cagra::build(dev_resources, index_params, dataset);
    cagra::serialize(dev_resources, index_file, index);
  }
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() 
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;

  // Configure standard search parameters
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  cagra::search_params filtered_search_params = search_params;
  filtered_search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  raft::resource::sync_stream(dev_resources);
  // warmup
  for (int i = 0; i < 5; i++) {
    cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  }
  raft::resource::sync_stream(dev_resources);

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs, 
                                                  query_freq, 
                                                  cat_freq, 
                                                  specificity_threshould);
  index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));
  std::cout << "Filtered index has " << index.size() << " vectors" << std::endl;
  std::cout << "Filtered graph has degree " << index.graph_degree() 
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;
  raft::resource::sync_stream(dev_resources);

  // Find valid queries
  std::vector<int64_t> valid_query_indices;
  n_queries = 0;
  for (int64_t i = 0; i < query_label_vecs.size(); i++) {
    if (cat_freq[query_label_vecs[i][0]] >= specificity_threshould) {
      valid_query_indices.push_back(i);
      n_queries ++;
    }
  }
  // Copy valid queries to new matrix
  auto filtered_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_queries, queries.extent(1));
  for (int64_t i = 0; i < n_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + valid_query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  // Create filtered query labels vector
  auto filtered_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  std::vector<uint32_t> h_filtered_labels(n_queries);
  
  // Map original labels to new indices using label_map
  for (int64_t i = 0; i < n_queries; i++) {
    h_filtered_labels[i] = query_label_vecs[valid_query_indices[i]][0];
  }
  // Copy mapped labels to device
  raft::update_device(filtered_query_labels.data_handle(),
                     h_filtered_labels.data(),
                     n_queries,
                     raft::resource::get_cuda_stream(dev_resources));

  // Resize result arrays for filtered queries
  neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, topk);
  distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);
  // Batch filtered search
  std::cout << "Total queries: " << n_queries << std::endl;
  std::cout << "Specificity threshold: " << (double)(specificity_threshould) / 10000000 << std::endl;
  time_it("filtered/batch", [&]() {
    cagra::filtereds_search(dev_resources, 
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
}

auto load_data_bin(const std::string& file_path) -> std::tuple<std::vector<uint8_t>, int64_t, int64_t> {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
      throw std::runtime_error("Cannot open file: " + file_path);
  }
  uint32_t N, dim;
  file.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
  
  std::vector<uint8_t> data(N * dim);
  for (uint32_t i = 0; i < N; i++) {
      file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(uint8_t));
  }
  
  return {data, N, dim};
}

int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 32;  // Default value
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int specificity_threshould = 0;
  if (argc > 2) {
    specificity_threshould = std::stoi(argv[2]);
  }
  
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.10M.u8bin";
  std::string data_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.sorted.metadata.10M.spmat";
  std::string query_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.public.100K.u8bin";
  std::string query_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.metadata.public.100K.spmat";
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";

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
  printf("N:%lld, Nq: %lld, dim:%lld\n", N, Nq, dim);
  auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, N, dim);
  auto queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, Nq, dim);

  // Copy datasets to GPU
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), stream);

  

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
                              specificity_threshould);
}
