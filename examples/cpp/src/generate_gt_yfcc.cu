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

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/refine.hpp>
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

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<std::vector<int>>& data_label_vecs) -> CombinedIndices 
{
  
  int label_number = label_data_vecs.size();
  int64_t total_rows = 0;
  int64_t total_data_labels = 0;
  int64_t n_valid_labels = 0;
  std::vector<uint32_t> host_label_size(label_number);
  std::vector<uint32_t> host_label_offset(label_number);
  for (uint32_t i = 0; i < label_number; i ++) {
    host_label_size[i] = label_data_vecs[i].size();
    host_label_offset[i] = total_rows;
    total_rows += label_data_vecs[i].size();
    n_valid_labels++;
  }
  for (uint32_t i = 0; i < data_label_vecs.size(); i++) {
    total_data_labels += data_label_vecs[i].size();
  }
  std::cout << "\n=== Index Combination ===" << std::endl;
  std::cout << "Number of valid labels: " << n_valid_labels << std::endl;
  std::cout << "Total rows: " << total_rows << std::endl;
  std::cout << "Total data labels: " << total_data_labels << std::endl;

  std::vector<uint32_t> host_index_map(total_rows);
  uint32_t iter = 0;
  for (uint32_t i = 0; i < label_number; i ++) {
    for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
      host_index_map[iter] = label_data_vecs[i][j];
      iter ++;
    }
  }

  auto index_map = raft::make_device_vector<uint32_t, int64_t>(res, total_rows);
  raft::resource::sync_stream(res);
  auto label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  raft::resource::sync_stream(res);

  // Copy label size and offset to device
  raft::update_device(label_size.data_handle(), 
                      host_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(label_offset.data_handle(),
                      host_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(index_map.data_handle(),
                      host_index_map.data(),
                      total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset)
  };
}

template<typename T>
void save_matrix_to_ibin(raft::resources const& res,
                         const std::string& filename, 
                         const raft::device_matrix_view<T, int64_t>& matrix) {
  // Get dimensions
  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  
  // Copy data to host
  std::vector<T> h_matrix(rows * cols);
  raft::copy(h_matrix.data(), matrix.data_handle(), rows * cols, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Convert to uint32_t for storage
  std::vector<uint32_t> output_data(rows * cols);
  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = 0; j < cols; j++) {
      output_data[i * cols + j] = static_cast<uint32_t>(h_matrix[i * cols + j]);
    }
  }
  
  // Save to file
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filename);
  }

  // Write dimensions
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));
  
  // Write data as uint32_t
  file.write(reinterpret_cast<const char*>(output_data.data()), rows * cols * sizeof(uint32_t));
  file.close();

  std::cout << "\nSaving matrix to " << filename << std::endl;
  std::cout << "Dimensions: [" << rows << " x " << cols << "]" << std::endl;
  std::cout << "First 5 rows of saved data:" << std::endl;
  for (int64_t i = 0; i < std::min(rows, int64_t(5)); i++) {
    std::cout << "Row " << i << ": ";
    for (int64_t j = 0; j < cols; j++) {
      std::cout << output_data[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

void filtered_bfs_build_search_variants(raft::resources const& dev_resources,
                                        raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                        raft::device_matrix_view<const uint8_t, int64_t> queries,
                                        const std::vector<std::vector<int>>& data_label_vecs,
                                        const std::vector<std::vector<int>>& label_data_vecs,
                                        const std::vector<int>& cat_freq,
                                        uint32_t label) {

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs);
  
  ivf_flat::index<uint8_t, int64_t> filtered_ivf_index(dev_resources,
                                                      cuvs::distance::DistanceType::L2Unexpanded,
                                                      label_data_vecs.size(),
                                                      false,  // adaptive_centers
                                                      true,   // conservative_memory_allocation
                                                      queries.extent(1));

  std::string filename = "/scratch/bdes/cmo1/CAGRA/yfcc_gt/index_all.bin";
  std::ifstream test_file(filename);
  if (test_file.good()) {
    ivf_flat::deserialize(dev_resources, filename, &filtered_ivf_index);
    test_file.close();
  } else {
    std::cout << "\n=== Building IVF Index ===" << std::endl;
    std::cout << "Building new index from scratch" << std::endl;
    build_filtered_IVF_index(dev_resources,
                            &filtered_ivf_index,
                            dataset,
                            combined_indices.index_map.view(),
                            combined_indices.label_size.view(),
                            combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    std::cout << "Saving index to: " << filename << std::endl;
    ivf_flat::serialize(dev_resources, filename, filtered_ivf_index);
    raft::resource::get_cuda_stream(dev_resources);
  }

  std::cout << "\n=== Index Information ===" << std::endl;
  std::cout << "Loading from file: " << filename << std::endl;
  std::cout << "Number of lists: " << filtered_ivf_index.n_lists() << std::endl;
  std::cout << "Total indexed points: " << combined_indices.index_map.size() << std::endl;
  
  // Prepare query labels and indices
  int64_t n_queries = queries.extent(0);
  std::vector<uint32_t> h_query_labels(n_queries, label);

  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  raft::update_device(query_labels.data_handle(), h_query_labels.data(), n_queries, raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Prepare for search
  const int k = 100; 
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, k);

  std::cout << "\n=== Search Parameters ===" << std::endl;
  std::cout << "Number of queries: " << n_queries << std::endl;
  std::cout << "Label: " << label << std::endl;
  std::cout << "Label size: " << cat_freq[label] << std::endl;
  std::cout << "Top-k: " << k << std::endl;
  
  // Warm up
  std::cout << "\n=== Warm-up Phase ===" << std::endl;
  for (int64_t i = 0; i < 5; i++) {
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        queries,
                        query_labels.view(),
                        combined_indices.label_size.view(),
                        neighbors.view(),
                        distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(dev_resources);
  }

  // Benchmark
  std::cout << "\n=== Performance Benchmark ===" << std::endl;
  const int num_runs = 100;
  double total_time = 0.0;
  
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    search_filtered_ivf(dev_resources,
                      filtered_ivf_index,
                      queries,
                      query_labels.view(),
                      combined_indices.label_size.view(),
                      neighbors.view(),
                      distances.view(),
                      cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(dev_resources);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time += duration.count() / 1000.0; // Convert to milliseconds
  }
  double avg_time_ms = total_time / num_runs;
  double qps = (n_queries * 1000.0) / avg_time_ms;
  std::cout << "Runs: " << num_runs << std::endl;
  std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
  std::cout << "Queries per second (QPS): " << std::scientific << std::setprecision(2) << qps << std::endl;

  // Generate ground truth
  auto h_neighbors = raft::make_host_matrix<int64_t, int64_t>(n_queries, k);
  raft::copy(h_neighbors.data_handle(), 
            neighbors.data_handle(), 
            neighbors.size(), 
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  
  std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/yfcc_gt/label_" + std::to_string(label) +
                         + "_groundtruth.ibin";
  save_matrix_to_ibin<int64_t>(dev_resources, gt_fname, neighbors.view());
  std::cout << "Results have been written to "<< gt_fname << std::endl;
}

int main(int argc, char** argv) {
  // Parse command line argument
  uint32_t label = 0;
  if (argc > 1) {
    label = std::stoi(argv[1]);
  }

  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

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
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(dev_resources));

  // run the interesting part of the program
  filtered_bfs_build_search_variants(dev_resources, 
                                    raft::make_const_mdspan(dataset.view()),
                                    raft::make_const_mdspan(queries.view()),
                                    data_label_vecs,
                                    label_data_vecs,
                                    cat_freq,
                                    label);
}
