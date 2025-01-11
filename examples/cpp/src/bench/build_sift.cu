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
  raft::device_matrix<uint32_t, int64_t> final_matrix;
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
};

// template<typename T>
// void save_matrix_to_ibin(raft::resources const& res,
//                          const std::string& filename, 
//                          const raft::device_matrix_view<T, int64_t>& matrix) {
//   std::ofstream file(filename, std::ios::binary);
//   if (!file) {
//     throw std::runtime_error("Cannot create file: " + filename);
//   }

//   // Write dimensions
//   int64_t rows = matrix.extent(0);
//   int64_t cols = matrix.extent(1);
//   file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
//   file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

//   // Copy data to host
//   std::vector<T> h_matrix(rows * cols);
//   raft::copy(h_matrix.data(), matrix.data_handle(), rows * cols, raft::resource::get_cuda_stream(res));
//   raft::resource::sync_stream(res);
  
//   // Write data
//   file.write(reinterpret_cast<const char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
//   file.close();
// }

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
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq,
                              int graph_degree) -> CombinedIndices {
  
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

  auto final_matrix = raft::make_device_matrix<uint32_t, int64_t>(res, total_rows, graph_degree);
  auto index_map = raft::make_device_vector<uint32_t, int64_t>(res, total_rows);
  auto label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  raft::resource::sync_stream(res);

  int64_t current_offset = 0;
  for (uint32_t i = 0; i < label_number; i++) {
    if (query_freq[i] == 0) {
      continue;
    }

    auto temp_index = cagra::index<float, uint32_t>(res);
    std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/label_" + std::to_string(i) + "_index_" +
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
    cagra::deserialize(res, index_path, &temp_index);
    auto graph = temp_index.graph();
    int64_t n_rows = graph.extent(0);
    raft::copy_async(final_matrix.data_handle() + current_offset * graph_degree,
                    graph.data_handle(),
                    n_rows * graph_degree,
                    raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    current_offset += n_rows;
  }

  // Copy label size and offset to device
  raft::update_device(index_map.data_handle(),
                      host_index_map.data(),
                      total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(label_size.data_handle(), 
                      host_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(label_offset.data_handle(),
                      host_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(final_matrix),
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset)
  };
}

template<typename T>
void cagra_build_subgraph(shared_resources::configured_raft_resources& dev_resources,
                          raft::device_matrix_view<const T, int64_t> dataset,
                          const std::vector<std::vector<int>>& label_data_vecs,
                          int graph_degree) { 
  
  int dim = dataset.extent(1);

  std::cout << "Building subgraphs ..." << std::endl;
  std::cout << "Total labels: " << label_data_vecs.size() << std::endl;

  int optimal_threads;
  char* env_threads = getenv("OMP_NUM_THREADS");
  optimal_threads = std::atoi(env_threads);
  omp_set_num_threads(optimal_threads);

  #pragma omp parallel for num_threads(optimal_threads)
  for (int i=0; i<label_data_vecs.size(); i++) {
    std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/label_" + std::to_string(i) + "_index_" +
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
    if (std::filesystem::exists(index_path)) {
      continue;
    }
    int thread_id = omp_get_thread_num();
    shared_resources::thread_id = thread_id;
    shared_resources::n_threads = optimal_threads;
    auto thread_resources = dev_resources;
    cudaStream_t thread_stream = thread_resources.get_sync_stream();

    size_t label_size = label_data_vecs[i].size();
    std::cout << "Building graph for label: " << i << " Size: " << label_size << std::endl;

    if (label_size < graph_degree * 2) continue;

    auto filtered_dataset = raft::make_device_matrix<T, int64_t>(thread_resources, label_size, dim);
    auto mapping_list = raft::make_device_vector<int>(thread_resources, label_size);
    raft::copy(mapping_list.data_handle(), label_data_vecs[i].data(), label_size, thread_stream);
    
    const int dataset_block_size = 256;
    const int dataset_grid_size = std::min<int>((label_size * dim + dataset_block_size - 1) / dataset_block_size, 65535);
    copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
        dataset, filtered_dataset.view(), mapping_list.view());
    
    cagra::index_params index_params;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_degree = graph_degree;
    auto index = cagra::build(thread_resources, index_params, raft::make_const_mdspan(filtered_dataset.view()));
    cagra::serialize(thread_resources, index_path, index);

    raft::resource::sync_stream(thread_resources);
  }
}

void cagra_build_variants(shared_resources::configured_raft_resources& dev_resources,
                          raft::device_matrix_view<const float, int64_t> dataset,
                          const std::vector<std::vector<int>>& data_label_vecs,
                          const std::vector<std::vector<int>>& label_data_vecs,
                          const std::vector<int>& cat_freq,
                          const std::vector<int>& query_freq,
                          int graph_degree) {

  std::cout << "\nBuilding CAGRA index (search graph)" << std::endl;
  std::string index_file = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/index_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::string graph_file = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  // Build index
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = graph_degree * 2;
  index_params.graph_degree = graph_degree;
  auto index = cagra::build(dev_resources, index_params, dataset);
  cagra::serialize(dev_resources, index_file, index);

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_freq, 
                                                  cat_freq,
                                                  graph_degree);
  index.update_graph(dev_resources, raft::make_const_mdspan(combined_indices.final_matrix.view()));
  save_matrix_to_ibin<uint32_t>(dev_resources, graph_file, combined_indices.final_matrix.view());
  raft::resource::sync_stream(dev_resources);

  std::cout << "\n=== Index Information ===" << std::endl;
  std::cout << "Loading index from: " << index_file << std::endl;
  std::cout << "CAGRA index size: " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph: [" << index.graph().extent(0) << ", " 
            << index.graph().extent(1) << "] (degree " << index.graph_degree() << ")" << std::endl;
}

void filtered_bfs_build_search_variants(raft::resources const& dev_resources,
                                        raft::device_matrix_view<const float, int64_t> dataset,
                                        raft::device_matrix_view<const float, int64_t> queries,
                                        const std::vector<std::vector<int>>& data_label_vecs,
                                        const std::vector<std::vector<int>>& label_data_vecs,
                                        const std::vector<std::vector<int>>& query_label_vecs,
                                        const std::vector<int>& cat_freq,
                                        const std::vector<int>& query_freq,
                                        int graph_degree) {

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_freq,
                                                  cat_freq,
                                                  graph_degree);
  
  ivf_flat::index<float, int64_t> filtered_ivf_index(dev_resources,
                                                    cuvs::distance::DistanceType::L2Unexpanded,
                                                    label_data_vecs.size(),
                                                    false,  // adaptive_centers
                                                    true,   // conservative_memory_allocation
                                                    queries.extent(1));

  std::string filename = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/spec_" + 
                        std::to_string(0) + 
                        "_index.bin";
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
  }

  std::cout << "\n=== Index Information ===" << std::endl;
  std::cout << "Loading from file: " << filename << std::endl;
  std::cout << "Number of lists: " << filtered_ivf_index.n_lists() << std::endl;
  std::cout << "Total indexed points: " << combined_indices.index_map.size() << std::endl;
  
  // Prepare query labels and indices
  int64_t n_queries = queries.extent(0);
  int64_t total_points = 0;
  std::vector<uint32_t> h_query_labels;
  h_query_labels.reserve(n_queries);

  // Second pass to collect queries
  for (size_t i = 0; i < query_label_vecs.size(); i++) {
    h_query_labels.push_back(query_label_vecs[i][0]);
    total_points += cat_freq[query_label_vecs[i][0]];
  }
  
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  raft::update_device(query_labels.data_handle(), 
                      h_query_labels.data(), 
                      n_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  
  auto filtered_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, queries.extent(1));
  for (int64_t i = 0; i < n_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + i * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);

  // Prepare for search
  const int k = 100; 
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, k);

  std::cout << "\n=== Search Parameters ===" << std::endl;
  std::cout << "Number of queries: " << n_queries << std::endl;
  std::cout << "Average points per query label: " << std::fixed << std::setprecision(2) << 
    static_cast<double>(total_points) / n_queries << std::endl;
  std::cout << "Top-k: " << k << std::endl;
  
  // Warm up
  std::cout << "\n=== Warm-up Phase ===" << std::endl;
  for (int64_t i = 0; i < 5; i++) {
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        filtered_queries.view(),
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
                      filtered_queries.view(),
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
  
  std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.groundtruth.neighbors.ibin";
  save_matrix_to_ibin<int64_t>(dev_resources, gt_fname, neighbors.view());
}


int main(int argc, char** argv) {
  // Parse command line argument
  int graph_degree = 16;  // Default value
  if (argc > 1) {
    graph_degree = std::stoi(argv[1]);
  }
    
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.fbin";
  std::string data_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.spmat";
  std::string query_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.fbin";
  std::string query_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.spmat";
  // std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.groundtruth.neighbors.ibin";

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

  cagra_build_subgraph<float>(dev_resources,
                              raft::make_const_mdspan(dataset.view()),
                              label_data_vecs,
                              graph_degree);
  
  cagra_build_variants(dev_resources, 
                      raft::make_const_mdspan(dataset.view()),
                      data_label_vecs,
                      label_data_vecs,
                      cat_freq,
                      query_freq,
                      graph_degree);
  
  filtered_bfs_build_search_variants(dev_resources, 
                                    raft::make_const_mdspan(dataset.view()),
                                    raft::make_const_mdspan(queries.view()),
                                    data_label_vecs,
                                    label_data_vecs,
                                    query_label_vecs,
                                    cat_freq,
                                    query_freq,
                                    graph_degree);
}
