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
#include <rmm/device_vector.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <fstream>
#include <vector>
#include <iomanip> 

#include <omp.h>
#include "common.cuh"
#include "../utils.h"

using namespace cuvs::neighbors;

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> cagra_index_map;
  raft::device_vector<uint32_t, int64_t> cagra_label_size;
  raft::device_vector<uint32_t, int64_t> cagra_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  std::vector<int> cat_freq;
};

void save_matrix_to_ibin(const std::string& filename, 
                         const raft::host_matrix_view<uint32_t, int64_t>& matrix) {
  // Get dimensions
  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  
  // Save to file
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filename);
  }

  // Write dimensions
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));
  
  // Write data as uint32_t
  file.write(reinterpret_cast<const char*>(matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();

  std::cout << "Saving graph to " << filename << std::endl;
}

void load_matrix_from_ibin(raft::resources const& res, 
                          const std::string& filename,
                          const raft::host_matrix_view<uint32_t, int64_t>& graph) {  // Pass graph as reference
  
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
  file.read(reinterpret_cast<char*>(graph.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
}

template<typename T>
auto build_load_graphs(shared_resources::configured_raft_resources& res,
                      raft::device_matrix_view<const T, int64_t> dataset,
                      cagra::index<T, uint32_t>& cagra_index,
                      ivf_flat::index<T, int64_t>& bfs_index,
                      const std::vector<std::vector<int>>& label_data_vecs,
                      const std::vector<int>& cat_freq,
                      int graph_degree,
                      int specificity_threshold,
                      std::string graph_fname = "",
                      std::string bfs_fname = "") -> CombinedIndices {
  
  int dim = dataset.extent(1);

  // Prepare metadata
  int label_number = label_data_vecs.size();
  int64_t cagra_total_rows = 0;
  int64_t bfs_total_rows = 0;
  int cagra_labels = 0;
  int bfs_labels = 0;
  std::vector<uint32_t> host_cagra_label_size(label_number);
  std::vector<uint32_t> host_cagra_label_offset(label_number);
  std::vector<uint32_t> host_bfs_label_size(label_number);
  std::vector<uint32_t> host_bfs_label_offset(label_number);
  for (uint32_t i = 0; i < label_number; i++) {
    auto n_rows = label_data_vecs[i].size();
    if (cat_freq[i] > specificity_threshold) {
      host_cagra_label_size[i] = n_rows;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = 0;
      host_bfs_label_offset[i] = bfs_total_rows;
      cagra_total_rows += n_rows;
      cagra_labels++;
    } else {
      host_cagra_label_size[i] = 0;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = n_rows;
      host_bfs_label_offset[i] = bfs_total_rows;
      bfs_total_rows += n_rows;
      bfs_labels++;
    }
  }

  std::vector<uint32_t> host_cagra_index_map(cagra_total_rows);
  std::vector<uint32_t> host_bfs_index_map(bfs_total_rows);
  uint32_t bfs_iter = 0;
  uint32_t cagra_iter = 0;
  for (uint32_t i = 0; i < label_number; i ++) {
    if (cat_freq[i] > specificity_threshold) {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
        host_cagra_index_map[cagra_iter] = label_data_vecs[i][j];
        cagra_iter ++;
      }
    }
    else {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
        host_bfs_index_map[bfs_iter] = label_data_vecs[i][j];
        bfs_iter ++;
      }
    }
  }

  auto cagra_index_map = raft::make_device_vector<uint32_t, int64_t>(res, cagra_total_rows);
  auto cagra_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto cagra_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto bfs_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

  raft::update_device(cagra_label_size.data_handle(), 
                      host_cagra_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_label_offset.data_handle(),
                      host_cagra_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_index_map.data_handle(),
                      host_cagra_index_map.data(),
                      cagra_total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(bfs_label_size.data_handle(), 
                      host_bfs_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;

  if (cagra_labels > 0) {
    cagra_index.update_dataset(res, raft::make_const_mdspan(dataset));
    auto host_final_graph = raft::make_host_matrix<uint32_t, int64_t>(cagra_total_rows, graph_degree);
    std::ifstream graph_file(graph_fname);
    if (graph_file.good()) {  
      std::cout << "Loading IVF-CAGRA index from " << graph_fname << std::endl;
      load_matrix_from_ibin(res, graph_fname, host_final_graph.view());
      cagra_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      graph_file.close();
    } else {
      std::cout << "Building IVF-CAGRA index from scratch ..." << std::endl;
      auto cagra_start_time = std::chrono::high_resolution_clock::now();
      int optimal_threads = 32;
      omp_set_num_threads(optimal_threads);
      std::atomic<size_t> completed_work{0};
      #pragma omp parallel for num_threads(optimal_threads)
      for (int i = 0; i < label_number; i++) {
        if (host_cagra_label_size[i] == 0) continue;

        int thread_id = omp_get_thread_num();
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = optimal_threads;
        auto thread_resources = res;
        cudaStream_t thread_stream = thread_resources.get_sync_stream();

        size_t label_size = label_data_vecs[i].size();
        // Progress calculation and display
        #pragma omp critical
        {
          completed_work += label_size;
          float progress = (float)completed_work / cagra_total_rows * 100;
          std::cout << "\rProgress: [";
          int pos = progress / 2;
          for (int j = 0; j < 50; j++) {
              if (j < pos) std::cout << "=";
              else if (j == pos) std::cout << ">";
              else std::cout << " ";
          }
          std::cout << "] " << std::fixed << std::setprecision(1) << progress << "% "
                    << "Label: " << i << " Size: " << label_size << std::flush;
        }

        const auto& matching_indices = label_data_vecs[i];  // Get all indices with this label
        int write_id = matching_indices.size();
        if (write_id == 0) continue;  // Skip if no matches found

        // Create filtered dataset with matching indices
        auto filtered_dataset = raft::make_device_matrix<float, int64_t>(res, write_id, dim);
        raft::resource::sync_stream(thread_resources);
        for (uint64_t j = 0; j < write_id; j++) {
          raft::copy_async(filtered_dataset.data_handle() + j * dim, 
                          dataset.data_handle() + (uint64_t)matching_indices[j] * dim, 
                          dim, 
                          thread_stream);
        }
        raft::resource::sync_stream(thread_resources);
        
        cagra::index_params index_params;
        index_params.intermediate_graph_degree = graph_degree * 2;
        index_params.graph_degree = graph_degree;
        index_params.attach_dataset_on_build = false; 
        auto index = cagra::build(thread_resources, index_params, raft::make_const_mdspan(filtered_dataset.view()));
        
        raft::copy(host_final_graph.data_handle() + host_cagra_label_offset[i] * graph_degree, 
                  index.graph().data_handle(), 
                  host_cagra_label_size[i] * graph_degree, 
                  thread_stream);
        raft::resource::sync_stream(thread_resources);
      }   
      auto cagra_end_time = std::chrono::high_resolution_clock::now();
      auto cagra_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cagra_end_time - cagra_start_time);
      std::cout << "\nIVF-CAGRA index building time: " << cagra_duration.count() / 1000.0 << " seconds" << std::endl;
      cagra_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      if (!graph_fname.empty()) save_matrix_to_ibin(graph_fname, host_final_graph.view());
      raft::resource::sync_stream(res);
    }
  }

  if (bfs_labels > 0) {
    std::ifstream bfs_file(bfs_fname);
    if (bfs_file.good()) {
      std::cout << "Loading IVF-BFS index from " << bfs_fname << std::endl;
      ivf_flat::deserialize(res, bfs_fname, &bfs_index);
      bfs_file.close();
    } else {
      std::cout << "Building IVF-BFS index from scratch ..." << std::endl;
      auto bfs_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
      auto bfs_index_map = raft::make_device_vector<uint32_t, int64_t>(res, bfs_total_rows);
      raft::update_device(bfs_label_offset.data_handle(),
                      host_bfs_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
      raft::update_device(bfs_index_map.data_handle(),
                      host_bfs_index_map.data(),
                      bfs_total_rows,
                      raft::resource::get_cuda_stream(res));
      auto bfs_start_time = std::chrono::high_resolution_clock::now();                 
      build_filtered_IVF_index(res,
                              &bfs_index,
                              dataset,
                              bfs_index_map.view(),
                              bfs_label_size.view(),
                              bfs_label_offset.view());
      auto bfs_end_time = std::chrono::high_resolution_clock::now();
      auto bfs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bfs_end_time - bfs_start_time);
      std::cout << "IVF-BFS graph building time: " << bfs_duration.count() << " ms" << std::endl;
      if (!bfs_fname.empty()) {
        ivf_flat::serialize(res, bfs_fname, bfs_index);
        std::cout << "Saving IVF-BFS index to " << bfs_fname << std::endl;
      }
    }
  }
  
  // CAGRA statistics
  std::cout << "\nIVF-CAGRA Index Stats:" << std::endl;
  std::cout << "  Total vectors:  " << cagra_index.size() << std::endl;
  std::cout << "  Number of labels: " << cagra_labels << std::endl;
  std::cout << "  Graph size:     [" << cagra_index.graph().extent(0) << " × " 
            << cagra_index.graph().extent(1) << "]" << std::endl;
  std::cout << "  Graph degree:   " << cagra_index.graph_degree() << std::endl;
  // IVF statistics
  std::cout << "\nIVF-BFS Index Stats:" << std::endl;
  std::cout << "  Number of labels: " << bfs_labels << std::endl;
  std::cout << "  Number of rows:  " << bfs_total_rows << std::endl;
  
  return CombinedIndices {
    std::move(cagra_index_map),
    std::move(cagra_label_size),
    std::move(cagra_label_offset),
    std::move(bfs_label_size),
    std::move(cat_freq)
  };
}

template<typename T>
void search_main(shared_resources::configured_raft_resources& res,
                raft::device_matrix_view<const T, int64_t> queries,
                const std::vector<std::vector<int>>& query_label_vecs,
                cagra::index<T, uint32_t>& cagra_index,
                ivf_flat::index<T, int64_t>& bfs_index,
                cagra::search_params& search_params,
                CombinedIndices& metadata,
                int specificity_threshold,
                raft::device_matrix_view<uint32_t, int64_t> neighbors,
                raft::device_matrix_view<float, int64_t> distances) {

  // Configure standard search parameters
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  int n_queries = queries.extent(0);
  int dim = queries.extent(1);

  int n_cagra_queries = 0;
  int n_bfs_queries = 0;
  for (int i = 0; i < n_queries; i++) {
    int label = query_label_vecs[i][0];
    if (metadata.cat_freq[label] > specificity_threshold) n_cagra_queries++;
    else n_bfs_queries++;
  }

  std::vector<int64_t> cagra_indices(n_cagra_queries);
  std::vector<uint32_t> host_cagra_query_labels(n_cagra_queries);
  std::vector<int64_t> bfs_indices(n_bfs_queries);
  std::vector<uint32_t> host_bfs_query_labels(n_bfs_queries);
  int cagra_iter = 0;
  int bfs_iter = 0;
  for (int i = 0; i < n_queries; i++) {
    int label = query_label_vecs[i][0];
    if (metadata.cat_freq[label] > specificity_threshold) {
      cagra_indices[cagra_iter] = i;
      host_cagra_query_labels[cagra_iter] = label;
      cagra_iter++;
    }
    else {
      bfs_indices[bfs_iter] = i;
      host_bfs_query_labels[bfs_iter] = label;
      bfs_iter++;
    }
  }
  
  auto cagra_queries = raft::make_device_matrix<T, int64_t>(res, n_cagra_queries, dim);
  auto bfs_queries = raft::make_device_matrix<T, int64_t>(res, n_bfs_queries, dim);
  auto cagra_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, n_cagra_queries);
  auto bfs_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, n_bfs_queries);
  
  for (int64_t i = 0; i < n_cagra_queries; i++) {
    raft::copy_async(cagra_queries.data_handle() + i * dim,
                    queries.data_handle() + cagra_indices[i] * dim,
                    dim,
                    raft::resource::get_cuda_stream(res));
  }
  for (int64_t i = 0; i < n_bfs_queries; i++) {
    raft::copy_async(bfs_queries.data_handle() + i * dim,
                    queries.data_handle() + bfs_indices[i] * dim,
                    dim,
                    raft::resource::get_cuda_stream(res));
  }
  raft::update_device(cagra_query_labels.data_handle(),
                      host_cagra_query_labels.data(),
                      n_cagra_queries,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(bfs_query_labels.data_handle(),
                      host_bfs_query_labels.data(),
                      n_bfs_queries,
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  int64_t topk = 10;
  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(res, n_cagra_queries, topk);
  auto bfs_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_bfs_queries, topk);
  auto bfs_distances = raft::make_device_matrix<float, int64_t>(res, n_bfs_queries, topk);
  raft::resource::sync_stream(res);

  if (n_cagra_queries > 0) {
    cagra::filtered_search(res, 
                          search_params,
                          cagra_index,
                          cagra_queries.view(),
                          cagra_neighbors.view(),
                          cagra_distances.view(),
                          cagra_query_labels.view(),
                          metadata.cagra_index_map.view(),
                          metadata.cagra_label_size.view(),
                          metadata.cagra_label_offset.view());
    raft::resource::sync_stream(res);
    for (int64_t i = 0; i < n_cagra_queries; i++) {
      raft::copy_async(neighbors.data_handle() + cagra_indices[i] * topk,
                      cagra_neighbors.data_handle() + i * topk,
                      topk,
                      raft::resource::get_cuda_stream(res));
      raft::copy_async(distances.data_handle() + cagra_indices[i] * topk,
                      cagra_distances.data_handle() + i * topk,
                      topk,
                      raft::resource::get_cuda_stream(res));
    }
    raft::resource::sync_stream(res);
  }

  if (n_bfs_queries > 0) {
    search_filtered_ivf(res,
                      bfs_index,
                      raft::make_const_mdspan(bfs_queries.view()),
                      bfs_query_labels.view(),
                      metadata.bfs_label_size.view(),
                      bfs_neighbors.view(),
                      bfs_distances.view(),
                      cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(res);
    auto temp_bfs_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_bfs_queries, topk);
    convert_neighbors_to_uint32(res,
                                bfs_neighbors.data_handle(),
                                temp_bfs_neighbors.data_handle(),
                                n_bfs_queries,
                                topk);
    for (int64_t i = 0; i < n_bfs_queries; i++) {
      raft::copy_async(neighbors.data_handle() + bfs_indices[i] * topk,
                      temp_bfs_neighbors.data_handle() + i * topk,
                      topk,
                      raft::resource::get_cuda_stream(res));
      raft::copy_async(distances.data_handle() + bfs_indices[i] * topk,
                      bfs_distances.data_handle() + i * topk,
                      topk,
                      raft::resource::get_cuda_stream(res));
    }
    raft::resource::sync_stream(res);
  }
}


int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 32;
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int specificity_threshold = 20000;
  if (argc > 2) {
    specificity_threshold = std::stoi(argv[2]);
  }
  int graph_degree = 16;
  if (argc > 3) {
    graph_degree = std::stoi(argv[3]);
  }
    
  shared_resources::configured_raft_resources res{};

  std::string data_fname = "workspace/sift1M/sift.base.fbin";
  std::string data_label_fname = "workspace/sift1M/sift.base.spmat";
  std::string query_fname = "workspace/sift1M/sift.query.fbin";
  std::string query_label_fname = "workspace/sift1M/sift.query.spmat";
  
  std::string graph_fname = "workspace/sift1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + "_spec_" +
                          std::to_string(specificity_threshold) + ".bin";
  std::string bfs_fname = "workspace/sift1M/spec_"+ 
                          std::to_string(specificity_threshold) + ".bin";

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
  auto dataset = raft::make_device_matrix<float, int64_t>(res, N, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(res, Nq, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), raft::resource::get_cuda_stream(res));
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(res));

  auto cagra_index = cagra::index<float, uint32_t>(res);
  auto bfs_index = ivf_flat::index<float, int64_t>(res,
                                                  cuvs::distance::DistanceType::L2Unexpanded,
                                                  data_label_vecs.size(),
                                                  false,
                                                  true,
                                                  dataset.extent(1));
  auto metadata = build_load_graphs<float>(res,
                                          raft::make_const_mdspan(dataset.view()),
                                          cagra_index,
                                          bfs_index,
                                          label_data_vecs,
                                          cat_freq,
                                          graph_degree,
                                          specificity_threshold,
                                          graph_fname,
                                          bfs_fname);             
  
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  int64_t topk = 10;
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float, int64_t>(res,queries.extent(0), topk);
  search_main<float>(res,
                    raft::make_const_mdspan(queries.view()),
                    query_label_vecs,
                    cagra_index,
                    bfs_index,
                    search_params,
                    metadata,
                    specificity_threshold,
                    neighbors.view(),
                    distances.view());
  std::string gt_fname = "workspace/sift1M/sift.groundtruth.neighbors.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_ground_truth_file(gt_fname, gt_indices);
  compute_recall(res, neighbors.view(), gt_indices);
}
