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
#include <iomanip> 

#include <omp.h>
#include "../common.cuh"
#include "../utils.h"

using namespace cuvs::neighbors;

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

  std::cout << "\nSaving graph to " << filename << std::endl;
}

template<typename T>
void build_graphs(shared_resources::configured_raft_resources& dev_resources,
                  raft::device_matrix_view<const T, int64_t> dataset,
                  const std::vector<std::vector<int>>& label_data_vecs,
                  const std::vector<int>& cat_freq,
                  int graph_degree,
                  int specificity_threshold,
                  std::string graph_fname = "",
                  std::string bfs_fname = "") { 
  
  int dim = dataset.extent(1);
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
  auto host_final_graph = raft::make_host_matrix<uint32_t, int64_t>(cagra_total_rows, graph_degree);

  std::vector<uint32_t> host_bfs_index_map(bfs_total_rows);
  uint32_t iter = 0;
  for (uint32_t i = 0; i < label_number; i ++) {
    if (cat_freq[i] > specificity_threshold) continue;
    for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
      host_bfs_index_map[iter] = label_data_vecs[i][j];
      iter ++;
    }
  }

  if (cagra_labels > 0) {
    std::cout << "\nBuilding IVF-CAGRA graphs ..." << std::endl;
    std::cout << "Total labels: " << cagra_labels << std::endl;
    std::cout << "Total rows: " << cagra_total_rows << std::endl;
    std::cout << "Specificity threshold: " << specificity_threshold << std::endl;

    int optimal_threads = 32;
    omp_set_num_threads(optimal_threads);

    // At the start of the loop, get total work amount
    std::atomic<size_t> completed_work{0};

    #pragma omp parallel for num_threads(optimal_threads)
    for (int i = 0; i < label_number; i++) {
      if (host_cagra_label_size[i] == 0) continue;

      int thread_id = omp_get_thread_num();
      shared_resources::thread_id = thread_id;
      shared_resources::n_threads = optimal_threads;
      auto thread_resources = dev_resources;
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
      auto filtered_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, write_id, dim);
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
    if (!graph_fname.empty()) save_matrix_to_ibin(graph_fname, host_final_graph.view());
  }

  if (bfs_labels > 0) {
    std::cout << "Building IVF-BFS graphs ..." << std::endl;
    std::cout << "Total labels: " << bfs_labels << std::endl;
    std::cout << "Total rows: " << bfs_total_rows << std::endl;
    std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
    ivf_flat::index<T, int64_t> bfs_index(dev_resources,
                                          cuvs::distance::DistanceType::L2Unexpanded,
                                          label_data_vecs.size(),
                                          false,  // adaptive_centers
                                          true,   // conservative_memory_allocation
                                          dim);                   
    auto index_map = raft::make_device_vector<uint32_t, int64_t>(dev_resources, bfs_total_rows);
    auto label_size = raft::make_device_vector<uint32_t, int64_t>(dev_resources, label_number);
    auto label_offset = raft::make_device_vector<uint32_t, int64_t>(dev_resources, label_number);
    raft::update_device(label_size.data_handle(), 
                        host_bfs_label_size.data(), 
                        label_number,
                        raft::resource::get_cuda_stream(dev_resources));
    raft::update_device(label_offset.data_handle(),
                        host_bfs_label_offset.data(),
                        label_number,
                        raft::resource::get_cuda_stream(dev_resources));
    raft::update_device(index_map.data_handle(),
                        host_bfs_index_map.data(),
                        bfs_total_rows,
                        raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);

    build_filtered_IVF_index(dev_resources,
                            &bfs_index,
                            dataset,
                            index_map.view(),
                            label_size.view(),
                            label_offset.view());
    if (!bfs_fname.empty()) {
      ivf_flat::serialize(dev_resources, bfs_fname, bfs_index);
       std::cout << "\nSaving IVF-BFS index to " << bfs_fname << std::endl;
    }
  }
}


int main(int argc, char** argv) {
  // Parse command line argument
  int graph_degree = 16;
  if (argc > 1) {
    graph_degree = std::stoi(argv[1]);
  }
  int specificity_threshold = 8000;
  if (argc > 2) {
    specificity_threshold = std::stoi(argv[2]);
  }
    
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "workspace/CAGRA/sift1M//sift.base.fbin";
  std::string data_label_fname = "workspace/CAGRA/sift1M/sift.base.spmat";
  std::string query_fname = "workspace/CAGRA/sift1M/sift.query.fbin";
  std::string query_label_fname = "workspace/CAGRA/sift1M/sift.query.spmat";
  
  std::string graph_fname = "workspace/CAGRA/sift1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + "_spec_" +
                          std::to_string(specificity_threshold) + ".bin";
  std::string bfs_fname = "workspace/CAGRA/sift1M/spec_"+ 
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
  auto dataset = raft::make_device_matrix<float, int64_t>(dev_resources, N, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(dev_resources, Nq, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(dev_resources));

  build_graphs<float>(dev_resources,
                      raft::make_const_mdspan(dataset.view()),
                      label_data_vecs,
                      cat_freq,
                      graph_degree,
                      specificity_threshold,
                      graph_fname,
                      bfs_fname);
}
