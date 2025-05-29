/*
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

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/device_vector.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <omp.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip> 

#include "vecflow_common.cuh"

namespace cuvs::neighbors::vecflow {

namespace detail {

template <typename data_t>
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const data_t, int64_t> dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname,
           const std::string& bfs_fname,
           bool force_rebuild) -> cuvs::neighbors::vecflow::index<data_t> {

  std::vector<int> cat_freq(data_label_vecs.size(), 0);

  int max_label = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      max_label = std::max(max_label, data_label_vecs[i][j]);
    }
  }

  cat_freq.resize(max_label + 1, 0);
  std::vector<std::vector<int>> label_data_vecs(max_label + 1);
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      cat_freq[data_label_vecs[i][j]] += 1;
      label_data_vecs[data_label_vecs[i][j]].push_back(i);
    }
  }

  // Prepare metadata
  uint32_t label_number = max_label + 1;
  int64_t cagra_total_rows = 0;
  int64_t bfs_total_rows = 0;
  int cagra_labels = 0;
  int bfs_labels = 0;
  std::vector<uint32_t> host_cagra_label_size(label_number);
  std::vector<uint32_t> host_cagra_label_offset(label_number);
  std::vector<uint32_t> host_bfs_label_size(label_number);
  std::vector<uint32_t> host_bfs_label_offset(label_number);
  std::vector<uint32_t> host_cat_freq(label_number);
  for (uint32_t i = 0; i < label_number; i++) {
    host_cat_freq[i] = cat_freq[i];
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
  auto d_cat_freq = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

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
  raft::update_device(d_cat_freq.data_handle(), 
                      host_cat_freq.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));

  auto ivf_graph_index = cagra::index<data_t, uint32_t>(res);
  auto ivf_bfs_index = ivf_flat::index<data_t, int64_t>(res,
                                                        cuvs::distance::DistanceType::L2Unexpanded,
                                                        label_number,
                                                        false,
                                                        true,
                                                        dataset.extent(1));

  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;

  if (cagra_labels > 0) {
    ivf_graph_index.update_dataset(res, raft::make_const_mdspan(dataset));
    auto host_final_graph = raft::make_host_matrix<uint32_t, int64_t>(cagra_total_rows, graph_degree);
    if (std::filesystem::exists(graph_fname) && !force_rebuild) {  
      load_matrix_from_ibin(graph_fname, host_final_graph.view());
      ivf_graph_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
    } else {
      std::cout << "Building IVF-Graph index from scratch ..." << std::endl;
      auto cagra_start_time = std::chrono::high_resolution_clock::now();
      int optimal_threads = 32;
      omp_set_num_threads(optimal_threads);
      std::atomic<size_t> completed_work{0};
      #pragma omp parallel for num_threads(optimal_threads)
      for (uint32_t i = 0; i < label_number; i++) {
        if (host_cagra_label_size[i] == 0 || label_data_vecs[i].size() == 0) continue;

        int thread_id = omp_get_thread_num();
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = optimal_threads;
        auto thread_resources = res;
        cudaStream_t thread_stream = thread_resources.get_sync_stream();

        // Progress calculation and display
        #pragma omp critical
        {
          completed_work += label_data_vecs[i].size();;
          float progress = (float)completed_work / cagra_total_rows * 100;
          std::cout << "\rProgress: [";
          int pos = progress / 2;
          for (int j = 0; j < 50; j++) {
              if (j < pos) std::cout << "=";
              else if (j == pos) std::cout << ">";
              else std::cout << " ";
          }
          std::cout << "] " << std::fixed << std::setprecision(1) << progress << "% "
                    << "Label: " << i << " Size: " << label_data_vecs[i].size() << std::flush;
        }

        auto filtered_dataset = raft::make_device_matrix<data_t, int64_t>(res, label_data_vecs[i].size(), dataset.extent(1));
        raft::resource::sync_stream(thread_resources);
        for (uint64_t j = 0; j < label_data_vecs[i].size(); j++) {
          raft::copy_async(filtered_dataset.data_handle() + j * dataset.extent(1), 
                           dataset.data_handle() + static_cast<uint64_t>(label_data_vecs[i][j]) * dataset.extent(1), 
                           dataset.extent(1), 
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
      std::cout << "\nIVF-Graph index building time: " << cagra_duration.count() << " ms" << std::endl;
      ivf_graph_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      if (!graph_fname.empty()) {
        save_matrix_to_ibin(graph_fname, host_final_graph.view());
      }
    }
  }

  if (bfs_labels > 0) {
    if (std::filesystem::exists(bfs_fname) && !force_rebuild) {
      std::cout << "Loading IVF-BFS index from " << bfs_fname << std::endl;
      ivf_flat::deserialize(res, bfs_fname, &ivf_bfs_index);
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
      build_filtered_bfs(res,
                         &ivf_bfs_index,
                         dataset,
                         bfs_index_map.view(),
                         bfs_label_size.view(),
                         bfs_label_offset.view());
      auto bfs_end_time = std::chrono::high_resolution_clock::now();
      auto bfs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bfs_end_time - bfs_start_time);
      std::cout << "IVF-BFS graph building time: " << bfs_duration.count() << " ms" << std::endl;
      if (!bfs_fname.empty()) {
        ivf_flat::serialize(res, bfs_fname, ivf_bfs_index);
        std::cout << "Saving IVF-BFS index to " << bfs_fname << std::endl;
      }
    }
  }
  
  // IVF-Graph statistics
  std::cout << "\nIVF-Graph Index Stats:" << std::endl;
  std::cout << "  Total vectors:  " << ivf_graph_index.size() << std::endl;
  std::cout << "  Number of labels: " << cagra_labels << std::endl;
  std::cout << "  Graph size:     [" << ivf_graph_index.graph().extent(0) << " × " 
            << ivf_graph_index.graph().extent(1) << "]" << std::endl;
  std::cout << "  Graph degree:   " << ivf_graph_index.graph_degree() << std::endl;
  // IVF statistics
  std::cout << "\nIVF-BFS Index Stats:" << std::endl;
  std::cout << "  Number of labels: " << bfs_labels << std::endl;
  std::cout << "  Number of rows:  " << bfs_total_rows << std::endl;

  return cuvs::neighbors::vecflow::index<data_t>{
    std::move(ivf_graph_index),
    std::move(ivf_bfs_index),
    specificity_threshold,
    std::move(cagra_index_map),
    std::move(cagra_label_size),
    std::move(cagra_label_offset),
    std::move(bfs_label_size),
    std::move(d_cat_freq)
  };
}

}  // namespace detail

template<typename data_t>
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const data_t, int64_t> dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname,
           const std::string& bfs_fname,
           bool force_rebuild) -> cuvs::neighbors::vecflow::index<data_t>
{
  return cuvs::neighbors::vecflow::detail::build<data_t>(
    res, dataset, data_label_vecs, graph_degree, specificity_threshold, graph_fname, bfs_fname, force_rebuild);
} 

} // namespace cuvs::neighbors::vecflow
