/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
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
           bool force_rebuild,
           bool multi_label) -> cuvs::neighbors::vecflow::index<data_t> {

  std::vector<int> label_freq(data_label_vecs.size(), 0);

  int max_label = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      max_label = std::max(max_label, data_label_vecs[i][j]);
    }
  }

  label_freq.resize(max_label + 1, 0);
  std::vector<std::vector<int>> label_data_vecs(max_label + 1);
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      label_freq[data_label_vecs[i][j]] += 1;
      label_data_vecs[data_label_vecs[i][j]].push_back(i);
    }
  }

  // Prepare metadata
  uint32_t label_number = max_label + 1;
  int64_t cagra_total_rows = 0;
  int64_t bfs_total_rows = 0;
  int cagra_labels = 0;
  int bfs_labels = 0;
  std::vector<uint32_t> host_graph_label_size(label_number);
  std::vector<uint32_t> host_graph_label_offset(label_number);
  std::vector<uint32_t> host_bfs_label_size(label_number);
  std::vector<uint32_t> host_bfs_label_offset(label_number);
  std::vector<uint32_t> host_label_freq(label_number);
  for (uint32_t i = 0; i < label_number; i++) {
    host_label_freq[i] = label_freq[i];
    auto n_rows = label_data_vecs[i].size();
    if (label_freq[i] > specificity_threshold) {
      host_graph_label_size[i] = n_rows;
      host_graph_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = 0;
      host_bfs_label_offset[i] = bfs_total_rows;
      cagra_total_rows += n_rows;
      cagra_labels++;
    } else {
      host_graph_label_size[i] = 0;
      host_graph_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = n_rows;
      host_bfs_label_offset[i] = bfs_total_rows;
      bfs_total_rows += n_rows;
      bfs_labels++;
    }
  }

  std::vector<uint32_t> host_graph_index_map(cagra_total_rows);
  std::vector<uint32_t> host_bfs_index_map(bfs_total_rows);
  uint32_t bfs_iter = 0;
  uint32_t cagra_iter = 0;
  for (uint32_t i = 0; i < label_number; i++) {
    if (label_freq[i] > specificity_threshold) {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j++) {
        host_graph_index_map[cagra_iter] = label_data_vecs[i][j];
        cagra_iter++;
      }
    } else {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j++) {
        host_bfs_index_map[bfs_iter] = label_data_vecs[i][j];
        bfs_iter++;
      }
    }
  }

  auto graph_index_map = raft::make_device_vector<uint32_t, int64_t>(res, cagra_total_rows);
  auto graph_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto graph_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto bfs_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto d_label_freq = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

  raft::update_device(graph_label_size.data_handle(), 
                      host_graph_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(graph_label_offset.data_handle(),
                      host_graph_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(graph_index_map.data_handle(),
                      host_graph_index_map.data(),
                      cagra_total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(bfs_label_size.data_handle(), 
                      host_bfs_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(d_label_freq.data_handle(), 
                      host_label_freq.data(), 
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
        if (host_graph_label_size[i] == 0 || label_data_vecs[i].size() == 0) continue;

        int thread_id = omp_get_thread_num();
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = optimal_threads;
        auto thread_resources = res;
        cudaStream_t thread_stream = thread_resources.get_sync_stream();

        // Progress calculation and display
        #pragma omp critical
        {
          completed_work += label_data_vecs[i].size();
          float progress = static_cast<float>(completed_work.load()) / cagra_total_rows * 100.0f;
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
        
        raft::copy(host_final_graph.data_handle() + host_graph_label_offset[i] * graph_degree, 
                   index.graph().data_handle(), 
                   host_graph_label_size[i] * graph_degree, 
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

  // Optional: prepare CSR dataset-label arrays for multi-label AND search.
  // Empty when `multi_label=false` (zero cost on the single-label path).
  auto dataset_labels        = raft::make_device_vector<uint32_t, int64_t>(res, 0);
  auto dataset_label_offsets = raft::make_device_vector<int64_t,  int64_t>(res, 0);
  if (multi_label) {
    prepare_dataset_label_csr(res, data_label_vecs, dataset_labels, dataset_label_offsets);
  }

  return cuvs::neighbors::vecflow::index<data_t>{
    std::move(ivf_graph_index),
    std::move(ivf_bfs_index),
    specificity_threshold,
    std::move(graph_index_map),
    std::move(graph_label_size),
    std::move(graph_label_offset),
    std::move(bfs_label_size),
    std::move(d_label_freq),
    std::move(dataset_labels),
    std::move(dataset_label_offsets)
  };
}

}  // namespace detail

template <typename data_t>
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const data_t, int64_t> dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname,
           const std::string& bfs_fname,
           bool force_rebuild,
           bool multi_label) -> cuvs::neighbors::vecflow::index<data_t>
{
  return cuvs::neighbors::vecflow::detail::build<data_t>(
    res, dataset, data_label_vecs, graph_degree, specificity_threshold,
    graph_fname, bfs_fname, force_rebuild, multi_label);
}

}  // namespace cuvs::neighbors::vecflow
