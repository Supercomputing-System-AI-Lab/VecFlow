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

  raft::device_vector<uint32_t, int64_t> data_labels;
  raft::device_vector<uint32_t, int64_t> data_label_size;
  raft::device_vector<uint32_t, int64_t> data_label_offset;
  raft::device_vector<int64_t, int64_t> query_labels;
};

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<std::vector<int>>& data_label_vecs,
                              const std::vector<std::vector<int>>& query_label_vecs,
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq,
                              int specificity_threshold = 2000) -> CombinedIndices 
{
  
  int label_number = label_data_vecs.size();
  int64_t total_rows = 0;
  int64_t total_data_labels = 0;
  int64_t n_valid_labels = 0;
  std::vector<uint32_t> host_label_size(label_number);
  std::vector<uint32_t> host_label_offset(label_number);
  for (uint32_t i = 0; i < label_number; i ++) {
    if (cat_freq[i] > specificity_threshold) {
      host_label_size[i] = 0;
      host_label_offset[i] = total_rows;
    } else {
      host_label_size[i] = label_data_vecs[i].size();
      host_label_offset[i] = total_rows;
      total_rows += label_data_vecs[i].size();
      n_valid_labels++;
    }
  }
  for (uint32_t i = 0; i < data_label_vecs.size(); i++) {
    total_data_labels += data_label_vecs[i].size();
  }
  std::cout << "\n=== Index Combination ===" << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
  std::cout << "Number of valid labels: " << n_valid_labels << std::endl;
  std::cout << "Total rows: " << total_rows << std::endl;
  std::cout << "Total data labels: " << total_data_labels << std::endl;

  std::vector<uint32_t> host_index_map(total_rows);
  uint32_t iter = 0;
  for (uint32_t i = 0; i < label_number; i ++) {
    if (cat_freq[i] > specificity_threshold) continue;
    for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
      host_index_map[iter] = label_data_vecs[i][j];
      iter ++;
    }
  }

  auto index_map = raft::make_device_vector<uint32_t, int64_t>(res, total_rows);
  auto label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

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

  auto data_labels = raft::make_device_vector<uint32_t, int64_t>(res, total_data_labels);
  auto data_label_size = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());
  auto data_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, data_label_vecs.size());

  std::vector<uint32_t> h_data_labels(total_data_labels);
  std::vector<uint32_t> h_data_label_size(data_label_vecs.size());
  std::vector<uint32_t> h_data_label_offset(data_label_vecs.size());

  int64_t label_offset_ = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    const auto& labels = data_label_vecs[i];
    h_data_label_size[i] = labels.size();
    h_data_label_offset[i] = label_offset_;
    
    for (size_t j = 0; j < labels.size(); j++) {
      h_data_labels[label_offset_ + j] = labels[j];
    }
    label_offset_ += labels.size();
  }

  std::vector<int64_t> h_query_labels;
  h_query_labels.reserve(query_label_vecs.size());
  int n_queries = 0;
  for (size_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1 && cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
      h_query_labels.push_back(-1);
      n_queries++;
    }
    else if (query_label_vecs[i].size() == 2) {
      int label1 = query_label_vecs[i][0];
      int label2 = query_label_vecs[i][1];
      if (min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) {
        h_query_labels.push_back((cat_freq[label1] >= cat_freq[label2]) ? label1 : label2);
        n_queries++;
      }
    }
  }

  auto query_labels = raft::make_device_vector<int64_t, int64_t>(res, n_queries);
  // Copy filter data to device
  raft::update_device(data_labels.data_handle(), h_data_labels.data(), total_data_labels, raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_size.data_handle(), h_data_label_size.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_offset.data_handle(), h_data_label_offset.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(query_labels.data_handle(), h_query_labels.data(), n_queries, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset),
    std::move(data_labels),
    std::move(data_label_size),
    std::move(data_label_offset),
    std::move(query_labels)
  };
}


void compute_recall(const raft::host_matrix_view<int64_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<int64_t>& valid_query_indices,
                    const std::vector<std::vector<int>>& query_label_vecs) {
  
  int n_queries = neighbors.extent(0);
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;

  // Read specificities
  std::vector<double> specificities = read_specificities("/u/cmo1/Filtered/cuvs/examples/cpp/src/query_specificities.txt", 
                                                        100000);
  

  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/ivf_recall_results.csv");
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open output file");
  }
  outfile << "query_id,labels,specificity,recall\n";
  for (int i = 0; i < neighbors.extent(0); i++) {
    int matches = 0;
    auto gt_idx = valid_query_indices[i];

    for (int j = 0; j < topk; j++) {
      int64_t neighbor_idx = neighbors(i, j);
      if (std::find(gt_indices[gt_idx].begin(), gt_indices[gt_idx].end(), neighbor_idx) != gt_indices[gt_idx].end()) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;

    // Write to CSV: query_id, number of labels, specificity, recall
    outfile << gt_idx << ",";
    outfile << query_label_vecs[gt_idx][0];
    if (query_label_vecs[gt_idx].size() > 1) {
      outfile << "," << query_label_vecs[gt_idx][1];
    } else {
      outfile << ",";  // Empty second label field if there isn't one
    }
    outfile << "," << specificities[gt_idx] << "," 
            << recall << "\n";
  }
  // Close the file
  outfile.close();

  // Print summary to console
  std::cout << "\n=== Recall Analysis ===" << std::endl;
  std::cout << "IVF queries (" << n_queries << "): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to ivf_recall_results.csv" << std::endl;
}

void filtered_bfs_build_search_variants(raft::resources const& dev_resources,
                                        raft::device_matrix_view<const uint8_t, int64_t> dataset,
                                        raft::device_matrix_view<const uint8_t, int64_t> queries,
                                        const std::vector<uint8_t>& h_data,
                                        const std::vector<std::vector<int>>& data_label_vecs,
                                        const std::vector<std::vector<int>>& label_data_vecs,
                                        const std::vector<std::vector<int>>& query_label_vecs,
                                        const std::vector<int>& cat_freq,
                                        const std::vector<int>& query_freq,
                                        int specificity_threshold = 2000) {

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_label_vecs,
                                                  query_freq,
                                                  cat_freq,
                                                  specificity_threshold);
  
  ivf_flat::index<uint8_t, int64_t> filtered_ivf_index(dev_resources,
                                                      cuvs::distance::DistanceType::L2Unexpanded,
                                                      label_data_vecs.size(),
                                                      false,  // adaptive_centers
                                                      true,   // conservative_memory_allocation
                                                      queries.extent(1));

  std::string filename = "/scratch/bdes/cmo1/CAGRA/filtered_ivf_indices_yfcc/spec_" + 
                        std::to_string(specificity_threshold) + 
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
  int64_t n_queries = 0;
  int64_t total_points = 0;
  std::vector<int64_t> query_indices;
  std::vector<uint32_t> h_query_labels;
  for (size_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1 && cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
      n_queries++;
      total_points += cat_freq[query_label_vecs[i][0]];
    }
    else if (query_label_vecs[i].size() == 2) {
      int label1 = query_label_vecs[i][0];
      int label2 = query_label_vecs[i][1];
      if (std::min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) {
        n_queries++;
        int label = (cat_freq[label1] < cat_freq[label2]) ? label1 : label2;
        total_points += cat_freq[label];
      }
    }
  }

  // Reserve space
  query_indices.reserve(n_queries);
  h_query_labels.reserve(n_queries);

  // Second pass to collect queries
  for (size_t i = 0; i < query_label_vecs.size(); i++) {
    if (query_label_vecs[i].size() == 1 && cat_freq[query_label_vecs[i][0]] <= specificity_threshold) {
      query_indices.push_back(i);
      h_query_labels.push_back(query_label_vecs[i][0]);
    }
    else if (query_label_vecs[i].size() == 2) {
      int label1 = query_label_vecs[i][0];
      int label2 = query_label_vecs[i][1];
      if (std::min(cat_freq[label1], cat_freq[label2]) <= specificity_threshold) {
        query_indices.push_back(i);
        h_query_labels.push_back((cat_freq[label1] < cat_freq[label2]) ? label1 : label2);
      }
    }
  }
  
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  raft::update_device(query_labels.data_handle(), 
                      h_query_labels.data(), 
                      n_queries,
                      raft::resource::get_cuda_stream(dev_resources));
  
  auto filtered_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, n_queries, queries.extent(1));
  for (int64_t i = 0; i < n_queries; i++) {
    raft::copy_async(filtered_queries.data_handle() + i * queries.extent(1),
                    queries.data_handle() + query_indices[i] * queries.extent(1),
                    queries.extent(1),
                    raft::resource::get_cuda_stream(dev_resources));
  }
  raft::resource::sync_stream(dev_resources);

  // Prepare for search
  const int k = 10; 
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, k);

  std::cout << "\n=== Search Parameters ===" << std::endl;
  std::cout << "Number of queries: " << n_queries << std::endl;
  std::cout << "Specificity threshold: " << specificity_threshold << std::endl;
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

  // Compute recall
  auto h_neighbors = raft::make_host_matrix<int64_t, int64_t>(n_queries, k);
  raft::copy(h_neighbors.data_handle(), 
            neighbors.data_handle(), 
            neighbors.size(), 
            raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_gt_file(gt_fname, gt_indices);
  compute_recall(h_neighbors.view(), gt_indices, query_indices, query_label_vecs);
}

int main(int argc, char** argv) {
  // Parse command line argument
  int specificity_threshold = 2000;
  if (argc > 1) {
    specificity_threshold = std::stoi(argv[1]);
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
                                    h_data,
                                    data_label_vecs,
                                    label_data_vecs,
                                    query_label_vecs,
                                    cat_freq,
                                    query_freq,
                                    specificity_threshold);
}
