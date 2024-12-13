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

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> index_map;
  raft::device_vector<uint32_t, int64_t> label_size;
  raft::device_vector<uint32_t, int64_t> label_offset;
};

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<int>& cat_freq,
                              int specificity_threshold = 1000) -> CombinedIndices 
{
  
  int label_number = label_data_vecs.size();
  std::cout << "Starting load_and_combine_indices with label_number: " << label_number << std::endl;

  int64_t total_rows = 0;
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
    }
  }

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

  std::cout << "Successfully completed load_and_combine_indices" << std::endl;
  return CombinedIndices{
    std::move(index_map),
    std::move(label_size),
    std::move(label_offset)
  };
}

template <typename T, typename U>
__global__ void convert_types(const T* input, U* output, size_t n, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // n is number of points
    for (size_t d = 0; d < dim; d++) {
      output[idx * dim + d] = static_cast<U>(input[idx * dim + d]);
    }
  }
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
                                        int specificity_threshold = 1000,
                                        int batch_size = 1000,
                                        int iter = 10) {

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs, 
                                                  cat_freq,
                                                  specificity_threshold);
  
  ivf_flat::index<uint8_t, int64_t> filtered_ivf_index(dev_resources,
                                                      cuvs::distance::DistanceType::L2Unexpanded,
                                                      label_data_vecs.size(),
                                                      false,  // adaptive_centers
                                                      true,   // conservative_memory_allocation - changed to true
                                                      queries.extent(1));

  std::string filename = "/scratch/bdes/cmo1/CAGRA/filtered_ivf_indices_yfcc/spec_" + 
                        std::to_string(specificity_threshold) + 
                        "_index.bin";
  std::ifstream test_file(filename);
  if (test_file.good()) {
    std::cout << "Loading filtered IVF index from file: " << filename << std::endl;
    ivf_flat::deserialize(dev_resources, filename, &filtered_ivf_index);
    test_file.close();
  } else {
    std::cout << "Building filtered IVF index from scratch" << std::endl;
    build_filtered_IVF_index(dev_resources,
                            &filtered_ivf_index,
                            dataset,
                            combined_indices.index_map.view(),
                            combined_indices.label_size.view(),
                            combined_indices.label_offset.view());
    raft::resource::sync_stream(dev_resources);
    std::cout << "Saving filtered IVF index...\n";
    ivf_flat::serialize(dev_resources, filename, filtered_ivf_index);
    std::cout << "Saving filtered IVF index finished\n";
  }

  std::cout << "Filtered IVF index has " << filtered_ivf_index.n_lists() << " labels"
          << " with frequency lower than " << specificity_threshold 
          << " and " << combined_indices.index_map.size() << " points" << std::endl;
  
  // Prepare query labels
  std::cout << "\nProcessing " << batch_size * iter << " queries in batches of " << batch_size << std::endl;
  auto n_queries = batch_size * iter;
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_queries);
  std::vector<uint32_t> h_query_labels(n_queries);
  int valid_labels = 0;
  int64_t total_points = 0;
  for (uint32_t i = 0; i < label_data_vecs.size(); i++) {
    if (valid_labels >= n_queries) break;
    if (cat_freq[i] <= specificity_threshold) {
      total_points += label_data_vecs[i].size();
      h_query_labels[valid_labels] = i;
      valid_labels++;
    }
  }
  raft::update_device(query_labels.data_handle(), 
                      h_query_labels.data(), 
                      n_queries,
                      raft::resource::get_cuda_stream(dev_resources));

  double avg_points_per_label = static_cast<double>(total_points) / valid_labels;
  std::cout << "  Statistics for labels under threshold " << specificity_threshold << ":\n"
            << "  Valid labels: " << valid_labels << "\n"
            << "  Total points: " << total_points << "\n"
            << "  Average points per label: " << avg_points_per_label << std::endl;

  // Prepare for search
  const int k = 10; 
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, k);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);
  // Search the queries
  auto batch_queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, batch_size, queries.extent(1));
  auto batch_query_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, batch_size);

  // warm up
  for (int64_t i = 0; i < n_queries; i += batch_size) {
    raft::copy(batch_queries.data_handle(), 
              queries.data_handle() + i * queries.extent(1), 
              batch_size * queries.extent(1), 
              raft::resource::get_cuda_stream(dev_resources));
    raft::copy(batch_query_labels.data_handle(), 
              query_labels.data_handle() + i, 
              batch_size, 
              raft::resource::get_cuda_stream(dev_resources));
    // Search
    search_filtered_ivf(dev_resources,
                        filtered_ivf_index,
                        batch_queries.view(),
                        batch_query_labels.view(),
                        combined_indices.label_size.view(),
                        neighbors.view(),
                        distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded);
  }

  raft::resource::sync_stream(dev_resources);
  time_it("filtered_bfs/batch", [&]() {
    for (int64_t i = 0; i < n_queries; i += batch_size) {
      raft::copy(batch_queries.data_handle(), 
                queries.data_handle() + i * queries.extent(1), 
                batch_size * queries.extent(1), 
                raft::resource::get_cuda_stream(dev_resources));
      raft::copy(batch_query_labels.data_handle(), 
                query_labels.data_handle() + i, 
                batch_size, 
                raft::resource::get_cuda_stream(dev_resources));
      // Search
      search_filtered_ivf(dev_resources,
                          filtered_ivf_index,
                          batch_queries.view(),
                          batch_query_labels.view(),
                          combined_indices.label_size.view(),
                          neighbors.view(),
                          distances.view(),
                          cuvs::distance::DistanceType::L2Unexpanded);
      raft::resource::sync_stream(dev_resources);
    }
  });


  std::cout << "\nVerifying results with brute force search..." << std::endl;

  auto query_label = h_query_labels[batch_size * (iter - 1)];
  auto label_size = label_data_vecs[query_label].size();  // Use actual size from label_data_vecs

  std::cout << "Query label: " << query_label << " with " << label_size << " points" << std::endl;

  auto single_neighbor = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, k);
  auto single_distance = raft::make_device_matrix<float, int64_t>(dev_resources, 1, k);
  auto single_dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, label_size, queries.extent(1));
  auto single_query = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, 1, queries.extent(1));

  // Copy query
  raft::copy(single_query.data_handle(),
            batch_queries.data_handle(),
            queries.extent(1),
            raft::resource::get_cuda_stream(dev_resources));

  // Copy dataset points for this label
  for (auto i = 0; i < label_size; i++) {
    auto data_idx = label_data_vecs[query_label][i];
    raft::copy(single_dataset.data_handle() + i * queries.extent(1),
              dataset.data_handle() + data_idx * queries.extent(1),
              queries.extent(1),
              raft::resource::get_cuda_stream(dev_resources));
  }

  // Convert to float for brute force
  auto float_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, label_size, queries.extent(1));
  auto float_query = raft::make_device_matrix<float, int64_t>(dev_resources, 1, queries.extent(1));

  // Then use it like:
  int block_size = 256;
  int grid_size = (label_size + block_size - 1) / block_size;  // n points

  convert_types<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
    single_dataset.data_handle(),
    float_dataset.data_handle(),
    label_size,  // number of points
    queries.extent(1)  // dimension
  );

  grid_size = 1;  // only one query point
  convert_types<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(dev_resources)>>>(
    single_query.data_handle(),
    float_query.data_handle(),
    1,  // one point
    queries.extent(1)  // dimension
  );

  auto single_index = brute_force::build(dev_resources, 
                                        float_dataset.view(), 
                                        cuvs::distance::DistanceType::L2Unexpanded);
  brute_force::search(dev_resources,
                      single_index,
                      float_query.view(),
                      single_neighbor.view(),
                      single_distance.view());

  auto h_single_neighbor = raft::make_host_matrix<int64_t, int64_t>(1, queries.extent(1));
  auto h_neighbors = raft::make_host_matrix<int64_t, int64_t>(1, queries.extent(1));
  auto h_single_distance = raft::make_host_matrix<float, int64_t>(1, k);
  auto h_distances = raft::make_host_matrix<float, int64_t>(1, k);

  raft::copy(h_single_neighbor.data_handle(),
            single_neighbor.data_handle(),
            k,
            raft::resource::get_cuda_stream(dev_resources));
  raft::copy(h_neighbors.data_handle(),
            neighbors.data_handle(),
            k,
            raft::resource::get_cuda_stream(dev_resources));
  raft::copy(h_single_distance.data_handle(),
            single_distance.data_handle(),
            k,
            raft::resource::get_cuda_stream(dev_resources));
  raft::copy(h_distances.data_handle(),
            distances.data_handle(),
            k,
            raft::resource::get_cuda_stream(dev_resources));
  
  // Print results comparison with correct index mapping
  std::cout << "\nComparing results for first query with label " << query_label << ":\n";
  std::cout << std::setw(50) << "Brute Force" << std::setw(50) << "Filtered IVF" << std::endl;
  std::cout << std::setw(25) << "Neighbor" << std::setw(25) << "Distance" 
            << std::setw(25) << "Neighbor" << std::setw(25) << "Distance" << std::endl;
  std::cout << std::string(100, '-') << std::endl;
  
  for (int i = 0; i < k; i++) {
    // For brute force, map relative index to actual dataset index
    int64_t bf_relative_idx = h_single_neighbor(0, i);
    int64_t bf_actual_idx = label_data_vecs[query_label][bf_relative_idx];  // Map to original index

    std::cout << std::fixed << std::setprecision(4)
              << std::setw(25) << bf_actual_idx  // Use mapped index
              << std::setw(25) << h_single_distance(0, i)
              << std::setw(25) << h_neighbors(0, i)
              << std::setw(25) << h_distances(0, i)
              << std::endl;
  }
}

int main(int argc, char** argv) {
  // Parse command line argument
  int specificity_threshould = 1000;
  if (argc > 1) {
    specificity_threshould = std::stoi(argv[1]);
  }
  int batch_size = 1000; 
  if (argc > 2) {
    batch_size = std::stoi(argv[2]);
  }
  int iter = 10;
  if (argc > 3) {
    iter = std::stoi(argv[3]);
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
  printf("N:%lld, Nq: %lld, dim:%lld\n", N, Nq, dim);
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
                                    specificity_threshould,
                                    batch_size,
                                    iter);
}
