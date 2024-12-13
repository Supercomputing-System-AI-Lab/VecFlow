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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

int main() {
  raft::device_resources dev_resources;
  auto stream = raft::resource::get_cuda_stream(dev_resources);

  // Smaller dataset for easier debugging 
  const int n_rows = 100;           
  const int dim = 8;                
  const int n_clusters = 4;         
  const int points_per_cluster = n_rows / n_clusters;
  const int k = 5;                  

  // Create unique dataset
  std::vector<float> h_data(n_rows * dim);
  for(int i = 0; i < n_rows; i++) {
    for(int j = 0; j < dim; j++) {
      // Create unique values that are still clustered
      h_data[i * dim + j] = static_cast<float>(i) / n_rows + 
                           static_cast<float>(i % points_per_cluster) / (points_per_cluster * 2.0f);
    }
  }

  // Create dataset on device
  auto dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_rows, dim);
  raft::update_device(dataset.data_handle(), h_data.data(), n_rows * dim, stream);

  // Create query using first point
  auto query = raft::make_device_matrix<float, int64_t>(dev_resources, 1, dim);
  raft::copy(query.data_handle(), dataset.data_handle(), dim, stream);

  // Create storage for results
  auto bf_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, k);
  auto bf_distances = raft::make_device_matrix<float, int64_t>(dev_resources, 1, k);
  auto ivf_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, k);
  auto ivf_distances = raft::make_device_matrix<float, int64_t>(dev_resources, 1, k);
  auto ref_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, k);
  auto ref_distances = raft::make_device_matrix<float, int64_t>(dev_resources, 1, k);

  // Run brute force search
  std::cout << "Running brute force search...\n";
  auto bf_index = cuvs::neighbors::brute_force::build(
    dev_resources, dataset.view(), cuvs::distance::DistanceType::L2Unexpanded);
  cuvs::neighbors::brute_force::search(
    dev_resources, bf_index, query.view(), bf_neighbors.view(), bf_distances.view());

  // Create cluster assignments
  auto query_label = raft::make_device_vector<uint32_t, int64_t>(dev_resources, 1);
  auto label_sizes = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_clusters);
  auto label_offsets = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_clusters);

  std::vector<uint32_t> h_label_sizes(n_clusters, points_per_cluster);
  std::vector<uint32_t> h_label_offsets(n_clusters);
  for(int i = 0; i < n_clusters; i++) {
    h_label_offsets[i] = i * points_per_cluster;
  }
  
  raft::update_device(label_sizes.data_handle(), h_label_sizes.data(), n_clusters, stream);
  raft::update_device(label_offsets.data_handle(), h_label_offsets.data(), n_clusters, stream);

  // Build IVF index
  std::cout << "Building IVF index...\n";
  cuvs::neighbors::ivf_flat::index<float, int64_t> ivf_index(
    dev_resources, 
    cuvs::distance::DistanceType::L2Unexpanded,
    n_clusters, 
    false, 
    true, 
    dim);

  auto index_map = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_rows);
  std::vector<uint32_t> h_index_map(n_rows);
  for(uint32_t i = 0; i < n_rows; i++) {
    h_index_map[i] = i;
  }
  raft::update_device(index_map.data_handle(), h_index_map.data(), n_rows, stream);

  cuvs::neighbors::build_filtered_IVF_index(
    dev_resources,
    &ivf_index,
    dataset.view(),
    index_map.view(),
    label_sizes.view(),
    label_offsets.view());

  // Query cluster 0
  std::vector<uint32_t> h_query_label = {0};
  raft::update_device(query_label.data_handle(), h_query_label.data(), 1, stream);

  // Run IVF search
  cuvs::neighbors::search_filtered_ivf(
    dev_resources,
    ivf_index,
    query.view(),
    query_label.view(),
    label_sizes.view(),
    ivf_neighbors.view(),
    ivf_distances.view(),
    cuvs::distance::DistanceType::L2Unexpanded);

  // Create candidates just from cluster 0
  auto candidates = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, points_per_cluster);
  std::vector<int64_t> h_cluster_0_indices(points_per_cluster);
  for(int i = 0; i < points_per_cluster; i++) {
    h_cluster_0_indices[i] = i;  // First cluster indices
  }
  raft::update_device(candidates.data_handle(), 
                     h_cluster_0_indices.data(), 
                     points_per_cluster, 
                     stream);

  // Run refinement
  cuvs::neighbors::refine(
    dev_resources,
    dataset.view(),
    query.view(),
    candidates.view(),
    ref_neighbors.view(),
    ref_distances.view(),
    cuvs::distance::DistanceType::L2Unexpanded);

  // Get results
  std::vector<int64_t> h_bf_neighbors(k);
  std::vector<float> h_bf_distances(k);
  std::vector<int64_t> h_ivf_neighbors(k);
  std::vector<float> h_ivf_distances(k);
  std::vector<int64_t> h_ref_neighbors(k);
  std::vector<float> h_ref_distances(k);

  raft::update_host(h_bf_neighbors.data(), bf_neighbors.data_handle(), k, stream);
  raft::update_host(h_bf_distances.data(), bf_distances.data_handle(), k, stream);
  raft::update_host(h_ivf_neighbors.data(), ivf_neighbors.data_handle(), k, stream);
  raft::update_host(h_ivf_distances.data(), ivf_distances.data_handle(), k, stream);
  raft::update_host(h_ref_neighbors.data(), ref_neighbors.data_handle(), k, stream);
  raft::update_host(h_ref_distances.data(), ref_distances.data_handle(), k, stream);

  // Print detailed comparison
  std::cout << "\nDetailed Results Comparison:\n";
  std::cout << std::left;
  std::cout << std::setw(15) << "Method" 
            << std::setw(10) << "Index" 
            << std::setw(15) << "Distance" 
            << "Point Values\n";
  std::cout << std::string(100, '-') << "\n";

  for (int i = 0; i < k; i++) {
    // Print BF result
    std::cout << std::setw(15) << "Brute Force" 
              << std::setw(10) << h_bf_neighbors[i]
              << std::setw(15) << h_bf_distances[i];
    
    int idx = h_bf_neighbors[i];
    for(int j = 0; j < std::min(4, dim); j++) {
      std::cout << h_data[idx * dim + j] << " ";
    }
    std::cout << "\n";

    // Print IVF result
    std::cout << std::setw(15) << "Filtered IVF" 
              << std::setw(10) << h_ivf_neighbors[i]
              << std::setw(15) << h_ivf_distances[i];
    
    idx = h_ivf_neighbors[i];
    for(int j = 0; j < std::min(4, dim); j++) {
      std::cout << h_data[idx * dim + j] << " ";
    }
    std::cout << "\n";

    // Print Refined result 
    std::cout << std::setw(15) << "Refined" 
              << std::setw(10) << h_ref_neighbors[i]
              << std::setw(15) << h_ref_distances[i];
    
    idx = h_ref_neighbors[i];
    for(int j = 0; j < std::min(4, dim); j++) {
      std::cout << h_data[idx * dim + j] << " ";
    }
    std::cout << "\n\n";
  }

  return 0;
}