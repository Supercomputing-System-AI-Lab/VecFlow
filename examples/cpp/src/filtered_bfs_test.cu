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

  // Larger dataset and batch size
  const int n_rows = 100;           
  const int dim = 8;                
  const int n_clusters = 4;         
  const int points_per_cluster = n_rows / n_clusters;
  const int k = 5;
  const int batch_size = 16;  // Increased from 1 to 16 for batch processing

  // Create unique dataset
  std::vector<float> h_data(n_rows * dim);
  for(int i = 0; i < n_rows; i++) {
    for(int j = 0; j < dim; j++) {
      h_data[i * dim + j] = static_cast<float>(i) / n_rows + 
                           static_cast<float>(i % points_per_cluster) / (points_per_cluster * 2.0f);
    }
  }

  // Create dataset on device
  auto dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_rows, dim);
  raft::update_device(dataset.data_handle(), h_data.data(), n_rows * dim, stream);

  // Create batch query using first batch_size points
  auto query = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, dim);
  raft::copy(query.data_handle(), dataset.data_handle(), batch_size * dim, stream);

  // Create storage for batch results
  auto bf_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, k);
  auto bf_distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);
  auto ivf_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, k);
  auto ivf_distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);
  auto ref_neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, k);
  auto ref_distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, k);

  // Run brute force search
  std::cout << "Running brute force search for batch size " << batch_size << "...\n";
  auto bf_index = cuvs::neighbors::brute_force::build(
    dev_resources, dataset.view(), cuvs::distance::DistanceType::L2Unexpanded);
  cuvs::neighbors::brute_force::search(
    dev_resources, bf_index, query.view(), bf_neighbors.view(), bf_distances.view());

  // Create cluster assignments for batch
  auto query_label = raft::make_device_vector<uint32_t, int64_t>(dev_resources, batch_size);
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

  // Assign queries to cluster 0 (for demonstration)
  std::vector<uint32_t> h_query_label(batch_size, 0);
  raft::update_device(query_label.data_handle(), h_query_label.data(), batch_size, stream);

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

  // Create candidates for batch (all from cluster 0)
  auto candidates = raft::make_device_matrix<int64_t, int64_t>(dev_resources, batch_size, points_per_cluster);
  std::vector<int64_t> h_cluster_0_indices(batch_size * points_per_cluster);
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < points_per_cluster; j++) {
      h_cluster_0_indices[i * points_per_cluster + j] = j;  // First cluster indices
    }
  }
  raft::update_device(candidates.data_handle(), 
                     h_cluster_0_indices.data(), 
                     batch_size * points_per_cluster, 
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
  std::vector<int64_t> h_bf_neighbors(batch_size * k);
  std::vector<float> h_bf_distances(batch_size * k);
  std::vector<int64_t> h_ivf_neighbors(batch_size * k);
  std::vector<float> h_ivf_distances(batch_size * k);
  std::vector<int64_t> h_ref_neighbors(batch_size * k);
  std::vector<float> h_ref_distances(batch_size * k);

  raft::update_host(h_bf_neighbors.data(), bf_neighbors.data_handle(), batch_size * k, stream);
  raft::update_host(h_bf_distances.data(), bf_distances.data_handle(), batch_size * k, stream);
  raft::update_host(h_ivf_neighbors.data(), ivf_neighbors.data_handle(), batch_size * k, stream);
  raft::update_host(h_ivf_distances.data(), ivf_distances.data_handle(), batch_size * k, stream);
  raft::update_host(h_ref_neighbors.data(), ref_neighbors.data_handle(), batch_size * k, stream);
  raft::update_host(h_ref_distances.data(), ref_distances.data_handle(), batch_size * k, stream);

  // Print results for first few queries in batch
  const int queries_to_print = std::min(3, batch_size);
  std::cout << "\nDetailed Results Comparison (First " << queries_to_print << " queries):\n";
  
  for(int q = 0; q < queries_to_print; q++) {
    std::cout << "\nQuery " << q << ":\n";
    std::cout << std::left;
    std::cout << std::setw(15) << "Method" 
              << std::setw(10) << "Index" 
              << std::setw(15) << "Distance" 
              << "Point Values\n";
    std::cout << std::string(100, '-') << "\n";

    for (int i = 0; i < k; i++) {
      int batch_offset = q * k + i;
      
      // Print BF result
      std::cout << std::setw(15) << "Brute Force" 
                << std::setw(10) << h_bf_neighbors[batch_offset]
                << std::setw(15) << h_bf_distances[batch_offset];
      
      int idx = h_bf_neighbors[batch_offset];
      for(int j = 0; j < std::min(4, dim); j++) {
        std::cout << h_data[idx * dim + j] << " ";
      }
      std::cout << "\n";

      // Print IVF result
      std::cout << std::setw(15) << "Filtered IVF" 
                << std::setw(10) << h_ivf_neighbors[batch_offset]
                << std::setw(15) << h_ivf_distances[batch_offset];
      
      idx = h_ivf_neighbors[batch_offset];
      for(int j = 0; j < std::min(4, dim); j++) {
        std::cout << h_data[idx * dim + j] << " ";
      }
      std::cout << "\n";

      // Print Refined result 
      std::cout << std::setw(15) << "Refined" 
                << std::setw(10) << h_ref_neighbors[batch_offset]
                << std::setw(15) << h_ref_distances[batch_offset];
      
      idx = h_ref_neighbors[batch_offset];
      for(int j = 0; j < std::min(4, dim); j++) {
        std::cout << h_data[idx * dim + j] << " ";
      }
      std::cout << "\n\n";
    }
  }

  return 0;
}