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
#include "../common.cuh"
#include "../utils.h"

// #include <cuvs/neighbors/cagra.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
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

using namespace raft::neighbors;

struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> data_labels;
  raft::device_vector<uint32_t, int64_t> data_label_size;
  raft::device_vector<uint32_t, int64_t> data_label_offset;
  raft::device_vector<int64_t, int64_t> query_labels;
};

struct cagra_filter {
  raft::device_vector_view<uint32_t, int64_t> data_labels_;
  raft::device_vector_view<uint32_t, int64_t> data_label_size_;
  raft::device_vector_view<uint32_t, int64_t> data_label_offset_;
  raft::device_vector_view<int64_t, int64_t> query_labels_;
  const uint32_t query_offset_;

  cagra_filter(raft::device_vector_view<uint32_t, int64_t> data_labels,
              raft::device_vector_view<uint32_t, int64_t> data_label_size,
              raft::device_vector_view<uint32_t, int64_t> data_label_offset,
              raft::device_vector_view<int64_t, int64_t> query_labels,
              const uint32_t query_offset = 0)
  : data_labels_{data_labels}, 
    data_label_size_{data_label_size},
    data_label_offset_{data_label_offset},
    query_labels_{query_labels},
    query_offset_{query_offset} {}
  
   inline _RAFT_HOST_DEVICE bool binary_search(
    const int target,
    const int start,
    const int end) const 
  {
    int left = start;
    int right = end;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      int mid_val = data_labels_(mid);
      if (mid_val == target) {
        return true;
      }
      if (mid_val < target) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return false;
  }

  
  inline _RAFT_HOST_DEVICE bool operator()(
    const uint32_t query_ix,    // Local query index within batch
    const uint32_t sample_ix) const 
  {
    
    uint32_t global_query_ix = query_ix + query_offset_;
    
    int left = data_label_offset_[sample_ix];
    int right = left + data_label_size_[sample_ix] - 1;

    
    // Search for first category
    int query_label = query_labels_[global_query_ix];
    if (!binary_search(query_label, left, right)) {
      return false;
    }
    return true;
  }

  cagra_filter update_filter(uint32_t new_offset) const {
    return cagra_filter(data_labels_, data_label_size_, data_label_offset_, query_labels_, new_offset);
  }
};

void save_matrix_to_ibin(raft::resources const& res,
                         const std::string& filename, 
                         const raft::device_matrix_view<uint32_t, int64_t>& matrix) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filename);
  }

  // Write dimensions
  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

  // Copy data to host
  std::vector<uint32_t> h_matrix(rows * cols);
  raft::copy(h_matrix.data(), matrix.data_handle(), rows * cols, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  
  // Write data
  file.write(reinterpret_cast<const char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
  file.close();
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

void read_ground_true_file(const std::string& fname, std::vector<std::vector<uint32_t>>& gt_indices) {

	std::ifstream file(fname, std::ios::binary);
	if (!file) {
		std::cout << "Warning: Ground truth file not found: " << fname << std::endl;
		return;
	}

	// Read dimensions
	int64_t rows, cols;
	file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
	file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

	// std::cout << "\n=== Reading Ground Truth File ===" << std::endl;
	// std::cout << "Dimensions: [" << rows << " x " << cols << "]" << std::endl;

	// Read data to temporary host buffer
	std::vector<uint32_t> h_matrix(rows * cols);
	file.read(reinterpret_cast<char*>(h_matrix.data()), rows * cols * sizeof(uint32_t));
	file.close();

	// Reshape into vector of vectors
	gt_indices.resize(rows);
	for (int64_t i = 0; i < rows; i++) {
		gt_indices[i].resize(cols);
		for (int64_t j = 0; j < cols; j++) {
			gt_indices[i][j] = h_matrix[i * cols + j];
		}
	}
}

auto load_and_combine_indices(raft::resources const& res, 
                              const std::vector<std::vector<int>>& label_data_vecs,
                              const std::vector<std::vector<int>>& data_label_vecs,
                              const std::vector<std::vector<int>>& query_label_vecs,
                              const std::vector<int>& query_freq,
                              const std::vector<int>& cat_freq) -> CombinedIndices {
  
  int64_t total_data_labels = 0;
  for (uint32_t i = 0; i < data_label_vecs.size(); i++) {
    total_data_labels += data_label_vecs[i].size();
  }

  auto data_labels = raft::make_device_vector<uint32_t, int64_t>(res, total_data_labels);
  raft::resource::sync_stream(res);
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
  
  int n_queries = query_label_vecs.size();
  std::vector<int64_t> h_query_labels(n_queries);
  for (size_t i = 0; i < n_queries; i++) {
    h_query_labels[i] = query_label_vecs[i][0];
  }

  auto query_labels = raft::make_device_vector<int64_t, int64_t>(res, n_queries);
  raft::resource::sync_stream(res);
  // Copy filter data to device
  raft::update_device(data_labels.data_handle(), h_data_labels.data(), total_data_labels, raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_size.data_handle(), h_data_label_size.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::update_device(data_label_offset.data_handle(), h_data_label_offset.data(), data_label_vecs.size(), raft::resource::get_cuda_stream(res));
  raft::copy(query_labels.data_handle(), h_query_labels.data(), h_query_labels.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return CombinedIndices{
    std::move(data_labels),
    std::move(data_label_size),
    std::move(data_label_offset),
    std::move(query_labels)
  };
}

void compute_recall(const raft::host_matrix_view<uint32_t, int64_t>& neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices,
                    const std::vector<std::vector<int>>& query_label_vecs) {
  
  int n_queries = neighbors.extent(0);
  int topk = neighbors.extent(1);
  float total_recall = 0.0f;

  // Open CSV file for writing
  std::ofstream outfile("/u/cmo1/Filtered/cuvs/examples/cpp/src/result/sift_if_recall_results.csv");
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open output file");
  }

  // Write CSV header
  outfile << "query_id,labels,recall\n";

  for (int i = 0; i < neighbors.extent(0); i++) {
    int matches = 0;
    for (int j = 0; j < topk; j++) {
      uint32_t neighbor_idx = neighbors(i, j);
      if (std::find(gt_indices[i].begin(), 
                  gt_indices[i].begin() + topk, 
                  neighbor_idx) != 
        gt_indices[i].begin() + topk) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;

    // Write to CSV: query_id, number of labels, specificity, recall
    outfile << i << ",";
    outfile << query_label_vecs[i][0];
    if (query_label_vecs[i].size() > 1) {
      outfile << "," << query_label_vecs[i][1];
    } else {
      outfile << ",";  // Empty second label field if there isn't one
    }
  }
  // Close the file
  outfile.close();

  // Print summary to console
  std::cout << "\n=== Recall Analysis ===" << std::endl;
  std::cout << "Overall recall (" << n_queries << "): " 
            << std::fixed << total_recall / n_queries << std::endl;
  std::cout << "Results have been written to sift_if_recall_results.csv" << std::endl;
}

void cagra_build_search_variants(shared_resources::configured_raft_resources& dev_resources,
                                raft::device_matrix_view<const float, int64_t> dataset,
                                raft::device_matrix_view<const float, int64_t> queries,
                                const std::vector<std::vector<int>>& data_label_vecs,
                                const std::vector<std::vector<int>>& label_data_vecs,
                                const std::vector<std::vector<int>>& query_label_vecs,
                                const std::vector<int>& cat_freq,
                                const std::vector<int>& query_freq,
                                int itopk_size,
                                int64_t topk,
                                int graph_degree) {

  int64_t n_queries = queries.extent(0);
  int dim = queries.extent(1);

  // Create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Build index
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = graph_degree * 2;
  index_params.graph_degree = graph_degree;
  std::cout << "Building CAGRA index (search graph)" << std::endl;
  std::string index_file = "/scratch/bdes/cmo1/CAGRA/indices_sift1M/raft_index_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + ".bin";
  std::ifstream test_file(index_file);
  auto index = cagra::index<float, uint32_t>(dev_resources);
  if (test_file.good()) {
    index = cagra::deserialize<float, uint32_t>(dev_resources, index_file);
    raft::resource::sync_stream(dev_resources);
    test_file.close();
  } else {
    std::cout << "Building index from scratch" << std::endl;
    index = cagra::build(dev_resources, index_params, dataset);
    cagra::serialize(dev_resources, index_file, index);
    raft::resource::sync_stream(dev_resources);
  }
  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;
  std::cout << "Loading index from: " << index_file << std::endl;
  std::cout << "CAGRA index size: " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph: [" << index.graph().extent(0) << ", " 
            << index.graph().extent(1) << "] (degree " << index.graph_degree() << ")" << std::endl;

  // Configure standard search parameters
  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;
  search_params.algo = cagra::search_algo::MULTI_CTA;

  // Load combined indices
  auto combined_indices = load_and_combine_indices(dev_resources, 
                                                  label_data_vecs,
                                                  data_label_vecs,
                                                  query_label_vecs, 
                                                  query_freq, 
                                                  cat_freq);
  raft::resource::sync_stream(dev_resources);

  int batch_size = 5000;
  int64_t num_batches = n_queries / batch_size;
  
  auto batch_queries = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, dim);
  auto batch_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, batch_size, topk);
  auto batch_distances = raft::make_device_matrix<float, int64_t>(dev_resources, batch_size, topk);
  raft::resource::sync_stream(dev_resources);

  // Performance measurement variables
  double total_time = 0.0;
  std::vector<double> batch_times;
  
  // Warm-up run
  auto filter = cagra_filter(combined_indices.data_labels.view(),
                            combined_indices.data_label_size.view(), 
                            combined_indices.data_label_offset.view(),
                            combined_indices.query_labels.view());
  raft::resource::sync_stream(dev_resources);
  std::cout << "Warm up ...\n";
  for (int i = 0; i < 10; i++) {
    for (uint32_t batch = 0; batch < num_batches; batch++) {
      uint32_t start_idx = batch * batch_size;
      uint32_t current_batch_size = batch_size;
      
      // Copy batch queries
      raft::copy(batch_queries.data_handle(),
                queries.data_handle() + start_idx * dim,
                current_batch_size * dim,
                raft::resource::get_cuda_stream(dev_resources));
      raft::resource::sync_stream(dev_resources);
      auto batch_filter = filter.update_filter(start_idx);
      cagra::search_with_filtering(dev_resources,
                                  search_params,
                                  index,
                                  raft::make_const_mdspan(batch_queries.view()),
                                  batch_neighbors.view(),
                                  batch_distances.view(),
                                  batch_filter);
      raft::resource::sync_stream(dev_resources);
      raft::copy(neighbors.data_handle() + start_idx * topk,
                batch_neighbors.data_handle(),
                current_batch_size * topk,
                raft::resource::get_cuda_stream(dev_resources));
      raft::resource::sync_stream(dev_resources);
    }
  }
  
  // Main measurement runs
  int num_runs = 10;
  for (int run = 0; run < num_runs; run++) {
    auto run_start = std::chrono::system_clock::now();
    
    for (uint32_t batch = 0; batch < num_batches; batch++) {
      uint32_t start_idx = batch * batch_size;
      uint32_t current_batch_size = batch_size;
      
      // Copy batch queries
      raft::copy(batch_queries.data_handle(),
                queries.data_handle() + start_idx * dim,
                current_batch_size * dim,
                raft::resource::get_cuda_stream(dev_resources));
      auto batch_filter = filter.update_filter(start_idx);
      
      auto batch_start = std::chrono::system_clock::now();
      cagra::search_with_filtering(dev_resources,
                                  search_params,
                                  index,
                                  raft::make_const_mdspan(batch_queries.view()),
                                  batch_neighbors.view(),
                                  batch_distances.view(),
                                  batch_filter);
      raft::resource::sync_stream(dev_resources);

      raft::copy(neighbors.data_handle() + start_idx * topk,
                batch_neighbors.data_handle(),
                current_batch_size * topk,
                raft::resource::get_cuda_stream(dev_resources));
      raft::resource::sync_stream(dev_resources);
      
      auto batch_end = std::chrono::system_clock::now();
      auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        batch_end - batch_start).count() / 1000.0;  // Convert to milliseconds
      batch_times.push_back(batch_duration);
    }
    
    auto run_end = std::chrono::system_clock::now();
    auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      run_end - run_start).count() / 1000.0;  // Convert to milliseconds
    total_time += run_duration;
  }

  double avg_time_ms = total_time / num_runs;
  double qps = (n_queries * 1000.0) / avg_time_ms;
  std::cout << "\n=== Single Index Search Performance ===" << std::endl;
  std::cout << "Queries: " << n_queries << std::endl;
  std::cout << "itopk_size: " << itopk_size << std::endl;
  std::cout << "topk: " << topk << std::endl;
  std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
  std::cout << "QPS: " << std::scientific << qps << std::endl;

  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), neighbors.data_handle(), neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  // Read ground truth neighbors
  std::string gt_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.groundtruth.neighbors.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_ground_true_file(gt_fname, gt_indices);
  compute_recall(h_neighbors.view(), gt_indices, query_label_vecs);
}


int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 512;  // Default value
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }

  int64_t topk = 10;  // Default value
  if (argc > 2) {
    topk = std::stoi(argv[2]);
  }

  int graph_degree = 16;  // Default value
  if (argc > 3) {
    graph_degree = std::stoi(argv[3]);
  }

  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.fbin";
  std::string data_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.base.spmat";
  std::string query_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.fbin";
  std::string query_label_fname = "/scratch/bdes/cmo1/CAGRA/dataset/sift1M/sift.query.spmat";

  std::vector<float> h_data;
  std::vector<float> h_queries;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);
  int max_N = 10000000;

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

  // Copy datasets to GPU
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), stream);

  // run the interesting part of the program
  cagra_build_search_variants(dev_resources, 
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              data_label_vecs,
                              label_data_vecs,
                              query_label_vecs,
                              cat_freq,
                              query_freq,
                              itopk_size,
                              topk,
                              graph_degree);
}
