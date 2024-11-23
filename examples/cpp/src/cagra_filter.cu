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

#include "common.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>

// A helper to split the dataset into chunks
template <typename DeviceMatrixOrView>
auto slice_matrix(DeviceMatrixOrView source,
                  typename DeviceMatrixOrView::index_type offset_rows,
                  typename DeviceMatrixOrView::index_type count_rows) {
  auto n_cols = source.extent(1);
  return raft::make_device_matrix_view<
      typename DeviceMatrixOrView::element_type,
      typename DeviceMatrixOrView::index_type>(
      source.data_handle() + offset_rows * n_cols, count_rows, n_cols);
}



auto load_and_combine_indices(raft::resources const& res, int label_number, 
                            std::unordered_map<uint32_t, cagra::label_info>& label_map) {
    // First pass: calculate total size and get graph width
    int64_t total_rows = 0;
    int64_t graph_width = 0;
    for (uint32_t i = 1; i <= label_number; i++) {
        cagra::index<float, uint32_t> temp_index(res);
        std::string index_path = "indices/label_" + std::to_string(i) + "_index_32_16.bin";
        cagra::deserialize(res, index_path, &temp_index);
        total_rows += temp_index.graph().extent(0);
        graph_width = temp_index.graph().extent(1);
    }
    
    // Create final matrix before the second pass
    auto final_matrix = raft::make_device_matrix<uint32_t, int64_t>(res, total_rows, graph_width);
    
    // Second pass: combine indices and build label map
    int64_t current_offset = 0;

    for (uint32_t i = 1; i <= label_number; i++) {
        auto temp_index = cagra::index<float, uint32_t>(res);
        
        std::string index_path = "indices/label_" + std::to_string(i) + "_index_32_16.bin";
        cagra::deserialize(res, index_path, &temp_index);
        
        auto graph = temp_index.graph();
        int64_t n_rows = graph.extent(0);

        // Store label info
        cagra::label_info info;
        info.offset = current_offset;
        info.size = n_rows;
        info.local_to_global.resize(n_rows);

        // Load the indices mapping
        std::string indices_path = "indices/label_" + std::to_string(i) + "_indices.bin";
        std::vector<uint32_t> host_indices(n_rows);
        std::ifstream file(indices_path, std::ios::binary);
        file.read(reinterpret_cast<char*>(host_indices.data()), n_rows * sizeof(uint32_t));
        file.close();

        info.local_to_global = host_indices;
        label_map[i] = info;
        
        raft::copy_async(final_matrix.data_handle() + current_offset * graph.extent(1),
                        graph.data_handle(),
                        n_rows * graph.extent(1),
                        raft::resource::get_cuda_stream(res));
        current_offset += n_rows * graph_width;
    }
    
    return final_matrix;
}

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

void cagra_build_search_variants(
    raft::device_resources const &res,
    raft::device_matrix_view<const float, int64_t> dataset,
    raft::device_matrix_view<const float, int64_t> queries) {
  using namespace cuvs::neighbors;

  // Number of neighbors to search
  int64_t topk = 10;
  // We split the queries set into three subsets for our experiment, one for a
  // sanity check and two for measuring the performance.
  int64_t n_queries_a = queries.extent(0) / 2;
  int64_t n_queries_b = queries.extent(0) - n_queries_a;

  auto queries_a = slice_matrix(queries, 0, n_queries_a);
  auto queries_b = slice_matrix(queries, n_queries_a, n_queries_b);

  // create output arrays
  auto neighbors =
      raft::make_device_matrix<uint32_t>(res, queries.extent(0), topk);
  auto distances =
      raft::make_device_matrix<float>(res, queries.extent(0), topk);
  // slice them same as queries
  auto neighbors_a = slice_matrix(neighbors, 0, n_queries_a);
  auto distances_a = slice_matrix(distances, 0, n_queries_a);
  auto neighbors_b = slice_matrix(neighbors, n_queries_a, n_queries_b);
  auto distances_b = slice_matrix(distances, n_queries_a, n_queries_b);

  // use default index parameters
  cagra::index<float, uint32_t> index(res);
  uint32_t label_number = 50;

  // Load and combine all indices
  std::unordered_map<uint32_t, cagra::label_info> label_map;
  auto combined_graph = load_and_combine_indices(res, label_number, label_map);

  // Update the index with the combined graph
  index.update_graph(res, raft::make_const_mdspan(combined_graph.view()));
  index.update_dataset(res, raft::make_const_mdspan(dataset));

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree()
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  cagra::search_params search_params;
  // get a decent recall by increasing the internal topk list
  search_params.itopk_size = 512;
  // Add the label-specific parameters
  search_params.label_map = &label_map;
  
  // Create and set random query labels (1 to 50)
  auto query_labels = raft::make_device_vector<uint32_t>(res, queries.extent(0));
  std::vector<uint32_t> host_query_labels(queries.extent(0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(1, 50);
  for (size_t i = 0; i < host_query_labels.size(); i++) {
      host_query_labels[i] = dist(gen);
  }
  raft::update_device(query_labels.data_handle(), host_query_labels.data(), queries.extent(0), 
                     raft::resource::get_cuda_stream(res));
  search_params.query_labels = query_labels.data_handle();

  // Print first 10 query labels for verification
  std::cout << "First 10 query labels: ";
  for (int i = 0; i < 10; i++) {
      std::cout << host_query_labels[i] << " ";
  }
  std::cout << std::endl;
  auto search_batch =
      [&res, &index](bool needs_sync, const cagra::search_params &ps,
                     raft::device_matrix_view<const float, int64_t> queries,
                     raft::device_matrix_view<uint32_t, int64_t> neighbors,
                     raft::device_matrix_view<float, int64_t> distances) {
        cagra::search(res, ps, index, queries, neighbors, distances);
        /*
        To make a fair comparison, standard implementation needs to synchronize
        with the device to make sure the kernel has finished the work.
        Persistent kernel does not make any use of CUDA streams and blocks till
        the results are available. Hence, synchronizing with the stream is a
        waste of time in this case.
         */
        if (needs_sync) {
          raft::resource::sync_stream(res);
        }
      };


  // Launch the baseline search: check the big-batch performance
  time_it("standard/batch A", search_batch, true, search_params, queries_a,
          neighbors_a, distances_a);
  time_it("standard/batch B", search_batch, true, search_params, queries_b,
          neighbors_b, distances_b);
}

int main() {
  raft::device_resources res;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  int64_t n_samples = 1000000;
  int64_t n_dim = 128;
  int64_t n_queries = 20000;
  auto dataset =
      raft::make_device_matrix<float, int64_t>(res, n_samples, n_dim);
  auto queries =
      raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);
  generate_dataset(res, dataset.view(), queries.view());

  // run the interesting part of the program
  cagra_build_search_variants(res, raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()));
}
