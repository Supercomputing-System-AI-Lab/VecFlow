// /*
//  * Copyright (c) 2024, NVIDIA CORPORATION.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include <cuvs/neighbors/cagra.hpp>
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

#include <omp.h>
#include "common.cuh"
#include "utils.h"


using namespace cuvs::neighbors;

template<typename T>
void cagra_build_subgraph(shared_resources::configured_raft_resources& dev_resources,
                          raft::device_matrix_view<const T, int64_t> dataset,
                          const std::vector<std::vector<int>>& label_data_vecs) 
{ 
  int dim = dataset.extent(1);

  std::cout << "Building subgraphs ..." << std::endl;
  std::cout << "Total labels: " << label_data_vecs.size() << std::endl;

  int optimal_threads;
  char* env_threads = getenv("OMP_NUM_THREADS");
  optimal_threads = std::atoi(env_threads);
  omp_set_num_threads(optimal_threads);

  #pragma omp parallel for num_threads(optimal_threads)
  for (int i=0; i<label_data_vecs.size(); i++) {
    std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
    if (std::filesystem::exists(index_path)) {
      continue;
    }
    int thread_id = omp_get_thread_num();
    shared_resources::thread_id = thread_id;
    shared_resources::n_threads = optimal_threads;
    auto thread_resources = dev_resources;
    cudaStream_t thread_stream = thread_resources.get_sync_stream();

    size_t label_size = label_data_vecs[i].size();
    std::cout << "Building graph for label: " << i << " Size: " << label_size << std::endl;

    if (label_size < 32) continue;

    auto filtered_dataset = raft::make_device_matrix<T, int64_t>(thread_resources, label_size, dim);
    auto mapping_list = raft::make_device_vector<int>(thread_resources, label_size);
    raft::copy(mapping_list.data_handle(), label_data_vecs[i].data(), label_size, thread_stream);
    
    const int dataset_block_size = 256;
    const int dataset_grid_size = std::min<int>((label_size * dim + dataset_block_size - 1) / dataset_block_size, 65535);
    copy_dataset_uint8<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
        dataset, filtered_dataset.view(), mapping_list.view());
    
    cagra::index_params index_params;
    index_params.intermediate_graph_degree = 32;
    index_params.graph_degree = 16;
    // std::string index_path = "/scratch/bdes/cmo1/CAGRA/indices_yfcc/label_" + std::to_string(i) + "_index_32_16.bin";
    auto index = cagra::build(thread_resources, index_params, raft::make_const_mdspan(filtered_dataset.view()));
    cagra::serialize(thread_resources, index_path, index);

    raft::resource::sync_stream(thread_resources);
  }
}


int main()
{
  shared_resources::configured_raft_resources dev_resources{};
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  std::string data_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.10M.u8bin";
  std::string data_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/base.sorted.metadata.10M.spmat";
  std::string query_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.public.100K.u8bin";
  std::string query_label_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/query.metadata.public.100K.spmat";
  std::string gt_fname = "/u/cmo1/Filtered/bench/data/yfcc100M/GT.public.ibin";

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
  size_t dim = h_data.size() / N;
  printf("N:%lld, dim:%lld\n", N, dim);
  auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, N, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);

  cagra_build_subgraph<uint8_t>(dev_resources,
                                raft::make_const_mdspan(dataset.view()),
                                label_data_vecs);
  return 0;
}