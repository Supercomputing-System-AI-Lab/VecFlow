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

#include <chrono>
#include <cstdint>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <omp.h>
#include <raft/neighbors/cagra.cuh>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/random/make_blobs.cuh>

//#include <raft/core/bitmap.cuh>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include "common.cuh"
#include "report.cpp"
#include "utils.h"
//#include "../../../cpp/src/neighbors/cagra.cuh" // Temporary workaround to get compile with filtered search

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

struct cagra_filter {
  const raft::device_vector_view<const int> data_cats_;     // Flattened device_vector for data samples
  const raft::device_matrix_view<const int> query_cats_;    // Matrix (n_queries x 2) for query samples
  const raft::device_vector_view<const int> data_offsets_;  // Offsets for data samples
  const raft::device_vector_view<const int> query_mapping_; // Maps local query indices to global indices
  const uint32_t query_offset_;                             // Global offset for the current batch
  const int single_label_flag_;

  cagra_filter(
              const raft::device_vector_view<const int> data_cats,
              const raft::device_matrix_view<const int> query_cats,
              const raft::device_vector_view<const int> data_offsets,
              const raft::device_vector_view<const int> query_mapping,
              const int single_label_flag,
              const uint32_t query_offset = 0)
  : data_cats_{data_cats}, 
    query_cats_{query_cats},
    data_offsets_{data_offsets},
    query_mapping_{query_mapping},
    single_label_flag_{single_label_flag},
    query_offset_{query_offset} {}

  // inline _RAFT_HOST_DEVICE bool operator()(
  //   const uint32_t query_ix,    // Local query index within batch
  //   const uint32_t sample_ix) 
  // const 
  // {
  //   uint32_t mapped_query_ix = query_mapping_(query_ix);
  //   uint32_t global_query_ix = mapped_query_ix + query_offset_;
    
  //   int sample_start = sample_ix == 0 ? 0 : data_offsets_(sample_ix - 1);
  //   int sample_end = data_offsets_(sample_ix);

  //   bool found = false;
  //   int query_cat = query_cats_(global_query_ix, 0);
  //   for (int j = sample_start; j < sample_end; j++) {
  //     if (query_cat == data_cats_(j)) {
  //       found = true;
  //       break;
  //     }
  //   }
  //   if (!found) return false;
    
  //   query_cat = query_cats_(global_query_ix, 1);
  //   if (query_cat != single_label_flag_) {
  //     found = false;
  //     for (int j = sample_start; j < sample_end; j++) {
  //       if (query_cat == data_cats_(j)) {
  //         found = true;
  //         break;
  //       }
  //     }
  //     if (!found) return false;
  //   }
  //   return true;
  // }

  // Binary search helper function
  inline _RAFT_HOST_DEVICE bool binary_search(
    const int target,
    const int start,
    const int end) const 
  {
    int left = start;
    int right = end - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      int mid_val = data_cats_(mid);
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
    const uint32_t sample_ix) 
  const 
  {
    uint32_t mapped_query_ix = query_mapping_(query_ix);
    uint32_t global_query_ix = mapped_query_ix + query_offset_;
    
    int sample_start = sample_ix == 0 ? 0 : data_offsets_(sample_ix - 1);
    int sample_end = data_offsets_(sample_ix);

    // if (query_cats_(global_query_ix, 0) == 95 && global_query_ix == 19) {
    //   printf("Debug - Query Label: %u, Global Query ID: %u\n", query_cats_(global_query_ix, 0), global_query_ix);
    //   printf("Data indices %d\n", sample_ix);
    //   // Print a few data_cats_ values in the range for verification
    // }
    
    // Search for first category
    int query_cat = query_cats_(global_query_ix, 0);
    if (!binary_search(query_cat, sample_start, sample_end)) {
      return false;
    }
    
    // Search for second category if needed
    query_cat = query_cats_(global_query_ix, 1);
    if (query_cat != single_label_flag_) {
      if (!binary_search(query_cat, sample_start, sample_end)) {
        return false;
      }
    }

    return true;
  }

  cagra_filter update_filter(uint32_t new_offset, 
                            raft::device_vector_view<const int> new_query_mapping) const {
    return cagra_filter(data_cats_, query_cats_, data_offsets_, new_query_mapping, single_label_flag_, new_offset);
  }
};


// void bf_search(raft::device_resources const& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk) {
  
//   // std::cout << "BF batch: offset=" << query_offset << " size=" << bf_search_size << std::endl;
  
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
//   int dim = dataset.extent(1);
  
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
//   copy_bf_query<<<32,32>>>(queries, bf_queries.view(), query_map.view(), query_offset);


//   for (int bf_idx = 0; bf_idx < bf_search_size; bf_idx++) {
//     std::vector<int> matching_indices;
//     auto query_labels = query_label_vecs[query_offset+bf_query_map[bf_idx]];
//     matching_indices.reserve(cat_freq[query_labels[0]]);
//     int write_id = 0;
//     if (query_labels.size() == 1) {
//       matching_indices = labels_data[query_labels[0]];
//       write_id = labels_data[query_labels[0]].size();
//     } else {
//       const auto& first_label_data = labels_data[query_labels[0]];
//       const auto& second_label_data = labels_data[query_labels[1]];
//       size_t i = 0, j = 0;
//       while (i < first_label_data.size() && j < second_label_data.size()) {
//         if (first_label_data[i] < second_label_data[j]) {
//           i++;
//         } else if (second_label_data[j] < first_label_data[i]) {
//           j++;
//         } else {
//           matching_indices.push_back(first_label_data[i]);
//           write_id++;
//           i++;
//           j++;
//         }
//       }
//     }

//     // std::cout << "Query " << bf_idx << " (global=" << query_offset+bf_query_map[bf_idx] 
//     //           << "): labels=[";
//     // for(int l : query_labels) std::cout << l << " ";
//     // std::cout << "], matches=" << write_id << std::endl;

//     auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, write_id, dim);
//     auto d_matching_indices = raft::make_device_vector<int>(dev_resources, write_id);
//     raft::copy(d_matching_indices.data_handle(), matching_indices.data(), write_id, stream);
//     copy_dataset<<<32,32>>>(dataset, d_filtered_dataset.view(), d_matching_indices.view());

//     auto bfs_index = cuvs::neighbors::brute_force::build(
//       dev_resources,
//       raft::make_const_mdspan(d_filtered_dataset.view())
//     );
    
//     auto d_distances = raft::make_device_matrix<float, int64_t>(dev_resources, 1, topk);
//     auto d_indices = raft::make_device_matrix<int64_t, int64_t>(dev_resources, 1, topk);

//     auto bf_single_query = raft::make_device_matrix<float, int64_t>(dev_resources, 1, dim);
//     raft::copy(bf_single_query.data_handle(), bf_queries.data_handle()+bf_idx*dim, dim, stream);
//     cuvs::neighbors::brute_force::search(
//       dev_resources,
//       bfs_index,
//       raft::make_const_mdspan(bf_single_query.view()),
//       d_indices.view(),
//       d_distances.view()
//     );

//     copy_bf_neighbors<<<1, topk>>>(d_indices.view(), all_neighbors, d_matching_indices.view(), 
//                                   query_map.view(), query_offset, bf_idx);
//     cudaDeviceSynchronize();
//   }
 
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk) {
  
//   int dim = dataset.extent(1);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  
//   // Initial data setup using main stream
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
//   copy_bf_query<<<32,32>>>(queries, bf_queries.view(), query_map.view(), query_offset);
  
//   int n_threads = omp_get_max_threads();
//   for (int start_idx = 0; start_idx < bf_search_size; start_idx += n_threads) {
//     // Calculate the actual number of items for this batch
//     int batch_size = std::min(n_threads, bf_search_size - start_idx);
//     #pragma omp parallel for num_threads(batch_size)
//     for (int batch_offset = 0; batch_offset < batch_size; batch_offset++) {
//       int bf_idx = start_idx + batch_offset;
      
//       // Get thread-specific stream
//       shared_resources::thread_id = batch_offset;
//       shared_resources::n_threads = batch_size;
//       auto thread_resources = dev_resources;
//       cudaStream_t thread_stream = raft::resource::get_cuda_stream(thread_resources);
      
    
//       // CPU preprocessing: Find matching indices based on labels
//       std::vector<int> matching_indices;
//       auto query_labels = query_label_vecs[query_offset+bf_query_map[bf_idx]];
//       matching_indices.reserve(cat_freq[query_labels[0]]);
//       int write_id = 0;
      
//       if (query_labels.size() == 1) {
//         matching_indices = labels_data[query_labels[0]];
//         write_id = labels_data[query_labels[0]].size();
//       } else {
//         const auto& first_label_data = labels_data[query_labels[0]];
//         const auto& second_label_data = labels_data[query_labels[1]];
//         size_t i = 0, j = 0;
//         while (i < first_label_data.size() && j < second_label_data.size()) {
//           if (first_label_data[i] < second_label_data[j]) {
//             i++;
//           } else if (second_label_data[j] < first_label_data[i]) {
//             j++;
//           } else {
//             matching_indices.push_back(first_label_data[i]);
//             write_id++;
//             i++;
//             j++;
//           }
//         }
//       }

//       // GPU processing using thread stream
//       auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, write_id, dim);
//       auto d_matching_indices = raft::make_device_vector<int>(thread_resources, write_id);
//       raft::copy(d_matching_indices.data_handle(), matching_indices.data(), write_id, thread_stream);
      
//       copy_dataset<<<32,32,0,thread_stream>>>(dataset, 
//                                              d_filtered_dataset.view(), 
//                                              d_matching_indices.view());

//       auto bfs_index = cuvs::neighbors::brute_force::build(
//         thread_resources,
//         raft::make_const_mdspan(d_filtered_dataset.view())
//       );
      
//       auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
//       auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);

//       auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
//       raft::copy(bf_single_query.data_handle(), 
//                  bf_queries.data_handle() + bf_idx*dim, 
//                  dim, 
//                  thread_stream);
      
//       cuvs::neighbors::brute_force::search(
//         thread_resources,
//         bfs_index,
//         raft::make_const_mdspan(bf_single_query.view()),
//         d_indices.view(),
//         d_distances.view()
//       );

//       copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//         d_indices.view(), 
//         all_neighbors, 
//         d_matching_indices.view(), 
//         query_map.view(), 
//         query_offset, 
//         bf_idx
//       );

//       // Synchronize thread stream after processing
//       // raft::resource::sync_stream(dev_resources);
//     }
//   }
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk) {

//   int dim = dataset.extent(1);
//   // printf("Debug - Input Parameters:");
//   // printf(" BF Search Size: %d", bf_search_size);
//   // printf(" Query Offset: %d\n", query_offset);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
//   copy_bf_query<<<32,32,0,stream>>>(queries, bf_queries.view(), query_map.view(), query_offset);
  

//   int n_threads = (int)std::thread::hardware_concurrency();
//   omp_set_num_threads(n_threads);
//   // printf("Maximum threads available: %d\n", n_threads);
//   #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
//   for (int bf_idx = 0; bf_idx < bf_search_size; bf_idx++) {
//     int thread_id = omp_get_thread_num();
//     shared_resources::thread_id = thread_id;
//     shared_resources::n_threads = n_threads;
//     auto thread_resources = dev_resources;
//     cudaStream_t thread_stream = thread_resources.get_sync_stream();
//     auto query_labels = query_label_vecs[query_offset+bf_query_map[bf_idx]];
//     std::vector<int> matching_indices;
//     matching_indices.reserve(cat_freq[query_labels[0]]);
//     int write_id = 0;
//     if (query_labels.size() == 1) {
//       matching_indices = labels_data[query_labels[0]];
//       write_id = labels_data[query_labels[0]].size();
//     } else {
//       const auto& first_label_data = labels_data[query_labels[0]];
//       const auto& second_label_data = labels_data[query_labels[1]];
//       size_t i = 0, j = 0;
//       while (i < first_label_data.size() && j < second_label_data.size()) {
//         if (first_label_data[i] < second_label_data[j]) {
//           i++;
//         } else if (second_label_data[j] < first_label_data[i]) {
//           j++;
//         } else {
//           matching_indices.push_back(first_label_data[i]);
//           write_id++;
//           i++;
//           j++;
//         }
//       }
//     }
//     auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, write_id, dim);
//     auto d_matching_indices = raft::make_device_vector<int>(thread_resources, write_id);
//     raft::copy(d_matching_indices.data_handle(), matching_indices.data(), write_id, thread_stream);
//     copy_dataset<<<32,32,0,thread_stream>>>(dataset, 
//                                             d_filtered_dataset.view(), 
//                                             d_matching_indices.view());
//     auto bfs_index = cuvs::neighbors::brute_force::build(
//       thread_resources,
//       raft::make_const_mdspan(d_filtered_dataset.view())
//     );
//     auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
//     auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);
//     auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
//     raft::copy(bf_single_query.data_handle(), 
//                 bf_queries.data_handle() + bf_idx*dim, 
//                 dim, 
//                 thread_stream);
//     cuvs::neighbors::brute_force::search(
//       thread_resources,
//       bfs_index,
//       raft::make_const_mdspan(bf_single_query.view()),
//       d_indices.view(),
//       d_distances.view()
//     );
//     copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//       d_indices.view(), 
//       all_neighbors, 
//       d_matching_indices.view(), 
//       query_map.view(), 
//       query_offset, 
//       bf_idx
//     );
//     // raft::resource::sync_stream(thread_resources,thread_stream);
//     cudaDeviceSynchronize();
//   }
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk) {

//   int dim = dataset.extent(1);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

//   // auto total_start = std::chrono::high_resolution_clock::now();
  
//   // Initial setup timing
//   // auto setup_start = std::chrono::high_resolution_clock::now();
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
//   const int block_size = 256;
//   const int total_elements = bf_search_size * dim;
//   const int num_blocks = min((total_elements + block_size - 1) / block_size, 65535);
//   copy_bf_query<<<num_blocks,block_size,0,stream>>>(queries, bf_queries.view(), query_map.view(), query_offset);
//   cudaStreamSynchronize(stream);
//   // auto setup_end = std::chrono::high_resolution_clock::now();
//   // printf("Initial setup time: %.3f ms\n", 
//   //        std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start).count()/1000.0);

//   int n_threads = (int)std::thread::hardware_concurrency();
//   omp_set_num_threads(n_threads);

//   // Per-thread timing accumulators
//   // std::vector<double> pre_filtering_times(n_threads, 0.0);
//   // std::vector<double> data_transfer_times(n_threads, 0.0);
//   // std::vector<double> index_build_times(n_threads, 0.0);
//   // std::vector<double> search_times(n_threads, 0.0);
//   // std::vector<double> post_process_times(n_threads, 0.0);
//   // std::vector<size_t> points_processed(n_threads, 0);

//   // #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
//   #pragma omp parallel for schedule(guided) num_threads(n_threads)
//   for (int bf_idx = 0; bf_idx < bf_search_size; bf_idx++) {
//     int thread_id = omp_get_thread_num();
//     shared_resources::thread_id = thread_id;
//     shared_resources::n_threads = n_threads;
//     auto thread_resources = dev_resources;
//     cudaStream_t thread_stream = thread_resources.get_sync_stream();

//     // Pre-filtering timing
//     // auto pre_filter_start = std::chrono::high_resolution_clock::now();
//     auto query_labels = query_label_vecs[query_offset+bf_query_map[bf_idx]];
//     std::vector<int> matching_indices;
//     matching_indices.reserve(cat_freq[query_labels[0]]);
//     int write_id = 0;
//     if (query_labels.size() == 1) {
//       matching_indices = labels_data[query_labels[0]];
//       write_id = labels_data[query_labels[0]].size();
//     } else {
//       const auto& first_label_data = labels_data[query_labels[0]];
//       const auto& second_label_data = labels_data[query_labels[1]];
//       size_t i = 0, j = 0;
//       while (i < first_label_data.size() && j < second_label_data.size()) {
//         if (first_label_data[i] < second_label_data[j]) {
//           i++;
//         } else if (second_label_data[j] < first_label_data[i]) {
//           j++;
//         } else {
//           matching_indices.push_back(first_label_data[i]);
//           write_id++;
//           i++;
//           j++;
//         }
//       }
//     }
//     // auto pre_filter_end = std::chrono::high_resolution_clock::now();
//     // pre_filtering_times[thread_id] += std::chrono::duration_cast<std::chrono::microseconds>(
//     //   pre_filter_end - pre_filter_start).count()/1000.0;
//     // points_processed[thread_id] += write_id;

//     // Data transfer timing
//     // auto transfer_start = std::chrono::high_resolution_clock::now();
//     auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, write_id, dim);
//     auto d_matching_indices = raft::make_device_vector<int>(thread_resources, write_id);
//     raft::copy(d_matching_indices.data_handle(), matching_indices.data(), write_id, thread_stream);
//     // copy_dataset<<<32,32,0,thread_stream>>>(dataset, d_filtered_dataset.view(), d_matching_indices.view());
//     const int dataset_block_size = 256;  // Full warp multiples for best performance
//     const int dataset_grid_size = min((write_id * dim + dataset_block_size - 1) / dataset_block_size, 65535);
//     copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
//         dataset, d_filtered_dataset.view(), d_matching_indices.view());
//     // auto transfer_end = std::chrono::high_resolution_clock::now();
//     // data_transfer_times[thread_id] += std::chrono::duration_cast<std::chrono::microseconds>(
//     //   transfer_end - transfer_start).count()/1000.0;

//     // Index building timing
//     // auto index_start = std::chrono::high_resolution_clock::now();
//     auto bfs_index = cuvs::neighbors::brute_force::build(
//       thread_resources,
//       raft::make_const_mdspan(d_filtered_dataset.view())
//     );
//     // auto index_end = std::chrono::high_resolution_clock::now();
//     // index_build_times[thread_id] += std::chrono::duration_cast<std::chrono::microseconds>(
//     //   index_end - index_start).count()/1000.0;

//     // Search timing
//     // auto search_start = std::chrono::high_resolution_clock::now();
//     auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
//     auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);
//     auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
//     raft::copy(bf_single_query.data_handle(), bf_queries.data_handle() + bf_idx*dim, dim, thread_stream);
//     cuvs::neighbors::brute_force::search(
//       thread_resources,
//       bfs_index,
//       raft::make_const_mdspan(bf_single_query.view()),
//       d_indices.view(),
//       d_distances.view()
//     );
//     // auto search_end = std::chrono::high_resolution_clock::now();
//     // search_times[thread_id] += std::chrono::duration_cast<std::chrono::microseconds>(
//     //   search_end - search_start).count()/1000.0;

//     // Post-processing timing
//     // auto post_start = std::chrono::high_resolution_clock::now();
//     copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//       d_indices.view(), 
//       all_neighbors, 
//       d_matching_indices.view(), 
//       query_map.view(), 
//       query_offset, 
//       bf_idx
//     );
//     cudaStreamSynchronize(thread_stream);
//     // auto post_end = std::chrono::high_resolution_clock::now();
//     // post_process_times[thread_id] += std::chrono::duration_cast<std::chrono::microseconds>(
//     //   post_end - post_start).count()/1000.0;
//   }

//   // auto total_end = std::chrono::high_resolution_clock::now();
//   // double total_time = std::chrono::duration_cast<std::chrono::microseconds>(
//   //   total_end - total_start).count()/1000.0;

//   // // Print timing breakdown
//   // printf("\nPer-thread timing breakdown (ms):\n");
//   // printf("Thread | Points | Pre-filter | Transfer | Index | Search | Post-proc\n");
//   // printf("-------|---------|------------|-----------|--------|---------|----------\n");
//   // for (int i = 0; i < n_threads; i++) {
//   //   printf("%6d | %7zu | %10.2f | %9.2f | %6.2f | %7.2f | %9.2f\n",
//   //          i, points_processed[i], pre_filtering_times[i], data_transfer_times[i],
//   //          index_build_times[i], search_times[i], post_process_times[i]);
//   // }

//   // // Calculate totals
//   // double total_pre_filter = std::accumulate(pre_filtering_times.begin(), pre_filtering_times.end(), 0.0);
//   // double total_transfer = std::accumulate(data_transfer_times.begin(), data_transfer_times.end(), 0.0);
//   // double total_index = std::accumulate(index_build_times.begin(), index_build_times.end(), 0.0);
//   // double total_search = std::accumulate(search_times.begin(), search_times.end(), 0.0);
//   // double total_post = std::accumulate(post_process_times.begin(), post_process_times.end(), 0.0);
//   // size_t total_points = std::accumulate(points_processed.begin(), points_processed.end(), 0ULL);

//   // printf("\nOverall Statistics:\n");
//   // printf("Total points processed: %zu\n", total_points);
//   // printf("Average points per query: %.1f\n", (float)total_points/bf_search_size);
//   // printf("\nTotal timing breakdown:\n");
//   // printf("Pre-filtering:    %.2f ms (%.1f%%)\n", total_pre_filter, total_pre_filter/total_time*100);
//   // printf("Data transfer:    %.2f ms (%.1f%%)\n", total_transfer, total_transfer/total_time*100);
//   // printf("Index building:   %.2f ms (%.1f%%)\n", total_index, total_index/total_time*100);
//   // printf("Search:           %.2f ms (%.1f%%)\n", total_search, total_search/total_time*100);
//   // printf("Post-processing:  %.2f ms (%.1f%%)\n", total_post, total_post/total_time*100);
//   // printf("Total time:       %.2f ms\n", total_time);
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk) {
  
//   int dim = dataset.extent(1);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
  
//   const int query_block_size = 256;
//   const int query_total_elements = bf_search_size * dim;
//   const int query_grid_size = std::min<int>((query_total_elements + query_block_size - 1) / query_block_size, 65535);
//   copy_bf_query<<<query_grid_size, query_block_size, 0, stream>>>(
//     queries, bf_queries.view(), query_map.view(), query_offset);
  
//   int optimal_threads;
//   char* env_threads = getenv("OMP_NUM_THREADS");
//   optimal_threads = std::atoi(env_threads);
  
//   const int batch_size = 32;
//   const int num_batches = (bf_search_size + batch_size - 1) / batch_size;

//   #pragma omp parallel for schedule(guided) num_threads(optimal_threads)
//   for (int batch = 0; batch < num_batches; batch++) {
//     int thread_id = omp_get_thread_num();
//     shared_resources::thread_id = thread_id;
//     shared_resources::n_threads = optimal_threads;
//     auto thread_resources = dev_resources;
//     cudaStream_t thread_stream = thread_resources.get_sync_stream();
    
//     const int batch_start = batch * batch_size;
//     const int batch_end = std::min(batch_start + batch_size, bf_search_size);
//     const int current_batch_size = batch_end - batch_start;
    
//     // Pre-allocate vectors for batch
//     std::vector<std::vector<int>> batch_matching_indices(current_batch_size);
//     std::vector<int> batch_write_ids(current_batch_size);
//     std::atomic<int> preprocess_idx{0};
//     std::atomic<bool> preprocessing_done{false};
    
//     #pragma omp parallel sections num_threads(2)
//     {
//       #pragma omp section
//       {
//         // Preprocessing thread
//         for (int i = 0; i < current_batch_size; i++) {
//           int bf_idx = batch_start + i;
//           auto query_labels = query_label_vecs[query_offset + bf_query_map[bf_idx]];
          
//           if (query_labels.size() == 1) {
//             batch_matching_indices[i] = labels_data[query_labels[0]];
//             batch_write_ids[i] = labels_data[query_labels[0]].size();
//           } else {
//             const auto& first_label_data = labels_data[query_labels[0]];
//             const auto& second_label_data = labels_data[query_labels[1]];
//             batch_matching_indices[i].reserve(cat_freq[query_labels[0]]);
//             size_t idx1 = 0, idx2 = 0;
//             while (idx1 < first_label_data.size() && idx2 < second_label_data.size()) {
//               if (first_label_data[idx1] < second_label_data[idx2]) idx1++;
//               else if (second_label_data[idx2] < first_label_data[idx1]) idx2++;
//               else {
//                 batch_matching_indices[i].push_back(first_label_data[idx1]);
//                 idx1++; idx2++;
//               }
//             }
//             batch_write_ids[i] = batch_matching_indices[i].size();
//           }
//           preprocess_idx.store(i + 1, std::memory_order_release);
//         }
//         preprocessing_done.store(true, std::memory_order_release);
//       }
      
//       #pragma omp section
//       {
//         // GPU processing thread
//         int gpu_index = 0;
//         while (!preprocessing_done.load(std::memory_order_acquire) || 
//                gpu_index < preprocess_idx.load(std::memory_order_acquire)) {
          
//           if (gpu_index < preprocess_idx.load(std::memory_order_acquire)) {
//             int write_id = batch_write_ids[gpu_index];
//             if (write_id > 0) {
//               int bf_idx = batch_start + gpu_index;
              
//               auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(
//                 thread_resources, write_id, dim);
//               auto d_matching_indices = raft::make_device_vector<int>(
//                 thread_resources, write_id);
                
//               raft::copy(d_matching_indices.data_handle(), 
//                 batch_matching_indices[gpu_index].data(), 
//                 write_id, 
//                 thread_stream);
              
//               const int dataset_block_size = 256;
//               const int dataset_grid_size = std::min<int>(
//                 (write_id * dim + dataset_block_size - 1) / dataset_block_size, 65535);
//               copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
//                 dataset, d_filtered_dataset.view(), d_matching_indices.view());
              
//               auto bfs_index = cuvs::neighbors::brute_force::build(
//                 thread_resources,
//                 raft::make_const_mdspan(d_filtered_dataset.view())
//               );
              
//               auto d_distances = raft::make_device_matrix<float, int64_t>(
//                 thread_resources, 1, topk);
//               auto d_indices = raft::make_device_matrix<int64_t, int64_t>(
//                 thread_resources, 1, topk);
//               auto bf_single_query = raft::make_device_matrix<float, int64_t>(
//                 thread_resources, 1, dim);
              
//               raft::copy(bf_single_query.data_handle(), 
//                 bf_queries.data_handle() + bf_idx*dim, 
//                 dim, 
//                 thread_stream);
              
//               cuvs::neighbors::brute_force::search(
//                 thread_resources,
//                 bfs_index,
//                 raft::make_const_mdspan(bf_single_query.view()),
//                 d_indices.view(),
//                 d_distances.view()
//               );
              
//               copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//                 d_indices.view(),
//                 all_neighbors,
//                 d_matching_indices.view(),
//                 query_map.view(),
//                 query_offset,
//                 bf_idx
//               );
//             }
//             gpu_index++;
//           }
//         }
//       }
//     }
//     cudaStreamSynchronize(thread_stream);
//   }
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//              raft::device_matrix_view<const float, int64_t> dataset, 
//              raft::device_matrix_view<const float, int64_t> queries,
//              const std::vector<std::vector<int>>& query_label_vecs,
//              const std::vector<std::vector<int>>& data_labels,
//              const std::vector<std::vector<int>>& labels_data,
//              const std::vector<int>& cat_freq,
//              int query_offset,
//              int bf_search_size,
//              const std::vector<int>& bf_query_map,
//              raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//              int topk,
//              cudaStream_t cagra_stream) {

//   // Prefiltering
//   auto preprocess_start = std::chrono::high_resolution_clock::now();
//   std::vector<std::vector<int>> all_matching_indices(bf_search_size);
//   std::vector<int> all_write_ids(bf_search_size);
//   int optimal_threads;
//   char* env_threads = getenv("OMP_NUM_THREADS");
//   optimal_threads = std::atoi(env_threads);
//   const int prefilter_batch_size = 32; // Adjust based on hardware capabilities
//   const int num_prefilter_batches = (bf_search_size + prefilter_batch_size - 1) / prefilter_batch_size;
//   #pragma omp parallel for schedule(guided) num_threads(optimal_threads)
//   for(int batch = 0; batch < num_prefilter_batches; batch++) {
//     const int start_idx = batch * prefilter_batch_size;
//     const int end_idx = std::min(start_idx + prefilter_batch_size, bf_search_size);
//     for(int i = start_idx; i < end_idx; i++) {
//       auto query_labels = query_label_vecs[query_offset + bf_query_map[i]];
//       if (query_labels.size() == 1) {
//         all_matching_indices[i] = labels_data[query_labels[0]];
//         all_write_ids[i] = labels_data[query_labels[0]].size();
//       } else {
//         const auto& first_label_data = labels_data[query_labels[0]];
//         const auto& second_label_data = labels_data[query_labels[1]];
//         all_matching_indices[i].reserve(cat_freq[query_labels[0]]);
//         size_t idx1 = 0, idx2 = 0;
//         while (idx1 < first_label_data.size() && idx2 < second_label_data.size()) {
//           if (first_label_data[idx1] < second_label_data[idx2]) idx1++;
//           else if (second_label_data[idx2] < first_label_data[idx1]) idx2++;
//           else {
//             all_matching_indices[i].push_back(first_label_data[idx1]);
//             idx1++; idx2++;
//           }
//         }
//         all_write_ids[i] = all_matching_indices[i].size();
//       }
//     }
//   }
//   auto preprocess_end = std::chrono::high_resolution_clock::now();
//   double pure_preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
//       preprocess_end - preprocess_start).count()/1000000.0;
//   cudaStreamSynchronize(cagra_stream);
//   std::cout << "Pure preprocessing time: " << pure_preprocess_time * 1000 << " ms" << std::endl;

//   int dim = dataset.extent(1);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
//   const int query_block_size = 256;
//   const int total_elements = bf_search_size * dim;
//   const int query_grid_size = std::min<int>((total_elements + query_block_size - 1) / query_block_size, 65535);
//   copy_bf_query<<<query_grid_size, query_block_size, 0, stream>>>(
//       queries, bf_queries.view(), query_map.view(), query_offset);

//   // Process GPU operations in batches
//   const int batch_size = 32; // Adjust based on hardware capabilities
//   const int num_batches = (bf_search_size + batch_size - 1) / batch_size;
//   auto bf_start = std::chrono::high_resolution_clock::now();
//   #pragma omp parallel for schedule(guided) num_threads(optimal_threads)
//   for(int batch = 0; batch < num_batches; batch++) {
//     int thread_id = omp_get_thread_num();
//     shared_resources::thread_id = thread_id;
//     shared_resources::n_threads = optimal_threads;
//     auto thread_resources = dev_resources;
//     cudaStream_t thread_stream = thread_resources.get_sync_stream();

//     const int batch_start = batch * batch_size;
//     const int batch_end = std::min(batch_start + batch_size, bf_search_size);

//     for(int i = batch_start; i < batch_end; i++) {
//       if (all_write_ids[i] == 0) continue;

//       auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, all_write_ids[i], dim);
//       auto d_matching_indices = raft::make_device_vector<int>(thread_resources, all_write_ids[i]);
//       raft::copy(d_matching_indices.data_handle(), all_matching_indices[i].data(), all_write_ids[i], thread_stream);
//       const int dataset_block_size = 256;
//       const int dataset_grid_size = std::min<int>((all_write_ids[i] * dim + dataset_block_size - 1) / dataset_block_size, 65535);
//       copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
//           dataset, d_filtered_dataset.view(), d_matching_indices.view());

//       auto bfs_index = cuvs::neighbors::brute_force::build(
//           thread_resources,
//           raft::make_const_mdspan(d_filtered_dataset.view())
//       );

//       auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
//       auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);
//       auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
//       raft::copy(bf_single_query.data_handle(), bf_queries.data_handle() + i*dim, dim, thread_stream);
//       cuvs::neighbors::brute_force::search(
//           thread_resources,
//           bfs_index,
//           raft::make_const_mdspan(bf_single_query.view()),
//           d_indices.view(),
//           d_distances.view()
//       );

//       copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//           d_indices.view(),
//           all_neighbors,
//           d_matching_indices.view(),
//           query_map.view(),
//           query_offset,
//           i
//       );
//     }
//     cudaStreamSynchronize(thread_stream);
//   }
//   auto bf_end = std::chrono::high_resolution_clock::now();
//   double bf_time = std::chrono::duration_cast<std::chrono::microseconds>(
//       bf_end - bf_start).count()/1000000.0;
//   std::cout << "BF time:         " << bf_time * 1000 << " ms" << std::endl;
// }

// void bf_search(shared_resources::configured_raft_resources& dev_resources,
//               raft::device_matrix_view<const float, int64_t> dataset,
//               raft::device_matrix_view<const float, int64_t> queries,
//               const std::vector<std::vector<int>>& query_label_vecs,
//               const std::vector<std::vector<int>>& data_labels,
//               const std::vector<std::vector<int>>& labels_data,
//               const std::vector<int>& cat_freq,
//               int query_offset,
//               int bf_search_size,
//               const std::vector<int>& bf_query_map,
//               raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
//               int topk,
//               cudaStream_t cagra_stream) {
              
//   int dim = dataset.extent(1);
//   cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  
//   auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
//   auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
//   raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
  
//   const int query_block_size = 256;
//   const int query_total_elements = bf_search_size * dim;
//   const int query_grid_size = std::min<int>((query_total_elements + query_block_size - 1) / query_block_size, 65535);
//   copy_bf_query<<<query_grid_size, query_block_size, 0, stream>>>(
//       queries, bf_queries.view(), query_map.view(), query_offset);
  
//   int optimal_threads;
//   char* env_threads = getenv("OMP_NUM_THREADS");
//   optimal_threads = std::atoi(env_threads);
  
//   const int batch_size = std::max(1, bf_search_size / optimal_threads);
//   const int num_batches = (bf_search_size + batch_size - 1) / batch_size;
  
//   int active_threads = std::min(optimal_threads, num_batches);
//   omp_set_num_threads(active_threads);

//   // Preprocessing phase - can be done in parallel before CAGRA sync
//   std::vector<std::vector<std::vector<int>>> all_batch_indices(num_batches);
//   std::vector<std::vector<int>> all_batch_write_ids(num_batches);
  
//   #pragma omp parallel for
//   for (int batch = 0; batch < num_batches; batch++) {
//     const int batch_start = batch * batch_size;
//     const int batch_end = std::min(batch_start + batch_size, bf_search_size);
//     const int current_batch_size = batch_end - batch_start;
//     all_batch_indices[batch].resize(current_batch_size);
//     all_batch_write_ids[batch].resize(current_batch_size);
//     for (int i = 0; i < current_batch_size; i++) {
//       int bf_idx = batch_start + i;
//       auto query_labels = query_label_vecs[query_offset + bf_query_map[bf_idx]];
//       if (query_labels.size() == 1) {
//         all_batch_indices[batch][i] = labels_data[query_labels[0]];
//         all_batch_write_ids[batch][i] = labels_data[query_labels[0]].size();
//       } else {
//         const auto& first_label_data = labels_data[query_labels[0]];
//         const auto& second_label_data = labels_data[query_labels[1]];
//         all_batch_indices[batch][i].reserve(cat_freq[query_labels[0]]);
//         size_t idx1 = 0, idx2 = 0;
//         while (idx1 < first_label_data.size() && idx2 < second_label_data.size()) {
//           if (first_label_data[idx1] < second_label_data[idx2]) idx1++;
//           else if (second_label_data[idx2] < first_label_data[idx1]) idx2++;
//           else {
//             all_batch_indices[batch][i].push_back(first_label_data[idx1]);
//             idx1++; idx2++;
//           }
//         }
//         all_batch_write_ids[batch][i] = all_batch_indices[batch][i].size();
//       }
//     }
//   }

//   cudaStreamSynchronize(cagra_stream);
  
//   #pragma omp parallel for
//   for (int batch = 0; batch < num_batches; batch++) {
//     int thread_id = omp_get_thread_num();
//     shared_resources::thread_id = thread_id;
//     shared_resources::n_threads = active_threads;
//     auto thread_resources = dev_resources;
//     cudaStream_t thread_stream = thread_resources.get_sync_stream();
    
//     const int batch_start = batch * batch_size;
//     const int batch_end = std::min(batch_start + batch_size, bf_search_size);
//     const int current_batch_size = batch_end - batch_start;
    
//     for (int i = 0; i < current_batch_size; i++) {
//       int bf_idx = batch_start + i;
//       int write_id = all_batch_write_ids[batch][i];
      
//       if (write_id > 0) {
//         auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, write_id, dim);
//         auto d_matching_indices = raft::make_device_vector<int>(thread_resources, write_id);
//         raft::copy(d_matching_indices.data_handle(), all_batch_indices[batch][i].data(), write_id, thread_stream);
        
//         const int dataset_block_size = 256;
//         const int dataset_grid_size = std::min<int>((write_id * dim + dataset_block_size - 1) / dataset_block_size, 65535);
//         copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
//             dataset, d_filtered_dataset.view(), d_matching_indices.view());
            
//         auto bfs_index = cuvs::neighbors::brute_force::build(
//             thread_resources,
//             raft::make_const_mdspan(d_filtered_dataset.view())
//         );
        
//         auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
//         auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);
//         auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
//         raft::copy(bf_single_query.data_handle(), bf_queries.data_handle() + bf_idx*dim, dim, thread_stream);
        
//         cuvs::neighbors::brute_force::search(
//             thread_resources,
//             bfs_index,
//             raft::make_const_mdspan(bf_single_query.view()),
//             d_indices.view(),
//             d_distances.view()
//         );
        
//         copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
//             d_indices.view(),
//             all_neighbors,
//             d_matching_indices.view(),
//             query_map.view(),
//             query_offset,
//             bf_idx
//         );
//       }
//     }
//     cudaStreamSynchronize(thread_stream);
//   }
// }



void bf_search(shared_resources::configured_raft_resources& dev_resources,
              raft::device_matrix_view<const float, int64_t> dataset,
              raft::device_matrix_view<const float, int64_t> queries,
              const std::vector<std::vector<int>>& query_label_vecs,
              const std::vector<std::vector<int>>& data_labels,
              const std::vector<std::vector<int>>& labels_data,
              const std::vector<int>& cat_freq,
              int query_offset,
              int bf_search_size,
              const std::vector<int>& bf_query_map,
              raft::device_matrix_view<uint32_t, int64_t> all_neighbors,
              int topk,
              cudaStream_t cagra_stream) {

  int dim = dataset.extent(1);
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  
  // Initial query setup
  auto bf_queries = raft::make_device_matrix<float, int64_t>(dev_resources, bf_search_size, dim);
  auto query_map = raft::make_device_vector<int>(dev_resources, bf_search_size);
  raft::copy(query_map.data_handle(), bf_query_map.data(), bf_search_size, stream);
  
  // Improved query copy with better grid dimensions
  const int query_block_size = 256;
  const int query_total_elements = bf_search_size * dim;
  const int query_grid_size = std::min<int>((query_total_elements + query_block_size - 1) / query_block_size, 65535);
  copy_bf_query<<<query_grid_size, query_block_size, 0, stream>>>(
      queries, bf_queries.view(), query_map.view(), query_offset);
  
  // Configure OpenMP threads
  int optimal_threads;
  char* env_threads = getenv("OMP_NUM_THREADS");
  optimal_threads = std::atoi(env_threads);
  omp_set_num_threads(optimal_threads);

  // Process queries in batches
  const int num_batches = optimal_threads;  // Number of threads
  const int base_batch_size = bf_search_size / num_batches;  // Minimum number of queries per thread
  const int remainder = bf_search_size % num_batches;  // Remaining queries to distribute

  std::vector<std::vector<double>> preprocess_times(optimal_threads);
  std::vector<std::vector<double>> gpu_process_times(optimal_threads);

  auto total_start = std::chrono::high_resolution_clock::now();
  // First level parallel processing for batches
  #pragma omp parallel for num_threads(optimal_threads)
  for (int batch = 0; batch < num_batches; batch++) {
    int thread_id = omp_get_thread_num();
    shared_resources::thread_id = thread_id;
    shared_resources::n_threads = optimal_threads;
    auto thread_resources = dev_resources;
    cudaStream_t thread_stream = thread_resources.get_sync_stream();
    
    // Calculate batch boundaries
    const int batch_start = batch * base_batch_size + std::min(batch, remainder);
    const int batch_end = batch_start + base_batch_size + (batch < remainder ? 1 : 0);
    const int current_batch_size = batch_end - batch_start;
    
    // Pre-process all queries in batch
    std::vector<std::vector<int>> batch_matching_indices;  // Don't pre-size the outer vector
    batch_matching_indices.resize(current_batch_size);  
    std::vector<int> batch_write_ids(current_batch_size);
    
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    // Second level parallel processing for preprocessing
    for (int i = 0; i < current_batch_size; i++) {
      int bf_idx = batch_start + i;
      auto query_labels = query_label_vecs[query_offset + bf_query_map[bf_idx]];
      if (query_labels.size() == 1) {
        batch_matching_indices[i] = labels_data[query_labels[0]];
        batch_write_ids[i] = labels_data[query_labels[0]].size();
      } else {
        int write_id = 0;
        const auto& first_label_data = labels_data[query_labels[0]];
        const auto& second_label_data = labels_data[query_labels[1]];
        int min_freq = std::min(cat_freq[query_labels[0]], cat_freq[query_labels[1]]);
        batch_matching_indices[i].resize(min_freq);
        size_t idx1 = 0, idx2 = 0;
        while (idx1 < first_label_data.size() && idx2 < second_label_data.size()) {
          if (first_label_data[idx1] < second_label_data[idx2]) idx1++;
          else if (second_label_data[idx2] < first_label_data[idx1]) idx2++;
          else {
            batch_matching_indices[i][write_id] = first_label_data[idx1];
            idx1++; idx2++; write_id++;
          }
        }
        batch_matching_indices[i].resize(write_id);
        batch_write_ids[i] = write_id;
      }
    }
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
        preprocess_end - preprocess_start).count()/1000.0;
    preprocess_times[thread_id].push_back(preprocess_time);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    // Process batch on GPU - sequential within each thread's batch
    for (int i = 0; i < current_batch_size; i++) {
      int bf_idx = batch_start + i;
      int write_id = batch_write_ids[i];
      if (write_id > 0) {
        auto d_filtered_dataset = raft::make_device_matrix<float, int64_t>(thread_resources, write_id, dim);
        auto d_matching_indices = raft::make_device_vector<int>(thread_resources, write_id);
        raft::copy(d_matching_indices.data_handle(), batch_matching_indices[i].data(), write_id, thread_stream);
        const int dataset_block_size = 256;
        const int dataset_grid_size = std::min<int>((write_id * dim + dataset_block_size - 1) / dataset_block_size, 65535);
        copy_dataset<<<dataset_grid_size, dataset_block_size, 0, thread_stream>>>(
            dataset, d_filtered_dataset.view(), d_matching_indices.view());
        auto bfs_index = cuvs::neighbors::brute_force::build(
            thread_resources,
            raft::make_const_mdspan(d_filtered_dataset.view())
        );
        auto d_distances = raft::make_device_matrix<float, int64_t>(thread_resources, 1, topk);
        auto d_indices = raft::make_device_matrix<int64_t, int64_t>(thread_resources, 1, topk);
        auto bf_single_query = raft::make_device_matrix<float, int64_t>(thread_resources, 1, dim);
        raft::copy(bf_single_query.data_handle(), bf_queries.data_handle() + bf_idx*dim, dim, thread_stream);
        cuvs::neighbors::brute_force::search(
            thread_resources,
            bfs_index,
            raft::make_const_mdspan(bf_single_query.view()),
            d_indices.view(),
            d_distances.view()
        );
        copy_bf_neighbors<<<1, topk, 0, thread_stream>>>(
            d_indices.view(),
            all_neighbors,
            d_matching_indices.view(),
            query_map.view(),
            query_offset,
            bf_idx
        );
      }
    }
    cudaStreamSynchronize(thread_stream);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
        gpu_end - gpu_start).count()/1000.0;
    gpu_process_times[thread_id].push_back(gpu_time);
  }
  auto total_end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      total_end - total_start).count()/1000.0;

  // Print timing statistics
  std::cout << "\nBF Search Size: " << bf_search_size << std::endl;
  std::cout << "Total Time (ms): " << total_time << std::endl;
  std::cout << "Timing breakdown per thread:" << std::endl;
  std::cout << "Thread | Avg Preprocess (ms) | Avg GPU (ms) | # Batches" << std::endl;
  std::cout << "-------|-------------------|-------------|----------" << std::endl;
  double total_preprocess_time = 0.0;
  double total_gpu_time = 0.0;
  int total_batches = 0;
  for (int t = 0; t < optimal_threads; t++) {
    if (preprocess_times[t].empty()) continue;
    double avg_preprocess = std::accumulate(preprocess_times[t].begin(), 
                                          preprocess_times[t].end(), 0.0) / preprocess_times[t].size();
    double avg_gpu = std::accumulate(gpu_process_times[t].begin(), 
                                   gpu_process_times[t].end(), 0.0) / gpu_process_times[t].size();
    total_preprocess_time += avg_preprocess;
    total_gpu_time += avg_gpu;
    total_batches += preprocess_times[t].size();
    std::cout << std::fixed << std::setprecision(2)
              << std::setw(6) << t << " | " 
              << std::setw(17) << avg_preprocess << " | "
              << std::setw(11) << avg_gpu << " | "
              << std::setw(9) << preprocess_times[t].size() << std::endl;
  }
  int active_threads = 0;
  for (int t = 0; t < optimal_threads; t++) {
    if (!preprocess_times[t].empty()) active_threads++;
  }
  if (active_threads > 0) {
  double avg_preprocess_per_thread = total_preprocess_time / active_threads;
  double avg_gpu_per_thread = total_gpu_time / active_threads;
  std::cout << "-------|-------------------|-------------|----------" << std::endl;
  std::cout << std::fixed << std::setprecision(2)
            << " AVG   | " 
            << std::setw(17) << avg_preprocess_per_thread << " | "
            << std::setw(11) << avg_gpu_per_thread << " | "
            << std::setw(9) << total_batches/active_threads << std::endl;
  }
}


template<typename T, typename IdxT = uint32_t>
void cagra_build_search_filtered(shared_resources::configured_raft_resources& dev_resources,
                                 raft::device_matrix_view<const T, int64_t> dataset,
                                 raft::device_matrix_view<const T, int64_t> query,
                                 const std::vector<std::vector<int>>& data_label_vecs,
                                 const std::vector<std::vector<int>>& label_data_vecs,
                                 const std::vector<std::vector<int>>& query_label_vecs,
                                 const std::vector<int>& cat_freq,
                                 const std::vector<std::vector<uint32_t>>& gt_indices,
                                 int batch_size,
                                 int iterations,
                                 int itopk,
                                 int max_iterations,
                                 int specificity_threshold,
                                 int graph_degree)
{
  using namespace raft::neighbors;
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  int total_queries = batch_size * iterations;
  
  std::cout << "Loading CAGRA index" << std::endl;
  std::string filename = "/u/cmo1/Filtered/cuvs-24.10/examples/cpp/src/index/raft_yfcc_" + 
                          std::to_string(graph_degree) + ".ibin";
  auto index = cagra::deserialize<T, IdxT>(dev_resources, filename);
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // Prepare brute-force dataset
  auto float_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, dataset.extent(0), dataset.extent(1));
  copy_int8_to_float<<<32,32>>>(dataset, float_dataset.view());
  auto h_queries = raft::make_host_matrix<T, int64_t>(query.extent(0), query.extent(1));
  raft::copy(h_queries.data_handle(), query.data_handle(), query.size(), stream);
  auto float_queries = raft::make_host_matrix<float, int64_t>(query.extent(0), query.extent(1));
  auto d_float_queries = raft::make_device_matrix<float, int64_t>(dev_resources, query.extent(0), query.extent(1));
  convert_uint8_to_float(h_queries.view(), float_queries.view());
  raft::copy(d_float_queries.data_handle(), float_queries.data_handle(), float_queries.size(), stream);

  // Prepare category and offset data
  auto h_data_offsets = raft::make_host_vector<int>(data_label_vecs.size());
  int32_t total_data_size = 0;
  for (int i = 0; i < data_label_vecs.size(); i++) {
    total_data_size += data_label_vecs[i].size();
    h_data_offsets(i) = total_data_size;
  }
  auto h_data_cats = raft::make_host_vector<int>(total_data_size);
  for (int i = 0; i < data_label_vecs.size(); i++) {
    for (int j = 0; j < data_label_vecs[i].size(); j++) {
      h_data_cats(h_data_offsets(i) - data_label_vecs[i].size() + j) = data_label_vecs[i][j];
    }
  }
  int Nq = query_label_vecs.size();
  auto h_query_cats = raft::make_host_matrix<int>(Nq, 2);
  auto h_query_mapping = raft::make_host_vector<int>(Nq);
  for (int i = 0; i < Nq; i++) {
    h_query_mapping(i) = i;
    h_query_cats.view()(i, 0) = query_label_vecs[i][0];
    if (query_label_vecs[i].size() > 1) {
      h_query_cats.view()(i, 1) = query_label_vecs[i][1];
    } else {
      h_query_cats.view()(i, 1) = cat_freq.size()-1;
    }
  }
  auto d_data_offsets = raft::make_device_vector<int>(dev_resources, h_data_offsets.size());
  raft::copy(d_data_offsets.data_handle(), h_data_offsets.data_handle(), h_data_offsets.size(), stream);
  auto d_data_cats = raft::make_device_vector<int>(dev_resources, total_data_size);
  raft::copy(d_data_cats.data_handle(), h_data_cats.data_handle(), total_data_size, stream);
  auto d_query_cats = raft::make_device_matrix<int>(dev_resources, Nq, 2);
  raft::copy(d_query_cats.data_handle(), h_query_cats.data_handle(), h_query_cats.size(), stream);
  auto d_query_mapping = raft::make_device_vector<int>(dev_resources, Nq);
  raft::copy(d_query_mapping.data_handle(), h_query_mapping.data_handle(), h_query_mapping.size(), stream);
  auto d_cat_freq = raft::make_device_vector<int>(dev_resources, cat_freq.size());
  raft::copy(d_cat_freq.data_handle(), cat_freq.data(), cat_freq.size(), stream);
  cagra_filter filter(
    d_data_cats.view(),
    d_query_cats.view(),
    d_data_offsets.view(),
    d_query_mapping.view(),
    cat_freq.size()-1
  );
  
  cagra::search_params search_params;
  search_params.itopk_size = itopk;
  search_params.max_iterations = max_iterations;
  int64_t topk = 10;
  auto all_neighbors = raft::make_device_matrix<IdxT, int64_t>(dev_resources, Nq, topk);
  auto all_distances = raft::make_device_matrix<float, int64_t>(dev_resources, Nq, topk);


  std::cout << "Starting warm-up..." << std::endl;
  int warm_up = 10;
  raft::device_resources cagra_resources;
  cudaStream_t cagra_stream = raft::resource::get_cuda_stream(cagra_resources);
  for (int iter = 0; iter < warm_up; iter++) {
    auto iteration_start_t = std::chrono::high_resolution_clock::now();
    int query_offset = (iter * batch_size) % Nq;
    shared_resources::thread_id = 0;
    shared_resources::n_threads = 1;
    std::vector<int> cagra_query_map;
    cagra_query_map.reserve(batch_size);
    std::vector<int> bf_query_map;
    bf_query_map.reserve(batch_size);
    int cagra_search_size = 0;
    int bf_search_size = 0;
    int N = dataset.extent(0);
    auto bf_flags = raft::make_device_vector<int>(cagra_resources, batch_size);
    classify_queries_kernel<<<32,32,0,cagra_stream>>>(
      d_query_cats.view(), 
      d_cat_freq.view(),
      bf_flags.view(),
      batch_size,
      query_offset,
      N,
      specificity_threshold);
    cudaDeviceSynchronize();
    auto h_bf_flags = raft::make_host_vector<int>(batch_size);
    raft::copy(h_bf_flags.data_handle(), bf_flags.data_handle(), batch_size, cagra_stream);
    for (int i=0; i<batch_size; i++) {
      if (h_bf_flags.view()(i) == 1) {
        bf_query_map.push_back(i);
        bf_search_size++;
      } else {
        cagra_query_map.push_back(i);
        cagra_search_size++;
      }
    }
    if (cagra_search_size > 0) {
      auto d_cagra_query_map = raft::make_device_vector<int>(cagra_resources, cagra_search_size);
      raft::copy(d_cagra_query_map.data_handle(), cagra_query_map.data(), cagra_search_size, cagra_stream);
      auto batch_filter = filter.update_filter(query_offset, d_cagra_query_map.view());
      auto cagra_batch_query = raft::make_device_matrix<T, int64_t>(cagra_resources, cagra_search_size, query.extent(1));
      copy_cagra_query<<<32,32,0,cagra_stream>>>(query, cagra_batch_query.view(), d_cagra_query_map.view(), query_offset);
      auto cagra_neighbors = raft::make_device_matrix<IdxT, int64_t>(cagra_resources, cagra_search_size, topk);
      auto cagra_distances = raft::make_device_matrix<float, int64_t>(cagra_resources, cagra_search_size, topk);
      cagra::search_with_filtering(cagra_resources, 
          search_params, index, raft::make_const_mdspan(cagra_batch_query.view()), 
          cagra_neighbors.view(), cagra_distances.view(), batch_filter);
      copy_cagra_neighbors<<<32,32,0,cagra_stream>>>(cagra_neighbors.view(), all_neighbors.view(), 
                              d_cagra_query_map.view(), query_offset, cagra_search_size, topk);
      cudaDeviceSynchronize();
    }
    if (bf_search_size > 0) {
      bf_search(dev_resources,
            float_dataset.view(),
            d_float_queries.view(),
            query_label_vecs,
            data_label_vecs,
            label_data_vecs,
            cat_freq,
            query_offset,
            bf_search_size,
            bf_query_map,
            all_neighbors.view(),
            topk,
            cagra_stream);
      cudaDeviceSynchronize();
    }
  }

  std::cout << "\nStarting benchmark:" << std::endl;
  std::cout << "Total queries: " << total_queries << std::endl;
  std::cout << "Batch size: " << batch_size << " queries" << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "itopk_size  " << itopk << std::endl;
  std::cout << "Specificity threshold  " << specificity_threshold << std::endl;

  std::vector<double> classification_times;
  std::vector<double> cagra_search_times;
  std::vector<double> bf_search_times;
  std::vector<int> cagra_sizes;
  std::vector<int> bf_sizes;
  classification_times.reserve(iterations);
  cagra_search_times.reserve(iterations);
  bf_search_times.reserve(iterations);
  cagra_sizes.reserve(iterations);
  bf_sizes.reserve(iterations);
  std::vector<double> iteration_times;
  iteration_times.reserve(iterations);
  std::vector<bool> run_bf_vecs;
  run_bf_vecs.reserve(iterations);

  GPUPerformanceMonitor cagra_monitor(1, false);
  
  for (int iter = 0; iter < iterations; iter++) {
    auto iteration_start_t = std::chrono::high_resolution_clock::now();
    int query_offset = (iter * batch_size) % Nq;
    shared_resources::thread_id = 0;
    shared_resources::n_threads = 1;

    // Start classification timing
    auto classification_start = std::chrono::high_resolution_clock::now();
    std::vector<int> cagra_query_map;
    cagra_query_map.reserve(batch_size);
    std::vector<int> bf_query_map;
    bf_query_map.reserve(batch_size);
    int cagra_search_size = 0;
    int bf_search_size = 0;
    int N = dataset.extent(0);
    auto bf_flags = raft::make_device_vector<int>(cagra_resources, batch_size);
    classify_queries_kernel<<<32,32,0,cagra_stream>>>(
      d_query_cats.view(), 
      d_cat_freq.view(),
      bf_flags.view(),
      batch_size,
      query_offset,
      N,
      specificity_threshold);
    auto h_bf_flags = raft::make_host_vector<int>(batch_size);
    raft::copy(h_bf_flags.data_handle(), bf_flags.data_handle(), batch_size, cagra_stream);
    for (int i=0; i<batch_size; i++) {
      if (h_bf_flags.view()(i) == 1) {
        bf_query_map.push_back(i);
        bf_search_size++;
        run_bf_vecs.push_back(true);
      } else {
        cagra_query_map.push_back(i);
        cagra_search_size++;
        run_bf_vecs.push_back(false);
      }
    }
    cagra_sizes.push_back(cagra_search_size);
    bf_sizes.push_back(bf_search_size);
    auto classification_end = std::chrono::high_resolution_clock::now();
    double classification_time = std::chrono::duration_cast<std::chrono::microseconds>(
        classification_end - classification_start).count()/1000000.0;
    classification_times.push_back(classification_time);

    // Start CAGRA timing
    auto cagra_start = std::chrono::high_resolution_clock::now();
    if (cagra_search_size > 0) {
      cagra_monitor.start_measurement(cagra_stream, cagra_search_size);
      auto d_cagra_query_map = raft::make_device_vector<int>(cagra_resources, cagra_search_size);
      raft::copy(d_cagra_query_map.data_handle(), cagra_query_map.data(), cagra_search_size, cagra_stream);
      auto batch_filter = filter.update_filter(query_offset, d_cagra_query_map.view());
      auto cagra_batch_query = raft::make_device_matrix<T, int64_t>(cagra_resources, cagra_search_size, query.extent(1));
      const int query_block_size = 256;
      const int total_elements = cagra_search_size * query.extent(1);
      const int query_grid_size = std::min<int>((total_elements + query_block_size - 1) / query_block_size, 65535);
      copy_cagra_query<<<query_grid_size, query_block_size, 0, cagra_stream>>>(
          query, cagra_batch_query.view(), d_cagra_query_map.view(), query_offset);
      auto cagra_neighbors = raft::make_device_matrix<IdxT, int64_t>(cagra_resources, cagra_search_size, topk);
      auto cagra_distances = raft::make_device_matrix<float, int64_t>(cagra_resources, cagra_search_size, topk);
      cagra::search_with_filtering(cagra_resources, 
          search_params, index, raft::make_const_mdspan(cagra_batch_query.view()), 
          cagra_neighbors.view(), cagra_distances.view(), batch_filter);
      const int neighbor_block_size = 256;  // Using a different name to avoid redeclaration
      const int neighbor_grid_size = std::min<int>((cagra_search_size * topk + neighbor_block_size - 1) / neighbor_block_size, 65535);
      copy_cagra_neighbors<<<neighbor_grid_size, neighbor_block_size, 0, cagra_stream>>>(
          cagra_neighbors.view(), 
          all_neighbors.view(), 
          d_cagra_query_map.view(), 
          query_offset, 
          cagra_search_size, 
          topk
      );
      cagra_monitor.stop_measurement(cagra_stream, cagra_search_size);
      // cudaStreamSynchronize(cagra_stream);
    }
    auto cagra_end = std::chrono::high_resolution_clock::now();
    double cagra_time = std::chrono::duration_cast<std::chrono::microseconds>(
        cagra_end - cagra_start).count()/1000000.0;
    cagra_search_times.push_back(cagra_time);
   
    // Start BF timing
    auto bf_start = std::chrono::high_resolution_clock::now();
    if (bf_search_size > 0) {
      bf_search(dev_resources,
            float_dataset.view(),
            d_float_queries.view(),
            query_label_vecs,
            data_label_vecs,
            label_data_vecs,
            cat_freq,
            query_offset,
            bf_search_size,
            bf_query_map,
            all_neighbors.view(),
            topk,
            cagra_stream);
      // cudaDeviceSynchronize();
    }
    auto bf_end = std::chrono::high_resolution_clock::now();
    double bf_time = std::chrono::duration_cast<std::chrono::microseconds>(
        bf_end - bf_start).count()/1000000.0;
    bf_search_times.push_back(bf_time);

    auto iteration_end_t = std::chrono::high_resolution_clock::now();
    double iteration_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            iteration_end_t - iteration_start_t).count()/1000000.0;
    iteration_times.push_back(iteration_time);
  }
  // At the end, process remaining metrics
  auto final_cagra_metrics = cagra_monitor.process_metrics(true);
  cagra_monitor.print_performance_stats("CAGRA Search Final", final_cagra_metrics);
  
  
  // Calculate average sizes
  double avg_cagra_size = std::accumulate(cagra_sizes.begin(), cagra_sizes.end(), 0.0) / iterations;
  double avg_bf_size = std::accumulate(bf_sizes.begin(), bf_sizes.end(), 0.0) / iterations;
  
  // Calculate timing statistics
  double avg_classification = std::accumulate(classification_times.begin(), classification_times.end(), 0.0) / iterations;
  double avg_cagra = std::accumulate(cagra_search_times.begin(), cagra_search_times.end(), 0.0) / iterations;
  double avg_bf = std::accumulate(bf_search_times.begin(), bf_search_times.end(), 0.0) / iterations;
  double total_time = avg_classification + avg_cagra + avg_bf;
  
  std::cout << "\nWorkload Distribution:" << std::endl;
  std::cout << "Average CAGRA batch size: " << avg_cagra_size << " queries (" 
            << (avg_cagra_size/batch_size)*100 << "% of batch)" << std::endl;
  std::cout << "Average BF batch size:    " << avg_bf_size << " queries (" 
            << (avg_bf_size/batch_size)*100 << "% of batch)" << std::endl;
  

  // std::cout << "\nTiming Breakdown (averaged over " << iterations << " iterations):" << std::endl;
  // std::cout << "Classification time: " << avg_classification * 1000 << " ms (" 
  //           << (avg_classification / total_time) * 100 << "%)" << std::endl;
  // std::cout << "CAGRA search time:  " << avg_cagra * 1000 << " ms (" 
  //           << (avg_cagra / total_time) * 100 << "%)" << std::endl;
  std::cout << "BF search time:     " << avg_bf * 1000 << " ms" << std::endl;
  //           << (avg_bf / total_time) * 100 << "%)" << std::endl;
  std::cout << "Total time:         " << total_time * 1000 << " ms" << std::endl;

  // Optional: Print time per query metrics
  // if (avg_cagra_size > 0) {
  //   std::cout << "\nPer-query times:" << std::endl;
  //   std::cout << "CAGRA: " << (avg_cagra_size / avg_cagra) << " QPS" << std::endl;
  // }
  // if (avg_bf_size > 0) {
  //   std::cout << "BF:    " << (avg_bf_size / avg_bf) << " QPS" << std::endl;
  // }
  std::cout << "Total QPS:    " << (batch_size / total_time) << std::endl;

  // Copy results to host and calculate metrics
  auto h_neighbors = raft::make_host_matrix<IdxT, int64_t>(Nq, topk);
  raft::copy(h_neighbors.data_handle(), all_neighbors.data_handle(), all_neighbors.size(), stream);
  calculate_recalls_and_qps<IdxT>(h_neighbors.view(), gt_indices, query_label_vecs, 
                                  run_bf_vecs, iteration_times, batch_size, iterations,
                                              specificity_threshold, itopk, graph_degree);
}

void usage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "./CAGRA_LABELS <data file> <data label file> <query file> <query label file> <gt file> "
            << "<max datasize> <itopk> <max_iterations> <batch_size> <iterations>  <specificity_threshold>" 
            << "<graph_degree>" << std::endl;
  exit(1);
}

int main(int argc, char* argv[])
{
  // raft::device_resources dev_resources;

  // rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
  //   rmm::mr::get_current_device_resource(), 30 * 1024 * 1024 * 1024ull);
  // rmm::mr::set_current_device_resource(&pool_mr);

  shared_resources::configured_raft_resources dev_resources{};

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  if(argc != 13) usage();

  std::string data_fname = (std::string)argv[1];
  std::string data_label_fname = (std::string)argv[2];
  std::string query_fname = (std::string)argv[3];
  std::string query_label_fname = (std::string)argv[4];
  std::string gt_fname = (std::string)argv[5];
  size_t max_N = atoi(argv[6]);
  int itopk = atoi(argv[7]);
  int max_iterations = atoi(argv[8]);
  int batch_size = atoi(argv[9]);
  int iterations = atoi(argv[10]);
  int specificity_threshold = atoi(argv[11]);
  int graph_degree = atoi(argv[12]);

  int total_queries = 100000;

  std::vector<uint8_t> h_data;
  std::vector<uint8_t> h_queries;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);

  read_labeled_data<uint8_t, int64_t>(data_fname, data_label_fname, query_fname, query_label_fname,
                  &h_data, &data_label_vecs, &label_data_vecs,
                  &h_queries, &query_label_vecs, &cat_freq, &query_freq, max_N);

  size_t N = data_label_vecs.size();
  size_t dim = h_data.size() / N;
  printf("N:%lld, dim:%lld\n", N, dim);

  auto dataset = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, N, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), stream);

  auto queries = raft::make_device_matrix<uint8_t, int64_t>(dev_resources, total_queries, dim);
  raft::copy(queries.data_handle(), h_queries.data(), total_queries * dim, stream);

  std::vector<std::vector<uint32_t>> gt_indices;
  read_gt_file(gt_fname, gt_indices);

  cagra_build_search_filtered<uint8_t>(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()),
                               data_label_vecs,
                               label_data_vecs,
                               query_label_vecs,
                               cat_freq,
                               gt_indices,
                               batch_size,
                               iterations,
                               itopk,
                               max_iterations,
                               specificity_threshold,
                               graph_degree);
  return 0;
}