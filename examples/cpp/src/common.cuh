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

#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <ctime> 

// Fill dataset and queries with synthetic data.
void generate_dataset(raft::device_resources const &dev_resources,
                      raft::device_matrix_view<float, int64_t> dataset,
                      raft::device_matrix_view<float, int64_t> queries) {
  auto labels = raft::make_device_vector<int64_t, int64_t>(dev_resources,
                                                           dataset.extent(0));
  raft::random::make_blobs(dev_resources, dataset, labels.view());
  raft::random::RngState r(1234ULL);
  raft::random::uniform(
      dev_resources, r,
      raft::make_device_vector_view(queries.data_handle(), queries.size()),
      -1.0f, 1.0f);
}

template<typename T, typename idxT>
void read_data(std::string data_fname, 
                       std::string query_fname, 
                       std::vector<T>* data, 
                       std::vector<T>* queries, 
                       int max_N, int* out_dim) {

  // Read datafile in
  std::ifstream datafile(data_fname, std::ifstream::binary);
  uint32_t N;
  uint32_t dim;
  datafile.read((char*)&N, sizeof(uint32_t));
  datafile.read((char*)&dim, sizeof(uint32_t));

  if(N > max_N) N = max_N;
  printf("N:%u, dim:%u\n", N, dim);

  data->resize(N*dim);
  datafile.read(reinterpret_cast<char*>(data->data()), N*dim);
  datafile.close();


  // read query data in
  std::ifstream queryfile(query_fname, std::ifstream::binary);
  
  uint32_t q_N;
  uint32_t q_dim;
  queryfile.read((char*)&q_N, sizeof(uint32_t));
  queryfile.read((char*)&q_dim, sizeof(uint32_t));
  printf("qN:%u, qdim:%u\n", q_N, q_dim);
  if(q_dim != dim) {
    printf("query dim and data dim don't match!\n");
    exit(1);
  }
  queries->resize(q_N*dim);
  queryfile.read(reinterpret_cast<char*>(queries->data()), q_N*dim);
  queryfile.close();

  *out_dim = dim;
}

void read_gt_file(const std::string& gt_fname, std::vector<std::vector<uint32_t>>& gt_indices) {
    std::ifstream gtfile(gt_fname.c_str(), std::ios::binary);
    if (!gtfile) {
        std::cerr << "Error opening ground truth file: " << gt_fname << std::endl;
        exit(1);
    }

    uint32_t num_queries, gt_k;
    gtfile.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    gtfile.read(reinterpret_cast<char*>(&gt_k), sizeof(uint32_t));

    gt_indices.resize(num_queries);
    for (uint32_t i = 0; i < num_queries; ++i) {
        gt_indices[i].resize(gt_k);
        gtfile.read(reinterpret_cast<char*>(gt_indices[i].data()), gt_indices[i].size() * sizeof(uint32_t));
    }
    gtfile.close();
}

template<typename T, typename idxT>
void read_labeled_data(std::string data_fname, 
                       std::string data_label_fname, 
                       std::string query_fname, 
                       std::string query_label_fname, 
                       std::vector<T>* data, 
                       std::vector<std::vector<int>>* data_labels, 
                       std::vector<std::vector<int>>* labels_data, 
                       std::vector<T>* queries, 
                       std::vector<std::vector<int>>* query_labels, 
                       std::vector<int>* cat_freq, 
                       std::vector<int>* query_freq, 
                       int max_N) {

  // Read datafile in
  std::ifstream datafile(data_fname, std::ifstream::binary);
  if (!datafile) {
    throw std::runtime_error("Unable to open data file: " + data_fname);
  }
  uint32_t N, dim;
  datafile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
  datafile.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
  if (N > static_cast<uint32_t>(max_N)) N = max_N;
  data->resize(N*dim);
  datafile.read(reinterpret_cast<char*>(data->data()), N*dim*sizeof(T));
  datafile.close();

  // read query data in
  std::ifstream queryfile(query_fname, std::ifstream::binary);
  if (!queryfile) {
    throw std::runtime_error("Unable to open query file: " + query_fname);
  }
  uint32_t q_N, q_dim;
  queryfile.read(reinterpret_cast<char*>(&q_N), sizeof(uint32_t));
  queryfile.read(reinterpret_cast<char*>(&q_dim), sizeof(uint32_t));
  if (q_dim != dim) {
    throw std::runtime_error("Query dim and data dim don't match!");
  }
  queries->resize(q_N*dim);
  queryfile.read(reinterpret_cast<char*>(queries->data()), q_N*dim*sizeof(T));
  queryfile.close();

  // read labels for data vectors
  std::ifstream labelfile(data_label_fname, std::ios::binary);
  if (!labelfile) {
    throw std::runtime_error("Unable to open data label file: " + data_label_fname);
  }
  std::vector<int64_t> sizes(3);
  labelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
  int64_t nrow = sizes[0], ncol = sizes[1], nnz = sizes[2];
  std::vector<int64_t> indptr(nrow + 1);
  labelfile.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
  if (nnz != indptr.back()) {
    throw std::runtime_error("Inconsistent nnz in data labels");
  }
  std::vector<int> indices(nnz);
  labelfile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
  if (!std::all_of(indices.begin(), indices.end(), [ncol](int i) { return i >= 0 && i < ncol; })) {
    throw std::runtime_error("Invalid indices in data labels");
  }
  labelfile.close();
  data_labels->reserve(N);
  labels_data->reserve(ncol);
  labels_data->resize(ncol);
  cat_freq->resize(ncol, 0);
  for (uint32_t i = 0; i < N; ++i) {
    std::vector<int> label_list;
    for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
      label_list.push_back(indices[j]);
      (*cat_freq)[indices[j]]++;
      (*labels_data)[indices[j]].push_back(i);
    }
    data_labels->push_back(std::move(label_list));
  }

  // read labels for queries
  std::ifstream qlabelfile(query_label_fname, std::ios::binary);
  if (!qlabelfile) {
    throw std::runtime_error("Unable to open query label file: " + query_label_fname);
  }
  qlabelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
  nrow = sizes[0], ncol = sizes[1], nnz = sizes[2];
  indptr.resize(nrow + 1);
  qlabelfile.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
  if (nnz != indptr.back()) {
    throw std::runtime_error("Inconsistent nnz in query labels");
  }
  indices.resize(nnz);
  qlabelfile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
  if (!std::all_of(indices.begin(), indices.end(), [ncol](int i) { return i >= 0 && i < ncol; })) {
    throw std::runtime_error("Invalid indices in query labels");
  }
  qlabelfile.close();
  query_labels->reserve(q_N);
  query_freq->resize(ncol, 0);
  for (uint32_t i = 0; i < q_N; ++i) {
    std::vector<int> label_list;
    for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
      label_list.push_back(indices[j]);
      (*query_freq)[indices[j]]++;
    }
    query_labels->push_back(std::move(label_list));
  }
}



std::vector<double> read_specificities(const std::string& filename, int total_queries) {
  std::vector<double> specificities(total_queries);
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Unable to open specificity file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    
    // Read query_id
    int query_id;
    iss >> query_id;
    
    // Read all tokens until the end
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    
    if (tokens.empty()) continue;
    
    // Last token is specificity
    std::string spec_str = tokens.back();
    
    // Convert the specificity string to double
    double specificity;
    try {
      specificity = std::stod(spec_str);
      if (query_id < total_queries) {
        specificities[query_id] = specificity;
      }
    } catch (const std::invalid_argument&) {
      std::cerr << "Warning: Invalid specificity value in line: " << line << std::endl;
    } catch (const std::out_of_range&) {
      std::cerr << "Warning: Specificity value out of range in line: " << line << std::endl;
    }
  }
  
  infile.close();
  return specificities;
}

template<typename IdxT>
void calculate_recalls_and_qps(const raft::host_matrix_view<IdxT, int64_t>& h_neighbors,
                               const std::vector<std::vector<uint32_t>>& gt_indices,
                               const std::vector<std::vector<int>>& query_label_vecs,
                               const std::vector<bool>& run_bf_vecs,
                               const std::vector<double>& iteration_times,
                               int batch_size,
                               int iterations,
                               int specificity_threshold,
                               int itopk,
                               int graph_degree) {
    
  int total_queries = 100000;
  // Read specificities from file
  std::vector<double> specificities = read_specificities("query_specificities.txt", total_queries);

  // Generate timestamped filename
  time_t now = time(nullptr);
  struct tm* timeinfo = localtime(&now);
  char timestamp[80];
  strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);
  std::string filename = "/u/cmo1/Filtered/cuvs-24.10/examples/cpp/src/draw/output/search_results_spec" + 
                        std::to_string(specificity_threshold) + "_bs" + 
                        std::to_string(batch_size) + "_itopk" + 
                        std::to_string(itopk) + "_gd" + 
                        std::to_string(graph_degree) + "_" + 
                        std::string(timestamp) + ".txt";

  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  // Write header with consistent spacing
  outfile << "query_id  labels  specificity  recall  qps  run_bf" << std::endl;

  float total_recall = 0.0f;
  int topk = 10;

  for (int q = 0; q < total_queries; ++q) {
    // Calculate recall
    int matches = 0;
    for (int i = 0; i < topk; ++i) {
      IdxT neighbor_idx = h_neighbors(q, i);
      if (std::find(gt_indices[q].begin(), gt_indices[q].end(), neighbor_idx) != gt_indices[q].end()) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;

    // Calculate QPS
    int batch_idx = q / batch_size;
    double qps = batch_size / iteration_times[batch_idx];

    // Convert labels to string
    std::string label_str;
    for (size_t i = 0; i < query_label_vecs[q].size(); ++i) {
      if (i > 0) label_str += " ";
      label_str += std::to_string(query_label_vecs[q][i]);
    }

    // Write to file with consistent spacing
    outfile << std::left << std::setw(10) << q << "  "
            << std::setw(20) << label_str << "  "
            << std::scientific << std::setprecision(7) << specificities[q] << "  "
            << std::fixed << std::setprecision(1) << recall << "  "
            << std::fixed << std::setprecision(1) << qps << "  "
            << std::boolalpha << run_bf_vecs[q] << std::endl;
  }

  outfile.close();

  std::cout << "\nOverall Statistics:" << std::endl;
  std::cout << "Average Recall@" << topk << ": " << total_recall / total_queries << std::endl;
  std::cout << "Results saved to '" << filename << "'" << std::endl;

  return;
}

void convert_uint8_to_float(raft::host_matrix_view<uint8_t, int64_t> in, raft::host_matrix_view<float, int64_t> out) {
  for(int i=0; i<in.extent(0); i++) {
    for(int j=0; j<in.extent(1); j++) {
      out(i,j) = in(i,j);
    }
  }
}

__global__ void copy_int8_to_float(raft::device_matrix_view<const uint8_t, int64_t> old_data, raft::device_matrix_view<float, int64_t> new_data) {
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<old_data.extent(0); i+=blockDim.x*gridDim.x) {
    for(int j=0; j<old_data.extent(1); j++) {
      new_data(i,j) = (float)old_data(i,j);
    }
  }
}

// __global__ void copy_cagra_query(raft::device_matrix_view<const uint8_t, int64_t> query, 
//                             raft::device_matrix_view<uint8_t, int64_t> batch_query,
//                             raft::device_vector_view<int> query_map,
//                             int query_offset) {
//   for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<batch_query.extent(0); i+=blockDim.x*gridDim.x) {
//     for(int j=0; j<batch_query.extent(1); j++) {
//       batch_query(i,j) = query(query_offset+query_map[i],j);
//     }
//   }
// }

__global__ void copy_cagra_query(raft::device_matrix_view<const uint8_t, int64_t> query,
                                raft::device_matrix_view<uint8_t, int64_t> batch_query,
                                raft::device_vector_view<int> query_map,
                                int query_offset) {
  // Calculate global thread index and stride
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_elements = batch_query.extent(0) * batch_query.extent(1);
  const int stride = blockDim.x * gridDim.x;
  const int dim = batch_query.extent(1);
  
  // Process elements using stride to handle all data points
  for(int idx = tid; idx < total_elements; idx += stride) {
    const int i = idx / dim;     // Row index
    const int j = idx % dim;     // Column index
    
    if(i < batch_query.extent(0)) {
      batch_query(i, j) = query(query_offset + query_map[i], j);
    }
  }
}


// __global__ void copy_bf_query(raft::device_matrix_view<const float, int64_t> query, 
//                             raft::device_matrix_view<float, int64_t> bf_query,
//                             raft::device_vector_view<int> query_map,
//                             int query_offset) {
//   for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<bf_query.extent(0); i+=blockDim.x*gridDim.x) {
//     for(int j=0; j<bf_query.extent(1); j++) {
//       bf_query(i,j) = query(query_offset+query_map[i],j);
//     }
//   }
// }

__global__ void copy_bf_query(raft::device_matrix_view<const float, int64_t> query,
                            raft::device_matrix_view<float, int64_t> bf_query,
                            raft::device_vector_view<int> query_map,
                            int query_offset) {
  // Calculate global thread index and stride
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_elements = bf_query.extent(0) * bf_query.extent(1);
  const int stride = blockDim.x * gridDim.x;
  const int dim = bf_query.extent(1);
  
  // Process elements using stride to handle all data points
  for(int idx = tid; idx < total_elements; idx += stride) {
    const int i = idx / dim;     // Row index
    const int j = idx % dim;     // Column index
    
    if(i < bf_query.extent(0)) {
      bf_query(i, j) = query(query_offset + query_map[i], j);
    }
  }
}

__global__ void classify_queries_kernel(raft::device_matrix_view<int> query_cats, 
                                        raft::device_vector_view<int> cat_freq,
                                        raft::device_vector_view<int> bf_flags,
                                        int batch_size,
                                        int query_offset,
                                        int N,
                                        int specificity_threshold) {
  for (int i=blockIdx.x*blockDim.x + threadIdx.x; i<batch_size; i+=blockDim.x*gridDim.x) {
    int label0 = query_cats(query_offset + i, 0);
    int label1 = query_cats(query_offset + i, 1);
    float freq_product = ((float)cat_freq[label0] / N) * ((float)cat_freq[label1] / N);
    bf_flags[i] = freq_product <= ((float)specificity_threshold / N) ? 1 : 0;
  }
}

// __global__ void classify_and_map_kernel(
//   raft::device_matrix_view<const int> query_cats,
//   raft::device_vector_view<const int> cat_freq,
//   raft::device_vector_view<int> single_label_map,
//   raft::device_vector_view<int> double_label_map,
//   raft::device_vector_view<int> cagra_map,
//   raft::device_vector_view<int> single_count,
//   raft::device_vector_view<int> double_count,
//   raft::device_vector_view<int> cagra_count,
//   int batch_size,
//   int query_offset,
//   int N,
//   int specificity_threshold) {
  
//   int gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid >= batch_size) return;

//   int label0 = query_cats(query_offset + gid, 0);
//   int label1 = query_cats(query_offset + gid, 1);
//   bool is_single = (label1 == cat_freq.size() - 1);
//   float freq_product = ((float)cat_freq[label0] / N) * ((float)cat_freq[label1] / N);
//   bool use_bf = freq_product <= ((float)specificity_threshold / N);
  
//   if (use_bf) {
//     if (is_single) {
//       int pos = atomicAdd(single_count.data_handle(), 1);
//       single_label_map[pos] = gid;
//     } else {
//       int pos = atomicAdd(double_count.data_handle(), 1);
//       double_label_map[pos] = gid;
//     }
//   } else {
//     int pos = atomicAdd(cagra_count.data_handle(), 1);
//     cagra_map[pos] = gid;
//   }
// }



// __global__ void copy_cagra_neighbors(raft::device_matrix_view<uint32_t, int64_t> cagra_neighbors, 
//                             raft::device_matrix_view<uint32_t, int64_t> neighbors,
//                             raft::device_vector_view<int> query_map,
//                             int query_offset,
//                             int cagra_search_size,
//                             int topk) {
//   for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<cagra_search_size; i+=blockDim.x*gridDim.x){
//     for(int j=0; j<topk; j++){
//       neighbors(query_offset+query_map[i], j) = cagra_neighbors(i, j);
//     }
//   }
// }

__global__ void copy_cagra_neighbors(raft::device_matrix_view<uint32_t, int64_t> cagra_neighbors, 
                                    raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                    raft::device_vector_view<int> query_map,
                                    int query_offset,
                                    int cagra_search_size,
                                    int topk) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_elements = cagra_search_size * topk;
  const int stride = blockDim.x * gridDim.x;
  
  for(int idx = tid; idx < total_elements; idx += stride) {
    const int i = idx / topk;     // Row index
    const int j = idx % topk;     // Column index
    
    if(i < cagra_search_size) {
      neighbors(query_offset + query_map[i], j) = cagra_neighbors(i, j);
    }
  }
}


// __global__ void copy_dataset(raft::device_matrix_view<const float, int64_t> dataset, 
//                             raft::device_matrix_view<float, int64_t> filtered_dataset,
//                             raft::device_vector_view<int> matching_indices) {
//   for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<filtered_dataset.extent(0); i+=blockDim.x*gridDim.x) {
//     for(int j=0; j<filtered_dataset.extent(1); j++) {
//       filtered_dataset(i,j) = dataset(matching_indices(i),j);
//     }
//   }
// }

__global__ void copy_dataset(raft::device_matrix_view<const float, int64_t> dataset, 
                           raft::device_matrix_view<float, int64_t> filtered_dataset,
                           raft::device_vector_view<int> matching_indices) {
  const int num_rows = filtered_dataset.extent(0);
  const int num_cols = filtered_dataset.extent(1);
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for(int idx = tid; idx < num_rows * num_cols; idx += blockDim.x * gridDim.x) {
    const int row = idx / num_cols;
    const int col = idx % num_cols;
    if(row < num_rows && col < num_cols) {
      filtered_dataset(row, col) = dataset(matching_indices(row), col);
    }
  }
}

__global__ void copy_dataset_uint8(raft::device_matrix_view<const uint8_t, int64_t> dataset, 
                                  raft::device_matrix_view<uint8_t, int64_t> filtered_dataset,
                                  raft::device_vector_view<int> matching_indices) {
  const int num_rows = filtered_dataset.extent(0);
  const int num_cols = filtered_dataset.extent(1);
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for(int idx = tid; idx < num_rows * num_cols; idx += blockDim.x * gridDim.x) {
    const int row = idx / num_cols;
    const int col = idx % num_cols;
    if(row < num_rows && col < num_cols) {
      filtered_dataset(row, col) = dataset(matching_indices(row), col);
    }
  }
}


__global__ void copy_neighbors(raft::device_matrix_view<int64_t, int64_t> bf_neighbors, 
                            raft::device_matrix_view<uint32_t, int64_t> neighbors,
                            raft::device_vector_view<int> matching_indices) {
  int i = threadIdx.x;
  neighbors(0, i) = (uint32_t)matching_indices[bf_neighbors(0, i)];
}


__global__ void copy_bf_neighbors(raft::device_matrix_view<int64_t, int64_t> bf_neighbors, 
                            raft::device_matrix_view<int64_t, int64_t> neighbors,
                            raft::device_vector_view<int> matching_indices,
                            raft::device_vector_view<int> query_map,
                            int query_offset,
                            int bf_idx) {
  int i = threadIdx.x;
  neighbors(query_offset+query_map(bf_idx), i) = (int64_t)matching_indices(bf_neighbors(0, i));
}

__global__ void copy_refine_neighbors(raft::device_matrix_view<int64_t, int64_t> refine_neighbors, 
                            raft::device_matrix_view<uint32_t, int64_t> neighbors,
                            raft::device_vector_view<int> query_map,
                            int batch_start,
                            int topk,
                            int total_elements,
                            int query_offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= total_elements) return;
  int k = i % topk;
  int n_id = i / topk;
  neighbors(query_offset+query_map(batch_start+n_id), k) = (uint32_t)refine_neighbors(n_id, k);
}


void convert_int64_to_uint32(raft::host_matrix_view<int64_t, int64_t> in, raft::host_matrix_view<uint32_t, int64_t> out) {
  for(int i=0; i<in.extent(0); i++) {
    for(int j=0; j<in.extent(1); j++) {
      out(i,j) = static_cast<uint32_t>(in(i,j));
    }
  }
}


__global__ void c_neighbors(raft::device_matrix_view<int64_t, int64_t> bf_neighbors, 
                            raft::device_matrix_view<uint32_t, int64_t> neighbors,
                            raft::device_vector_view<int> matching_indices) {
  int i = threadIdx.x;
  neighbors(0, i) = (uint32_t)matching_indices[bf_neighbors(0, i)];
}

__global__ void convert_int64_to_uint32(const int64_t* input, uint32_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<uint32_t>(input[idx]);
    }
}

void convert_neighbors_to_uint32(raft::resources const& res,
                               const int64_t* input,
                               uint32_t* output,
                               int n_queries,
                               int k) {
  int total_elements = n_queries * k;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  
  convert_int64_to_uint32<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
      input, output, total_elements);
}

