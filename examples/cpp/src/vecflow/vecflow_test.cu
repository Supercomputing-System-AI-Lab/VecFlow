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

#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>

#include "common.cuh"

using namespace cuvs::neighbors;

int main(int argc, char** argv) {
  // Parse command line argument
  int itopk_size = 32;
  if (argc > 1) {
    itopk_size = std::stoi(argv[1]);
  }
  int specificity_threshold = 20000;
  if (argc > 2) {
    specificity_threshold = std::stoi(argv[2]);
  }
  int graph_degree = 16;
  if (argc > 3) {
    graph_degree = std::stoi(argv[3]);
  }
  
  std::string data_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/sift.base.fbin";
  std::string data_label_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/sift.base.spmat";
  std::string query_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/sift.query.fbin";
  std::string query_label_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/sift.query.spmat";
  
  std::string graph_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/graph_"+ 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + "_spec_" +
                          std::to_string(specificity_threshold) + ".bin";
  std::string bfs_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/spec_"+ 
                          std::to_string(specificity_threshold) + ".bin";

  std::vector<float> h_data;
  std::vector<float> h_queries;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);
  int max_N = 1000000;
  read_labeled_data<float, int64_t>(data_fname, data_label_fname, query_fname, query_label_fname,
                  &h_data, &data_label_vecs, &label_data_vecs,
                  &h_queries, &query_label_vecs, &cat_freq, &query_freq, max_N);
  
  vecflow::index<float> idx;
  size_t N = data_label_vecs.size();
  size_t Nq = query_label_vecs.size();
  size_t dim = h_data.size() / N;
  printf("\n=== Dataset Information ===\n");
  printf("Base dataset size: N=%lld, dim=%lld\n", N, dim);
  printf("Query dataset size: N=%lld, dim=%lld\n", Nq, dim);
  auto dataset = raft::make_device_matrix<float, int64_t>(idx.res, N, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(idx.res, Nq, dim);
  raft::copy(dataset.data_handle(), h_data.data(), h_data.size(), raft::resource::get_cuda_stream(idx.res));
  raft::copy(queries.data_handle(), h_queries.data(), h_queries.size(), raft::resource::get_cuda_stream(idx.res));

  std::vector<uint32_t> h_query_labels(Nq);
  for (int i = 0; i < Nq; i++) {
    h_query_labels[i] = query_label_vecs[i][0];
  }
  auto query_labels = raft::make_device_vector<uint32_t, int64_t>(idx.res, Nq);
  raft::copy(query_labels.data_handle(),
             h_query_labels.data(),
             Nq,
             raft::resource::get_cuda_stream(idx.res));
  
  vecflow::build(idx,
                raft::make_const_mdspan(dataset.view()),
                data_label_fname,
                graph_degree,
                specificity_threshold,
                graph_fname,
                bfs_fname);

  int64_t topk = 10;
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float, int64_t>(idx.res, queries.extent(0), topk);
  vecflow::search(idx,
                  raft::make_const_mdspan(queries.view()),
                  query_labels.view(),
                  itopk_size,
                  neighbors.view(),
                  distances.view());
  std::string gt_fname = "/scratch/bcjw/cmo1/CAGRA/sift1M/sift.groundtruth.neighbors.ibin";
  std::vector<std::vector<uint32_t>> gt_indices;
  read_ground_truth_file(gt_fname, gt_indices);
  compute_recall(idx.res, neighbors.view(), gt_indices);
}
