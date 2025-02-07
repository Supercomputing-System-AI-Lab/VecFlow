#include "vecflow.hpp"

#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>


using namespace cuvs::neighbors;

PyVecFlow::PyVecFlow() : idx() {};

void PyVecFlow::build(py::array_t<float> dataset,
                      std::string data_label_fname,
                      int graph_degree,
                      int specificity_threshold,
                      std::string graph_fname,
                      std::string bfs_fname) {
  
  auto dataset_info = dataset.request();
  int n = dataset_info.shape[0];
  int dim = dataset_info.shape[1];

  // Copy queries to device.
  auto d_dataset = raft::make_device_matrix<float, int64_t>(idx.res, n, dim);
  raft::copy(d_dataset.data_handle(),
             static_cast<float*>(dataset_info.ptr),
             n * dim,
             raft::resource::get_cuda_stream(idx.res));
  raft::resource::sync_stream(idx.res);

  vecflow::build(idx,
                raft::make_const_mdspan(d_dataset.view()),
                data_label_fname,
                graph_degree,
                specificity_threshold,
                graph_fname,
                bfs_fname);
  raft::resource::sync_stream(idx.res);
}

std::tuple<py::array_t<uint32_t>, py::array_t<float>>
PyVecFlow::search(py::array_t<float> queries,
                  py::array_t<int> query_labels,
                  int itopk_size) {
  // Get queries info.
  auto queries_info = queries.request();
  int num_queries = queries_info.shape[0];
  int dim = queries_info.shape[1];

  // Copy queries to device.
  auto d_queries = raft::make_device_matrix<float, int64_t>(idx.res, num_queries, dim);
  raft::copy(d_queries.data_handle(),
             static_cast<float*>(queries_info.ptr),
             num_queries * dim,
             raft::resource::get_cuda_stream(idx.res));

  // Copy query labels to device.
  auto query_labels_info = query_labels.request();
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(idx.res, num_queries);
  raft::copy(d_query_labels.data_handle(),
             static_cast<uint32_t*>(query_labels_info.ptr),
             num_queries,
             raft::resource::get_cuda_stream(idx.res));
  
  // Allocate device matrices for the output.
  auto topk = 10;
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, num_queries, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(idx.res, num_queries, topk);

  // Call the search routine.
  vecflow::search(idx,
                  raft::make_const_mdspan(d_queries.view()),
                  d_query_labels.view(),
                  itopk_size,
                  d_neighbors.view(),
                  d_distances.view());

  // Copy results from device to host.
  py::array_t<uint32_t> neighbors_array({num_queries, topk});
  py::array_t<float> distances_array({num_queries, topk});
  auto neigh_buf = neighbors_array.request();
  auto dist_buf = distances_array.request();
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
              d_neighbors.data_handle(),
              num_queries * topk,
              raft::resource::get_cuda_stream(idx.res));
  raft::copy(static_cast<float*>(dist_buf.ptr),
            d_distances.data_handle(),
            num_queries * topk,
            raft::resource::get_cuda_stream(idx.res));

  return std::make_tuple(neighbors_array, distances_array);
}

float PyVecFlow::compute_recall(py::array_t<uint32_t> neighbors, py::array_t<uint32_t> gt_indices) {
  auto neigh_info = neighbors.request();
  auto gt_info = gt_indices.request();
  
  if (neigh_info.shape[0] != gt_info.shape[0]) {
    throw std::runtime_error("Number of queries must match between neighbors and ground truth");
  }
  
  int n_queries = neigh_info.shape[0];
  int topk = neigh_info.shape[1];
  int gt_k = gt_info.shape[1];
  
  if (topk > gt_k) {
    throw std::runtime_error("k in neighbors cannot be larger than ground truth k");
  }
  
  uint32_t* neigh_ptr = static_cast<uint32_t*>(neigh_info.ptr);
  uint32_t* gt_ptr = static_cast<uint32_t*>(gt_info.ptr);
  
  float total_recall = 0.0f;
  
  // For each query
  for (int i = 0; i < n_queries; i++) {
    int matches = 0;
    for (int j = 0; j < topk; j++) {
      uint32_t neighbor_idx = neigh_ptr[i * topk + j];
      bool found = false;
      for (int gt_j = 0; gt_j < topk; gt_j++) {
        if (gt_ptr[i * gt_k + gt_j] == neighbor_idx) {
          found = true;
          break;
        }
      }
      
      if (found) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;
  }
  return total_recall / n_queries;
}