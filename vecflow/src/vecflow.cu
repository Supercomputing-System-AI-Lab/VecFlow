#include "vecflow.hpp"

#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/copy.cuh>

#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>

#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

// Helper function to check if a pointer is on GPU device memory
bool is_gpu_pointer(void* ptr) {
	if (ptr == nullptr) return false;
	
	cudaPointerAttributes attributes;
	cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
	
	if (err != cudaSuccess) {
		cudaGetLastError(); // Clear the error
		return false;
	}
	
	// Check if it's device memory or managed memory accessible from device
	return (attributes.type == cudaMemoryTypeDevice || 
			attributes.type == cudaMemoryTypeManaged);
}

// Helper function to check if NumPy array data is on GPU
bool is_gpu_numpy_array(py::array_t<float>& arr) {
	auto buf_info = arr.request();
	return is_gpu_pointer(buf_info.ptr);
}

// Helper function to check if NumPy array data is on GPU (int version)
bool is_gpu_numpy_array(py::array_t<int>& arr) {
	auto buf_info = arr.request();
	return is_gpu_pointer(buf_info.ptr);
}

void PyVecFlow::build(py::array_t<float> dataset,
                      std::vector<std::vector<int>> data_labels,
                      int graph_degree,
                      int specificity_threshold,
                      std::string graph_fname,
                      std::string bfs_fname,
                      bool force_rebuild,
                      bool multi_label) {
  
  auto dataset_info = dataset.request();
  int64_t n = dataset_info.shape[0];
  int64_t dim = dataset_info.shape[1];

  // Copy dataset to device
  auto d_dataset = raft::make_device_matrix<float, int64_t>(res, n, dim);
  raft::copy(d_dataset.data_handle(),
             static_cast<float*>(dataset_info.ptr),
             n * dim,
             raft::resource::get_cuda_stream(res));
  d_dataset = raft::make_device_matrix<float, int64_t>(res, n, dim);
  raft::copy(d_dataset.data_handle(),
             static_cast<float*>(dataset_info.ptr),
             n * dim,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Build VecFlow index using the same API as C++ example and store internally
  auto built_index = vecflow::build(res,
                                    raft::make_const_mdspan(d_dataset.view()),
                                    data_labels,
                                    graph_degree,
                                    specificity_threshold,
                                    graph_fname,
                                    bfs_fname,
                                    force_rebuild,
                                    multi_label);
  idx = std::make_unique<vecflow::index<float>>(std::move(built_index));
  raft::resource::sync_stream(res);
}

std::tuple<py::array_t<uint32_t>, py::array_t<float>>
PyVecFlow::search(py::array_t<float> queries,
                  py::array_t<int> query_labels,
                  int itopk_size,
                  int topk) {
  // Check if index has been built
  if (!idx) {
    throw std::runtime_error("Index has not been built. Call build() first.");
  }

  // Get queries info
  auto queries_info = queries.request();
  int n_queries = queries_info.shape[0];
  int dim = queries_info.shape[1];

  // Handle queries - check if already on GPU to avoid unnecessary copy
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
  bool queries_on_gpu = is_gpu_numpy_array(queries);
  
  if (queries_on_gpu) {
    raft::copy(d_queries.data_handle(),
               static_cast<float*>(queries_info.ptr),
               n_queries * dim,
               raft::resource::get_cuda_stream(res));
  } else {
    // Data is on host - copy from host to device
    raft::copy(d_queries.data_handle(),
               static_cast<float*>(queries_info.ptr),
               n_queries * dim,
               raft::resource::get_cuda_stream(res));
  }

  // Handle query labels - check if already on GPU and convert int to uint32_t
  auto query_labels_info = query_labels.request();
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  bool labels_on_gpu = is_gpu_numpy_array(query_labels);
  
  if (labels_on_gpu) {
    raft::copy(d_query_labels.data_handle(),
               static_cast<uint32_t*>(query_labels_info.ptr),
               n_queries,
               raft::resource::get_cuda_stream(res));
  } else {
    std::vector<uint32_t> h_query_labels(n_queries);
    for (int i = 0; i < n_queries; i++) {
      h_query_labels[i] = static_cast<uint32_t>(static_cast<int*>(query_labels_info.ptr)[i]);
    }
    raft::copy(d_query_labels.data_handle(),
               h_query_labels.data(),
               n_queries,
               raft::resource::get_cuda_stream(res));
  }
  
  // Allocate device matrices for the output
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, topk);

  // Call the search routine using the same API as C++ example
  vecflow::search(res,
                  *idx,
                  raft::make_const_mdspan(d_queries.view()),
                  d_query_labels.view(),
                  itopk_size,
                  d_neighbors.view(),
                  d_distances.view());
  raft::resource::sync_stream(res);

  // Copy results from device to host
  std::vector<py::ssize_t> shape = {n_queries, topk};
  py::array_t<uint32_t> neighbors_array(shape);
  py::array_t<float> distances_array(shape);
  auto neigh_buf = neighbors_array.request();
  auto dist_buf = distances_array.request();
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
             d_neighbors.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(res));
  raft::copy(static_cast<float*>(dist_buf.ptr),
             d_distances.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return std::make_tuple(neighbors_array, distances_array);
}

// ─── Multi-label AND search ───────────────────────────────────────────────
std::tuple<py::array_t<uint32_t>, py::array_t<float>>
PyVecFlow::search_multi(py::array_t<float> queries,
                        py::array_t<int> query_labels_a,
                        py::array_t<int> query_labels_b,
                        int itopk_size,
                        int topk) {
  if (!idx) {
    throw std::runtime_error("Index has not been built. Call build(... multi_label=True) first.");
  }
  if (idx->dataset_labels.size() == 0) {
    throw std::runtime_error(
      "search_multi requires an index built with multi_label=True; "
      "the dataset_labels CSR is empty. Rebuild with multi_label=True.");
  }

  auto q_info = queries.request();
  const int n_queries = static_cast<int>(q_info.shape[0]);
  const int dim       = static_cast<int>(q_info.shape[1]);

  // Move queries to device (host-or-GPU pointer; raft::copy handles both).
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
  raft::copy(d_queries.data_handle(),
             static_cast<float*>(q_info.ptr),
             n_queries * dim,
             raft::resource::get_cuda_stream(res));

  // Move both label arrays to device, converting int → uint32_t as needed.
  auto d_lbl_a = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto d_lbl_b = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto copy_labels = [&](py::array_t<int>& src, raft::device_vector<uint32_t, int64_t>& dst) {
    auto info = src.request();
    if (is_gpu_numpy_array(src)) {
      raft::copy(dst.data_handle(),
                 static_cast<uint32_t*>(info.ptr),
                 n_queries,
                 raft::resource::get_cuda_stream(res));
    } else {
      std::vector<uint32_t> h(n_queries);
      auto* p = static_cast<int*>(info.ptr);
      for (int i = 0; i < n_queries; i++) h[i] = static_cast<uint32_t>(p[i]);
      raft::copy(dst.data_handle(), h.data(), n_queries, raft::resource::get_cuda_stream(res));
    }
  };
  copy_labels(query_labels_a, d_lbl_a);
  copy_labels(query_labels_b, d_lbl_b);

  auto d_nbr = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, topk);
  auto d_dst = raft::make_device_matrix<float,    int64_t>(res, n_queries, topk);

  // raft::device_vector<uint32_t> isn't directly viewable as `const uint32_t`;
  // wrap the underlying handles into device_vector_view<const uint32_t,...>.
  auto a_view = raft::make_device_vector_view<const uint32_t, int64_t>(d_lbl_a.data_handle(), n_queries);
  auto b_view = raft::make_device_vector_view<const uint32_t, int64_t>(d_lbl_b.data_handle(), n_queries);

  vecflow::search_multi_labels(res,
                               *idx,
                               raft::make_const_mdspan(d_queries.view()),
                               a_view, b_view,
                               itopk_size,
                               d_nbr.view(),
                               d_dst.view());
  raft::resource::sync_stream(res);

  // Copy results back to host.
  std::vector<py::ssize_t> shape = {n_queries, topk};
  py::array_t<uint32_t> nbr_arr(shape);
  py::array_t<float>    dst_arr(shape);
  auto nb = nbr_arr.request();
  auto db = dst_arr.request();
  raft::copy(static_cast<uint32_t*>(nb.ptr), d_nbr.data_handle(), n_queries * topk, raft::resource::get_cuda_stream(res));
  raft::copy(static_cast<float*>(db.ptr),    d_dst.data_handle(), n_queries * topk, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return std::make_tuple(nbr_arr, dst_arr);
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
                                 int topk) {
  
  int total_elements = n_queries * topk;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  convert_int64_to_uint32<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
      input, output, total_elements);
}

void save_matrix_to_ibin(raft::resources const& res,
                         std::string& filename,
                         raft::device_matrix_view<uint32_t, int64_t> matrix) {
	
	int32_t rows = matrix.extent(0);
	int32_t cols = matrix.extent(1);

	auto host_matrix = raft::make_host_matrix<uint32_t, int64_t>(rows, cols);
	raft::copy(host_matrix.data_handle(),
						 matrix.data_handle(),
						 matrix.size(),
						 raft::resource::get_cuda_stream(res));
	
	std::ofstream file(filename, std::ios::binary);
	if (!file) throw std::runtime_error("Cannot create file: " + filename);
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(host_matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
  std::cout << "Saving matrix to " << filename << std::endl;
}

void load_matrix_from_ibin(raft::resources const& res,
                           std::string& filename,
                           raft::device_matrix_view<uint32_t, int64_t> matrix) {
  
  std::ifstream file(filename, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);
  int32_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));

	auto host_matrix = raft::make_host_matrix<uint32_t, int64_t>(rows, cols);
  if (rows != matrix.extent(0) || cols != matrix.extent(1))
    throw std::runtime_error("File dimensions do not match pre-allocated matrix dimensions");

  file.read(reinterpret_cast<char*>(host_matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();

	raft::copy(matrix.data_handle(),
						 host_matrix.data_handle(),
						 host_matrix.size(),
						 raft::resource::get_cuda_stream(res));
}


py::array_t<uint32_t>
PyVecFlow::generate_ground_truth(py::array_t<float> dataset,
                                 py::array_t<float> queries,
                                 std::vector<std::vector<int>> data_label_vecs,
                                 std::vector<std::vector<int>> query_label_vecs,
                                 int topk,
                                 std::string gt_fname) {

  auto dataset_info = dataset.request();
  auto queries_info = queries.request();
  int64_t n_queries = queries_info.shape[0];
  int64_t dim       = dataset_info.shape[1];
  int64_t k         = topk;

  if (dim != queries_info.shape[1]) {
    throw std::runtime_error("Dataset and query dimensions do not match");
  }

  auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);

  std::ifstream file(gt_fname);
  if (file.good()) {
    load_matrix_from_ibin(res, gt_fname, gt_neighbors.view());
    std::vector<py::ssize_t> shape = {n_queries, k};
    py::array_t<uint32_t> neighbors_array(shape);
    auto neigh_buf = neighbors_array.request();
    raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
               gt_neighbors.data_handle(),
               n_queries * k,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return neighbors_array;
  }

  if (static_cast<int64_t>(query_label_vecs.size()) != n_queries) {
    throw std::runtime_error("Number of query labels doesn't match number of queries");
  }

  // Per-label brute-force: for each label, gather its database vectors and
  // the queries assigned to it, run an unfiltered brute_force::search on that
  // subset, then map local indices back to global ones.  Mirrors the C++
  // example in examples/cpp/src/common.cuh and avoids the bitmap_filter path
  // in brute_force::search which crashes on this cuVS build with
  // "Unsupported sample filter type".

  int max_label = 0;
  for (const auto& v : data_label_vecs)
    for (int l : v) max_label = std::max(max_label, l);
  for (const auto& v : query_label_vecs)
    for (int l : v) max_label = std::max(max_label, l);
  int64_t n_labels = static_cast<int64_t>(max_label) + 1;

  std::vector<std::vector<int>> label_data_vecs(n_labels);
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (int l : data_label_vecs[i]) {
      if (l >= 0 && l < n_labels) label_data_vecs[l].push_back(static_cast<int>(i));
    }
  }

  std::vector<std::vector<int>> label_query_vecs(n_labels);
  for (int64_t q = 0; q < n_queries; q++) {
    for (int l : query_label_vecs[q]) {
      if (l >= 0 && l < n_labels) label_query_vecs[l].push_back(static_cast<int>(q));
    }
  }

  const float* h_dataset = static_cast<const float*>(dataset_info.ptr);
  const float* h_queries = static_cast<const float*>(queries_info.ptr);

  // Host result matrix (initialised to UINT32_MAX = "no result").
  auto h_gt = raft::make_host_matrix<uint32_t, int64_t>(n_queries, k);
  std::fill(h_gt.data_handle(), h_gt.data_handle() + n_queries * k, UINT32_MAX);

  cuvs::neighbors::brute_force::index_params bf_index_params;
  bf_index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  cuvs::neighbors::brute_force::search_params bf_search_params;

  for (int64_t label = 0; label < n_labels; label++) {
    const auto& db_indices = label_data_vecs[label];
    const auto& q_indices  = label_query_vecs[label];
    if (db_indices.empty() || q_indices.empty()) continue;

    int64_t n_db = static_cast<int64_t>(db_indices.size());
    int64_t n_q  = static_cast<int64_t>(q_indices.size());
    int64_t actual_k = std::min(k, n_db);

    auto h_sub_db = raft::make_host_matrix<float, int64_t>(n_db, dim);
    for (int64_t i = 0; i < n_db; i++)
      std::copy_n(h_dataset + static_cast<int64_t>(db_indices[i]) * dim, dim,
                  h_sub_db.data_handle() + i * dim);
    auto d_sub_db = raft::make_device_matrix<float, int64_t>(res, n_db, dim);
    raft::copy(d_sub_db.data_handle(), h_sub_db.data_handle(),
               n_db * dim, raft::resource::get_cuda_stream(res));

    auto h_sub_q = raft::make_host_matrix<float, int64_t>(n_q, dim);
    for (int64_t i = 0; i < n_q; i++)
      std::copy_n(h_queries + static_cast<int64_t>(q_indices[i]) * dim, dim,
                  h_sub_q.data_handle() + i * dim);
    auto d_sub_q = raft::make_device_matrix<float, int64_t>(res, n_q, dim);
    raft::copy(d_sub_q.data_handle(), h_sub_q.data_handle(),
               n_q * dim, raft::resource::get_cuda_stream(res));

    auto d_nbrs = raft::make_device_matrix<int64_t, int64_t>(res, n_q, actual_k);
    auto d_dist = raft::make_device_matrix<float,   int64_t>(res, n_q, actual_k);
    auto bf_idx = cuvs::neighbors::brute_force::build(
        res, bf_index_params, raft::make_const_mdspan(d_sub_db.view()));
    cuvs::neighbors::brute_force::search(
        res, bf_search_params, bf_idx,
        raft::make_const_mdspan(d_sub_q.view()),
        d_nbrs.view(), d_dist.view());
    raft::resource::sync_stream(res);

    auto h_nbrs = raft::make_host_matrix<int64_t, int64_t>(n_q, actual_k);
    raft::copy(h_nbrs.data_handle(), d_nbrs.data_handle(),
               n_q * actual_k, raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    for (int64_t i = 0; i < n_q; i++) {
      for (int64_t j = 0; j < actual_k; j++) {
        int64_t local = h_nbrs(i, j);
        if (local >= 0 && local < n_db)
          h_gt(q_indices[i], j) = static_cast<uint32_t>(db_indices[local]);
      }
    }
  }

  raft::copy(gt_neighbors.data_handle(), h_gt.data_handle(),
             n_queries * k, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  save_matrix_to_ibin(res, gt_fname, gt_neighbors.view());

  std::vector<py::ssize_t> shape = {n_queries, k};
  py::array_t<uint32_t> neighbors_array(shape);
  auto neigh_buf = neighbors_array.request();
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
             gt_neighbors.data_handle(),
             n_queries * k,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  std::cout << "Generated ground truth for " << n_queries << " queries" << std::endl;

  return neighbors_array;
}