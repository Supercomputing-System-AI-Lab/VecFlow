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
                      bool force_rebuild) {
  
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
                                    force_rebuild);
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
  
  // Get dataset and query dimensions
  auto dataset_info = dataset.request();
  auto queries_info = queries.request();
  int64_t n_database = dataset_info.shape[0];
  int64_t n_queries = queries_info.shape[0];
  int64_t dim = dataset_info.shape[1];
  
  // Check if dimensions match
  if (dim != queries_info.shape[1]) {
    throw std::runtime_error("Dataset and query dimensions do not match");
  }
  
  // Initialize device matrices for results
  auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, topk);

  // Check if ground truth file already exists, load if it does
  std::ifstream file(gt_fname);
  if (file.good()) {
    load_matrix_from_ibin(res, gt_fname, gt_neighbors.view());
    std::vector<py::ssize_t> shape = {n_queries, topk};
    py::array_t<uint32_t> neighbors_array(shape);
    auto neigh_buf = neighbors_array.request();
    raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
               gt_neighbors.data_handle(),
               n_queries * topk,
               raft::resource::get_cuda_stream(res));
    return neighbors_array;
  }

  // Read label data
  int max_label = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      max_label = std::max(max_label, data_label_vecs[i][j]);
    }
  }
  int num_labels = max_label + 1;
  std::vector<std::vector<int>> label_data_vecs(num_labels);
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      label_data_vecs[data_label_vecs[i][j]].push_back(i);
    }
  }

  // Verify query labels match expected count
  if (query_label_vecs.size() != n_queries) {
    throw std::runtime_error("Number of query labels doesn't match number of queries");
  }

  // Copy queries to device
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
  raft::copy(d_queries.data_handle(),
             static_cast<float*>(queries_info.ptr),
             n_queries * dim,
             raft::resource::get_cuda_stream(res));
  
  // Copy dataset to device (only once)
  auto d_dataset = raft::make_device_matrix<float, int64_t>(res, n_database, dim);
  raft::copy(d_dataset.data_handle(),
             static_cast<float*>(dataset_info.ptr),
             n_database * dim,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  
  // Generate bitmap for filtered search
  int64_t words_per_row = (n_database + 31) / 32;  // Round up to nearest 32 bits
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, words_per_row);
  
  // Clear the bitmap
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      bitmap.size() * sizeof(uint32_t),
      raft::resource::get_cuda_stream(res)));
  
  // For each query, set bits for data points that match its label
  for (int64_t q_idx = 0; q_idx < n_queries; q_idx++) {
    const auto& query_labels = query_label_vecs[q_idx];
    for (int label : query_labels) {
      // Skip invalid labels
      if (label < 0 || label >= static_cast<int>(label_data_vecs.size())) {
        continue;
      }
      
      const auto& matching_data_points = label_data_vecs[label];
      std::vector<uint32_t> h_matching_bits(words_per_row, 0);
      
      for (int data_idx : matching_data_points) {
        if (data_idx >= n_database) continue;
        int word_idx = data_idx / 32;
        int bit_pos = data_idx % 32;
        h_matching_bits[word_idx] |= (1u << bit_pos);
      }
      
      raft::update_device(bitmap.data_handle() + q_idx * words_per_row,
                          h_matching_bits.data(),
                          words_per_row,
                          raft::resource::get_cuda_stream(res));
    }
  }
  
  // Build brute force index
  auto bf_index = cuvs::neighbors::brute_force::build(res,
                                                      d_dataset.view(),
                                                      cuvs::distance::DistanceType::L2Expanded);
  
  // Create bitmap view and filter for brute force search
  auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
      bitmap.data_handle(), n_queries, n_database);
  auto filter = cuvs::neighbors::filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);
  
  // Temporary storage for int64_t neighbors
  auto temp_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, topk);
  auto gt_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, topk);
  
  // Perform filtered exact search
  cuvs::neighbors::brute_force::search(res,
                                       bf_index,
                                       d_queries.view(),
                                       temp_neighbors.view(),
                                       gt_distances.view(),
                                       filter);
  raft::resource::sync_stream(res);

  // Convert int64_t neighbors to uint32_t
  convert_neighbors_to_uint32(res, temp_neighbors.data_handle(), gt_neighbors.data_handle(), n_queries, topk);
  
  // Save ground truth to file
  save_matrix_to_ibin(res, gt_fname, gt_neighbors.view());
  
  // Copy results from device to host
  std::vector<py::ssize_t> shape = {n_queries, topk};
  py::array_t<uint32_t> neighbors_array(shape);
  auto neigh_buf = neighbors_array.request();
  
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
             gt_neighbors.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(res));
  std::cout << "Generated filtered ground truth for " << n_queries << " queries" << std::endl;
  
  return neighbors_array;
}