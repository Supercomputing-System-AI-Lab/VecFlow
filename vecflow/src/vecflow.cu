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

template<typename T, typename idxT>
void read_labeled_data(std::string data_label_fname, 
                       std::string query_label_fname, 
                       std::vector<std::vector<int>>* labels_data, 
                       std::vector<std::vector<int>>* query_labels) {

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
  labels_data->reserve(ncol);
  labels_data->resize(ncol);
  
  // Process data labels and build reverse mappings
  for (int64_t i = 0; i < nrow; ++i) {
    std::vector<int> label_list;
    for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
      label_list.push_back(indices[j]);
      (*labels_data)[indices[j]].push_back(i);
    }
  }

  // read labels for queries
  std::ifstream qlabelfile(query_label_fname, std::ios::binary);
  if (!qlabelfile) {
    throw std::runtime_error("Unable to open query label file: " + query_label_fname);
  }
  qlabelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
  int64_t q_nrow = sizes[0], q_ncol = sizes[1], q_nnz = sizes[2];
  
  indptr.resize(q_nrow + 1);
  qlabelfile.read(reinterpret_cast<char*>(indptr.data()), (q_nrow + 1) * sizeof(int64_t));
  if (q_nnz != indptr.back()) {
    throw std::runtime_error("Inconsistent nnz in query labels");
  }
  indices.resize(q_nnz);
  qlabelfile.read(reinterpret_cast<char*>(indices.data()), q_nnz * sizeof(int));
  if (!std::all_of(indices.begin(), indices.end(), [q_ncol](int i) { return i >= 0 && i < q_ncol; })) {
    throw std::runtime_error("Invalid indices in query labels");
  }
  qlabelfile.close();
  
  query_labels->reserve(q_nrow);
  for (int64_t i = 0; i < q_nrow; ++i) {
    std::vector<int> label_list;
    for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
      int label = indices[j];
      // Ensure the label is in the range of dataset labels
      if (label < 0 || label >= static_cast<int>(labels_data->size())) {
        throw std::runtime_error("Query label " + std::to_string(label) + 
                              " at query index " + std::to_string(i) + 
                              " is outside the range of dataset labels (0 to " + 
                              std::to_string(labels_data->size() - 1) + ")");
      }
      label_list.push_back(label);
    }
    query_labels->push_back(std::move(label_list));
  }
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


void PyVecFlow::build(py::array_t<float> dataset,
                      std::string data_label_fname,
                      int graph_degree,
                      int specificity_threshold,
                      std::string graph_fname,
                      std::string bfs_fname) {
  
  auto dataset_info = dataset.request();
  int n = dataset_info.shape[0];
  int dim = dataset_info.shape[1];

  // Copy dataset to device.
  auto d_dataset = raft::make_device_matrix<float, int64_t>(idx.res, n, dim);
  raft::copy(d_dataset.data_handle(),
            static_cast<float*>(dataset_info.ptr),
            n * dim,
            raft::resource::get_cuda_stream(idx.res));
  d_dataset = raft::make_device_matrix<float, int64_t>(idx.res, n, dim);
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
  int n_queries = queries_info.shape[0];
  int dim = queries_info.shape[1];

  // Copy queries to device.
  auto d_queries = raft::make_device_matrix<float, int64_t>(idx.res, n_queries, dim);
  raft::copy(d_queries.data_handle(),
             static_cast<float*>(queries_info.ptr),
             n_queries * dim,
             raft::resource::get_cuda_stream(idx.res));

  // Copy query labels to device.
  auto query_labels_info = query_labels.request();
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(idx.res, n_queries);
  raft::copy(d_query_labels.data_handle(),
             static_cast<uint32_t*>(query_labels_info.ptr),
             n_queries,
             raft::resource::get_cuda_stream(idx.res));
  
  // Allocate device matrices for the output.
  auto topk = 10;
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, n_queries, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(idx.res, n_queries, topk);

  // Call the search routine.
  vecflow::search(idx,
                  raft::make_const_mdspan(d_queries.view()),
                  d_query_labels.view(),
                  itopk_size,
                  d_neighbors.view(),
                  d_distances.view());

  // Copy results from device to host.
  py::array_t<uint32_t> neighbors_array({n_queries, topk});
  py::array_t<float> distances_array({n_queries, topk});
  auto neigh_buf = neighbors_array.request();
  auto dist_buf = distances_array.request();
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
              d_neighbors.data_handle(),
              n_queries * topk,
              raft::resource::get_cuda_stream(idx.res));
  raft::copy(static_cast<float*>(dist_buf.ptr),
            d_distances.data_handle(),
            n_queries * topk,
            raft::resource::get_cuda_stream(idx.res));

  return std::make_tuple(neighbors_array, distances_array);
}

py::array_t<uint32_t>
PyVecFlow::generate_ground_truth(py::array_t<float> dataset,
                                py::array_t<float> queries,
                                int64_t topk,
                                std::string data_label_fname,
                                std::string query_label_fname,
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
  auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, n_queries, topk);

  // Check if ground truth file already exists, load if it does
  std::ifstream file(gt_fname);
  if (file.good()) {
    load_matrix_from_ibin(idx.res, gt_fname, gt_neighbors.view());
    
    // Create host arrays and copy results
    py::array_t<uint32_t> neighbors_array({n_queries, topk});
    auto neigh_buf = neighbors_array.request();
    
    // Copy neighbors from device to host
    raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
              gt_neighbors.data_handle(),
              n_queries * topk,
              raft::resource::get_cuda_stream(idx.res));
    
    raft::resource::sync_stream(idx.res);
    return neighbors_array;
  }

  // Read label data
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  read_labeled_data<float, int64_t>(data_label_fname, query_label_fname, &label_data_vecs, &query_label_vecs);

  // Verify query labels match expected count
  if (query_label_vecs.size() != n_queries) {
    throw std::runtime_error("Number of query labels doesn't match number of queries");
  }

  // Copy queries to device
  auto d_queries = raft::make_device_matrix<float, int64_t>(idx.res, n_queries, dim);
  raft::copy(d_queries.data_handle(),
            static_cast<float*>(queries_info.ptr),
            n_queries * dim,
            raft::resource::get_cuda_stream(idx.res));
  
  // Copy dataset to device (only once)
  auto d_dataset = raft::make_device_matrix<float, int64_t>(idx.res, n_database, dim);
  raft::copy(d_dataset.data_handle(),
            static_cast<float*>(dataset_info.ptr),
            n_database * dim,
            raft::resource::get_cuda_stream(idx.res));
  raft::resource::sync_stream(idx.res);
  
  // Generate bitmap for filtered search
  int64_t words_per_row = (n_database + 31) / 32;  // Round up to nearest 32 bits
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(idx.res, n_queries, words_per_row);
  
  // Clear the bitmap
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      bitmap.size() * sizeof(uint32_t),
      raft::resource::get_cuda_stream(idx.res)));
  
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
                          raft::resource::get_cuda_stream(idx.res));
    }
  }
  
  // Build brute force index
  auto bf_index = cuvs::neighbors::brute_force::build(idx.res,
                                                    d_dataset.view(),
                                                    cuvs::distance::DistanceType::L2Expanded);
  
  // Create bitmap view and filter for brute force search
  auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
      bitmap.data_handle(), n_queries, n_database);
  auto filter = cuvs::neighbors::filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);
  
  // Temporary storage for int64_t neighbors
  auto temp_neighbors = raft::make_device_matrix<int64_t, int64_t>(idx.res, n_queries, topk);
  auto gt_distances = raft::make_device_matrix<float, int64_t>(idx.res, n_queries, topk);
  
  // Perform filtered exact search
  cuvs::neighbors::brute_force::search(idx.res,
                                      bf_index,
                                      d_queries.view(),
                                      temp_neighbors.view(),
                                      gt_distances.view(),
                                      filter);
  raft::resource::sync_stream(idx.res);

  // Convert int64_t neighbors to uint32_t
  convert_neighbors_to_uint32(idx.res, temp_neighbors.data_handle(), gt_neighbors.data_handle(), n_queries, topk);
  
  // Save ground truth to file
  save_matrix_to_ibin(idx.res, gt_fname, gt_neighbors.view());
  
  // Copy results from device to host
  py::array_t<uint32_t> neighbors_array({n_queries, topk});
  auto neigh_buf = neighbors_array.request();
  
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
            gt_neighbors.data_handle(),
            n_queries * topk,
            raft::resource::get_cuda_stream(idx.res));
  
  raft::resource::sync_stream(idx.res);
  std::cout << "Generated filtered ground truth for " << n_queries << " queries" << std::endl;
  
  return neighbors_array;
}
