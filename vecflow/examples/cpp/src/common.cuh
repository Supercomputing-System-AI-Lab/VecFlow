#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>

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

void read_ground_truth_file(const std::string& fname, std::vector<std::vector<uint32_t>>& gt_indices) {

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

void compute_recall(const raft::resources& res,
										raft::device_matrix_view<uint32_t, int64_t> neighbors,
                    const std::vector<std::vector<uint32_t>>& gt_indices) {
	
  int n_queries = neighbors.extent(0);
  int topk = neighbors.extent(1);
	auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
	raft::copy(h_neighbors.data_handle(),
						 neighbors.data_handle(),
						 n_queries * topk,
						 raft::resource::get_cuda_stream(res));
  float total_recall = 0.0f;

  for (int i = 0; i < n_queries; i++) {
    int matches = 0;
    // Count matches between found and ground truth neighbors
    for (int j = 0; j < topk; j++) {
      uint32_t neighbor_idx = h_neighbors.view()(i, j);
      if (std::find(gt_indices[i].begin(), 
                  gt_indices[i].begin() + topk, 
                  neighbor_idx) != 
        gt_indices[i].begin() + topk) {
        matches++;
      }
    }
    
    float recall = static_cast<float>(matches) / topk;
    total_recall += recall;
  }
  // Print summary statistics
  std::cout << "Overall recall (" << n_queries << " queries): " 
          << std::fixed << std::setprecision(6) 
          << total_recall / n_queries << std::endl;
}
