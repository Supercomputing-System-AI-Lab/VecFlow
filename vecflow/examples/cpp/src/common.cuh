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
	if (q_N != nrow) {
    throw std::runtime_error("Number of rows in queries (" + std::to_string(N) + 
                            ") doesn't match number of rows in query label file (" + 
                            std::to_string(nrow) + ")");
  }
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

void txt2spmat(const std::string& input_file, const std::string& output_file) {
  std::ifstream infile(input_file);
  if (!infile) {
    std::cerr << "Error: Could not open input file: " << input_file << std::endl;
    return;
  }

  // Read all lines from file
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }
  infile.close();
  
  // Parse labels from each line
  std::vector<std::vector<int32_t>> labels_list;
  for (const auto& line : lines) {
    std::string trimmed_line = line;
    // Trim whitespace
    trimmed_line.erase(0, trimmed_line.find_first_not_of(" \t\r\n"));
    trimmed_line.erase(trimmed_line.find_last_not_of(" \t\r\n") + 1);
    
    if (!trimmed_line.empty() && trimmed_line != "0") {
      // Handle comma-separated values and decrement by 1
      std::vector<int32_t> labels;
      std::istringstream iss(trimmed_line);
      std::string token;
      
      while (std::getline(iss, token, ',')) {
        int32_t label = std::stoi(token) - 1; // Convert from 1-indexed to 0-indexed
        labels.push_back(label);
      }
      labels_list.push_back(labels);
    } else {
      // Empty line or "0" means no labels
      labels_list.push_back(std::vector<int32_t>());
    }
  }
  
  // Calculate CSR parameters
  int64_t nrow = labels_list.size();
  int64_t ncol = 0;
  int64_t nnz = 0;
  
  if (nrow > 0) {
    // Find maximum label value
    for (const auto& labels : labels_list) {
      if (!labels.empty()) {
        int32_t max_in_row = *std::max_element(labels.begin(), labels.end());
        ncol = std::max(ncol, static_cast<int64_t>(max_in_row) + 1);
      }
      nnz += labels.size();
    }
  }
  
  // Build CSR representation
  std::vector<int64_t> indptr(nrow + 1, 0);
  std::vector<int32_t> indices(nnz, 0);
  
  int64_t idx = 0;
  for (int64_t i = 0; i < nrow; i++) {
    indptr[i] = idx;
    for (const auto& label : labels_list[i]) {
      indices[idx++] = label;
    }
  }
  indptr[nrow] = nnz;
  
  // Write to binary file
  std::ofstream outfile(output_file, std::ios::binary);
  if (!outfile) {
    std::cerr << "Error: Could not open output file: " << output_file << std::endl;
    return;
  }
  
  // Write header: nrow, ncol, nnz
  outfile.write(reinterpret_cast<const char*>(&nrow), sizeof(int64_t));
  outfile.write(reinterpret_cast<const char*>(&ncol), sizeof(int64_t));
  outfile.write(reinterpret_cast<const char*>(&nnz), sizeof(int64_t));
  
  // Write indptr
  outfile.write(reinterpret_cast<const char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
  
  // Write indices
  outfile.write(reinterpret_cast<const char*>(indices.data()), nnz * sizeof(int32_t));
  
  outfile.close();
  
  std::cout << "Converted text file to spmat format:" << std::endl
            << "  Rows: " << nrow << std::endl
            << "  Columns: " << ncol << std::endl
            << "  Non-zeros: " << nnz << std::endl
            << "Output saved to " << output_file << std::endl;
}