#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>
#include <cuvs/neighbors/brute_force.hpp>
#include <sstream>

// Helper function to check if a file is in text format
bool is_text_format(const std::string& filename) {
	return filename.find(".txt") != std::string::npos;
}

// Helper function to read labels from text format
void read_labels_from_text(const std::string& filename, 
													 std::vector<std::vector<int>>& row_labels,
													 int64_t& ncol) {
	
	std::ifstream infile(filename);
	if (!infile) {
		throw std::runtime_error("Unable to open text label file: " + filename);
	}

	row_labels.clear();
	ncol = 0;
	
	std::string line;
	while (std::getline(infile, line)) {
		std::string trimmed_line = line;
		// Trim whitespace
		trimmed_line.erase(0, trimmed_line.find_first_not_of(" \t\r\n"));
		trimmed_line.erase(trimmed_line.find_last_not_of(" \t\r\n") + 1);

		std::vector<int> labels;
		if (!trimmed_line.empty() && trimmed_line != "-1") {
			// Handle comma-separated values
			std::istringstream iss(trimmed_line);
			std::string token;
			
			while (std::getline(iss, token, ',')) {
				int label = std::stoi(token);
				labels.push_back(label);
				ncol = std::max(ncol, static_cast<int64_t>(label + 1));
			}
		}
		// For empty lines or "-1", labels vector remains empty
		row_labels.push_back(std::move(labels));
	}
	infile.close();
}

// Helper function to read labels from spmat format
void read_labels_from_spmat(const std::string& filename,
													  std::vector<std::vector<int>>& row_labels,
													  int64_t& ncol) {
	
	std::ifstream labelfile(filename, std::ios::binary);
	if (!labelfile) {
		throw std::runtime_error("Unable to open spmat label file: " + filename);
	}
	
	std::vector<int64_t> sizes(3);
	labelfile.read(reinterpret_cast<char*>(sizes.data()), 3 * sizeof(int64_t));
	int64_t nrow = sizes[0];
	ncol = sizes[1];
	int64_t nnz = sizes[2];
	
	std::vector<int64_t> indptr(nrow + 1);
	labelfile.read(reinterpret_cast<char*>(indptr.data()), (nrow + 1) * sizeof(int64_t));
	if (nnz != indptr.back()) {
		throw std::runtime_error("Inconsistent nnz in spmat labels");
	}
	
	std::vector<int> indices(nnz);
	labelfile.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int));
	if (!std::all_of(indices.begin(), indices.end(), [ncol](int i) { return i >= 0 && i < ncol; })) {
		throw std::runtime_error("Invalid indices in spmat labels");
	}
	labelfile.close();
	
	row_labels.clear();
	row_labels.reserve(nrow);
	
	for (int64_t i = 0; i < nrow; ++i) {
		std::vector<int> label_list;
		for (int64_t j = indptr[i]; j < indptr[i+1]; ++j) {
			label_list.push_back(indices[j]);
		}
		row_labels.push_back(std::move(label_list));
	}
}

template<typename T, typename idxT>
void read_labeled_data(std::string data_fname,
											 std::string data_label_fname,
											 std::string query_fname,
											 std::string query_label_fname,
											 std::vector<T>* data,
											 std::vector<T>* queries,
											 std::vector<std::vector<int>>* label_data_vecs,
											 std::vector<std::vector<int>>* data_label_vecs,
											 std::vector<std::vector<int>>* query_labels,
											 uint32_t* N_out,
											 uint32_t* q_N_out,
											 uint32_t* dim_out) {

	// Read datafile in
	std::ifstream datafile(data_fname, std::ifstream::binary);
	if (!datafile) {
		throw std::runtime_error("Unable to open data file: " + data_fname);
	}
	uint32_t N, dim;
	datafile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
	datafile.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
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

	// Read data labels (supporting both text and spmat formats)
	std::vector<std::vector<int>> data_row_labels;
	int64_t data_ncol;
	
	if (is_text_format(data_label_fname)) {
		std::cout << "Reading data labels from text format: " << data_label_fname << std::endl;
		read_labels_from_text(data_label_fname, data_row_labels, data_ncol);
	} else {
		std::cout << "Reading data labels from spmat format: " << data_label_fname << std::endl;
		read_labels_from_spmat(data_label_fname, data_row_labels, data_ncol);
	}
	
	// Validate data label dimensions
	if (data_row_labels.size() != N) {
		throw std::runtime_error("Number of data points (" + std::to_string(N) + 
								") doesn't match number of rows in data label file (" + 
								std::to_string(data_row_labels.size()) + ")");
	}
	
	// Populate both mappings:
	// 1. data_label_vecs: data point index -> list of labels
	// 2. label_data_vecs: label index -> list of data points
	*data_label_vecs = data_row_labels;  // Direct assignment
	
	label_data_vecs->clear();
	label_data_vecs->resize(data_ncol);
	for (uint32_t i = 0; i < N; ++i) {
		for (int label : data_row_labels[i]) {
			(*label_data_vecs)[label].push_back(i);
		}
	}

	// Read query labels (supporting both text and spmat formats)
	std::vector<std::vector<int>> query_row_labels;
	int64_t query_ncol;
	
	if (is_text_format(query_label_fname)) {
		std::cout << "Reading query labels from text format: " << query_label_fname << std::endl;
		read_labels_from_text(query_label_fname, query_row_labels, query_ncol);
	} else {
		std::cout << "Reading query labels from spmat format: " << query_label_fname << std::endl;
		read_labels_from_spmat(query_label_fname, query_row_labels, query_ncol);
	}
	
	// Validate query label dimensions
	if (query_row_labels.size() != q_N) {
		throw std::runtime_error("Number of queries (" + std::to_string(q_N) +
								") doesn't match number of rows in query label file (" +
								std::to_string(query_row_labels.size()) + ")");
	}
	
	// Validate that query labels are within the range of data labels
	for (uint32_t i = 0; i < q_N; ++i) {
		for (int label : query_row_labels[i]) {
			if (label < 0 || label >= static_cast<int>(data_ncol)) {
				throw std::runtime_error("Query label " + std::to_string(label) +
										" at query index " + std::to_string(i) +
										" is outside the range of dataset labels (0 to " +
										std::to_string(data_ncol - 1) + ")");
			}
		}
	}
	
	*query_labels = std::move(query_row_labels);

	if (N_out != nullptr) *N_out = N;
	if (q_N_out != nullptr) *q_N_out = q_N;
	if (dim_out != nullptr) *dim_out = dim;
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
												 raft::device_matrix_view<uint32_t, int64_t>& matrix) {

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
													 raft::device_matrix_view<uint32_t, int64_t>& matrix) {
	
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

void generate_ground_truth(raft::resources const& res,
													 raft::device_matrix_view<const float, int64_t> dataset,
													 raft::device_matrix_view<const float, int64_t> queries,
													 const std::vector<std::vector<int>>& label_data_vecs,
													 const std::vector<std::vector<int>>& query_label_vecs,
													 raft::device_matrix_view<uint32_t, int64_t> gt_neighbors,
													 std::string gt_fname) {

	std::ifstream file(gt_fname);
	if (file.good()) {
		load_matrix_from_ibin(res, gt_fname, gt_neighbors);
		return;
	}

	// Part 1: Generate bitmap for filtered search
	int64_t n_queries = query_label_vecs.size();
	int64_t n_database = dataset.extent(0);
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
			const auto& matching_data_points = label_data_vecs[label];
			std::vector<uint32_t> h_matching_bits(bitmap.extent(1), 0);
			for (int data_idx : matching_data_points) {
				if (data_idx >= n_database) continue;
				int word_idx = data_idx / 32;
				int bit_pos = data_idx % 32;
				h_matching_bits[word_idx] |= (1u << bit_pos);
			}
			raft::update_device(bitmap.data_handle() + q_idx * bitmap.extent(1),
													h_matching_bits.data(),
													bitmap.extent(1),
													raft::resource::get_cuda_stream(res));
		}
	}

	// Build brute force index
	auto bf_index = cuvs::neighbors::brute_force::build(res,
																										  dataset,
																										  cuvs::distance::DistanceType::L2Expanded);

	// Create bitmap view and filter for brute force search
	auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(
			bitmap.data_handle(), n_queries, n_database);
	auto filter = cuvs::neighbors::filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);

	// Temporary storage for int64_t neighbors
	auto temp_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, queries.extent(0), gt_neighbors.extent(1));

	// Perform filtered exact search
	auto gt_distances = raft::make_device_matrix<float, int64_t>(res, queries.extent(0), gt_neighbors.extent(1));
	cuvs::neighbors::brute_force::search(res,
																			 bf_index,
																			 queries,
																			 temp_neighbors.view(),
																			 gt_distances.view(),
																			 filter);
	raft::resource::sync_stream(res);

	convert_neighbors_to_uint32(res, temp_neighbors.data_handle(), gt_neighbors.data_handle(), n_queries, gt_neighbors.extent(1));
	save_matrix_to_ibin(res, gt_fname, gt_neighbors);
	std::cout << "Generated filtered ground truth for " << n_queries << " queries" << std::endl;
}

double compute_recall(const raft::resources& res,
                      raft::device_matrix_view<uint32_t, int64_t> neighbors,
                      raft::device_matrix_view<uint32_t, int64_t> gt_indices) {

	int n_queries = neighbors.extent(0);
	int topk = neighbors.extent(1);

	// Create host matrices for both neighbors and ground truth
	auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
	auto h_gt = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);

	// Copy data from device to host
	raft::copy(h_neighbors.data_handle(),
						 neighbors.data_handle(),
						 n_queries * topk,
						 raft::resource::get_cuda_stream(res));

	raft::copy(h_gt.data_handle(),
						 gt_indices.data_handle(),
						 n_queries * topk,
						 raft::resource::get_cuda_stream(res));

	// Synchronize to ensure copies are complete
	RAFT_CUDA_TRY(cudaStreamSynchronize(raft::resource::get_cuda_stream(res)));

	float total_recall = 0.0f;

	for (int i = 0; i < n_queries; i++) {
		int matches = 0;
		// Count matches between found and ground truth neighbors
		for (int j = 0; j < topk; j++) {
			uint32_t neighbor_idx = h_neighbors.view()(i, j);

			// Check if neighbor_idx is in the ground truth
			for (int k = 0; k < topk; k++) {
				if (neighbor_idx == h_gt.view()(i, k)) {
					matches++;
					break;
				}
			}
		}

		float recall = static_cast<float>(matches) / topk;
		total_recall += recall;
	}

  double recall = static_cast<double>(total_recall) / static_cast<double>(n_queries);
  return recall;
}
