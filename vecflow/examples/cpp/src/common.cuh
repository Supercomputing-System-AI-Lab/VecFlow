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

	int64_t n_queries = query_label_vecs.size();
	int64_t n_database = dataset.extent(0);

	// Per-label brute-force: for each label, gather its database vectors and
	// the queries assigned to it, run an unfiltered brute_force::search on that
	// subset, then map local indices back to global ones.  This avoids the
	// bitmap_filter path in brute_force::search which crashes on this cuVS build.

	int64_t n_labels = (int64_t)label_data_vecs.size();

	// Copy full dataset and queries to host once for gathering subsets.
	auto h_dataset = raft::make_host_matrix<float, int64_t>(n_database, queries.extent(1));
	auto h_queries = raft::make_host_matrix<float, int64_t>(n_queries, queries.extent(1));
	raft::copy(h_dataset.data_handle(), dataset.data_handle(),
	           n_database * queries.extent(1), raft::resource::get_cuda_stream(res));
	raft::copy(h_queries.data_handle(), queries.data_handle(),
	           n_queries * queries.extent(1), raft::resource::get_cuda_stream(res));
	raft::resource::sync_stream(res);

	int64_t dim = queries.extent(1);
	int64_t k   = gt_neighbors.extent(1);

	// Build label -> [query_idx] mapping.
	std::vector<std::vector<int>> label_query_vecs(n_labels);
	for (int64_t q = 0; q < n_queries; q++) {
		for (int label : query_label_vecs[q]) {
			if (label >= 0 && label < n_labels) label_query_vecs[label].push_back((int)q);
		}
	}

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

		int64_t n_db = (int64_t)db_indices.size();
		int64_t n_q  = (int64_t)q_indices.size();
		int64_t actual_k = std::min(k, n_db);

		// Gather label-subset of dataset onto device.
		auto h_sub_db = raft::make_host_matrix<float, int64_t>(n_db, dim);
		for (int64_t i = 0; i < n_db; i++)
			std::copy_n(h_dataset.data_handle() + db_indices[i] * dim, dim,
			            h_sub_db.data_handle() + i * dim);
		auto d_sub_db = raft::make_device_matrix<float, int64_t>(res, n_db, dim);
		raft::copy(d_sub_db.data_handle(), h_sub_db.data_handle(),
		           n_db * dim, raft::resource::get_cuda_stream(res));

		// Gather label-subset of queries onto device.
		auto h_sub_q = raft::make_host_matrix<float, int64_t>(n_q, dim);
		for (int64_t i = 0; i < n_q; i++)
			std::copy_n(h_queries.data_handle() + q_indices[i] * dim, dim,
			            h_sub_q.data_handle() + i * dim);
		auto d_sub_q = raft::make_device_matrix<float, int64_t>(res, n_q, dim);
		raft::copy(d_sub_q.data_handle(), h_sub_q.data_handle(),
		           n_q * dim, raft::resource::get_cuda_stream(res));

		// Unfiltered brute-force search on the subset.
		auto d_nbrs = raft::make_device_matrix<int64_t, int64_t>(res, n_q, actual_k);
		auto d_dist = raft::make_device_matrix<float,   int64_t>(res, n_q, actual_k);
		auto bf_idx = cuvs::neighbors::brute_force::build(
		    res, bf_index_params, raft::make_const_mdspan(d_sub_db.view()));
		cuvs::neighbors::brute_force::search(
		    res, bf_search_params, bf_idx,
		    raft::make_const_mdspan(d_sub_q.view()),
		    d_nbrs.view(), d_dist.view());
		raft::resource::sync_stream(res);

		// Map local indices back to global indices.
		auto h_nbrs = raft::make_host_matrix<int64_t, int64_t>(n_q, actual_k);
		raft::copy(h_nbrs.data_handle(), d_nbrs.data_handle(),
		           n_q * actual_k, raft::resource::get_cuda_stream(res));
		raft::resource::sync_stream(res);

		for (int64_t i = 0; i < n_q; i++) {
			for (int64_t j = 0; j < actual_k; j++) {
				int64_t local = h_nbrs(i, j);
				if (local >= 0 && local < n_db)
					h_gt(q_indices[i], j) = (uint32_t)db_indices[local];
			}
		}
	}

	raft::copy(gt_neighbors.data_handle(), h_gt.data_handle(),
	           n_queries * k, raft::resource::get_cuda_stream(res));
	raft::resource::sync_stream(res);
	save_matrix_to_ibin(res, gt_fname, gt_neighbors);
	printf("Generated ground truth for %ld queries\n", n_queries);
}

/**
 * Brute-force ground truth for 2-label AND search. For each query, the
 * candidate set is the INTERSECTION of points carrying label_a and points
 * carrying label_b; we then take the top-k nearest among that intersection.
 * Same on-disk cache contract as `generate_ground_truth`.
 */
inline void generate_ground_truth_multi(raft::resources const& res,
                                        raft::device_matrix_view<const float, int64_t> dataset,
                                        raft::device_matrix_view<const float, int64_t> queries,
                                        const std::vector<std::vector<int>>& label_data_vecs,
                                        const std::vector<int>& query_labels_a,
                                        const std::vector<int>& query_labels_b,
                                        raft::device_matrix_view<uint32_t, int64_t> gt_neighbors,
                                        std::string gt_fname) {
	std::ifstream f(gt_fname);
	if (f.good()) {
		load_matrix_from_ibin(res, gt_fname, gt_neighbors);
		return;
	}

	int64_t n_queries     = static_cast<int64_t>(query_labels_a.size());
	int64_t n_database    = dataset.extent(0);
	int64_t words_per_row = (n_database + 31) / 32;

	auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, words_per_row);
	RAFT_CUDA_TRY(cudaMemsetAsync(bitmap.data_handle(),
	                              0,
	                              bitmap.size() * sizeof(uint32_t),
	                              raft::resource::get_cuda_stream(res)));

	// Per-row bitmap: bits set for points that contain BOTH label_a AND label_b.
	for (int64_t q = 0; q < n_queries; q++) {
		const auto& pts_a = label_data_vecs[query_labels_a[q]];
		const auto& pts_b = label_data_vecs[query_labels_b[q]];
		// Build set of pts_a for O(log n) lookups, then intersect with pts_b.
		std::vector<int> a_sorted(pts_a.begin(), pts_a.end());
		std::sort(a_sorted.begin(), a_sorted.end());
		std::vector<uint32_t> h_bits(words_per_row, 0);
		for (int p : pts_b) {
			if (p >= n_database) continue;
			if (std::binary_search(a_sorted.begin(), a_sorted.end(), p)) {
				h_bits[p / 32] |= (1u << (p % 32));
			}
		}
		raft::update_device(bitmap.data_handle() + q * words_per_row,
		                    h_bits.data(),
		                    words_per_row,
		                    raft::resource::get_cuda_stream(res));
	}

	// Brute-force search with the intersection bitmap as filter.
	cuvs::neighbors::brute_force::index_params bf_idx_p;
	bf_idx_p.metric = cuvs::distance::DistanceType::L2Expanded;
	auto bf_idx     = cuvs::neighbors::brute_force::build(res, bf_idx_p, dataset);

	auto bitmap_view = raft::core::bitmap_view<uint32_t, int64_t>(
	  bitmap.data_handle(), n_queries, n_database);
	auto filter = cuvs::neighbors::filtering::bitmap_filter<uint32_t, int64_t>(bitmap_view);

	auto temp_nbr = raft::make_device_matrix<int64_t, int64_t>(res, queries.extent(0), gt_neighbors.extent(1));
	auto gt_dist  = raft::make_device_matrix<float,   int64_t>(res, queries.extent(0), gt_neighbors.extent(1));
	cuvs::neighbors::brute_force::search_params bf_search_p;
	cuvs::neighbors::brute_force::search(res, bf_search_p, bf_idx, queries,
	                                     temp_nbr.view(), gt_dist.view(), filter);
	raft::resource::sync_stream(res);

	convert_neighbors_to_uint32(res, temp_nbr.data_handle(), gt_neighbors.data_handle(),
	                            n_queries, gt_neighbors.extent(1));
	save_matrix_to_ibin(res, gt_fname, gt_neighbors);
	std::cout << "Generated AND ground truth for " << n_queries << " queries" << std::endl;
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
