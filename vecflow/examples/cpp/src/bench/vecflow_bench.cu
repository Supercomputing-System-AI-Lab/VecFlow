#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <nlohmann/json.hpp>
#include <bitset>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include "../common.cuh"

using namespace cuvs::neighbors;
using json = nlohmann::json;

void build_cagra_index(shared_resources::configured_raft_resources& dev_resources,
                       cagra::index<float, uint32_t>& index,
                       const raft::device_matrix_view<const float, int64_t>& dataset,
                       const std::string& index_path,
                       int graph_degree) {

  if (std::filesystem::exists(index_path)) {
    std::cout << "Loading existing CAGRA index from " << index_path << std::endl;
    cagra::deserialize(dev_resources, index_path, &index);
    index.update_dataset(dev_resources, dataset);
  } else {
    std::cout << "Building new CAGRA index..." << std::endl;
    cagra::index_params index_params;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_degree = graph_degree;

    index = cagra::build(dev_resources, index_params, dataset);

    std::cout << "Saving CAGRA index to " << index_path << std::endl;
    cagra::serialize(dev_resources, index_path, index, false);
  }

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;
}

__global__ void filter_neighbors_kernel(const uint32_t* neighbors,
                                        const uint32_t* query_labels,
                                        const uint32_t* label_bits,
                                        uint32_t* filtered_neighbors,
                                        int64_t n_queries,
                                        int64_t itopk_size,
                                        int64_t topk,
                                        int num_labels,
                                        int64_t n_vectors) {

  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_idx >= n_queries) return;

  constexpr size_t BITS_PER_WORD = 32;
  size_t NUM_WORDS = (num_labels + BITS_PER_WORD - 1) / BITS_PER_WORD;

  uint32_t query_label = query_labels[query_idx];
  int filtered_count = 0;

  if (query_label >= num_labels) {
     while (filtered_count < topk) {
       filtered_neighbors[query_idx * topk + filtered_count] = UINT32_MAX;
       filtered_count++;
     }
     return;
  }

  size_t query_word_idx = query_label / BITS_PER_WORD;
  size_t query_bit_idx = query_label % BITS_PER_WORD;
  uint32_t query_bit_mask = 1u << query_bit_idx;

  int match_count = 0;
  for (int j = 0; j < itopk_size && filtered_count < topk; ++j) {
    uint32_t neighbor_idx = neighbors[query_idx * itopk_size + j];

    if (neighbor_idx == UINT32_MAX || neighbor_idx >= n_vectors) continue;

    size_t bit_array_idx = neighbor_idx * NUM_WORDS + query_word_idx;

    // Check bounds ONLY to prevent crash, actual data might still be wrong
    if (bit_array_idx >= (size_t)(n_vectors * NUM_WORDS)) continue;

    uint32_t word = label_bits[bit_array_idx];

    if ((word & query_bit_mask) != 0) {
      filtered_neighbors[query_idx * topk + filtered_count] = neighbor_idx;
      filtered_count++;
      match_count++;
    }
  }

  // Fill remaining slots with UINT32_MAX
  while (filtered_count < topk) {
    filtered_neighbors[query_idx * topk + filtered_count] = UINT32_MAX;
    filtered_count++;
  }
}

template<typename index_t = int, typename bitmap_t = uint32_t>
__global__ void create_bitmap_filter_kernel(const index_t* __restrict__ row_offsets,
                                            const index_t* __restrict__ indices,
                                            const uint32_t* __restrict__ query_labels,
                                            bitmap_t* __restrict__ bitmap,
                                            const index_t num_queries,
                                            const index_t num_cols,
                                            const index_t words_per_row) {

  const index_t query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  // Get start and end indices for this query's label
  // Note: Assuming label_data_vecs was used to build row_offsets and indices correctly
  // such that query_labels[query_idx] maps to the correct row in that structure.
  const index_t label_row = query_labels[query_idx];
  // Add bounds check for label_row if needed, depending on how row_offsets is sized
  // if (label_row >= num_unique_labels) return; // Example check

  const index_t start = row_offsets[label_row];
  const index_t end = row_offsets[label_row + 1];

  // Each thread handles one index in the label list
  for (index_t i = threadIdx.x; i < (end - start); i += blockDim.x) {
    const index_t idx = indices[start + i]; // This is the vector_idx that has the label
    if (idx >= num_cols) continue;  // Check against n_database

    // Calculate position in bitmap for this query and this vector_idx
    const index_t word_idx = query_idx * words_per_row + (idx / (sizeof(bitmap_t) * 8));
    const unsigned bit_offset = idx % (sizeof(bitmap_t) * 8);

    // Set bit using atomic operation
    atomicOr(&bitmap[word_idx], bitmap_t(1) << bit_offset);
  }
}

void create_bitmap_filter_fast(raft::resources const& handle,
                               const std::vector<std::vector<int>>& label_data_vecs,
                               const raft::device_vector_view<const uint32_t, int64_t>& query_labels_view,
                               int64_t n_queries,
                               int64_t n_database,
                               const raft::device_matrix_view<uint32_t, int64_t>& bitmap) {

  // Calculate total size needed for indices based on label_data_vecs
  int64_t total_indices = 0;
  for (const auto& vec : label_data_vecs) {
    total_indices += vec.size();
  }

  // Create and fill row offsets on host
  std::vector<int64_t> h_row_offsets(label_data_vecs.size() + 1, 0);
  int64_t current_offset = 0;
  for (size_t i = 0; i < label_data_vecs.size(); i++) {
    h_row_offsets[i] = current_offset;
    current_offset += label_data_vecs[i].size();
  }
  h_row_offsets[label_data_vecs.size()] = current_offset;

  // Create indices array on host (vector IDs for each label)
  std::vector<int64_t> h_indices(total_indices);
  current_offset = 0;
  for (const auto& vec : label_data_vecs) {
    // Important: Convert int to int64_t if needed, ensure types match d_indices
    std::transform(vec.begin(), vec.end(), h_indices.begin() + current_offset,
                   [](int val){ return static_cast<int64_t>(val); });
    current_offset += vec.size();
  }

  // Copy CSR structure data to GPU
  auto d_row_offsets = raft::make_device_vector<int64_t, int64_t>(handle, h_row_offsets.size());
  auto d_indices = raft::make_device_vector<int64_t, int64_t>(handle, h_indices.size());

  raft::update_device(d_row_offsets.data_handle(),
                      h_row_offsets.data(),
                      h_row_offsets.size(),
                      raft::resource::get_cuda_stream(handle));

  raft::update_device(d_indices.data_handle(),
                      h_indices.data(),
                      h_indices.size(),
                      raft::resource::get_cuda_stream(handle));

  // Get the actual words_per_row from the bitmap view
  const int64_t words_per_row = bitmap.extent(1);

  // Zero initialize bitmap
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_queries * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(handle)));

  // Launch kernel
  const int block_size = 256;
  // GridDim is n_queries - each block handles one query
  create_bitmap_filter_kernel<<<n_queries, block_size, 0, raft::resource::get_cuda_stream(handle)>>>(
    d_row_offsets.data_handle(),
    d_indices.data_handle(),
    query_labels_view.data_handle(), // Pass device pointer
    bitmap.data_handle(),            // Pass device pointer
    n_queries,
    n_database,  // This is num_cols in the kernel
    words_per_row
  );

  raft::resource::sync_stream(handle); // Wait for kernel completion
}

// Function for CAGRA search with inline filtering
double cagra_search_inline_filtering(shared_resources::configured_raft_resources& dev_resources,
                                     cagra::index<float, uint32_t>& cagra_index,
                                     const raft::device_matrix_view<const float, int64_t>& queries,
                                     const raft::device_vector_view<const uint32_t, int64_t>& query_labels,
                                     const std::vector<std::vector<int>>& label_data_vecs,
                                     int itopk_size,
                                     int topk,
                                     int num_runs,
                                     int warmup_runs,
                                     raft::device_matrix_view<uint32_t, int64_t> filtered_neighbors) {

  int64_t n_queries = queries.extent(0);
  int64_t n_database = cagra_index.size();

  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;

  auto cagra_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);

  // Create bitmap for this iteration
  const int64_t bits_per_uint32 = sizeof(uint32_t) * 8;
  const int64_t words_per_row = (n_database + bits_per_uint32 - 1) / bits_per_uint32;
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, words_per_row);
  create_bitmap_filter_fast(dev_resources,
                            label_data_vecs,
                            query_labels,
                            n_queries,
                            n_database,
                            bitmap.view());
  auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(bitmap.data_handle(), n_queries, n_database);
  auto filter = cuvs::neighbors::filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);

  // --- Warmup Section ---
  for (int i = 0; i < warmup_runs; i++) {
    cagra::search(dev_resources,
                  search_params,
                  cagra_index,
                  queries,
                  filtered_neighbors,
                  cagra_distances.view(),
                  filter);
  }
  raft::resource::sync_stream(dev_resources);

  // --- Timed Section ---
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; i++) {
    cagra::search(dev_resources,
                  search_params,
                  cagra_index,
                  queries,
                  filtered_neighbors,
                  cagra_distances.view(),
                  filter);
  }
  raft::resource::sync_stream(dev_resources);
  auto end_time = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration<double>(end_time - start_time).count();
  double qps = num_runs * n_queries / total_time;

  return qps;
}

double cagra_search_with_post_processing(shared_resources::configured_raft_resources& dev_resources,
                                         cagra::index<float, uint32_t>& cagra_index,
                                         const raft::device_matrix_view<const float, int64_t>& queries,
                                         const raft::device_vector_view<const uint32_t, int64_t>& query_labels,
                                         const std::vector<std::vector<int>>& label_data_vecs,
                                         int num_labels,
                                         int itopk_size,
                                         int topk,
                                         int num_runs,
                                         int warmup_runs,
                                         raft::device_matrix_view<uint32_t, int64_t> filtered_neighbors) {

  auto stream = raft::resource::get_cuda_stream(dev_resources);
  int64_t n_queries = queries.extent(0);
  int64_t n_vectors = cagra_index.size();  // Use actual index size

  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, itopk_size);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, itopk_size);
  auto d_query_labels = raft::make_device_vector<uint32_t>(dev_resources, n_queries);
  raft::copy(d_query_labels.data_handle(), query_labels.data_handle(), n_queries, stream);

  constexpr size_t BITS_PER_WORD = 32;
  size_t NUM_WORDS = (num_labels + BITS_PER_WORD - 1) / BITS_PER_WORD;
  size_t total_label_words = (size_t)n_vectors * NUM_WORDS; // Use size_t

  std::vector<uint32_t> h_label_bits(total_label_words, 0);

  size_t max_vectors_for_labels = std::min(static_cast<size_t>(n_vectors), label_data_vecs.size());
  for (int label = 0; label < num_labels; label++) {
    if (label >= label_data_vecs.size()) continue; // Skip if label index is out of bounds for label_data_vecs
    const auto& vectors_with_this_label = label_data_vecs[label];
    size_t word_idx = label / BITS_PER_WORD;
    size_t bit_idx = label % BITS_PER_WORD;
    uint32_t bit_mask = 1u << bit_idx;
    for (int vector_idx : vectors_with_this_label) {
      if (vector_idx >= 0 && vector_idx < n_vectors) { // Check against n_vectors
        size_t array_idx = (size_t)vector_idx * NUM_WORDS + word_idx; // Cast vector_idx
        if (array_idx < h_label_bits.size()) {
          h_label_bits[array_idx] |= bit_mask;
        }
      }
    }
  }

  auto d_label_bits = raft::make_device_vector<uint32_t>(dev_resources, total_label_words);

  raft::copy(d_label_bits.data_handle(),
             h_label_bits.data(),
             h_label_bits.size(),
             stream);
  // Ensure copy is complete before kernel launch
  raft::resource::sync_stream(dev_resources);

  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;

  // Warmup runs
  for (int i = 0; i < warmup_runs; i++) {
    cagra::search(dev_resources, search_params, cagra_index, queries,
                  cagra_neighbors.view(), cagra_distances.view());
    int block_size = 256;
    int grid_size = (n_queries + block_size - 1) / block_size;
    filter_neighbors_kernel<<<grid_size, block_size, 0, stream>>>(
      cagra_neighbors.data_handle(),
      d_query_labels.data_handle(),
      d_label_bits.data_handle(),
      filtered_neighbors.data_handle(),
      n_queries,
      itopk_size,
      topk,
      num_labels,
      n_vectors
    );
    raft::resource::sync_stream(dev_resources);
  }
  raft::resource::sync_stream(dev_resources);

  // Timed runs
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; i++) {
    cagra::search(dev_resources, search_params, cagra_index, queries,
                  cagra_neighbors.view(), cagra_distances.view());
    int block_size = 256;
    int grid_size = (n_queries + block_size - 1) / block_size;
    filter_neighbors_kernel<<<grid_size, block_size, 0, stream>>>(
      cagra_neighbors.data_handle(),
      d_query_labels.data_handle(),
      d_label_bits.data_handle(),
      filtered_neighbors.data_handle(),
      n_queries,
      itopk_size,
      topk,
      num_labels,
      n_vectors
    );
    raft::resource::sync_stream(dev_resources);
  }
  raft::resource::sync_stream(dev_resources);
  auto end_time = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration<double>(end_time - start_time).count();
  double qps = num_runs * n_queries / total_time;

  return qps;
}


int main(int argc, char** argv) {
	// Check if config file is provided
	std::string config_file;
	if (argc < 3 || std::string(argv[1]) != "--config") {
		printf("No config file provided. Using default configuration file './config/default_config.json'.\n");
		config_file = "../src/bench/config.json";
	} else {
		config_file = argv[2];
	}

	// Variables to store configuration
	std::string data_dir;
	std::string data_fname;
	std::string query_fname;
	std::string data_label_fname;
	std::string query_label_fname;
	std::string ivf_graph_fname;
	std::string ivf_bfs_fname;
	std::string cagra_index_fname;
	std::string ground_truth_fname;
	std::vector<int> itopk_sizes;
	int specificity_threshold;
	int graph_degree;
	int topk;
	int num_runs;
	int warmup_runs;
	bool force_rebuild = false;
	std::vector<std::string> algorithms_to_run;
	std::string output_json_file;

	// Load configuration from file
	std::ifstream file(config_file);
	if (!file.is_open()) {
		fprintf(stderr, "Unable to open config file: %s\n", config_file.c_str());
		return 1;
	}

	try {
		printf("Loading configuration from %s\n", config_file.c_str());
		json config;
		file >> config;

		// Load all parameters directly from config
		data_dir = config["data_dir"];
		data_fname = config["data_fname"];
		query_fname = config["query_fname"];
		data_label_fname = config["data_label_fname"];
		query_label_fname = config["query_label_fname"];
		itopk_sizes = config["itopk_size"].get<std::vector<int>>();
		specificity_threshold = config["spec_threshold"];
		graph_degree = config["graph_degree"];
		topk = config["topk"];
		num_runs = config["num_runs"];
		warmup_runs = config["warmup_runs"];
		force_rebuild = config["force_rebuild"];

		ivf_graph_fname = config["ivf_graph_fname"];
		ivf_bfs_fname = config["ivf_bfs_fname"];
		cagra_index_fname = config["cagra_index_fname"];
		ground_truth_fname = config["ground_truth_fname"];

		// Load new parameters
		algorithms_to_run = config["algorithms_to_run"].get<std::vector<std::string>>();
		output_json_file = config["output_json_file"];

	} catch (const std::exception& e) {
		fprintf(stderr, "Error parsing JSON config file: %s\n", e.what());
		return 1;
	}

	// Check if itopk_sizes is empty
	if (itopk_sizes.empty()) {
		fprintf(stderr, "Error: 'itopk_size' array in config file is empty.\n");
		return 1;
	}
	// Sort itopk_sizes for potentially clearer output, though not strictly necessary
	std::sort(itopk_sizes.begin(), itopk_sizes.end());

	// Construct full file paths
	std::string full_data_fname = data_dir + data_fname;
	std::string full_query_fname = data_dir + query_fname;
	std::string full_data_label_fname = data_dir + data_label_fname;
	std::string full_query_label_fname = data_dir + query_label_fname;
	std::string full_ivf_graph_fname = data_dir + ivf_graph_fname;
	std::string full_ivf_bfs_fname = data_dir + ivf_bfs_fname;
	std::string full_cagra_index_fname = data_dir + cagra_index_fname;
	std::string full_ground_truth_fname = data_dir + ground_truth_fname;

	// Print configuration
	printf("\n=== Configuration ===\n");
	printf("iTopK sizes: [ ");
	for(int itopk : itopk_sizes) { printf("%d ", itopk); }
	printf("]\n");
	printf("Specificity threshold: %d\n", specificity_threshold);
	printf("Graph degree: %d\n", graph_degree);
	printf("TopK: %d\n", topk);
	printf("Number of runs: %d\n", num_runs);
	printf("Warmup runs: %d\n", warmup_runs);
	printf("Algorithms to run: [ ");
	for(const auto& algo : algorithms_to_run) { printf("%s ", algo.c_str()); }
	printf("]\n");
	printf("Output JSON file: %s\n", output_json_file.c_str());

	std::vector<float> h_data;
	std::vector<float> h_queries;
	std::vector<std::vector<int>> label_data_vecs;
	std::vector<std::vector<int>> data_label_vecs;
	std::vector<std::vector<int>> query_label_vecs;
	uint32_t N, Nq, dim;
	read_labeled_data<float, int64_t>(full_data_fname, full_data_label_fname, full_query_fname, full_query_label_fname,
                                    &h_data, &h_queries,
                                    &label_data_vecs, &data_label_vecs, &query_label_vecs,
                                    &N, &Nq, &dim);

	printf("\n=== Dataset Information ===\n");
	printf("Base dataset size: N=%u, dim=%u\n", N, dim);
	printf("Query dataset size: N=%u, dim=%u\n", Nq, dim);

	shared_resources::configured_raft_resources res;

	// Prepare device data
	auto stream = raft::resource::get_cuda_stream(res);
	auto d_data = raft::make_device_matrix<float, int64_t>(res, N, dim);
	raft::copy(d_data.data_handle(), h_data.data(), N * dim, stream);

	auto d_queries = raft::make_device_matrix<float, int64_t>(res, Nq, dim);
	raft::copy(d_queries.data_handle(), h_queries.data(), Nq * dim, stream);

	// Prepare query labels (taking the first label if available, UINT32_MAX otherwise)
	std::vector<uint32_t> h_query_labels(Nq);
	for (int64_t i = 0; i < Nq; ++i) {
		h_query_labels[i] = query_label_vecs[i].empty() ? UINT32_MAX : static_cast<uint32_t>(query_label_vecs[i][0]);
	}
	auto d_query_labels_main = raft::make_device_vector<uint32_t, int64_t>(res, Nq);
	raft::copy(d_query_labels_main.data_handle(), h_query_labels.data(), Nq, stream);

  auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
	generate_ground_truth(res,
												raft::make_const_mdspan(d_data.view()),
												raft::make_const_mdspan(d_queries.view()),
												label_data_vecs,
												query_label_vecs,
												gt_neighbors.view(),
												full_ground_truth_fname);

	// Initialize the JSON array for results
	json results_json = json::array();

	// VecFlow
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow") != algorithms_to_run.end()) {
		printf("\n=== VecFlow Index Building and Search ===\n");
		auto idx = vecflow::build(res,
                              raft::make_const_mdspan(d_data.view()),
                              data_label_vecs,
                              graph_degree,
                              specificity_threshold,
                              full_ivf_graph_fname,
                              full_ivf_bfs_fname,
                              force_rebuild);

		auto vecflow_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
		auto vecflow_distances = raft::make_device_matrix<float, int64_t>(res, Nq, topk);

		// ===== VecFlow Search Loop =====
		printf("\n=== VecFlow Search Benchmarking ===\n");
		for (int current_itopk : itopk_sizes) {
			printf("-- Running VecFlow Search (itopk=%d) --\n", current_itopk);
			// Warmup runs for VecFlow
			for (int i = 0; i < warmup_runs; i++) {
				vecflow::search(res,
                        idx,
                        raft::make_const_mdspan(d_queries.view()),
                        d_query_labels_main.view(),
                        current_itopk,
                        vecflow_neighbors.view(),
                        vecflow_distances.view());
				raft::resource::sync_stream(res);
			}
			// Timed runs for VecFlow
			auto start_time = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < num_runs; i++) {
				vecflow::search(res,
                        idx,
                        raft::make_const_mdspan(d_queries.view()),
                        d_query_labels_main.view(),
                        current_itopk,
                        vecflow_neighbors.view(),
                        vecflow_distances.view());
				raft::resource::sync_stream(res);
			}
			raft::resource::sync_stream(res);
			auto end_time = std::chrono::high_resolution_clock::now();

			auto total_time = std::chrono::duration<double>(end_time - start_time).count();
			double qps = num_runs * Nq / total_time;
			double recall = compute_recall(res, vecflow_neighbors.view(), gt_neighbors.view());
			printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
			results_json.push_back({{"algorithm", "vecflow"},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", recall}});
		}
	}

	// CAGRA Post-Processing
  cagra::index<float, uint32_t> cagra_index(res);
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "cagra_post_processing") != algorithms_to_run.end()) {
		// Build CAGRA index if not already built
		printf("\n=== Building CAGRA Index (for Post-Processing) ===\n");
		build_cagra_index(res, cagra_index, raft::make_const_mdspan(d_data.view()),
						          full_cagra_index_fname, graph_degree);

		printf("\n=== CAGRA Search with Post-Processing Benchmarking ===\n");
		auto filtered_neighbors_pp = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
		for (int current_itopk : itopk_sizes) {
			printf("-- Running CAGRA Search + PP (itopk=%d) --\n", current_itopk);
			double qps = cagra_search_with_post_processing(res, cagra_index,
										raft::make_const_mdspan(d_queries.view()),
										d_query_labels_main.view(), label_data_vecs,
										label_data_vecs.size(), current_itopk, topk,
										num_runs, warmup_runs, filtered_neighbors_pp.view());
			double recall = compute_recall(res, filtered_neighbors_pp.view(), gt_neighbors.view());
			printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
			results_json.push_back({{"algorithm", "cagra_post_processing"},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", recall}});
		}
	}

	// CAGRA Inline Filtering
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "cagra_inline_filtering") != algorithms_to_run.end()) {
    printf("\n=== Building CAGRA Index (for Inline Filtering) ===\n");
    build_cagra_index(res, cagra_index, raft::make_const_mdspan(d_data.view()),
              full_cagra_index_fname, graph_degree);

    printf("\n=== CAGRA Search with Inline Filtering Benchmarking ===\n");
    auto filtered_neighbors_inline = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
    for (int current_itopk : itopk_sizes) {
      printf("-- Running CAGRA Search + Inline Filter (itopk=%d) --\n", current_itopk);
      double qps = cagra_search_inline_filtering(res, cagra_index,
                            raft::make_const_mdspan(d_queries.view()),
                            d_query_labels_main.view(), label_data_vecs,
                            current_itopk, topk, num_runs, warmup_runs,
                            filtered_neighbors_inline.view());
      double recall = compute_recall(res, filtered_neighbors_inline.view(), gt_neighbors.view());
      printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
      results_json.push_back({{"algorithm", "cagra_inline_filtering"},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", recall}});
    }
	}

	// --- Write Results to JSON ---
	printf("\nWriting results to %s\n", output_json_file.c_str());
	std::ofstream output_file(output_json_file);
	if (output_file.is_open()) {
		output_file << results_json.dump(2); // Indent with 2 spaces
		output_file.close();
		printf("Results successfully written.\n");
	} else {
		fprintf(stderr, "Error: Unable to open output file: %s\n", output_json_file.c_str());
	}

	return 0;
}
