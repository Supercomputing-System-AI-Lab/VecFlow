#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>

#include <chrono>

#include "common.cuh"

using namespace cuvs::neighbors;

int main(int argc, char** argv) {
  // Default values
  std::string data_dir = "../../data";
  int itopk_size = 32;
  int specificity_threshold = 1000;
  int graph_degree = 16;
  
  // Parse command line arguments
  for (int i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    if (arg == "--data_dir" && i + 1 < argc) data_dir = argv[i + 1];
    if (arg == "--itopk_size" && i + 1 < argc) itopk_size = std::stoi(argv[i + 1]);
    if (arg == "--spec_threshold" && i + 1 < argc) specificity_threshold = std::stoi(argv[i + 1]);
    if (arg == "--graph_degree" && i + 1 < argc) graph_degree = std::stoi(argv[i + 1]);
  }
  
  // Construct file paths
  std::string base_path = data_dir + "/sift1M";
  std::string data_fname = base_path + "/sift.base.fbin";
  std::string query_fname = base_path + "/sift.query.fbin";
	std::string data_label_fname = base_path + "/sift.base.spmat";
  std::string query_label_fname = base_path + "/sift.query.spmat";

	// Check if the binary spmat files exist and convert from text format if needed
	std::ifstream datafile(data_label_fname);
	if (!datafile) {
		std::string txt_file = base_path + "/sift.base.txt";
		txt2spmat(txt_file, data_label_fname);
	}
	std::ifstream queryfile(query_label_fname);
	if (!queryfile) {
		std::string txt_file = base_path + "/sift.query.txt";
		txt2spmat(txt_file, query_label_fname);
	}
  
  std::string graph_fname = base_path + "/graph_" + 
                          std::to_string(graph_degree * 2) + "_" + 
                          std::to_string(graph_degree) + "_spec_" +
                          std::to_string(specificity_threshold) + ".bin";
  std::string bfs_fname = base_path + "/spec_" + 
                          std::to_string(specificity_threshold) + ".bin";
  
  std::string gt_fname = base_path + "/sift.groundtruth.neighbors.ibin";

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
  // Warmup runs
  printf("\n=== Performing Search ===\n");
  for (int i = 0; i < 10; i++) {
    vecflow::search(idx,
                    raft::make_const_mdspan(queries.view()),
                    query_labels.view(),
                    itopk_size,
                    neighbors.view(),
                    distances.view());
		raft::resource::sync_stream(idx.res);
  }
  // Timed runs
  int num_searches = 1000;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_searches; i++) {
    vecflow::search(idx,
                    raft::make_const_mdspan(queries.view()),
                    query_labels.view(),
                    itopk_size,
                    neighbors.view(),
                    distances.view());
		raft::resource::sync_stream(idx.res);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration<double>(end_time - start_time).count();
  double avg_ms = (total_time * 1000.0) / num_searches;
  double qps = num_searches * queries.extent(0) / total_time;
  printf("Search timing (%d runs):\n", num_searches);
  printf("- Total time: %.2f ms\n", total_time * 1000.0);
  printf("- Average per search: %.2f ms\n", avg_ms);
  printf("- QPS: %.2f\n", qps);
  
  auto gt_neighbors = raft::make_device_matrix<uint32_t, int64_t>(idx.res, queries.extent(0), topk);
  auto gt_distances = raft::make_device_matrix<float, int64_t>(idx.res, queries.extent(0), topk);
  generate_ground_truth(idx.res,
                        raft::make_const_mdspan(dataset.view()),
                        raft::make_const_mdspan(queries.view()),
                        label_data_vecs,
                        query_label_vecs,
                        gt_neighbors.view(),
                        gt_distances.view(),
                        gt_fname);
  compute_recall(idx.res, neighbors.view(), gt_neighbors.view());
}
