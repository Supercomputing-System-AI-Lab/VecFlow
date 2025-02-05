#include "vecflow.hpp"           // Include the declarations
#include "common.cuh"            // Other project-specific headers

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <rmm/device_vector.hpp>

#include <fstream>
#include <chrono>
#include <iostream>
#include <vector>
#include <atomic>
#include <omp.h>
#include <cstring>
#include <unordered_set>

//----------------------------------------------------------------------
// Implementation of helper functions and template routines
//----------------------------------------------------------------------

using namespace cuvs::neighbors;

// Template function to build and load the graphs/indexes.
template<typename T>
CombinedIndices build_load_graphs(raft::resources& res,
  raft::device_matrix_view<const T, int64_t> dataset,
  cagra::index<T, uint32_t>& cagra_index,
  ivf_flat::index<T, int64_t>& bfs_index,
  const std::vector<std::vector<int>>& label_data_vecs,
  const std::vector<int>& cat_freq,
  int graph_degree,
  int specificity_threshold,
  std::string graph_fname,
  std::string bfs_fname) {

  int dim = dataset.extent(1);
  int label_number = label_data_vecs.size();
  int64_t cagra_total_rows = 0;
  int64_t bfs_total_rows = 0;
  int cagra_labels = 0;
  int bfs_labels = 0;
  std::vector<uint32_t> host_cagra_label_size(label_number);
  std::vector<uint32_t> host_cagra_label_offset(label_number);
  std::vector<uint32_t> host_bfs_label_size(label_number);
  std::vector<uint32_t> host_bfs_label_offset(label_number);
  std::vector<uint32_t> host_cat_freq(label_number);

  for (uint32_t i = 0; i < label_number; i++) {
    host_cat_freq[i] = cat_freq[i];
    auto n_rows = label_data_vecs[i].size();
    if (cat_freq[i] > specificity_threshold) {
      host_cagra_label_size[i] = n_rows;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = 0;
      host_bfs_label_offset[i] = bfs_total_rows;
      cagra_total_rows += n_rows;
      cagra_labels++;
    } else {
      host_cagra_label_size[i] = 0;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = n_rows;
      host_bfs_label_offset[i] = bfs_total_rows;
      bfs_total_rows += n_rows;
      bfs_labels++;
    }
  }

  std::vector<uint32_t> host_cagra_index_map(cagra_total_rows);
  std::vector<uint32_t> host_bfs_index_map(bfs_total_rows);
  uint32_t bfs_iter = 0;
  uint32_t cagra_iter = 0;
  for (uint32_t i = 0; i < label_number; i++) {
    if (cat_freq[i] > specificity_threshold) {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j++) {
        host_cagra_index_map[cagra_iter++] = label_data_vecs[i][j];
      }
    } else {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j++) {
        host_bfs_index_map[bfs_iter++] = label_data_vecs[i][j];
      }
    }
  }

  auto cagra_index_map = raft::make_device_vector<uint32_t, int64_t>(res, cagra_total_rows);
  auto cagra_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto cagra_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto bfs_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto d_cat_freq = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

  raft::update_device(cagra_label_size.data_handle(),
                      host_cagra_label_size.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_label_offset.data_handle(),
                      host_cagra_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_index_map.data_handle(),
                      host_cagra_index_map.data(),
                      cagra_total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(bfs_label_size.data_handle(),
                      host_bfs_label_size.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(d_cat_freq.data_handle(),
                      host_cat_freq.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Index building: First, for the CAGRA (graph) part.
  if (cagra_labels > 0) {
    cagra_index.update_dataset(res, raft::make_const_mdspan(dataset));
    auto host_final_graph = raft::make_host_matrix<uint32_t, int64_t>(cagra_total_rows, graph_degree);
    std::ifstream graph_file(graph_fname);
    if (graph_file.good()) {
      std::cout << "Loading IVF-CAGRA index from " << graph_fname << std::endl;
      load_matrix_from_ibin(res, graph_fname, host_final_graph.view());
      cagra_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      graph_file.close();
    } else {
      std::cout << "Building IVF-CAGRA index from scratch ..." << std::endl;
      int optimal_threads = 32;
      omp_set_num_threads(optimal_threads);
      std::atomic<size_t> completed_work{0};

      #pragma omp parallel for num_threads(optimal_threads)
      for (int i = 0; i < label_number; i++) {
        if (host_cagra_label_size[i] == 0)
          continue;

        int thread_id = omp_get_thread_num();
        // (Set thread-specific resources if needed.)
        size_t label_size = label_data_vecs[i].size();
        #pragma omp critical
        {
          completed_work += label_size;
          float progress = (float)completed_work / cagra_total_rows * 100;
          std::cout << "\rProgress: " << progress << "% Label: " << i << " Size: " << label_size << std::flush;
        }

        // Create a filtered dataset for this label.
        auto filtered_dataset = raft::make_device_matrix<float, int64_t>(res, label_size, dim);
        for (size_t j = 0; j < label_size; j++) {
          raft::copy_async(filtered_dataset.data_handle() + j * dim,
                           dataset.data_handle() + (uint64_t)label_data_vecs[i][j] * dim,
                           dim,
                           raft::resource::get_cuda_stream(res));
        }
        raft::resource::sync_stream(res);

        cagra::index_params index_params;
        index_params.intermediate_graph_degree = graph_degree * 2;
        index_params.graph_degree = graph_degree;
        index_params.attach_dataset_on_build = false;
        auto index = cagra::build(res, index_params, raft::make_const_mdspan(filtered_dataset.view()));

        raft::copy(host_final_graph.data_handle() + host_cagra_label_offset[i] * graph_degree,
                   index.graph().data_handle(),
                   host_cagra_label_size[i] * graph_degree,
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
      }
      std::cout << "\nFinished building CAGRA index." << std::endl;
      cagra_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      if (!graph_fname.empty())
        save_matrix_to_ibin(graph_fname, host_final_graph.view());
      raft::resource::sync_stream(res);
    }
  }

  // Now for the BFS (IVF) part.
  if (bfs_labels > 0) {
    std::ifstream bfs_file(bfs_fname);
    if (bfs_file.good()) {
      std::cout << "Loading IVF-BFS index from " << bfs_fname << std::endl;
      ivf_flat::deserialize(res, bfs_fname, &bfs_index);
      bfs_file.close();
    } else {
      std::cout << "Building IVF-BFS index from scratch ..." << std::endl;
      auto bfs_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
      auto bfs_index_map = raft::make_device_vector<uint32_t, int64_t>(res, bfs_total_rows);
      raft::update_device(bfs_label_offset.data_handle(),
                          host_bfs_label_offset.data(),
                          label_number,
                          raft::resource::get_cuda_stream(res));
      raft::update_device(bfs_index_map.data_handle(),
                          host_bfs_index_map.data(),
                          bfs_total_rows,
                          raft::resource::get_cuda_stream(res));
      auto bfs_start_time = std::chrono::high_resolution_clock::now();
      build_filtered_IVF_index(res,
                               &bfs_index,
                               dataset,
                               bfs_index_map.view(),
                               bfs_label_size.view(),
                               bfs_label_offset.view());
      auto bfs_end_time = std::chrono::high_resolution_clock::now();
      auto bfs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bfs_end_time - bfs_start_time);
      std::cout << "IVF-BFS index building time: " << bfs_duration.count() << " ms" << std::endl;
      if (!bfs_fname.empty()) {
        ivf_flat::serialize(res, bfs_fname, bfs_index);
        std::cout << "Saving IVF-BFS index to " << bfs_fname << std::endl;
      }
    }
  }
  
  std::cout << "\nIVF-CAGRA Index Stats:" << std::endl;
  std::cout << "  Total vectors:  " << cagra_index.size() << std::endl;
  std::cout << "  Number of labels: " << cagra_labels << std::endl;
  std::cout << "  Graph size:     [" << cagra_index.graph().extent(0) << " × " 
            << cagra_index.graph().extent(1) << "]" << std::endl;
  std::cout << "  Graph degree:   " << cagra_index.graph_degree() << std::endl;
  
  std::cout << "\nIVF-BFS Index Stats:" << std::endl;
  std::cout << "  Number of labels: " << bfs_labels << std::endl;
  std::cout << "  Number of rows:  " << bfs_total_rows << std::endl;
  
  return CombinedIndices{
    std::move(cagra_index_map),
    std::move(cagra_label_size),
    std::move(cagra_label_offset),
    std::move(bfs_label_size),
    std::move(d_cat_freq)
  };
}

// Template function to perform the search. It takes device views for the queries,
// calls the appropriate search routines, and then fills the device matrices for the results.
template<typename T>
void search_main(raft::resources& res,
                 raft::device_matrix_view<const T, int64_t> queries,
                 raft::device_vector_view<uint32_t, int64_t> query_labels,
                 cagra::index<T, uint32_t>& cagra_index,
                 ivf_flat::index<T, int64_t>& bfs_index,
                 cagra::search_params& search_params,
                 CombinedIndices& metadata,
                 int specificity_threshold,
                 raft::device_matrix_view<uint32_t, int64_t> neighbors,
                 raft::device_matrix_view<float, int64_t> distances) {
  // Set the search algorithm.
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;

  // Classify the queries between those handled by CAGRA and those by BFS.
  auto query_info = classify_queries<T>(res,
                                        queries,
                                        query_labels,
                                        metadata.cat_freq.view(),
                                        specificity_threshold);

  int64_t topk = distances.extent(1);
  int n_cagra_queries = query_info.cagra_query_map.size();
  int n_bfs_queries = query_info.bfs_query_map.size();

  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(res, n_cagra_queries, topk);
  auto bfs_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_bfs_queries, topk);
  auto bfs_distances = raft::make_device_matrix<float, int64_t>(res, n_bfs_queries, topk);
  raft::resource::sync_stream(res);

  if (n_cagra_queries > 0) {
    cagra::filtered_search(res,
                           search_params,
                           cagra_index,
                           query_info.cagra_queries.view(),
                           cagra_neighbors.view(),
                           cagra_distances.view(),
                           query_info.cagra_query_labels.view(),
                           metadata.cagra_index_map.view(),
                           metadata.cagra_label_size.view(),
                           metadata.cagra_label_offset.view());
    raft::resource::sync_stream(res);
  }

  if (n_bfs_queries > 0) {
    search_filtered_ivf(res,
                        bfs_index,
                        raft::make_const_mdspan(query_info.bfs_queries.view()),
                        query_info.bfs_query_labels.view(),
                        metadata.bfs_label_size.view(),
                        bfs_neighbors.view(),
                        bfs_distances.view(),
                        cuvs::distance::DistanceType::L2Unexpanded);
    raft::resource::sync_stream(res);
  }
  
  merge_search_results<T>(res,
                          neighbors,
                          distances,
                          query_info,
                          bfs_neighbors.view(),
                          bfs_distances.view(),
                          cagra_neighbors.view(),
                          cagra_distances.view(),
                          topk);
}

// Explicit instantiation for float.
template CombinedIndices build_load_graphs<float>(
  raft::resources&,
  raft::device_matrix_view<const float, int64_t>,
  cagra::index<float, uint32_t>&,
  ivf_flat::index<float, int64_t>&,
  const std::vector<std::vector<int>>&,
  const std::vector<int>&,
  int,
  int,
  std::string,
  std::string);

template void search_main<float>(
  raft::resources&,
  raft::device_matrix_view<const float, int64_t>,
  raft::device_vector_view<uint32_t, int64_t>,
  cagra::index<float, uint32_t>&,
  ivf_flat::index<float, int64_t>&,
  cagra::search_params&,
  CombinedIndices&,
  int,
  raft::device_matrix_view<uint32_t, int64_t>,
  raft::device_matrix_view<float, int64_t>);

//----------------------------------------------------------------------
// Implementation of the PyVecFlow class methods
//----------------------------------------------------------------------

PyVecFlow::PyVecFlow() :
  res{}, 
  cagra_index(res),
  bfs_index(res, cuvs::distance::DistanceType::L2Unexpanded, 0, false, true, 0),
  metadata(res),
  dataset(raft::make_device_matrix<float, int64_t>(res, 0, 0)) {}

void PyVecFlow::load_data(py::array_t<float> dataset, py::object labels_obj, int num_labels) {
  // Get dataset dimensions.
  auto dataset_buf = dataset.request();
  int64_t rows = dataset_buf.shape[0];
  int64_t cols = dataset_buf.shape[1];

  // Convert the labels_obj to a py::list.
  py::list labels_list = labels_obj.cast<py::list>();
  if (labels_list.size() != rows) {
    throw std::runtime_error("The number of label entries must equal the number of data points");
  }

  // Initialize the label data vectors and category frequency counters with known size
  label_data_vecs.clear();
  label_data_vecs.resize(num_labels);
  cat_freq.clear();
  cat_freq.resize(num_labels, 0);

  // Directly build the inverted index
  for (int i = 0; i < rows; i++) {
    py::list inner = labels_list[i].cast<py::list>();
    for (size_t j = 0; j < inner.size(); j++) {
      int lbl = inner[j].cast<int>();
      if (lbl >= num_labels) {
        throw std::runtime_error("Label index exceeds the specified number of labels");
      }
      label_data_vecs[lbl].push_back(i);
      cat_freq[lbl]++;
    }
  }

  // Copy the dataset to device memory.
  this->dataset = raft::make_device_matrix<float, int64_t>(res, rows, cols);
  raft::copy(this->dataset.data_handle(),
             static_cast<float*>(dataset_buf.ptr),
             rows * cols,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
}

void PyVecFlow::build_load_index(int graph_degree, int specificity_threshold,
                                 std::string graph_fname, std::string bfs_fname) {

  metadata = build_load_graphs<float>(
    res,
    raft::make_const_mdspan(this->dataset.view()),
    cagra_index,
    bfs_index,
    label_data_vecs,
    cat_freq,
    graph_degree,
    specificity_threshold,
    graph_fname,
    bfs_fname
  );
}

std::tuple<py::array_t<uint32_t>, py::array_t<float>>
PyVecFlow::search(py::array_t<float> queries,
                  py::array_t<int> query_labels,
                  int itopk_size,
                  int specificity_threshold) {
  // Get queries info.
  auto queries_info = queries.request();
  int num_queries = queries_info.shape[0];
  int dim = queries_info.shape[1];

  // Copy queries to device.
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, num_queries, dim);
  raft::copy(d_queries.data_handle(),
             static_cast<float*>(queries_info.ptr),
             num_queries * dim,
             raft::resource::get_cuda_stream(res));

  // Copy query labels to device.
  auto query_labels_info = query_labels.request();
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, num_queries);
  raft::copy(d_query_labels.data_handle(),
             static_cast<uint32_t*>(query_labels_info.ptr),
             num_queries,
             raft::resource::get_cuda_stream(res));
  
  // Allocate device matrices for the output.
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, num_queries, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, num_queries, topk);

  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;

  // Call the search routine.
  search_main<float>(res, d_queries.view(), d_query_labels.view(),
                     cagra_index, bfs_index, search_params,
                     metadata, specificity_threshold,
                     d_neighbors.view(), d_distances.view());

  // Copy results from device to host.
  py::array_t<uint32_t> neighbors_array({num_queries, topk});
  py::array_t<float> distances_array({num_queries, topk});
  auto neigh_buf = neighbors_array.request();
  auto dist_buf = distances_array.request();
  raft::copy(static_cast<uint32_t*>(neigh_buf.ptr),
            d_neighbors.data_handle(),
            num_queries * topk,
            raft::resource::get_cuda_stream(res));
  raft::copy(static_cast<float*>(dist_buf.ptr),
            d_distances.data_handle(),
            num_queries * topk,
            raft::resource::get_cuda_stream(res));

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