#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <raft/core/device_mdarray.hpp>
#include <rmm/device_vector.hpp>
#include "shared_resources.hpp"

namespace py = pybind11;

// Add a constructor so that you can initialize CombinedIndices from a raft::resources object.
struct CombinedIndices {
  raft::device_vector<uint32_t, int64_t> cagra_index_map;
  raft::device_vector<uint32_t, int64_t> cagra_label_size;
  raft::device_vector<uint32_t, int64_t> cagra_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  raft::device_vector<uint32_t, int64_t> cat_freq;

  // Constructor: initialize each device_vector with an initial size of 0.
  CombinedIndices(raft::resources& res)
    : cagra_index_map(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cagra_label_size(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cagra_label_offset(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      bfs_label_size(raft::make_device_vector<uint32_t, int64_t>(res, 0)),
      cat_freq(raft::make_device_vector<uint32_t, int64_t>(res, 0))
  {}

  // Overload constructor for aggregate return.
  CombinedIndices(raft::device_vector<uint32_t, int64_t>&& c_idx_map,
                  raft::device_vector<uint32_t, int64_t>&& c_label_size,
                  raft::device_vector<uint32_t, int64_t>&& c_label_offset,
                  raft::device_vector<uint32_t, int64_t>&& b_label_size,
                  raft::device_vector<uint32_t, int64_t>&& cat_f)
    : cagra_index_map(std::move(c_idx_map)),
      cagra_label_size(std::move(c_label_size)),
      cagra_label_offset(std::move(c_label_offset)),
      bfs_label_size(std::move(b_label_size)),
      cat_freq(std::move(cat_f))
  {}
};

class PyVecFlow {
private:
  shared_resources::configured_raft_resources res;
  cuvs::neighbors::cagra::index<float, uint32_t> cagra_index;
  cuvs::neighbors::ivf_flat::index<float, int64_t> bfs_index;
  CombinedIndices metadata;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<int> cat_freq;
  raft::device_matrix<float, int64_t> dataset;
  int topk = 10;

public:
  PyVecFlow();

  void load_data(py::array_t<float> dataset, py::object labels_obj, int num_labels);
  void build_load_index(int graph_degree,
                        int specificity_threshold,
                        std::string graph_fname = "",
                        std::string bfs_fname = "");
  std::tuple<py::array_t<uint32_t>, py::array_t<float>> search(py::array_t<float> queries,
                                                              py::array_t<int> query_labels,
                                                              int itopk_size,
                                                              int specificity_threshold);
  float compute_recall(py::array_t<uint32_t> neighbors, py::array_t<uint32_t> gt_indices);
};
