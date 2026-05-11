#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <memory>

namespace py = pybind11;
using namespace cuvs::neighbors;

class PyVecFlow {

private:
  shared_resources::configured_raft_resources res;
  std::unique_ptr<vecflow::index<float>> idx;

public:
  PyVecFlow() {
  }

  void build(py::array_t<float> dataset,
             std::vector<std::vector<int>> data_labels,
             int graph_degree,
             int specificity_threshold,
             std::string graph_fname = "",
             std::string bfs_fname = "",
             bool force_rebuild = false,
             bool multi_label = false);

  std::tuple<py::array_t<uint32_t>, py::array_t<float>>
  search(py::array_t<float> queries,
         py::array_t<int> query_labels,
         int itopk_size,
         int topk = 10);

  // Multi-label AND search. Each query carries two labels (the order is
  // irrelevant — the C++ side auto-picks the larger-cat_freq label as
  // primary). Requires the index to have been built with multi_label=true.
  std::tuple<py::array_t<uint32_t>, py::array_t<float>>
  search_multi(py::array_t<float> queries,
               py::array_t<int> query_labels_a,
               py::array_t<int> query_labels_b,
               int itopk_size,
               int topk = 10);

  py::array_t<uint32_t>
  generate_ground_truth(py::array_t<float> dataset,
                        py::array_t<float> queries,
                        std::vector<std::vector<int>> data_labels,
                        std::vector<std::vector<int>> query_labels,
                        int topk,
                        std::string gt_fname);
};
