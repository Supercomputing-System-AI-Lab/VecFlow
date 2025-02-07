#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuvs/neighbors/vecflow.hpp>

namespace py = pybind11;
using namespace cuvs::neighbors;

class PyVecFlow {

private:
  vecflow::index<float> idx;

public:
  PyVecFlow();

  void build(py::array_t<float> dataset,
            std::string data_label_fname,
            int graph_degree,
            int specificity_threshold,
            std::string graph_fname = "",
            std::string bfs_fname = "");
  
  std::tuple<py::array_t<uint32_t>, py::array_t<float>> 
  search(py::array_t<float> queries,
        py::array_t<int> query_labels,
        int itopk_size);
  
  float compute_recall(py::array_t<uint32_t> neighbors, py::array_t<uint32_t> gt_indices);
};
