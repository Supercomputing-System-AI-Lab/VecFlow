#include <pybind11/pybind11.h>
#include "vecflow.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vecflow, m) {
    py::class_<PyVecFlow>(m, "VecFlow")
	   .def(py::init<>())
		 .def("load_data", &PyVecFlow::load_data,
		    py::arg("dataset"),
				py::arg("labels"),
				py::arg("num_labels"))
	   .def("build", &PyVecFlow::build_load_index,
		    py::arg("graph_degree"),
				py::arg("specificity_threshold"),
				py::arg("graph_fname") = "",
				py::arg("bfs_fname") = "")
	   .def("search", &PyVecFlow::search,
		   py::arg("queries"),
		   py::arg("query_labels"),
		   py::arg("itopk_size"),
		   py::arg("specificity_threshold"))
	   .def("compute_recall", &PyVecFlow::compute_recall,
		   py::arg("neighbors"),
		   py::arg("gt_indices"));
}