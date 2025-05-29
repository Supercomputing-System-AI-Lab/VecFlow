#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vecflow.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vecflow, m) {
	py::class_<PyVecFlow>(m, "VecFlow")
		.def(py::init<>())
		.def("build", &PyVecFlow::build,
			py::arg("dataset"),
			py::arg("data_labels"),
			py::arg("graph_degree"),
			py::arg("specificity_threshold"),
			py::arg("graph_fname") = "",
			py::arg("bfs_fname") = "",
			py::arg("force_rebuild") = false)
		.def("search", &PyVecFlow::search,
			py::arg("queries"),
			py::arg("query_labels"),
			py::arg("itopk_size"),
			py::arg("topk") = 10)
		.def("generate_ground_truth", &PyVecFlow::generate_ground_truth,
			py::arg("dataset"),
			py::arg("queries"),
			py::arg("data_labels"),
			py::arg("query_labels"),
			py::arg("topk"),
			py::arg("gt_fname"));
}
