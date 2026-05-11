// SPDX-License-Identifier: Apache-2.0
//
// pybind11 entry point for the VecFlow Python wrapper. Exposes the
// PyVecFlow class (defined in include/vecflow.hpp, implemented in
// src/vecflow.cu) as `vecflow.VecFlow`.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vecflow.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vecflow, m) {
    m.doc() = R"pbdoc(
        VecFlow — label-aware ANN search on GPUs.

        Composes cuVS's CAGRA (high-specificity labels) and IVF-BFS
        (low-specificity labels) into a dual-structured index gated by
        a configurable specificity threshold. See the project README
        for the algorithm description.
    )pbdoc";

    py::class_<PyVecFlow>(m, "VecFlow", R"pbdoc(
        VecFlow index handle.

        Construct via ``VecFlow()`` (zero-arg). Then call ``build(...)``
        once to construct the dual-structured index, and ``search(...)``
        repeatedly to query it.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Create an empty VecFlow index. Call ``build(...)`` next.
        )pbdoc")

        .def("build", &PyVecFlow::build,
             py::arg("dataset"),
             py::arg("data_labels"),
             py::arg("graph_degree"),
             py::arg("specificity_threshold"),
             py::arg("graph_fname")          = "",
             py::arg("bfs_fname")            = "",
             py::arg("force_rebuild")        = false,
             py::arg("multi_label")          = false,
             R"pbdoc(
            Build (or reload from disk) the VecFlow dual index.

            Parameters
            ----------
            dataset : numpy.ndarray, shape (n_rows, dim), dtype float32
                Source vectors. Host or GPU memory both accepted.
            data_labels : list[list[int]]
                Per-row label list. ``data_labels[i]`` is the labels
                attached to ``dataset[i]``.
            graph_degree : int
                Degree of the underlying CAGRA graph (typical: 16-32).
            specificity_threshold : int
                Labels with at least this many points go to the IVF-CAGRA
                lane; rarer labels go to the IVF-BFS lane.
            graph_fname : str, optional
                Cache path for the IVF-CAGRA graph (default: no caching).
            bfs_fname : str, optional
                Cache path for the IVF-BFS index (default: no caching).
            force_rebuild : bool, optional
                If True, ignore cache files and rebuild from scratch.
            multi_label : bool, optional
                If True, also prepare the CSR ``dataset_labels`` /
                ``dataset_label_offsets`` buffers on the index so that
                ``search_multi(...)`` can run 2-label AND queries. Adds
                a one-time pass over ``data_labels`` at build time;
                zero overhead on the single-label ``search(...)`` path.
        )pbdoc")

        .def("search", &PyVecFlow::search,
             py::arg("queries"),
             py::arg("query_labels"),
             py::arg("itopk_size"),
             py::arg("topk")                 = 10,
             R"pbdoc(
            Run a label-aware top-k search.

            Parameters
            ----------
            queries : numpy.ndarray, shape (n_queries, dim), dtype float32
                Query vectors. Host or GPU memory both accepted.
            query_labels : numpy.ndarray, shape (n_queries,), dtype int32
                Label gating each query (-1 disables gating for that row).
            itopk_size : int
                Internal top-k buffer used by CAGRA's iterative search.
                Higher = better recall, slower; typical: 32-128.
            topk : int, optional
                Number of neighbors to return per query (default 10).

            Returns
            -------
            (neighbors, distances) : tuple[numpy.ndarray, numpy.ndarray]
                ``neighbors`` shape (n_queries, topk) dtype uint32.
                ``distances`` shape (n_queries, topk) dtype float32.
        )pbdoc")

        .def("search_multi", &PyVecFlow::search_multi,
             py::arg("queries"),
             py::arg("query_labels_a"),
             py::arg("query_labels_b"),
             py::arg("itopk_size"),
             py::arg("topk")                 = 10,
             R"pbdoc(
            Run a 2-label AND label-aware top-k search.

            A dataset point is a candidate only if it contains BOTH of the
            query's labels. The binding auto-selects which of the two labels
            has the larger specificity (more common label) and uses it as
            the primary IVF-list selector; the other is checked inline by
            the kernel. Order of `query_labels_a` vs `query_labels_b` is
            therefore irrelevant.

            Requires the index to have been built with ``multi_label=True``.

            Parameters
            ----------
            queries : numpy.ndarray, shape (n_queries, dim), dtype float32
            query_labels_a : numpy.ndarray, shape (n_queries,), dtype int32
                First label per query.
            query_labels_b : numpy.ndarray, shape (n_queries,), dtype int32
                Second label per query.
            itopk_size : int
                Internal top-k buffer for CAGRA. Higher = better recall, slower.
            topk : int, optional
                Neighbors returned per query (default 10).

            Returns
            -------
            (neighbors, distances) : tuple[numpy.ndarray, numpy.ndarray]
        )pbdoc")

        .def("generate_ground_truth", &PyVecFlow::generate_ground_truth,
             py::arg("dataset"),
             py::arg("queries"),
             py::arg("data_labels"),
             py::arg("query_labels"),
             py::arg("topk"),
             py::arg("gt_fname"),
             R"pbdoc(
            Generate ground-truth neighbors via brute-force, with the
            same label gating as ``search``. Used for recall benchmarking.

            Writes the result to ``gt_fname`` (.ibin format) and returns
            it as an array of shape (n_queries, topk) dtype uint32.
        )pbdoc");
}
