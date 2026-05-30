// SPDX-License-Identifier: Apache-2.0
//
// pybind11 entry point for the vecflow-chamfer Python wrapper. Mirrors the
// C++ shape: explicit Dataset / QuerySet / Index / IndexParams / SearchParams
// classes plus free build()/search()/serialize()/deserialize() functions.
// The single-class PyVecFlow pattern used by vecflow's binding is a poor fit
// here because chamfer's search reads dataset.doc_embeddings independently of
// the index, so the dataset/index split is real and worth preserving.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <vecflow_chamfer/vecflow_chamfer.hpp>
#include <vecflow_chamfer/kernels.hpp>

namespace py = pybind11;
namespace vc = vecflow_chamfer;

namespace {

// Lazy module-level GPU context. The RMM pool MUST be installed before
// raft::device_resources is constructed — without it stage-1 search
// regresses ~2.5× (per the vecflow-chamfer perf notes / example main()).
// Function-local statics are initialized in declaration order on first call,
// so pool-install runs before the device_resources constructor here.
raft::device_resources& gpu_res() {
  static int _install_pool = [] {
    rmm::mr::set_current_device_resource(
      rmm::mr::pool_memory_resource(
        rmm::mr::get_current_device_resource_ref(),
        2ull * 1024 * 1024 * 1024));  // 2 GiB
    return 0;
  }();
  (void)_install_pool;
  static raft::device_resources res;
  return res;
}

// Owning wrapper around vecflow_chamfer::dataset. release_dataset() picks the
// right deallocator (free vs cudaFree) based on the alloc flag the loader set.
struct PyDataset {
  vc::dataset ds;
  PyDataset() = default;
  ~PyDataset() { vc::release_dataset(ds); }
  PyDataset(const PyDataset&)            = delete;
  PyDataset& operator=(const PyDataset&) = delete;
  PyDataset(PyDataset&&)                 = default;
  PyDataset& operator=(PyDataset&&)      = default;
};

// Move-only wrapper around vecflow_chamfer::index. The underlying type holds
// a cuvs::neighbors::cagra::index which is move-only by design.
struct PyIndex {
  vc::index idx;
  explicit PyIndex(vc::index&& i) : idx(std::move(i)) {}
  PyIndex(const PyIndex&)            = delete;
  PyIndex& operator=(const PyIndex&) = delete;
  PyIndex(PyIndex&&)                 = default;
  PyIndex& operator=(PyIndex&&)      = default;
};

// Plain holder for query_set — std::vector<__half> on host, trivially destructed.
struct PyQuerySet {
  vc::query_set qs;
};

// Copy a row-major device matrix back to a numpy array, sync, return.
template <typename T>
py::array_t<T> device_matrix_to_numpy(raft::device_resources const& res,
                                       T const* d_ptr, int64_t rows, int64_t cols) {
  py::array_t<T> arr({rows, cols});
  cudaMemcpyAsync(arr.mutable_data(), d_ptr,
                  static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(T),
                  cudaMemcpyDeviceToHost,
                  raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return arr;
}

}  // namespace

PYBIND11_MODULE(_vecflow_chamfer, m) {
  m.doc() = R"pbdoc(
    vecflow_chamfer — multi-vector (ColBERT-style) retrieval on GPUs with a
    memory hierarchy (HBM next to GPU-addressable host RAM).

    Pipeline: MaxIVF anchor index (CAGRA-routed) → optional VPQ rerank →
    full-precision Chamfer rerank → top-K.
  )pbdoc";

  // ── Dataset ────────────────────────────────────────────────────────────
  py::class_<PyDataset>(m, "Dataset", R"pbdoc(
    Document set used by both indexing and search.

    Holds the per-token fp16 embedding buffer plus CSR-style offsets. The
    buffer lives in host RAM with a CUDA mem-advise hint so the GPU can
    reach it directly (over C2C on GH200, or as managed memory on PCIe-attached
    GPUs — selected automatically at load time).

    Construct via ``Dataset.load(dataset_dir, prefix)``.
  )pbdoc")
    .def_static("load",
      [](const std::string& dir, const std::string& prefix) {
        auto p = std::make_unique<PyDataset>();
        p->ds  = vc::load_dataset(dir, prefix);
        return p;
      },
      py::arg("dataset_dir"), py::arg("prefix"),
      R"pbdoc(
        Load doc embeddings + offsets from ``<dir>/<prefix>.doc.{embeddings.fp16,offsets}.{fbin,bin}``.
      )pbdoc")
    .def_property_readonly("num_docs",       [](const PyDataset& p) { return p.ds.num_docs; })
    .def_property_readonly("num_doc_tokens", [](const PyDataset& p) { return p.ds.num_doc_tokens; })
    .def_property_readonly("embedding_dim",  [](const PyDataset& p) { return p.ds.embedding_dim; });

  // ── QuerySet ──────────────────────────────────────────────────────────
  py::class_<PyQuerySet>(m, "QuerySet", R"pbdoc(
    Query set: search-only. fp16 embeddings, fixed tokens-per-query (32 by default).
  )pbdoc")
    .def_static("load",
      [](const std::string& dir, const std::string& prefix,
         int embedding_dim, int num_tokens_per_query) {
        PyQuerySet p;
        p.qs = vc::load_queries(dir, prefix, embedding_dim, num_tokens_per_query);
        return p;
      },
      py::arg("dataset_dir"), py::arg("prefix"),
      py::arg("embedding_dim"), py::arg("num_tokens_per_query") = 32,
      R"pbdoc(
        Load query embeddings from ``<dir>/<prefix>.query.embeddings.fp16.fbin``.
      )pbdoc")
    .def_property_readonly("num_queries",          [](const PyQuerySet& p) { return p.qs.num_queries; })
    .def_property_readonly("num_tokens_per_query", [](const PyQuerySet& p) { return p.qs.num_tokens_per_query; })
    .def_property_readonly("embedding_dim",        [](const PyQuerySet& p) { return p.qs.embedding_dim; });

  // ── IndexParams ───────────────────────────────────────────────────────
  py::class_<vc::index_params>(m, "IndexParams", R"pbdoc(
    MaxIVF build-time parameters.

    ``pq_bits=0`` disables VPQ side-encoding (default). When ``pq_bits > 0``
    a PQ codebook is trained over per-token residuals against the anchors.
    ``pq_dim`` must divide ``embedding_dim``.
  )pbdoc")
    .def(py::init([](int n_anchors, int graph_degree, int itopk_size, int n_iter,
                     uint32_t pq_bits, uint32_t pq_dim) {
        vc::index_params p;
        p.n_anchors    = n_anchors;
        p.graph_degree = graph_degree;
        p.itopk_size   = itopk_size;
        p.n_iter       = n_iter;
        p.pq_bits      = pq_bits;
        p.pq_dim       = pq_dim;
        return p;
      }),
      py::arg("n_anchors"), py::arg("graph_degree"),
      py::arg("itopk_size"), py::arg("n_iter"),
      py::arg("pq_bits") = 0u, py::arg("pq_dim") = 0u)
    .def_readwrite("n_anchors",    &vc::index_params::n_anchors)
    .def_readwrite("graph_degree", &vc::index_params::graph_degree)
    .def_readwrite("itopk_size",   &vc::index_params::itopk_size)
    .def_readwrite("n_iter",       &vc::index_params::n_iter)
    .def_readwrite("pq_bits",      &vc::index_params::pq_bits)
    .def_readwrite("pq_dim",       &vc::index_params::pq_dim);

  // ── SearchParams ──────────────────────────────────────────────────────
  py::class_<vc::search_params>(m, "SearchParams", R"pbdoc(
    MaxIVF search-time parameters. ``use_vpq=True`` requires that the index
    was built with ``pq_bits > 0``; otherwise the VPQ stages are skipped.
  )pbdoc")
    .def(py::init([](uint32_t itopk, uint32_t search_topk, float refine_rate,
                     uint32_t final_topk, bool use_vpq, float refine_rate_vpq) {
        vc::search_params p;
        p.itopk           = itopk;
        p.search_topk     = search_topk;
        p.refine_rate     = refine_rate;
        p.final_topk      = final_topk;
        p.use_vpq         = use_vpq;
        p.refine_rate_vpq = refine_rate_vpq;
        return p;
      }),
      py::arg("itopk"), py::arg("search_topk"),
      py::arg("refine_rate"), py::arg("final_topk"),
      py::arg("use_vpq") = false, py::arg("refine_rate_vpq") = 1.5f)
    .def_readwrite("itopk",           &vc::search_params::itopk)
    .def_readwrite("search_topk",     &vc::search_params::search_topk)
    .def_readwrite("refine_rate",     &vc::search_params::refine_rate)
    .def_readwrite("final_topk",      &vc::search_params::final_topk)
    .def_readwrite("use_vpq",         &vc::search_params::use_vpq)
    .def_readwrite("refine_rate_vpq", &vc::search_params::refine_rate_vpq);

  // ── Index ─────────────────────────────────────────────────────────────
  // Opaque handle. Construction goes through build() or deserialize().
  py::class_<PyIndex>(m, "Index", R"pbdoc(
    MaxIVF index handle. Opaque; construct via ``build(...)`` or ``deserialize(...)``.
  )pbdoc");

  // ── SearchStats ───────────────────────────────────────────────────────
  py::class_<vc::search_stats>(m, "SearchStats", R"pbdoc(
    Per-stage timing breakdown produced when ``search(..., with_stats=True)``.
    All ``avg_*_ms`` fields are per-query means.
  )pbdoc")
    .def_readonly("avg_candidate_generation_ms",     &vc::search_stats::avg_candidate_generation_ms)
    .def_readonly("avg_unique_id_finding_ms",        &vc::search_stats::avg_unique_id_finding_ms)
    .def_readonly("avg_anchor_chamfer_reranking_ms", &vc::search_stats::avg_anchor_chamfer_reranking_ms)
    .def_readonly("avg_first_sorting_ms",            &vc::search_stats::avg_first_sorting_ms)
    .def_readonly("avg_vpq_reranking_ms",            &vc::search_stats::avg_vpq_reranking_ms)
    .def_readonly("avg_vpq_sorting_ms",              &vc::search_stats::avg_vpq_sorting_ms)
    .def_readonly("avg_full_chamfer_reranking_ms",   &vc::search_stats::avg_full_chamfer_reranking_ms)
    .def_readonly("avg_second_sorting_ms",           &vc::search_stats::avg_second_sorting_ms)
    .def_readonly("avg_candidates_per_query",        &vc::search_stats::avg_candidates_per_query)
    .def_readonly("avg_per_query_ms",                &vc::search_stats::avg_per_query_ms)
    .def_readonly("total_wall_ms",                   &vc::search_stats::total_wall_ms)
    .def_readonly("throughput_qps",                  &vc::search_stats::throughput_qps)
    .def_readonly("avg_stage4_input_count",          &vc::search_stats::avg_stage4_input_count)
    .def_readonly("avg_stage4_output_count",         &vc::search_stats::avg_stage4_output_count)
    .def_readonly("avg_stage_vpq_output_count",      &vc::search_stats::avg_stage_vpq_output_count)
    .def_readonly("avg_stage6_output_count",         &vc::search_stats::avg_stage6_output_count);

  // ── build() ───────────────────────────────────────────────────────────
  m.def("build",
    [](PyDataset& ds, const vc::index_params& params) {
      return std::make_unique<PyIndex>(vc::build(gpu_res(), ds.ds, params));
    },
    py::arg("dataset"), py::arg("params"),
    R"pbdoc(
      Build a MaxIVF index over ``dataset`` with the given build params.
      Returns an opaque ``Index``; persist with ``serialize()``.
    )pbdoc");

  // ── search() ──────────────────────────────────────────────────────────
  m.def("search",
    [](PyIndex& idx, PyDataset& ds, PyQuerySet& qs,
       const vc::search_params& params, bool with_stats) -> py::object {
      auto& res = gpu_res();
      const int64_t n_queries = qs.qs.num_queries;
      const int64_t k         = params.final_topk;

      auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
      auto d_distances = raft::make_device_matrix<float,    int64_t>(res, n_queries, k);

      if (with_stats) {
        vc::search_stats stats;
        vc::search(res, params, idx.idx, ds.ds, qs.qs,
                   d_neighbors.view(), d_distances.view(), stats);
        auto h_neighbors = device_matrix_to_numpy(res, d_neighbors.data_handle(), n_queries, k);
        auto h_distances = device_matrix_to_numpy(res, d_distances.data_handle(), n_queries, k);
        return py::make_tuple(std::move(h_neighbors), std::move(h_distances), std::move(stats));
      } else {
        vc::search(res, params, idx.idx, ds.ds, qs.qs,
                   d_neighbors.view(), d_distances.view());
        auto h_neighbors = device_matrix_to_numpy(res, d_neighbors.data_handle(), n_queries, k);
        auto h_distances = device_matrix_to_numpy(res, d_distances.data_handle(), n_queries, k);
        return py::make_tuple(std::move(h_neighbors), std::move(h_distances));
      }
    },
    py::arg("index"), py::arg("dataset"), py::arg("queries"),
    py::arg("params"), py::arg("with_stats") = false,
    R"pbdoc(
      Run MaxIVF search and return ``(neighbors, distances)`` as numpy arrays of
      shape ``(n_queries, params.final_topk)``. If ``with_stats=True``, returns
      a 3-tuple with an additional ``SearchStats`` instance.

      Rows that produced fewer than ``final_topk`` valid hits are padded with
      ``UINT32_MAX`` doc IDs and ``-FLT_MAX`` scores. Doc IDs are in descending
      score order.
    )pbdoc");

  // ── serialize() / deserialize() ───────────────────────────────────────
  m.def("serialize",
    [](PyIndex& idx, const std::string& stem) {
      vc::serialize(gpu_res(), stem, idx.idx);
    },
    py::arg("index"), py::arg("stem"),
    R"pbdoc(
      Persist ``index`` to ``<stem>.maxivf`` + ``<stem>.meta``.
    )pbdoc");

  m.def("deserialize",
    [](const std::string& stem) {
      return std::make_unique<PyIndex>(vc::deserialize(gpu_res(), stem));
    },
    py::arg("stem"),
    R"pbdoc(
      Load an index previously written by ``serialize()``. Both ``<stem>.maxivf``
      and ``<stem>.meta`` must be present.
    )pbdoc");

  // ── cuda_sync() ───────────────────────────────────────────────────────
  m.def("cuda_sync",
    []() {
      auto err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
      }
    },
    R"pbdoc(
      Block until all GPU work issued so far has completed. Useful for getting
      a clean wall-clock measurement around ``search()`` / ``build()`` etc.
    )pbdoc");

  // ── ChamferKernelBench ────────────────────────────────────────────────
  // Holds *non-owning* device pointers for the kernel's inputs, sourced from
  // any objects implementing __cuda_array_interface__ (cupy, torch w/ CUDA,
  // numba device arrays). The caller is responsible for keeping the source
  // arrays alive — we stash py::object references so Python won't GC them
  // while the bench exists, even if the user drops their local handle.
  //
  // The output scores buffer is the only piece this class allocates; it's
  // the one thing the kernel writes that has no natural source array.
  struct PyChamferKernelBench {
    const __half*   d_query        = nullptr;
    const __half*   d_dataset      = nullptr;
    const uint32_t* d_doc_offsets  = nullptr;
    const uint32_t* d_doc_ids      = nullptr;
    float*          d_scores       = nullptr;
    uint32_t  n_docs            = 0;
    uint32_t  tokens_per_query  = 0;
    uint32_t  dim               = 0;

    // Keepalives: hold a strong ref to each input so the caller can drop the
    // local variable and the underlying device memory still won't be freed.
    py::object _keep_query, _keep_dataset, _keep_doc_offsets, _keep_doc_ids;

    PyChamferKernelBench()                                       = default;
    PyChamferKernelBench(const PyChamferKernelBench&)            = delete;
    PyChamferKernelBench& operator=(const PyChamferKernelBench&) = delete;

    ~PyChamferKernelBench() {
      if (d_scores) cudaFree(d_scores);
      // Input device memory is owned by the python-side arrays we kept alive.
    }
  };

  py::class_<PyChamferKernelBench>(m, "ChamferKernelBench", R"pbdoc(
    Bench harness for the raw chamfer_score kernel.

    All four input arrays must already live on the GPU. The class accepts
    any object implementing the ``__cuda_array_interface__`` protocol —
    typically cupy arrays, but torch CUDA tensors and numba device arrays
    work the same way. Construction does *no* H→D copy; pointers are
    extracted from the interface dict and used directly. ``run()`` is
    kernel-only. ``scores()`` does one D→H to return the latest result.

    The bench keeps a strong reference to each input array, so the source
    objects won't be GC'd (and their device memory freed) while the bench
    is alive.
  )pbdoc")
    .def(py::init([](py::object query,
                     py::object dataset,
                     py::object doc_offsets,
                     py::object doc_ids,
                     uint32_t tokens_per_query,
                     uint32_t dim) {
        // ---- helpers ---------------------------------------------------------
        auto fetch_cai = [](const py::object& o, const char* name) -> py::dict {
          if (!py::hasattr(o, "__cuda_array_interface__")) {
            throw std::invalid_argument(
              std::string(name) + " must be a GPU array (cupy / torch CUDA / numba device) "
                                  "providing __cuda_array_interface__");
          }
          return o.attr("__cuda_array_interface__").cast<py::dict>();
        };

        // typestr canonical forms: '<f2', '=f2', or 'f2' for fp16; '<u4', '=u4', 'u4' for uint32.
        auto typestr_matches = [](const std::string& s, char kind, int bytes) -> bool {
          const char* p = s.c_str();
          if (*p == '<' || *p == '>' || *p == '=' || *p == '|') ++p;
          if (*p != kind) return false;
          ++p;
          return std::atoi(p) == bytes;
        };

        struct CaiView {
          uintptr_t ptr;
          std::vector<py::ssize_t> shape;
          std::string typestr;
          bool readonly;
        };

        auto parse_cai = [&](const py::dict& d, const char* name) -> CaiView {
          CaiView v;
          auto shape_t = d["shape"].cast<py::tuple>();
          v.shape.reserve(shape_t.size());
          for (auto x : shape_t) v.shape.push_back(x.cast<py::ssize_t>());
          v.typestr = d["typestr"].cast<std::string>();
          auto data_t = d["data"].cast<py::tuple>();
          v.ptr      = data_t[0].cast<uintptr_t>();
          v.readonly = data_t.size() > 1 ? data_t[1].cast<bool>() : false;
          // strides == None means C-contiguous; reject explicit strides because the
          // kernel reads packed buffers.
          if (d.contains("strides") && !d["strides"].is_none()) {
            throw std::invalid_argument(
              std::string(name) + " must be C-contiguous (strides=None in __cuda_array_interface__)");
          }
          if (v.ptr == 0) {
            throw std::invalid_argument(std::string(name) + " has a null device pointer");
          }
          return v;
        };

        // ---- parse ----------------------------------------------------------
        CaiView q   = parse_cai(fetch_cai(query,       "query"),       "query");
        CaiView ds  = parse_cai(fetch_cai(dataset,     "dataset"),     "dataset");
        CaiView off = parse_cai(fetch_cai(doc_offsets, "doc_offsets"), "doc_offsets");
        CaiView ids = parse_cai(fetch_cai(doc_ids,     "doc_ids"),     "doc_ids");

        // ---- validate -------------------------------------------------------
        if (!typestr_matches(q.typestr,   'f', 2)) throw std::invalid_argument("query must be float16");
        if (!typestr_matches(ds.typestr,  'f', 2)) throw std::invalid_argument("dataset must be float16");
        if (!typestr_matches(off.typestr, 'u', 4)) throw std::invalid_argument("doc_offsets must be uint32");
        if (!typestr_matches(ids.typestr, 'u', 4)) throw std::invalid_argument("doc_ids must be uint32");

        if (q.shape.size()  != 2) throw std::invalid_argument("query must be 2-D");
        if (ds.shape.size() != 2) throw std::invalid_argument("dataset must be 2-D");
        if (off.shape.size() != 1) throw std::invalid_argument("doc_offsets must be 1-D");
        if (ids.shape.size() != 1) throw std::invalid_argument("doc_ids must be 1-D");

        if (static_cast<uint32_t>(q.shape[1])  != dim ||
            static_cast<uint32_t>(ds.shape[1]) != dim) {
          throw std::invalid_argument("query and dataset must have shape (..., dim)");
        }
        const uint32_t n_docs = static_cast<uint32_t>(ids.shape[0]);
        if (static_cast<uint32_t>(off.shape[0]) != n_docs + 1) {
          throw std::invalid_argument("doc_offsets must have shape (n_docs + 1,)");
        }

        // ---- build ----------------------------------------------------------
        auto bench = std::make_unique<PyChamferKernelBench>();
        bench->n_docs           = n_docs;
        bench->tokens_per_query = tokens_per_query;
        bench->dim              = dim;
        bench->d_query        = reinterpret_cast<const __half*>(q.ptr);
        bench->d_dataset      = reinterpret_cast<const __half*>(ds.ptr);
        bench->d_doc_offsets  = reinterpret_cast<const uint32_t*>(off.ptr);
        bench->d_doc_ids      = reinterpret_cast<const uint32_t*>(ids.ptr);
        bench->_keep_query        = query;
        bench->_keep_dataset      = dataset;
        bench->_keep_doc_offsets  = doc_offsets;
        bench->_keep_doc_ids      = doc_ids;

        // Only the output scores buffer is bench-owned.
        auto err = cudaMalloc(&bench->d_scores, n_docs * sizeof(float));
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMalloc scores: ") + cudaGetErrorString(err));
        }

        // Sync once now to flush any async H→D the producer (e.g. cupy
        // asarray) is doing on its own stream. CAI v3 leaves stream
        // synchronization to the consumer; doing device-wide sync at
        // construction is the simplest correct policy and is not in the
        // perf-critical path (the timing loop calls only run()).
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
        }

        return bench;
      }),
      py::arg("query"), py::arg("dataset"),
      py::arg("doc_offsets"), py::arg("doc_ids"),
      py::arg("tokens_per_query"), py::arg("dim"),
      R"pbdoc(
        Build a bench harness wrapping device-resident inputs.

        Args:
          query:        GPU array (cupy/torch/etc), float16, shape (tokens_per_query, dim).
          dataset:      GPU array, float16, shape (total_doc_tokens, dim).
          doc_offsets:  GPU array, uint32, shape (n_docs + 1,), CSR-style.
          doc_ids:      GPU array, uint32, shape (n_docs,) — docs to score.
          tokens_per_query, dim: passed through to the kernel for shape validation.

        Each input must implement ``__cuda_array_interface__`` and be C-contiguous
        (strides=None). The bench retains a Python reference to each, so the
        backing device memory is guaranteed alive for the bench's lifetime.
      )pbdoc")
    .def("run",
      [](PyChamferKernelBench& b) {
        vc::kernels::chamfer_score(b.n_docs, b.tokens_per_query, b.dim,
                                   b.d_query, b.d_dataset, b.d_doc_ids, b.d_doc_offsets,
                                   b.d_scores);
      },
      R"pbdoc(
        Launch the kernel on the device-resident inputs. No copy, no sync.
        Call ``cuda_sync()`` outside the timing loop if you need a wall
        clock that reflects completion.
      )pbdoc")
    .def("scores",
      [](PyChamferKernelBench& b) {
        // Build the numpy array with EXPLICIT shape and strides. The
        // convenience overload `array_t<T>(ssize_t)` produces an array with
        // strides=(0,) in this pybind11/numpy combination — every Python
        // element access then aliases position 0. Passing both shape and
        // strides explicitly sidesteps the auto-compute path.
        py::array_t<float> out(
            std::vector<py::ssize_t>{static_cast<py::ssize_t>(b.n_docs)},
            std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(float))});
        auto err = cudaMemcpy(out.mutable_data(), b.d_scores,
                              b.n_docs * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("memcpy scores back: ") + cudaGetErrorString(err));
        }
        return out;
      },
      R"pbdoc(
        Copy the latest score buffer back to a numpy.float32 array of
        length n_docs.
      )pbdoc")
    .def_property_readonly("n_docs",           [](const PyChamferKernelBench& b) { return b.n_docs; })
    .def_property_readonly("tokens_per_query", [](const PyChamferKernelBench& b) { return b.tokens_per_query; })
    .def_property_readonly("dim",              [](const PyChamferKernelBench& b) { return b.dim; });
}
