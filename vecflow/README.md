# VecFlow — Install, API Usage, Build From Source, Examples

This subdirectory holds the standalone Python wrapper and end-to-end
examples on the SIFT1M dataset. It also documents the C++-only install
path, API usage (Python + C++), building from source, and bundled
SIFT1M examples. The [top-level README](../README.md) covers the
high-level pitch and Python install in one shot.

## What's here

| Path | What |
|---|---|
| `binding/binding.cpp` | pybind11 module entry point |
| `include/vecflow.hpp` | `PyVecFlow` class declaration |
| `src/vecflow.cu` | CUDA-side implementation of the binding |
| `vecflow/__init__.py` | Python package surface (`from vecflow import VecFlow`) |
| `vecflow/vecflow.pyi` | Type stubs for IDEs / mypy |
| `pyproject.toml` | scikit-build-core build config |
| `CMakeLists.txt` | drives the pybind11 module compilation |
| `examples/` | end-to-end Python + C++ examples on SIFT1M |

C++ source for the algorithm itself lives in:
- `../cpp/src/neighbors/vecflow/` (composite IVF-CAGRA + IVF-BFS)
- `../cpp/src/neighbors/filtered_bfs/` (label-gated IVF-Flat)
- `../cpp/src/neighbors/detail/cagra/filtered_search_single_cta*` (CAGRA fork)

## Install (C++ users, no Python)

Precompiled conda packages on the [VecFlow Anaconda channel](https://anaconda.org/VecFlow) for Linux x86_64 and Linux aarch64, CUDA 12. Compute capabilities baked in: `sm_80`, `sm_90`, `sm_90a` (A100, H100, GH200).

```bash
mamba install -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
              libcuvs-vecflow-cu12
```

Then in your CMake project:

```cmake
find_package(cuvs CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cuvs::cuvs)
```

VecFlow's headers are exposed as `<cuvs/neighbors/vecflow.hpp>`, `<cuvs/neighbors/filtered_bfs.hpp>`, and `cagra::filtered_search` overloads in `<cuvs/neighbors/cagra.hpp>`.

### Coexistence with upstream cuVS

`libcuvs-vecflow-cu12` and rapidsai's stock `libcuvs` ship the same `libcuvs.so` filename, so they can't share a conda env. Use a fresh env, or `mamba remove libcuvs cuvs` before installing VecFlow's variant.

(Python users: `mamba install ... vecflow-cu12` pulls `libcuvs-vecflow-cu12` in transitively — see the [top-level README](../README.md).)

## API Usage

### Python

```python
from vecflow import VecFlow

# Initialize an empty index
vf = VecFlow()

# Build the dual-structured index
vf.build(
    dataset=dataset,                  # numpy array (n_vectors x dim), float32
    data_labels=data_labels,          # list[list[int]], one label list per vector
    graph_degree=16,                  # CAGRA graph degree for high-specificity lane
    specificity_threshold=2000,       # labels with ≥ this many points → CAGRA; rarer → BFS
    graph_fname="ivf_graph.bin",      # cache path for the IVF-CAGRA graph
    bfs_fname="ivf_bfs.bin",          # cache path for the IVF-BFS index
    multi_label=False,                # set True to also prep CSR for search_multi(...) below
)

# Single-label search
neighbors, distances = vf.search(
    queries=query_vectors,            # numpy array (n_queries x dim), float32
    query_labels=query_labels,        # numpy array (n_queries,), int32
    itopk_size=32,                    # internal top-k buffer (higher = better recall, slower)
    topk=10,                          # neighbors returned per query
)

# 2-label AND search (requires the index to be built with multi_label=True)
neighbors, distances = vf.search_multi(
    queries=query_vectors,            # numpy array (n_queries x dim), float32
    query_labels_a=labels_a,          # numpy array (n_queries,), int32 — any order
    query_labels_b=labels_b,          # numpy array (n_queries,), int32 — order vs `_a` is irrelevant
    itopk_size=32,
    topk=10,
)
```

`help(vecflow.VecFlow.build)` / `help(vecflow.VecFlow.search)` /
`help(vecflow.VecFlow.search_multi)` show the full docstrings. Type stubs in
`vecflow/vecflow.pyi` give IDE auto-completion.

### C++

```cpp
#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <raft/core/device_mdarray.hpp>

using namespace cuvs::neighbors;

int main() {
    shared_resources::configured_raft_resources res;

    // Build VecFlow index. data_labels is std::vector<std::vector<int>>.
    // Set multi_label=true to also prep the CSR label arrays for AND search.
    auto idx = vecflow::build(
        res,
        raft::make_const_mdspan(dataset.view()),   // device matrix [n × dim]
        data_labels,
        /*graph_degree*/           16,
        /*specificity_threshold*/  2000,
        /*graph_fname*/            "ivf_graph.bin",
        /*bfs_fname*/              "ivf_bfs.bin",
        /*force_rebuild*/          false,
        /*multi_label*/            false);          // set true to enable search_multi_labels

    // Single-label search.
    vecflow::search(
        res, idx,
        raft::make_const_mdspan(queries.view()),
        query_labels.view(),                       // [n_queries], uint32_t
        /*itopk_size*/ 32,
        neighbors.view(),                          // device matrix [n_queries × topk]
        distances.view());

    // 2-label AND search (requires multi_label=true at build time). Order
    // of `query_labels_a` vs `_b` is irrelevant — the impl auto-picks the
    // larger-frequency label as the primary IVF selector.
    vecflow::search_multi_labels(
        res, idx,
        raft::make_const_mdspan(queries.view()),
        query_labels_a.view(),                     // [n_queries], uint32_t
        query_labels_b.view(),                     // [n_queries], uint32_t
        /*itopk_size*/ 32,
        neighbors.view(),
        distances.view());

    return 0;
}
```

The full set of public APIs:
- `cuvs::neighbors::vecflow::{build, search, search_multi_labels, index<T>}` — composite top-level (single- and 2-label AND search)
- `cuvs::neighbors::filtered_bfs::{build_filtered_bfs, search_filtered_bfs}` — IVF-Flat with one-probe label gate; `search_filtered_bfs` accepts optional `dataset_labels_ptr`/`dataset_label_offsets_ptr`/`query_labels_second_ptr` for inline AND filtering
- `cuvs::neighbors::cagra::filtered_search` — CAGRA with per-query label gating; same optional trailing pointers for inline AND filtering

## 1. Building from Source

### Environment setup

```bash
# CUDA 12.9 (x86_64). Other env files in the same dir cover aarch64 + CUDA 13.1.
conda env create --name vecflow -f ../conda/environments/all_cuda-129_arch-x86_64.yaml
conda activate vecflow
```

### Build the cuVS C++ library

From the VecFlow repo root:

```bash
cd ..
./build.sh libcuvs --install
```

Produces `libcuvs.so` (with the VecFlow patches integrated) and installs it
into `$CONDA_PREFIX/lib/`.

### Build the Python package

```bash
cd vecflow
./build.sh python                  # convenience wrapper
```

`scikit-build-core` drives a CMake build of the pybind11 module against the
just-installed `libcuvs.so`, then packages the resulting `.so` into a wheel
that's installed into the active env.

Verify:

```bash
python -c "import vecflow; print(vecflow.__version__); print(vecflow.VecFlow())"
```

## 2. Dataset Setup (SIFT1M)

```bash
pip install gdown
mkdir -p examples/datasets/sift1M
gdown 'https://drive.google.com/drive/folders/1v4PfcefSKQvJzDz_5BnRzaPSIk4CEQ_S?usp=sharing' \
      -O examples/datasets/ --folder
```

### Files you should end up with

| File | Purpose |
|---|---|
| `base.fbin` | base vectors (1M × 128 floats) |
| `query.fbin` | query vectors |
| `base.txt` / `base.spmat` | labels for base vectors |
| `query.txt` / `query.spmat` | labels for query vectors |

### Label formats

**Text (`.txt`)**: one line per data point; labels are comma-separated integers; a single `-1` means "no labels".

**Binary (`.spmat`)**: header (three 64-bit ints — `nrow`, `ncol`, `nnz`) → row pointers (`nrow+1` 64-bit ints) → label values (`nnz` 32-bit ints).

## 3. Configuration

Both the Python and C++ examples read a JSON config:

```json
{
  "data_dir": "../../datasets/sift1M/",
  "data_fname": "base.fbin",
  "query_fname": "query.fbin",
  "data_label_fname": "base.txt",
  "query_label_fname": "query.txt",
  "itopk_size": [16, 32, 64, 128],
  "spec_threshold": 1000,
  "graph_degree": 16,
  "topk": 10,
  "num_runs": 1000,
  "warmup_runs": 10,
  "force_rebuild": false,
  "ivf_graph_fname": "ivf_graph.bin",
  "ivf_bfs_fname": "ivf_bfs.bin",
  "ground_truth_fname": "groundtruth.neighbors.10.ibin"
}
```

| Key | Meaning |
|---|---|
| `spec_threshold` | specificity cutoff: labels with ≥ this many points go to IVF-CAGRA; rarer labels go to IVF-BFS |
| `graph_degree` | CAGRA graph degree for the high-specificity lane |
| `topk` | neighbors returned per query |
| `force_rebuild` | ignore cached index files and rebuild |
| `ivf_graph_fname` / `ivf_bfs_fname` | cache locations for the two index halves |

## 4. Running the Examples

### Python

```bash
cd examples
python python/vecflow_example.py                       # uses default config
python python/vecflow_example.py --config path/to/config.json
```

### C++

The C++ example requires the cuVS C++ library installed (step 1 above). A convenience script wraps the cmake/make dance:

```bash
cd vecflow              # repo's vecflow/ subdir, where build.sh lives
./build.sh examples     # configures + builds; binary lands in examples/cpp/build/VECFLOW_EXAMPLE
```

Run it with the default config (paths in `config.json` are relative to `examples/cpp/src/`):

```bash
cd examples/cpp/src
../build/VECFLOW_EXAMPLE                                # uses ./config.json
../build/VECFLOW_EXAMPLE --config path/to/config.json   # custom config
```

`./build.sh` also accepts:

| Command | What |
|---|---|
| `./build.sh` | Same as `./build.sh examples` |
| `./build.sh examples` | Configure + build the C++ example |
| `./build.sh python` | `pip install .` the Python wrapper into the active env |
| `./build.sh clean` | Remove `examples/cpp/build/` |
| `./build.sh examples python` | Both targets in one go |
| `./build.sh -j 8 examples` | Parallel jobs (default: `nproc`) |
| `./build.sh -v examples` | Verbose (echoes every command) |
| `./build.sh -h` | Help |


### What both examples do

1. Load the dataset + JSON config.
2. Build the dual-structure index (IVF-CAGRA for high-specificity labels, IVF-BFS for low-specificity).
3. Generate ground truth via brute force (once, reused for every itopk_size).
4. Sweep over each `itopk_size` in the config: warmup → timed runs → recall.
5. Print one progress line per itopk value with QPS / avg latency / recall.

`itopk_size` can be a single integer or an array. With an array (default
config: `[16, 32, 64, 128]`) the sweep shows the speed/recall trade-off:
small itopk = faster but lower recall, large itopk = higher recall but
slower. Example output (NVIDIA GH200, bundled SIFT1M config; absolute
QPS/latency depend on GPU and dataset):

```
=== Performing Search Sweep ===
  itopk=  16  qps= 8466428.5  avg= 1.181 ms  recall=0.8743
  itopk=  32  qps= 6035911.2  avg= 1.657 ms  recall=0.9397
  itopk=  64  qps= 3299089.0  avg= 3.031 ms  recall=0.9831
  itopk= 128  qps= 1573093.1  avg= 6.357 ms  recall=0.9968
```

### Multi-label AND example (2 labels per query)

A separate pair of examples — `vecflow_example_multi.{cu,py}` — exercises
the 2-label AND search path through `vecflow::search_multi_labels`. They
build the index with `multi_label=true`, brute-force AND ground truth, then
sweep itopk the same way as the single-label sweep.

The bundled SIFT1M `query.txt` is single-label. Generate the 2-label query
file from it once:

```bash
cd vecflow/examples/python
python generate_multi_query.py \
    --base-labels  ../datasets/sift1M/base.txt \
    --query-labels ../datasets/sift1M/query.txt \
    --out-txt      ../datasets/sift1M/query_multi.txt \
    --out-spmat    ../datasets/sift1M/query_multi.spmat \
    --min-and-size 500
```

`--out-spmat` is optional. The script's built-in default for `--min-and-size` is 50;
the bundled examples use 500 for a stricter benchmark (each kept query has ≥500
AND-valid candidates, so recall numbers are meaningful and not bottlenecked by
tiny intersection sets).

The generator picks the second label such that the **AND intersection**
`points(primary) ∩ points(secondary)` has at least `--min-and-size`
members. Queries for which no such secondary exists are emitted as `-1`
rows in `query_multi.txt`, and both the C++ and Python examples auto-skip
them, so every query that reaches `search_multi_labels` has a meaningfully
large ground-truth set. The generator prints the distribution of intersection
sizes at the end so you can tune the threshold (raise it for stricter
benchmarks, lower it to keep more queries).

Then run either example:

```bash
# C++
./build.sh examples                # already builds VECFLOW_EXAMPLE_MULTI alongside
~/VecFlow/vecflow/examples/cpp/build/VECFLOW_EXAMPLE_MULTI

# Python
cd vecflow/examples/python
python vecflow_example_multi.py
```

The config files (`config_multi.json` in each example dir) point at
`query_multi.txt` and use a separate `groundtruth.multi.neighbors.10.ibin`
cache, so the multi-label run won't clobber the single-label ground truth.

## 5. Utility helpers worth knowing

**Data loading**:
- Python — `load_labels_auto()` in `examples/python/vecflow_example.py`
- C++ — `read_labeled_data()` in `examples/cpp/src/common.cuh`

**Ground truth generation**:
- Python — `generate_ground_truth()` in `examples/python/vecflow_example.py`
- C++ — `generate_ground_truth()` in `examples/cpp/src/common.cuh`
- AND ground truth (multi-label): `generate_ground_truth_multi()` in `examples/cpp/src/common.cuh`; `brute_force_and_ground_truth()` in `examples/python/vecflow_example_multi.py`

**Multi-label query file generation**:
- `examples/python/generate_multi_query.py` — converts a single-label
  `query.txt` into a 2-label `query_multi.txt` (and optionally `.spmat`)
  by sampling co-occurring secondary labels from the base label distribution.
