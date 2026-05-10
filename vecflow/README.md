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
)

# Search
neighbors, distances = vf.search(
    queries=query_vectors,            # numpy array (n_queries x dim), float32
    query_labels=query_labels,        # numpy array (n_queries,), int32
    itopk_size=32,                    # internal top-k buffer (higher = better recall, slower)
    topk=10,                          # neighbors returned per query
)
```

`help(vecflow.VecFlow.build)` and `help(vecflow.VecFlow.search)` show the
full docstrings with parameter types and shapes. Type stubs in
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
    auto idx = vecflow::build(
        res,
        raft::make_const_mdspan(dataset.view()),   // device matrix [n × dim]
        data_labels,
        /*graph_degree*/           16,
        /*specificity_threshold*/  2000,
        /*graph_fname*/            "ivf_graph.bin",
        /*bfs_fname*/              "ivf_bfs.bin");

    // Search. itopk = internal top-k buffer; topk = neighbors per query.
    vecflow::search(
        res, idx,
        raft::make_const_mdspan(queries.view()),
        query_labels.view(),
        /*itopk_size*/ 32,
        neighbors.view(),                          // device matrix [n_queries × topk]
        distances.view());

    return 0;
}
```

The full set of public APIs:
- `cuvs::neighbors::vecflow::{build, search, index<T>}` — composite top-level
- `cuvs::neighbors::filtered_bfs::{build_filtered_bfs, search_filtered_bfs}` — IVF-Flat with one-probe label gate
- `cuvs::neighbors::cagra::filtered_search` — CAGRA with per-query label gating

## 1. Building from Source

### Environment setup

```bash
# CUDA 12
conda env create --name vecflow -f ../conda/environments/all_cuda-128_arch-x86_64.yaml
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
pip install . --no-build-isolation
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

The C++ example requires the cuVS C++ library installed (step 1 above):

```bash
cd examples/cpp
mkdir build && cd build
cmake .. && make
./VECFLOW_EXAMPLE                                       # uses default config
./VECFLOW_EXAMPLE --config path/to/config.json
```

### What both examples do

1. Load the dataset + JSON config.
2. Build the dual-structure index (IVF-CAGRA for high-specificity labels, IVF-BFS for low-specificity).
3. Generate ground truth via brute force (once, reused for every itopk_size).
4. Sweep over each `itopk_size` in the config: warmup → timed runs → recall.
5. Print a per-itopk row + a compact summary table at the end.

`itopk_size` can be a single integer or an array. With an array (default
config: `[16, 32, 64, 128]`) the sweep shows the speed/recall trade-off:
small itopk = faster but lower recall, large itopk = higher recall but
slower. Example output:

```
=== Summary ===
  itopk_size         qps    avg_ms   recall
  ----------  ----------  --------  -------
          16     85432.1     0.117   0.8512
          32     63215.4     0.158   0.9234
          64     42108.2     0.237   0.9678
         128     28503.7     0.351   0.9891
```

## 5. Utility helpers worth knowing

**Data loading**:
- Python — `load_labels_auto()` in `examples/python/vecflow_example.py`
- C++ — `read_labeled_data()` in `examples/cpp/src/common.cuh`

**Ground truth generation**:
- Python — `generate_ground_truth()` in `examples/python/vecflow_example.py`
- C++ — `generate_ground_truth()` in `examples/cpp/src/common.cuh`
