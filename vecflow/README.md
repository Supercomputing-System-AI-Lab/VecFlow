# VecFlow

VecFlow is a high-performance vector data management system for filtered-search on GPUs, developed as an extension to NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) library. It achieves unprecedented high throughput and recall while obtaining low latency for filtered-ANNS.

VecFlow introduces the concept of "label specificity" - the number of data points associated with a particular label. Using a configurable specificity threshold T, it builds a dual-structured index: an IVF-CAGRA index for data points with labels that appear frequently (high specificity, ≥ T points), and an IVF-BFS index with interleaved vector storage for rare data points (low specificity, < T points). This dual-index approach optimizes GPU memory access patterns and achieves high performance across varying label distributions.

## Recent News
* [5/30/2025] 🚀 VecFlow v0.0.1 released
* [5/23/2025] 🎉 VecFlow accepted by SIGMOD 2026! 
* [2/27/2025] ⚙️ Introduced JSON-based configuration files for easier parameter management
* [2/25/2025] 🔍 Added support for ground truth generator to help with results validation and benchmarking
* [2/19/2025] 🎉 Added support for both binary (.spmat) and text (.txt) label formats

<details>
<summary><strong>Show older updates</strong></summary>

</details>


## Features

- Built on NVIDIA's cuVS library for high-performance vector search
- High throughput and low latency for filtered-ANNS
- Efficient handling of both high-specificity and low-specificity labels
- GPU acceleration with CUDA
- Python and C++ APIs

## Installation Pre-compiled Python Packages 

Create a new conda environment:

```bash
conda create -n vecflow python=3.12
conda activate vecflow
```

Install required dependencies:

```bash
pip install numpy cupy-cuda11x
```

Install VecFlow using the provided source distribution:

```bash
pip install ./dist/vecflow-0.0.1.tar.gz
```

Note: Make sure you have CUDA 11 compatible GPUs and drivers installed on your system before installation.

For instructions on building from source (both Python and C++), please refer to our [build instructions](examples/README.md#building-from-source).

## Quick Start

### Python API

```python
from vecflow import VecFlow 

# Initialize VecFlow
vf = VecFlow()

# Build index
vf.build(dataset=dataset,                    # numpy array of vectors (n_vectors x dim)
         data_labels=data_labels,            # list of lists containing labels for each vector
         graph_degree=16,                    # graph degree for high-specificity data
         specificity_threshold=2000,         # threshold for label specificity
         ivf_graph_fname="graph.bin",        # path to save/load IVF-CAGRA graph
         ivf_bfs_fname="bfs.bin")            # path to save/load IVF-BFS index

# Search
neighbors, distances = vf.search(queries=query_vectors,       # numpy array of query vectors
                                 query_labels=query_labels,   # numpy array of query labels
                                 itopk_size=32)               # Internal topk size for search (higher values increase accuracy but reduce throughput)
```

### C++ API

```cpp
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

using namespace cuvs::neighbors;

int main() {
    // Initialize RAFT shared resources
    shared_resources::configured_raft_resources res;
    
    // Build VecFlow index
    auto idx = vecflow::build(res,
                              raft::make_const_mdspan(dataset.view()),  // device matrix of vectors
                              data_labels,                              // vector of label lists
                              16,                                       // graph degree
                              2000,                                     // specificity threshold
                              "graph.bin",                              // IVF-CAGRA graph file
                              "bfs.bin");                               // IVF-BFS index file
    
    // Search
    vecflow::search(res, idx,
                    raft::make_const_mdspan(queries.view()),     // device matrix of queries
                    query_labels.view(),                         // device vector of labels
                    itopk,                                       // Internal topk size for search (higher values increase accuracy but reduce throughput)
                    neighbors.view(),                            // output device matrix for neighbors
                    distances.view());                           // output device matrix for distances
    
    return 0;
}
```

## Examples

For detailed usage examples and tutorials, please refer to the [examples](examples/README.md) in the repository.

## License

Apache License 2.0

## Acknowledgments

This project is built on top of [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's high-performance GPU-accelerated vector search library. We thank the NVIDIA RAPIDS team for providing this foundation.
