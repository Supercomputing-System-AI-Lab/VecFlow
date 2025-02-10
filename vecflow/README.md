# VecFlow

VecFlow is a high-performance vector data management system for filtered-search on GPUs, developed as an extension to NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) library. It achieves unprecedented high throughput and recall while obtaining low latency for filtered-ANNS.

VecFlow introduces the concept of "label specificity" - the number of data points associated with a particular label. Using a configurable specificity threshold T, it builds a dual-structured index: an IVF-CAGRA index for data points with labels that appear frequently (high specificity, ≥ T points), and an IVF-BFS index with interleaved vector storage for rare data points (low specificity, < T points). This dual-index approach optimizes GPU memory access patterns and achieves high performance across varying label distributions.

## Features

- Built on NVIDIA's cuVS library for high-performance vector search
- High throughput
- Efficient handling of both high-specificity and low-specificity labels
- GPU acceleration with CUDA
- Python and C++ APIs

## Installation Pre-compiled Python Packages 

Create a new conda environment:

```bash
conda create -n vecflow python=3.12
conda activate vecflow
```

Install NVIDIA cuVS library:

```bash
pip install cuvs-cu11 --extra-index-url=https://pypi.nvidia.com
```

Install VecFlow using the provided wheel file:

```bash
pip install dist/vecflow-0.0.1-cp312-cp312-linux_x86_64.whl
```

Note: Make sure you have CUDA 11 compatible GPUs and drivers installed on your system before installation.

## Quick Start

### Python API

```python
from vecflow import VecFlow 

# Initialize VecFlow
vf = VecFlow()

# Build index
vf.build(dataset=vectors,                   # numpy array of vectors (n_vectors x dim)
         data_label_fname="labels.spmat",   # path to data label file
         graph_degree=16,                   # graph degree for high-specificity data
         specificity_threshold=2000,        # threshold for label specificity
         graph_fname="graph.bin",           # path to save or load IVF-CAGRA graph
         bfs_fname="bfs.bin")               # path to save or load IVF-BFS index

# Search
neighbors, distances = vf.search(queries=query_vectors,       # numpy array of query vectors
                                 query_labels=query_labels,   # numpy array of query labels
                                 itopk_size=32)               # Internal topk size for search (higher values increase accuracy but reduce throughput)
```

### C++ API

```cpp
#include <cuvs/neighbors/vecflow.hpp>

using namespace cuvs::neighbors;

int main() {
    // Initialize index
    vecflow::index<float> idx;
    
    // Build index
    vecflow::build(idx,
                  dataset,                  // device matrix of vectors
                  "labels.txt",             // path to label file
                  16,                       // graph degree
                  2000,                     // specificity threshold
                  "graph.bin",              // path to save or load IVF-CAGRA graph
                  "bfs.bin");               // path to save or load IVF-BFS index
    
    // Search
    vecflow::search(idx,
                   queries,                 // device matrix of queries
                   query_labels,            // device vector of labels
                   32,                      // Internal topk size for search (higher values increase accuracy but reduce throughput)
                   neighbors,               // output device matrix for neighbors
                   distances);              // output device matrix for distances
}
```

## Examples

For detailed usage examples and tutorials, please refer to the [examples](examples/README.md) in the repository.

## License

Apache License 2.0

## Acknowledgments

This project is built on top of [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's high-performance GPU-accelerated vector search library. We thank the NVIDIA RAPIDS team for providing this foundation.
