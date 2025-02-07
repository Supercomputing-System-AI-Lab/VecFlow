# VecFlow

VecFlow is a high-performance vector data management system for filtered-search on GPUs, developed as an extension to NVIDIA's cuVS (CUDA Vector Search) library. It achieves unprecedented high throughput and recall while obtaining low latency for filtered-ANNS.

## Features

- Built on NVIDIA's cuVS library for high-performance vector search
- High throughput
- Efficient handling of both high-specificity and low-specificity labels
- GPU acceleration with CUDA
- Python and C++ APIs

## Installation

[TBD: Building from source instructions]

[TBD: Python package installation via pip]

## Quick Start

### Python API

```python
from vecflow import VecFlow 

# Initialize VecFlow
vf = VecFlow()

# Build index
vf.build(dataset=vectors,              		# numpy array of vectors (n_vectors x dim)
         data_label_fname="labels.spmat", # path to label file
         graph_degree=16,              		# graph degree for high-specificity labels
         specificity_threshold=2000,    	# threshold for label specificity
         graph_fname="graph.bin",     		# path to save IVF-CAGRA graph
         bfs_fname="bfs.bin")         		# path to save IVF-BFS index

# Search
neighbors, distances = vf.search(queries=query_vectors,     # numpy array of query vectors
                               query_labels=query_labels,   # numpy array of query labels
                               itopk_size=32)              	# internal topk parameter
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
                  "graph.bin",              // path to save IVF-CAGRA graph
                  "bfs.bin");              	// path to save IVF-BFS index
    
    // Search
    vecflow::search(idx,
                   queries,                 // device matrix of queries
                   query_labels,            // device vector of labels
                   32,                      // internal topk parameter
                   neighbors,               // output device matrix for neighbors
                   distances);              // output device matrix for distances
}
```

## License

Apache License 2.0

## Acknowledgments

This project is built on top of [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's high-performance GPU-accelerated vector search library. We thank the NVIDIA RAPIDS team for providing this foundation.

## Citation

[Add paper citation when published]