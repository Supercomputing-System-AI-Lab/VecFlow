## Overview

The benchmark suite includes:

Standard search benchmark (`vecflow_bench.cu`) - Compare VecFlow against alternative filtering methods for large batch queries

## Configuration

The benchmark tool uses JSON configuration files for flexibility:

### Standard Benchmark (`config.json`)

```json
{
  "data_dir": "/path/to/data/directory/",
  "data_fname": "base.fbin",
  "query_fname": "query.fbin",
  "data_label_fname": "base.txt",
  "query_label_fname": "query.txt",
  "algorithms_to_run": ["vecflow", "cagra_inline_filtering", "cagra_post_processing"],
  "itopk_size": [8, 16, 32, 64, 128, 256, 512],
  "spec_threshold": 2000,
  "graph_degree": 16,
  "topk": 10,
  "num_runs": 1000,
  "warmup_runs": 10,
  "ivf_cagra_index_fname": "ivf_cagra_16_spec_2000.bin",
  "ivf_bfs_index_fname": "ivf_bfs_spec_2000.bin",
  "cagra_index_fname": "cagra_16.bin",
  "ground_truth_fname": "ground_truth_10.bin",
  "output_json_file": "/path/to/output/results.json"
}
```

## Parameters

- `data_dir`: Directory containing dataset files
- `data_fname`: Filename for base vectors (typically .fbin format)
- `query_fname`: Filename for query vectors (typically .fbin format)
- `data_label_fname`: Filename for base vector labels (.txt or .spmat)
- `query_label_fname`: Filename for query vector labels (.txt or .spmat)
- `algorithms_to_run`: List of algorithms to benchmark
- `itopk_size`: Array of internal topk sizes to test
- `spec_threshold`: Specificity threshold for VecFlow indexing
- `graph_degree`: Graph degree for CAGRA and VecFlow indices
- `topk`: Number of nearest neighbors to return
- `num_runs`: Number of benchmark runs to perform
- `warmup_runs`: Number of warmup runs before timing
- `ivf_cagra_index_fname`: Filename for IVF-CAGRA index
- `ivf_bfs_index_fname`: Filename for IVF-BFS index
- `cagra_index_fname`: Filename for CAGRA index
- `ground_truth_fname`: Filename for ground truth data
- `output_json_file`: Path for benchmark results output

## Usage

### Building the Benchmark Tools

First, make sure you've built the VecFlow library as described in the main examples README. Then compile the benchmark tools:

bash

```bash
# From the cpp directory
mkdir build
cd build
cmake ..
make
```

### Running Benchmarks

#### Standard Benchmark

bash

```bash
# Run with default config file
./VECFLOW_BENCH

# Or specify a custom config file
./VECFLOW_BENCH --config path/to/config.json
```

This benchmark compares VecFlow against alternative filtering methods for large batch queries:

- VecFlow: Our optimized dual-structure approach
- CAGRA with inline filtering: Using bitmap filters during search
- CAGRA with post-processing: Standard search then filter results

## Output

The benchmark outputs JSON files with detailed performance metrics:

- QPS (Queries Per Second)
- Recall@K

Example output:

json

```json
[
  {
    "algorithm": "vecflow",
    "itopk": 32,
    "qps": 24687.5,
    "recall": 0.9815
  },
  {
    "algorithm": "cagra_inline_filtering",
    "itopk": 32,
    "qps": 14325.6,
    "recall": 0.9756
  }
]
```

## Data Format Requirements

See the main examples README for details on data format requirements, including:

- Vector file formats (.fbin)
- Label formats (.txt and .spmat)
- Converting between label formats



## Cached Files

The benchmark generates cached files to speed up repeated runs:

- Index files for each algorithm
- Ground truth files for evaluation

These are automatically created if not present, or loaded if they exist.