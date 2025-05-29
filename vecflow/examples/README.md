# VecFlow Examples

This guide walks you through building VecFlow from source and running the provided examples with the SIFT1M dataset.

## 1. Building from Source

### Environment Setup
Set up the build environment using conda:

```bash
# For CUDA 12.8+
conda env create --name vecflow -f conda/environments/all_cuda-128_arch-x86_64.yaml
conda activate vecflow

# OR for CUDA 11.8
conda env create --name vecflow -f conda/environments/all_cuda-118_arch-x86_64.yaml
conda activate vecflow
```

### Build cuVS Library
Go to the VecFlow root directory and build the C/C++ libraries:

```bash
cd ../../
./build.sh libcuvs
```

This creates `libcuvs.so` (C++ library) and `libcuvs_c.so` (C API library), installed by default in `$INSTALL_PREFIX/lib`.

### Build Python Package
After building cuVS, build the Python package:

```bash
cd vecflow
mkdir build
cd build
cmake ..
make install
```

## 2. Dataset Setup

### Download SIFT1M Dataset

Install gdown and download the dataset:

```bash
pip install gdown
mkdir -p datasets/sift1M
gdown https://drive.google.com/drive/folders/1v4PfcefSKQvJzDz_5BnRzaPSIk4CEQ_S?usp=sharing -O datasets/ --folder
```

### Dataset Files
- `base.fbin`: Base vectors (1M vectors, 128 dimensions)
- `query.fbin`: Query vectors for testing
- `base.txt` / `base.spmat`: Labels for base vectors
- `query.txt` / `query.spmat`: Labels for query vectors

### Label Formats

**Text Format (.txt):**
- Each line corresponds to one data point
- Labels are comma-separated integers (range: 0 to N-1)
- Line containing only "-1" indicates no labels

**Binary Format (.spmat):**
Three sequential parts:
1. **Header** (three 64-bit integers): nrow, ncol, nnz
2. **Row Pointers** (nrow+1 64-bit integers): Index boundaries for each data point
3. **Label Values** (nnz 32-bit integers): Actual label values

## 3. Configuration

Both Python and C++ examples use this JSON configuration:

```json
{
  "data_dir": "../../datasets/sift1M/",
  "data_fname": "base.fbin",
  "query_fname": "query.fbin",
  "data_label_fname": "base.txt",
  "query_label_fname": "query.txt",
  "itopk_size": 32,
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

### Key Parameters
- **spec_threshold**: Specificity threshold for dual-structure indexing
- **graph_degree**: Degree of CAGRA graph for high-specificity data
- **topk**: Number of nearest neighbors to retrieve
- **force_rebuild**: Force rebuilding index files even if they exist
- **Cache files**: Automatically generated for performance optimization

## 4. Running Examples

### Python Example

```bash
# Default config
python python/vecflow_example.py

# Custom config
python python/vecflow_example.py --config path/to/config.json
```

### C++ Example

The C++ example requires building VecFlow from source first.

Build and run:

```bash
cd vecflow/examples/cpp
mkdir build && cd build
cmake .. && make

# Run with default or custom config
./VECFLOW_EXAMPLE
./VECFLOW_EXAMPLE --config path/to/config.json
```

### Execution Flow
Both examples follow the same process:
1. Load dataset and configuration
2. Build dual-structure index (IVF-CAGRA + IVF-BFS)
3. Generate ground truth if needed
4. Perform warmup runs
5. Execute benchmark runs
6. Report performance metrics (throughput and recall)

## 5. Utility Functions

For advanced usage, see these utility functions:

**Data Loading:**
- Python: `load_labels_auto()` in `vecflow_example.py`
- C++: `read_labeled_data()` in `common.cuh`

**Ground Truth Generation:**
- Python: `generate_ground_truth()` in `vecflow_example.py`
- C++: `generate_ground_truth()` in `common.cuh`

---

**Note:** Ensure dataset files are in the locations specified in your config file before running examples.
