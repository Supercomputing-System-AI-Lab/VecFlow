# Examples

## Building from Source
To build Python and C++ from source, we first need to build cuVS from source since VecFlow integrates with it. Here's the step-by-step process:

First, set up the build environment using conda:

```bash
# For CUDA 12.8+
conda env create --name cuvs -f conda/environments/all_cuda-128_arch-x86_64.yaml
conda activate cuvs

# OR for CUDA 11.8
conda env create --name cuvs -f conda/environments/all_cuda-118_arch-x86_64.yaml
conda activate cuvs
```

### Build cuVS Library
Go to the VecFlow root directory and build the C/C++ libraries:

```bash
cd ../../
./build.sh libcuvs
```

This creates:
* libcuvs.so (C++ library)
* libcuvs_c.so (C API library)

These are installed by default in `$INSTALL_PREFIX/lib`. After this step, the VecFlow headers are available for use in your C++ applications.

### Build Python Package from Source
After building cuVS, build the Python package:

```bash
cd vecflow
mkdir build
cd build
cmake ..
make install
```

## Getting Started with SIFT1M Dataset

### Download Dataset

First install gdown to download from Google Drive:

```bash
pip install gdown
```

Then create data directory and download files:

```bash
# Create directory
mkdir -p data/sift1M

# Download dataset files
gdown https://drive.google.com/drive/folders/1v4PfcefSKQvJzDz_5BnRzaPSIk4CEQ_S?usp=sharing -O data/ --folder
```

The dataset contains:
* `base.fbin`: Base vectors (1M vectors, 128 dimensions)
* `base.spmat` or `base.txt`: Labels for base vectors (spmat or text format)
* `query.fbin`: Query vectors for testing
* `query.spmat` or `query.txt`: Labels for query vectors (spmat or text format)

#### Spmat Label Format (.spmat)
The .spmat files store labels in binary format with three sequential parts:
1. Header (three 64-bit integers):
   - nrow: Number of data points, where each point has multiple labels
   - ncol: Maximum label value + 1 (defines valid label range 0 to ncol-1)
   - nnz:  Total number of labels across all data points
2. Row Pointers (array of 64-bit integers):
   - Length: nrow + 1 
   - indptr[i] and indptr[i+1] mark where data point i's labels are stored in the indices array
3. Label Values (array of 32-bit integers):
   - Length: nnz
   - Contains actual label values, each in range [0, ncol-1]

#### Text Label Format (.txt)
We also support a text-based label format (.txt):
* Each line corresponds to one data point
* Labels are comma-separated integers
* Valid label values range from 0 to N-1
* A line containing only "-1" indicates a data point with no labels

**Important**: Text format labels (.txt) must be converted to binary .spmat format before passing to the VecFlow::build function.

#### Utility Functions
For conversion and loading utilities, see:
- Python: [txt2spmat](https://github.com/Supercomputing-System-AI-Lab/VecFlow/blob/test/vecflow/examples/python/vecflow_example.py#L51) in vecflow_example.py 
- C++: [txt2spmat](https://github.com/Supercomputing-System-AI-Lab/VecFlow/blob/test/vecflow/examples/cpp/src/common.cuh#L312) in common.cuh
- For loading spmat files, see [vecflow_example.py](https://github.com/Supercomputing-System-AI-Lab/VecFlow/blob/test/vecflow/examples/python/vecflow_example.py#L17)

The ground truth generation utilities, see:
- Python: [generate_ground_truth](https://github.com/Supercomputing-System-AI-Lab/VecFlow/blob/test/vecflow/examples/python/vecflow_example.py#L253) in vecflow_example.py 
- C++: [generate_ground_truth](https://github.com/Supercomputing-System-AI-Lab/VecFlow/blob/test/vecflow/examples/cpp/src/common.cuh#L185) in common.cuh
- `generate_ground_truth`: A function to generate and save ground truth for evaluation

### Configuration and Indexing

#### Configuration Parameters
Both Python and C++ examples use a consistent JSON configuration:

```json
{
  "data_dir": "../../data/sift1M/",
  "data_fname": "base.fbin",
  "query_fname": "query.fbin",
  "data_label_fname": "base.txt",
  "query_label_fname": "query.txt",
  "itopk_size": 32,
  "spec_threshold": 1000,
  "graph_degree": 16,
  "topk": 10,
  "num_runs": 1000,
  "warmup_runs": 10
}
```

Parameter explanations:
- data_dir: Dataset directory
- data_fname: Base vectors filename
- query_fname: Query vectors filename
- data_label_fname: Base labels filename
- query_label_fname: Query labels filename
- itopk_size: Internal topk size for search
- spec_threshold: Specificity threshold
- graph_degree: Graph degree for index
- topk: Number of nearest neighbors to retrieve
- num_runs: Benchmark runs
- warmup_runs: Warmup runs

#### Cached Indexing and Evaluation Files
VecFlow generates three key files to optimize performance and evaluation:

1. **IVF-CAGRA Index** (`ivf_cagra_{graph_degree}_spec_{threshold}.bin`): IVF-CAGRA index file for data with high specificity labels.

2. **IVF-BFS Index** (`ivf_bfs_spec_{specificity_threshold}.bin`): IVF-BFS index file for data with low specificity labels.

3. **Ground Truth Neighbors** (`groundtruth.neighbors.{topk}.ibin`): Stores exact nearest neighbors for benchmark evaluation.

These files are automatically created and cached, reducing initialization time for repeated experiments.

### Running the Python Example

```bash
# Run with default config file
python python/vecflow_example.py

# Or specify a custom config file
python python/vecflow_example.py --config path/to/config.json
```

The program will:

1. Load the dataset based on configuration
2. Build the dual-structure index
3. Perform warmup runs followed by benchmark runs
4. Generate ground truth if needed
5. Report performance metrics including throughput and recall

### Running the C++ Example

#### Build C++ Example
To build the C++ example:

```bash
cd vecflow/examples/cpp
mkdir build
cd build
cmake ..
make
```

Run the compiled example with a config file:

```bash
# Run with default config file
./VECFLOW_EXAMPLE

# Or specify a custom config file
./VECFLOW_EXAMPLE --config path/to/config.json
```

The configuration parameters are the same as those for the Python example (see above).

The program will:

1. Load the dataset based on configuration
2. Build the dual-structure index
3. Perform warmup runs followed by benchmark runs
4. Generate ground truth if needed
5. Report performance metrics including throughput and recall

Make sure you have the dataset files in the locations specified in your config file before running the example.
