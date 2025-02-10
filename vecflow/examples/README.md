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

* `sift.base.fbin`: Base vectors (1M vectors, 128 dimensions)
* `sift.base.spmat`: Labels for base vectors  
* `sift.query.fbin`: Query vectors for testing
* `sift.query.spmat`: Labels for query vectors
* `sift.groundtruth.neighbors.ibin`: Ground truth neighbors for evaluating recall
* `sift.groundtruth.distances.fbin`: Ground truth distances

### Running the Python Example

```bash
python python/vecflow_example.py --data_dir data/ \
                                --itopk_size 32 \
                                --spec_threshold 1000 \
                                --graph_degree 16
```

Parameters:

- `--data_dir`: Directory containing the dataset files
- `--itopk_size`: Internal topk size for search (higher values increase accuracy but reduce throughput)
- `--spec_threshold`: Specificity threshold for index building
- `--graph_degree`: Graph degree for high-specificity index

The script will:

1. Load the dataset
2. Build the dual-structured index
3. Perform searches 
4. Compute and report timing and recall metrics

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

Run the compiled example with parameters:

```bash
./VECFLOW_EXAMPLE --data_dir ../../data/ --itopk_size 32 --spec_threshold 1000 --graph_degree 16
```

Parameters:

- 32: Internal topk size
- 1000: Specificity threshold
- 16: Graph degree

The program will:

1. Load the SIFT1M dataset
2. Build the dual-structure index
3. Perform searches
4. Report performance metrics including throughput and recall

Make sure you have the SIFT1M dataset files in the expected location before running the example.
