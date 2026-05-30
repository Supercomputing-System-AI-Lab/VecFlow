# VecFlow

VecFlow is a family of high-performance, GPU-accelerated vector data management systems for efficient vector search, developed as extensions to NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) library. This repository contains two complementary systems:

- **[VecFlow](vecflow/)** — filtered approximate nearest-neighbor search (filtered-ANNS). It provides a dual-structured index keyed off label specificity (the number of data points per label) with a configurable threshold T: an IVF-Graph index for high-specificity labels (≥ T points) and an IVF-BFS index with interleaved vector storage for low-specificity labels (< T points). This split optimizes GPU memory access patterns and achieves high throughput, high recall, and low latency across varying label distributions.
- **[VecFlow-Chamfer](vecflow-chamfer/)** — multi-vector (ColBERT-style) retrieval built for GPU architectures with a memory hierarchy (fast HBM next to capacious, GPU-addressable host RAM). A two-stage pipeline combining a CAGRA-routed anchor index (`maxivf`) for candidate generation, an optimized GPU chamfer-scoring kernel (`chamferkernel`) for single-digit-millisecond reranking over tens of thousands of candidates, and a tiered storage layer (`zerocomp`) that keeps full-precision embeddings on the larger host tier and streams them on demand over the GPU↔host interconnect.

## Recent News
* [5/10/2026] 🚀 VecFlow v0.1.0 released — rebased onto cuVS 26.06; added multi-label AND query support (`vecflow::search_multi_labels`); precompiled conda packages on [anaconda.org/VecFlow](https://anaconda.org/VecFlow) for Linux x86_64 + aarch64, CUDA 12, Python 3.11–3.14
* [11/24/2025] 🎉 VecFlow-Chamfer accepted by SIGMOD 2026!
* [5/23/2025] 🎉 VecFlow accepted by SIGMOD 2026!
* [5/30/2025] 🚀 VecFlow v0.0.1 released
* [2/27/2025] ⚙️ Introduced JSON-based configuration files for easier parameter management
* [2/25/2025] 🔍 Added support for ground truth generator to help with results validation and benchmarking
* [2/19/2025] 🎉 Added support for both binary (.spmat) and text (.txt) label formats

<details>
<summary><strong>Show older updates</strong></summary>

</details>


## Features

- Built on NVIDIA's cuVS library for high-performance vector search
- **VecFlow**: high throughput and low latency for filtered-ANNS; efficient handling of both high-specificity and low-specificity labels; multi-label AND queries
- **VecFlow-Chamfer**: low-latency, high-recall multi-vector / ColBERT-style retrieval on Grace-Hopper Superchips via `maxivf` indexing, fused `chamferkernel` scoring, and `zerocomp` tiered storage
- GPU acceleration with CUDA
- Python and C++ APIs (VecFlow); C++/CUDA library + example binary (VecFlow-Chamfer)

## Install (precompiled)

VecFlow ships precompiled conda packages on the [VecFlow Anaconda channel](https://anaconda.org/VecFlow) for Linux x86_64 and Linux aarch64, CUDA 12. Compute capabilities baked in: `sm_80`, `sm_90`, `sm_90a` (A100, H100, GH200). Supported Python versions: **3.11, 3.12, 3.13, 3.14**.

```bash
# VecFlow Python wrapper (transitively pulls libcuvs-vecflow-cu12)
mamba create -n vecflow-py -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       vecflow-cu12 python=3.12

# Or VecFlow C++ only (no Python)
mamba create -n vecflow-cpp -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       libcuvs-vecflow-cu12

# VecFlow-Chamfer Python wrapper (transitively pulls libvecflow-chamfer-cu12 + libcuvs-vecflow-cu12)
mamba create -n vecflow-chamfer-py -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       vecflow-chamfer-cu12 python=3.12

# Or VecFlow-Chamfer C++ only (no Python)
mamba create -n vecflow-chamfer-cpp -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       libvecflow-chamfer-cu12
```

## Build from source

Both libraries share the same conda env. Create it once from the repo root:

Pick the env file under `conda/environments/` that matches your CUDA version and arch (e.g. `all_cuda-129_arch-x86_64.yaml`, `all_cuda-131_arch-aarch64.yaml`):

```bash
conda env create --name vecflow -f conda/environments/<your-env>.yaml
conda activate vecflow
```

Then build whichever subproject you need:

```bash
./build.sh libcuvs --install                                       # patched libcuvs.so → $CONDA_PREFIX/lib/
cd vecflow         && ./build.sh examples python && cd ..          # VecFlow C++ example + Python wrapper
cd vecflow-chamfer && ./build.sh vecflow-chamfer examples python   # vecflow-chamfer lib + example + Python wrapper
```

## Subproject docs

Each subproject's README owns its API reference, dataset download script, configuration knobs, and end-to-end examples:

| Subproject | README | Covers |
|---|---|---|
| VecFlow | [`vecflow/README.md`](vecflow/README.md) | Python + C++ API, CMake snippet, SIFT1M dataset (`examples/download_dataset.sh`), single- and multi-label AND examples |
| VecFlow-Chamfer | [`vecflow-chamfer/README.md`](vecflow-chamfer/README.md) | C++/CUDA API, lifestyle.test dataset (`examples/download_dataset.sh`), anchor-index + chamfer-rerank pipeline, raw-kernel benchmark |

## Citation

If you use VecFlow in your research, please cite our papers:

```bibtex
@article{xi2025vecflow,
  author    = {Xi, Jingyi and Mo, Chenghao and Karsin, Ben and Chirkin, Artem and Li, Mingqin and Zhang, Minjia},
  title     = {VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs},
  journal   = {Proc. ACM Manag. Data},
  volume    = {3},
  number    = {4},
  articleno = {271},
  numpages  = {27},
  year      = {2025},
  doi       = {10.1145/3749189},
}

@article{mo2026vecflowchamfer,
  author    = {Mo, Chenghao and Karsin, Ben and Adams, Philip and Zhang, Minjia},
  title     = {VecFlow-Chamfer: A GPU-based Data Management System for High-Performance Multi-Vector Search on Superchips},
  journal   = {Proc. ACM Manag. Data},
  volume    = {4},
  number    = {1},
  articleno = {92},
  numpages  = {26},
  year      = {2026},
  month     = apr,
  publisher = {Association for Computing Machinery},
  url       = {https://doi.org/10.1145/3786706},
  doi       = {10.1145/3786706},
}
```

## License

Apache License 2.0

## Acknowledgments

This project is built on top of [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's high-performance GPU-accelerated vector search library. We thank the NVIDIA RAPIDS team for providing this foundation.
