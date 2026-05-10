# VecFlow

VecFlow is a high-performance vector data management system for filtered-search on GPUs, developed as an extension to NVIDIA's [cuVS](https://github.com/rapidsai/cuvs) library. It achieves unprecedented high throughput and recall while obtaining low latency for filtered-ANNS.

VecFlow introduces the concept of "label specificity" - the number of data points associated with a particular label. Using a configurable specificity threshold T, it builds a dual-structured index: an IVF-CAGRA index for data points with labels that appear frequently (high specificity, ≥ T points), and an IVF-BFS index with interleaved vector storage for rare data points (low specificity, < T points). This dual-index approach optimizes GPU memory access patterns and achieves high performance across varying label distributions.

## Recent News
* [5/10/2026] 🚀 VecFlow v0.1.0 released — rebased onto cuVS 26.06; precompiled conda packages on [anaconda.org/VecFlow](https://anaconda.org/VecFlow) for Linux x86_64 + aarch64, CUDA 12, Python 3.11–3.14
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
- High throughput and low latency for filtered-ANNS
- Efficient handling of both high-specificity and low-specificity labels
- GPU acceleration with CUDA
- Python and C++ APIs

## Installation

Precompiled conda packages on the [VecFlow Anaconda channel](https://anaconda.org/VecFlow) for Linux x86_64 and Linux aarch64, CUDA 12. Compute capabilities baked in: `sm_80`, `sm_90`, `sm_90a` (A100, H100, GH200). Two packages are published from the same release tag:

| Package | Install with | Includes |
|---|---|---|
| `vecflow-cu12` | `mamba install vecflow-cu12` | Python wrapper + transitively pulls the C++ library |
| `libcuvs-vecflow-cu12` | `mamba install libcuvs-vecflow-cu12` | C++ library only (`libcuvs.so` + headers + cmake config) |

### Python users

Supported Python versions: **3.11, 3.12, 3.13, 3.14** (matches cuVS upstream — the C++ runtime ships per-Python wheels for each).

```bash
mamba create -n vecflow -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       vecflow-cu12 python=3.12      # or 3.11 / 3.13 / 3.14
mamba activate vecflow
python -c "import vecflow; print(vecflow.VecFlow())"
```

### C++ users (no Python)

```bash
mamba install -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
              libcuvs-vecflow-cu12
```

For the CMake snippet, coexistence note with upstream cuVS, full Python and C++ API usage, build-from-source instructions, and end-to-end SIFT1M examples, see [`vecflow/README.md`](vecflow/README.md) (source under [`vecflow/examples/`](vecflow/examples/)).

## Citation

If you use VecFlow in your research, please cite our paper:

```bibtex
@article{vecflow2025,
  author    = {Xi, Jingyi and Mo, Chenghao and Karsin, Ben and Chirkin, Artem and Li, Mingqin and Zhang, Minjia},
  title     = {VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs},
  journal   = {arXiv preprint arXiv:2506.00812},
  year      = {2025},
}
```

## License

Apache License 2.0

## Acknowledgments

This project is built on top of [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's high-performance GPU-accelerated vector search library. We thank the NVIDIA RAPIDS team for providing this foundation.
