# vecflow-chamfer — API Usage, Examples

GPU-accelerated multi-vector (ColBERT-style) retrieval. Two-stage pipeline:

1. **Anchor index** — cluster every document's token embeddings into anchors,
   build a CAGRA graph over the anchors. Search reduces a query's tokens to
   the anchors they hit, then expands those anchors to candidate documents.
2. **Chamfer rerank** — score the candidate documents with the full chamfer
   similarity kernel, then take the top-K.

The library is a thin public C++ API + a C++/CUDA shared library; the
example binary `VECFLOW_CHAMFER` drives the pipeline end-to-end on a
JSON config. The shared conda env setup is documented in the
[top-level README](../README.md).

## What's here

| Path | What |
|---|---|
| `cpp/include/vecflow_chamfer/` | Public headers (`build`, `search`, `serialize`, `kernels`, `io`, `types`) |
| `cpp/src/` | Implementation: indexing, search stages, chamfer kernels, IO |
| `binding/binding.cpp` | pybind11 module entry point |
| `vecflow_chamfer/` | Python package surface + `.pyi` type stubs |
| `examples/cpp/src/vecflow_chamfer.cu` | C++ end-to-end demo (load → build → search → recall) |
| `examples/cpp/src/chamfer_core_kernel.cu` | Raw chamfer kernel benchmark (no cuVS dep) |
| `examples/cpp/config/` | Default JSON configs for the two C++ example binaries |
| `examples/python/vecflow_chamfer.py` | Python end-to-end demo (same pipeline as the C++ one) |
| `examples/python/chamfer_core_kernel.py` | Python raw kernel benchmark (mirrors the C++ binary) |
| `examples/python/config.json`, `config_kernel.json` | Default JSON configs for the Python examples |
| `examples/download_dataset.sh` | Fetches the bundled lifestyle dataset from Google Drive |
| `build.sh` | One-shot build orchestrator |

## Install

### Option A — Precompiled (recommended)

Conda packages on the [VecFlow Anaconda channel](https://anaconda.org/VecFlow)
for Linux x86_64 / aarch64, CUDA 12. Compute capabilities baked in: `sm_80`,
`sm_90`, `sm_90a` (A100, H100, GH200).

```bash
# Python wrapper (transitively pulls libvecflow-chamfer-cu12 + libcuvs-vecflow-cu12)
mamba create -n vecflow-chamfer -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       vecflow-chamfer-cu12 python=3.12      # or 3.11 / 3.13 / 3.14
mamba activate vecflow-chamfer

# Or C++ only
mamba create -n vecflow-chamfer -y \
       -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \
       libvecflow-chamfer-cu12
mamba activate vecflow-chamfer
```

### Option B — Build from source

vecflow-chamfer links against VecFlow's patched `libcuvs.so`, so the build
order from a fresh checkout is:

```bash
# 1. From the repo root — installs the patched libcuvs.so into $CONDA_PREFIX/lib/.
#    Only needed once per env (or after pulling cuVS changes). See top-level
#    README for env creation.
cd $REPO_ROOT
./build.sh libcuvs --install

# 2. From this directory — builds the vecflow-chamfer library + example binaries
#    against the libcuvs.so installed in step 1.
cd $REPO_ROOT/vecflow-chamfer
./build.sh vecflow-chamfer examples

# 3. (Optional) install the Python wrapper into the active env. Requires step 2
#    to have produced cpp/build/ so find_package(vecflow_chamfer) can resolve.
./build.sh python
```

Artifacts produced:
- step 2 → `cpp/build/libvecflow_chamfer.so`, `cpp/build/libvecflow_chamfer_kernels.so`, `examples/cpp/build/VECFLOW_CHAMFER`, `examples/cpp/build/CHAMFER_CORE_KERNEL`
- step 3 → `vecflow_chamfer` Python module installed into `$CONDA_PREFIX`

`./build.sh -h` lists the rest of the flags (`-j`, `-v`, `-g`, `--gpu-arch=...`).

## API Usage

End-to-end C++ — see `examples/cpp/src/vecflow_chamfer.cu` for the full driver
(the Python equivalent is `examples/python/vecflow_chamfer.py`):

```cpp
#include <vecflow_chamfer/vecflow_chamfer.hpp>
#include <raft/core/device_resources.hpp>

namespace vc = vecflow_chamfer;

int main() {
    raft::device_resources res;

    // Load docs (host memory, GPU-accessible via C2C on GH200) + queries.
    vc::dataset   ds      = vc::load_dataset("examples/datasets/lifestyle", "lifestyle.test");
    vc::query_set queries = vc::load_queries("examples/datasets/lifestyle", "lifestyle.test",
                                             ds.embedding_dim, /*tokens_per_query*/ 32);

    // Build (or deserialize) the anchor index.
    vc::index_params bp{ .n_anchors    = 500000,
                         .graph_degree = 32,
                         .itopk_size   = 256,
                         .n_iter       = 5 };
    vc::index idx = vc::build(res, ds, bp);
    // vc::serialize(res, "lifestyle.test.500000", idx);    // optional cache
    // auto idx = vc::deserialize(res, "lifestyle.test.500000");

    // Search.
    vc::search_params sp{ .itopk        = 256,
                          .search_topk  = 16,
                          .refine_rate  = 10.0f,
                          .final_topk   = 100 };
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries.num_queries, sp.final_topk);
    auto distances = raft::make_device_matrix<float,    int64_t>(res, queries.num_queries, sp.final_topk);
    vc::search(res, sp, idx, ds, queries, neighbors.view(), distances.view());

    // Optional per-stage timing breakdown.
    // vc::search_stats stats;
    // vc::search(res, sp, idx, ds, queries, neighbors.view(), distances.view(), stats);

    vc::release_dataset(ds);
}
```

Public API surface (all under `namespace vecflow_chamfer`):
- `build(res, dataset, index_params) -> index` — k-means anchors + CAGRA graph
- `search(res, search_params, index, dataset, query_set, neighbors, distances [, search_stats])` — two-stage Chamfer rerank
- `serialize(res, stem, index)` / `deserialize(res, stem) -> index` — disk cache (`<stem>.maxivf` + `<stem>.meta`)
- `load_dataset(dir, prefix) -> dataset`, `load_queries(dir, prefix, dim, tokens_per_query=32) -> query_set`, `release_dataset(dataset&)`
- `kernels.hpp` — raw Chamfer scoring kernel launchers if you want to bypass the index path

## 1. Dataset

Bundled example dataset: **lifestyle.test** (a LoTTE-derived multi-vector
collection, ColBERT-v2 embeddings, fp16, dim=128, ~32 tokens/query).
Pulled from Google Drive:

```bash
./examples/download_dataset.sh
```

Files land in `examples/datasets/lifestyle/`:

| File | Purpose |
|---|---|
| `lifestyle.test.doc.embeddings.fp16.fbin` | concatenated per-doc token embeddings (raw fp16) |
| `lifestyle.test.doc.offsets.bin` | `uint32` CSR-style offsets: doc `i` owns tokens `[off[i], off[i+1])` |
| `lifestyle.test.query.embeddings.fp16.fbin` | concatenated per-query token embeddings, fixed `tokens_per_query` (32) |

Ground truth is **not** downloaded — the example computes it on the GPU at
run time via the `chamfer_score` kernel (batched over docs, top-K via a host
min-heap per query). It tries `<prefix>.ground.truth.ibin` first and falls
back to the kernel path if the file is missing. With `output.cache_gt: true`
(the default) the kernel-computed GT is written back to that file, so the
next run skips the kernel pass. To force recompute, delete the file. To skip
GT entirely, set `output.evaluate_recall: false`.

The script installs `gdown` via `pip --user` if it isn't already on PATH.
Re-running is safe — it skips files that already exist with non-zero size.

### Using your own dataset

Stick to the same naming convention (`<prefix>.doc.embeddings.fp16.fbin`,
`<prefix>.doc.offsets.bin`, `<prefix>.query.embeddings.fp16.fbin`) and point
`dataset.dir` + `dataset.prefix` in the JSON at it. The ground-truth file is
optional — when `<prefix>.ground.truth.ibin` is absent the example computes
it on the GPU via the chamfer kernel and (with `output.cache_gt: true`)
caches it next to the dataset for reuse.

## 2. Config

`examples/cpp/config/config.json` (`examples/python/config.json` is the same shape):

```json
{
  "dataset": { "dir": "../datasets/lifestyle", "prefix": "lifestyle.test" },
  "index":   {
    "n_anchors": 500000, "graph_degree": 32, "build_itopk": 256,
    "n_iter": 5, "force_rebuild": false,
    "vpq": { "pq_bits": 0, "pq_dim": 0 }
  },
  "search":  {
    "search_itopk": [256, 256, 256, 256],
    "search_topk":  [8,   16,  24,  32],
    "final_topk":   100,
    "n_queries":    -1,
    "rerank": {
      "use_vpq":            false,
      "refine_rate_anchor": 10.0,
      "refine_rate_vpq":    2.0
    }
  },
  "output":  { "max_queries_to_print": 0, "show_performance": true,
               "evaluate_recall": true, "recall_k": 0,
               "gt_doc_batch_size": 100000, "cache_gt": true }
}
```

| Key | Meaning |
|---|---|
| `dataset.dir` | resolved relative to the config file's location, so `../datasets/lifestyle` lands at `examples/datasets/lifestyle/` |
| `index.n_anchors` | total clusters across all docs (k-means stage) |
| `index.graph_degree` | CAGRA graph degree over the anchor centroids |
| `index.build_itopk` | iterative top-K used while constructing the CAGRA graph |
| `index.n_iter` | k-means iterations during anchor construction |
| `index.vpq.pq_bits` | `0` disables VPQ side-encode; otherwise `[4..16]` (cuVS constraint, `8` is the common pick) |
| `index.vpq.pq_dim` | number of PQ subspaces; must divide `embedding_dim` |
| `search.search_itopk[i]` / `search.search_topk[i]` | paired (itopk, top-K) for sweep — each row in the search table |
| `search.final_topk` | neighbors returned per query |
| `search.n_queries` | `-1` = all queries; otherwise truncate |
| `search.rerank.use_vpq` | enable the optional VPQ rerank stage (requires the index to be built with `pq_bits > 0`) |
| `search.rerank.refine_rate_anchor` | stage-3→4 trim multiplier (`final_topk × refine_rate_anchor` candidates kept) |
| `search.rerank.refine_rate_vpq` | stage-5→6 trim multiplier when `use_vpq` is true |
| `output.recall_k` | `0` → use `final_topk` |
| `output.gt_doc_batch_size` | doc-batch size when computing exact GT |
| `output.cache_gt` | write the kernel-computed GT to `<prefix>.ground.truth.ibin` so subsequent runs skip the kernel pass (default `true`) |

Pipeline: `anchor proxy → top-K(refine_rate_anchor·final_topk) → [VPQ → top-K(refine_rate_vpq·final_topk) →] full-chamfer → top-K(final_topk)`.
The VPQ stage runs only when `search.rerank.use_vpq` is true *and* the loaded
index has VPQ data.

### When to enable VPQ

Full-precision Chamfer rerank (stage 7) reads doc token embeddings from
host RAM. The cost of that read depends on how the GPU reaches host memory:

- **Grace-Hopper / C2C-connected Superchips** — host RAM is reachable at
  NVLink-C2C bandwidth (~450 GB/s). Full-precision rerank is cheap, so the
  default `pq_bits: 0` / `use_vpq: false` is the recommended path — skip the
  VPQ stage entirely and let the full-chamfer kernel chew through the
  candidate set.
- **PCIe-attached GPUs (A100, H100 PCIe, etc.)** — host RAM is reachable
  only at PCIe bandwidth (~64 GB/s on PCIe 5). The full rerank stage becomes
  the bottleneck. Enable VPQ to narrow the candidate set in HBM with
  compressed codes before paying for the PCIe fetch: build with
  `index.vpq.pq_bits: 8`, `index.vpq.pq_dim: embedding_dim/2` (`64` for
  ColBERT-v2), and search with `search.rerank.use_vpq: true`.

Index caches land flat next to the dataset as two files sharing a stem
`<prefix>.<n_anchors>`:

| File | Contents |
|---|---|
| `<prefix>.<n_anchors>.maxivf` | CAGRA index over the anchor centroids (cuVS native format) |
| `<prefix>.<n_anchors>.meta`   | vecflow-chamfer sidecar: anchor labels, inverted lists, doc offsets, optional VPQ codebook + codes |

Subsequent runs deserialize from disk unless `force_rebuild: true`.

## 3. Run

### C++

```bash
./examples/cpp/build/VECFLOW_CHAMFER                          # default config (examples/cpp/config/config.json)
./examples/cpp/build/VECFLOW_CHAMFER path/to/config.json      # custom config
```

### Python

Requires step 3 of [Install → Build from source](#option-b--build-from-source) (the Python wrapper):

```bash
python examples/python/vecflow_chamfer.py                   # default config (examples/python/config.json)
python examples/python/vecflow_chamfer.py path/to/config.json
```

Both drivers do the same thing, per the config above:

1. Install a 2 GiB RMM pool, construct `raft::device_resources`.
2. Load embeddings (doc embeddings stay in system RAM, GPU-accessible via
   C2C on GH200; no HBM staging copy).
3. Build (or deserialize) the hierarchical anchor index.
4. Warmup, then run search for each `(search_itopk, search_topk)` pair.
5. Print per-stage latency breakdown, throughput, and recall@K.

The C++ driver additionally computes/caches exact ground truth on the GPU
via the `chamfer_score` kernel (the `generate_ground_truth_exact` helper
isn't exposed through the pybind module). The Python driver loads
`<prefix>.ground.truth.ibin` if it exists — run the C++ example once with
`output.cache_gt: true` (the default) and Python will pick up the cached
GT file on its next run. Without it, Python reports timing only and skips
recall.

Sample output (NVIDIA GH200, bundled lifestyle.test dataset; absolute
numbers depend on GPU and dataset):

```
[4/4] Search
      queries:         661
      configurations:  4

  --- config 2/4 ---
      params:          itopk=64  search_topk=16  refine_rate_anchor=10  final_topk=100  rerank=anchor->full
      per-query latency:
        stage 1   CAGRA candidate search                0.229 ms
        stage 2   hash-table dedup                      0.039 ms
        stage 3   anchor-chamfer proxy rerank           0.170 ms
        stage 4   top-K  (4307 -> 1000)                 0.040 ms
        stage 5   full-chamfer rerank                   0.154 ms
        stage 6   top-K  (1000 -> 100)                  0.030 ms
        --------------------------------------------------------
        total                                           0.676 ms

      throughput:         1397.1 q/s   (e2e 1415.1 q/s)
      candidates/query:   3895.0 (avg)
      recall@100:         96.54 %
  ...
```

## 4. Raw chamfer kernel benchmark

`CHAMFER_CORE_KERNEL` exercises just the chamfer scoring kernel —
synthetic data, no anchor index, CPU verification. Useful for kernel-level
profiling.

### C++

Driven by `examples/cpp/config/config_kernel.json`:

```bash
./examples/cpp/build/CHAMFER_CORE_KERNEL
./examples/cpp/build/CHAMFER_CORE_KERNEL path/to/config_kernel.json
```

### Python

Driven by `examples/python/config_kernel.json`. The Python wrapper
exposes the raw kernel via the `vc.ChamferKernelBench` class — construct
it with **GPU arrays** (anything implementing `__cuda_array_interface__`:
cupy arrays, torch CUDA tensors, numba device arrays) and the bench
uses their device pointers directly, with no H↔D inside the timing loop.
`.run()` is kernel-only; `.scores()` pulls the latest score buffer back
to numpy. The example uses cupy for the upload step — install with
`mamba install cupy` if it isn't already in the env. The CPU reference
uses a single numpy matmul instead of the C++ triple loop, so
verification is fast even at `num_docs=10000`.

```bash
python examples/python/chamfer_core_kernel.py
python examples/python/chamfer_core_kernel.py path/to/config_kernel.json
```
