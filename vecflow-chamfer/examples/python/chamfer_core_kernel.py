#!/usr/bin/env python
"""Direct usage of the public chamfer_score kernel from Python.

Mirrors examples/cpp/src/chamfer_core_kernel.cu — generates a random fp16
query (tokens_per_query x dim) and a random fp16 dataset (num_docs x
tokens_per_doc x dim), uploads them once into a ``vc.ChamferKernelBench``
harness, then calls ``bench.run()`` in a warmup + timing loop. Inputs stay
device-resident across iterations, so the per-iteration timing reflects
pure kernel cost (no H↔D copy). Optionally verifies the GPU scores against
a NumPy fp32 reference.

Usage:
    python chamfer_core_kernel.py [config_kernel.json]
    (defaults to ./config_kernel.json relative to this script)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# `examples/python/vecflow_chamfer.py` (the e2e example) lives next to this
# script and shares the package's name. Python adds the script's own directory
# to sys.path[0] before running, so `import vecflow_chamfer` would resolve to
# that file rather than the installed package. Strip the directory first.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import vecflow_chamfer as vc


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    return cfg


def cpu_chamfer_score_all(query_f32: np.ndarray, dataset_f32: np.ndarray,
                          doc_offsets: np.ndarray) -> np.ndarray:
    """Reference Chamfer score per doc, in fp32.

    Score = sum_i max_j <q_i, d_j>. Computed via a single (Tq x total_tokens)
    matmul plus a per-doc reduce — vastly faster than the C++ triple loop
    while staying pure numpy.
    """
    # q.dot(d.T) -> [tokens_per_query, total_doc_tokens]
    sim = query_f32 @ dataset_f32.T

    n_docs = doc_offsets.shape[0] - 1
    scores = np.empty(n_docs, dtype=np.float32)
    for d in range(n_docs):
        s, e = int(doc_offsets[d]), int(doc_offsets[d + 1])
        # per-doc slice: [Tq, tokens_in_doc]
        scores[d] = sim[:, s:e].max(axis=1).sum()
    return scores


def run(config_path: Path) -> int:
    cfg = load_config(config_path)

    num_docs         = int(cfg["data"]["num_docs"])
    tokens_per_doc   = int(cfg["data"]["tokens_per_doc"])
    dim              = int(cfg["data"]["dim"])
    tokens_per_query = int(cfg["data"]["tokens_per_query"])
    warmup_runs      = int(cfg["benchmark"]["warmup_runs"])
    num_runs         = int(cfg["benchmark"]["num_runs"])
    sample_compare   = int(cfg.get("output", {}).get("sample_compare_count", 10))
    verify_cpu       = bool(cfg.get("output", {}).get("verify_against_cpu", True))

    print(f"[config] {config_path}")
    print(f"  num_docs={num_docs}, tokens_per_doc={tokens_per_doc}, dim={dim}, "
          f"tokens_per_query={tokens_per_query}, "
          f"warmup_runs={warmup_runs}, num_runs={num_runs}")

    rng = np.random.default_rng(42)
    # Generate in fp32 (for the CPU reference) then cast to fp16 for the kernel.
    query_f32   = rng.uniform(-1.0, 1.0, size=(tokens_per_query, dim)).astype(np.float32)
    dataset_f32 = rng.uniform(-1.0, 1.0,
                              size=(num_docs * tokens_per_doc, dim)).astype(np.float32)
    query_f16   = query_f32.astype(np.float16)
    dataset_f16 = dataset_f32.astype(np.float16)

    doc_offsets = np.arange(num_docs + 1, dtype=np.uint32) * tokens_per_doc
    doc_ids     = np.arange(num_docs, dtype=np.uint32)

    # Upload to the GPU once via cupy. ChamferKernelBench takes any object
    # implementing __cuda_array_interface__ and uses its device pointer
    # directly — no H→D inside the bench, no copies during the timing loop.
    query_gpu       = cp.asarray(query_f16)
    dataset_gpu     = cp.asarray(dataset_f16)
    doc_offsets_gpu = cp.asarray(doc_offsets)
    doc_ids_gpu     = cp.asarray(doc_ids)

    bench = vc.ChamferKernelBench(query_gpu, dataset_gpu, doc_offsets_gpu, doc_ids_gpu,
                                  tokens_per_query=tokens_per_query, dim=dim)

    # Warmup — discard timing, just want the GPU caches hot.
    for _ in range(warmup_runs):
        bench.run()
    vc.cuda_sync()

    # Timed runs. Kernel-only — no H↔D copy inside the loop.
    t0 = time.perf_counter()
    for _ in range(num_runs):
        bench.run()
    vc.cuda_sync()
    avg_ms = (time.perf_counter() - t0) * 1000.0 / num_runs
    gpu_scores = bench.scores()  # one D→H, after the loop

    print(f"\n[kernel] avg time:   {avg_ms:.4f} ms for {num_docs} docs")
    print(f"[kernel] throughput: {num_docs * 1000.0 / avg_ms:.2f} docs/s")

    if verify_cpu:
        cpu_t0 = time.perf_counter()
        cpu_scores = cpu_chamfer_score_all(query_f32, dataset_f32, doc_offsets)
        cpu_ms = (time.perf_counter() - cpu_t0) * 1000.0
        print(f"[cpu] verify time: {cpu_ms:.2f} ms")

        diff = np.abs(gpu_scores.astype(np.float64) - cpu_scores.astype(np.float64))
        print(f"[verify] mean_abs_diff={diff.mean():.6f}, max_abs_diff={diff.max():.6f}")

        show = min(sample_compare, num_docs)
        print(f"\nSample comparisons (first {show}):")
        print("DocID  GPU          CPU          Diff")
        for i in range(show):
            d = float(gpu_scores[i]) - float(cpu_scores[i])
            print(f"{i:5d}  {gpu_scores[i]:12.6f}  {cpu_scores[i]:12.6f}  {d:12.6f}")

        idx = int(diff.argmax())
        print(f"\nWorst at DocID {idx}: GPU={gpu_scores[idx]}, "
              f"CPU={cpu_scores[idx]}, Diff={gpu_scores[idx] - cpu_scores[idx]}")

    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="vecflow-chamfer raw kernel benchmark (Python)")
    default_cfg = Path(__file__).resolve().parent / "config_kernel.json"
    p.add_argument("config", nargs="?", default=str(default_cfg),
                   help="path to config_kernel.json (default: %(default)s)")
    args = p.parse_args(argv)
    return run(Path(args.config))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
