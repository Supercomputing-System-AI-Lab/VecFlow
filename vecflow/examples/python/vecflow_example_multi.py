#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SIFT1M example for VecFlow's multi-label AND search.

Each query carries TWO labels; a dataset point is a candidate only if it
contains BOTH. The example:
  1. Builds the VecFlow index with multi_label=True so the CSR label arrays
     are prepared internally.
  2. Generates ground truth via brute force AND (cached on disk).
  3. Sweeps itopk_size and prints per-itopk QPS, latency, and recall.

Generate query_multi.txt first via `generate_multi_query.py` if it doesn't
exist.
"""
import argparse
import ctypes
import json
import os
import time

import numpy as np
import cupy as cp

from vecflow import VecFlow

# Reuse loaders from the single-label example so we don't duplicate code.
from vecflow_example import load_fbin, load_labels_auto, compute_recall


def brute_force_and_ground_truth(dataset, queries, data_labels,
                                 query_labels_a, query_labels_b, topk,
                                 cache_path=None):
    """Top-k nearest neighbors by L2, restricted to points that have BOTH
    query labels. Cached to `cache_path` (.ibin) as a (n_q, topk) int32 matrix.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached AND ground truth: {cache_path}")
        with open(cache_path, "rb") as f:
            n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            k = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            gt = np.fromfile(f, dtype=np.int32, count=n * k).reshape(n, k)
        if n == queries.shape[0] and k == topk:
            return gt

    print("  computing AND ground truth (brute force)...")
    # Per-point label sets (Python sets for O(1) AND lookup).
    label_sets = [set(ls) for ls in data_labels]

    n_q = queries.shape[0]
    gt = np.full((n_q, topk), -1, dtype=np.int32)
    for qi in range(n_q):
        if (qi % 500) == 0:
            print(f"    {qi}/{n_q}", end="\r")
        a, b = int(query_labels_a[qi]), int(query_labels_b[qi])
        # Candidate indices: points containing BOTH a and b.
        cand = [i for i, s in enumerate(label_sets) if (a in s) and (b in s)]
        if not cand:
            continue
        diffs = dataset[cand] - queries[qi]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        k_eff = min(topk, len(cand))
        order = np.argpartition(d2, k_eff - 1)[:k_eff]
        order = order[np.argsort(d2[order])]
        gt[qi, :k_eff] = np.asarray(cand, dtype=np.int32)[order]
    print()

    if cache_path:
        print(f"  caching AND ground truth: {cache_path}")
        with open(cache_path, "wb") as f:
            np.array([n_q, topk], dtype=np.int32).tofile(f)
            gt.tofile(f)
    return gt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="config_multi.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_dir          = cfg["data_dir"]
    data_fname        = os.path.join(data_dir, cfg["data_fname"])
    query_fname       = os.path.join(data_dir, cfg["query_fname"])
    data_label_fname  = os.path.join(data_dir, cfg["data_label_fname"])
    query_label_fname = os.path.join(data_dir, cfg["query_label_fname"])
    ivf_graph_fname   = os.path.join(data_dir, cfg["ivf_graph_fname"])
    ivf_bfs_fname     = os.path.join(data_dir, cfg["ivf_bfs_fname"])
    gt_fname          = os.path.join(data_dir, cfg["ground_truth_fname"])

    itopk_raw         = cfg.get("itopk_size", 32)
    itopk_sizes       = ([int(v) for v in itopk_raw] if isinstance(itopk_raw, (list, tuple))
                         else [int(itopk_raw)])
    spec_threshold    = cfg.get("spec_threshold", 1000)
    graph_degree      = cfg.get("graph_degree", 16)
    topk              = cfg.get("topk", 10)
    num_runs          = cfg.get("num_runs", 100)
    warmup_runs       = cfg.get("warmup_runs", 5)
    force_rebuild     = cfg.get("force_rebuild", False)

    print("=== Configuration ===")
    print(f"  itopk sizes:           {itopk_sizes}")
    print(f"  specificity threshold: {spec_threshold}")
    print(f"  graph degree:          {graph_degree}")
    print(f"  topk:                  {topk}")
    print(f"  num_runs / warmup:     {num_runs} / {warmup_runs}")

    print("\n=== Loading Dataset ===")
    dataset = load_fbin(data_fname)
    queries = load_fbin(query_fname)
    data_labels, _ = load_labels_auto(data_label_fname)
    query_labels, _ = load_labels_auto(query_label_fname)

    print(f"  base:  N={dataset.shape[0]}, dim={dataset.shape[1]}")
    print(f"  query: N={queries.shape[0]}, dim={queries.shape[1]}")

    # Filter to queries that carry exactly 2 labels. The generator emits an
    # empty row for queries whose AND result set is too small (< --min-and-size);
    # those are dropped here so every retained query has a guaranteed-large
    # AND intersection per the generator's threshold.
    keep_idx = [i for i, ls in enumerate(query_labels) if len(ls) >= 2]
    n_total = len(query_labels)
    if not keep_idx:
        raise SystemExit(
            f"ERROR: 0 queries with ≥2 labels in {query_label_fname}.\n"
            f"       Regenerate with a lower --min-and-size."
        )
    if len(keep_idx) != n_total:
        print(f"  WARNING: dropping {n_total - len(keep_idx)} queries with <2 labels; "
              f"using {len(keep_idx)} valid queries")

    queries        = queries[keep_idx]
    query_labels_a = np.array([query_labels[i][0] for i in keep_idx], dtype=np.uint32)
    query_labels_b = np.array([query_labels[i][1] for i in keep_idx], dtype=np.uint32)

    print("\n=== Building VecFlow Index (multi_label=True) ===")
    vf = VecFlow()
    t0 = time.time()
    vf.build(dataset,
             data_labels,
             graph_degree,
             spec_threshold,
             ivf_graph_fname,
             ivf_bfs_fname,
             force_rebuild,
             True)  # multi_label
    print(f"  build time: {time.time() - t0:.2f} s")

    # Push queries + labels to device (the binding expects GPU-resident inputs).
    print("\n=== Moving queries to GPU ===")
    queries_gpu = cp.asarray(queries, dtype=cp.float32)
    qa_gpu = cp.asarray(query_labels_a, dtype=cp.uint32)
    qb_gpu = cp.asarray(query_labels_b, dtype=cp.uint32)
    # Wrap as numpy views over the device pointers for the binding.
    queries_np_gpu = np.ctypeslib.as_array(
        ctypes.cast(queries_gpu.data.ptr, ctypes.POINTER(ctypes.c_float)),
        shape=queries_gpu.shape)
    qa_np_gpu = np.ctypeslib.as_array(
        ctypes.cast(qa_gpu.data.ptr, ctypes.POINTER(ctypes.c_uint32)),
        shape=qa_gpu.shape)
    qb_np_gpu = np.ctypeslib.as_array(
        ctypes.cast(qb_gpu.data.ptr, ctypes.POINTER(ctypes.c_uint32)),
        shape=qb_gpu.shape)

    print("\n=== Generating Ground Truth (AND, brute force, cached) ===")
    gt = brute_force_and_ground_truth(dataset, queries, data_labels,
                                      query_labels_a, query_labels_b,
                                      topk, cache_path=gt_fname)

    print("\n=== Performing Search Sweep ===")
    for itopk in itopk_sizes:
        for _ in range(warmup_runs):
            _, _ = vf.search_multi(queries_np_gpu, qa_np_gpu, qb_np_gpu, itopk)

        t_start = time.perf_counter()
        for _ in range(num_runs):
            neighbors, distances = vf.search_multi(queries_np_gpu, qa_np_gpu, qb_np_gpu, itopk)
        total = time.perf_counter() - t_start

        avg_ms = (total * 1000) / num_runs
        qps    = num_runs * queries.shape[0] / total
        recall = compute_recall(neighbors, gt)
        print(f"  itopk={itopk:>4}  qps={qps:>10.1f}  avg={avg_ms:>6.3f} ms  recall={recall:.4f}")


if __name__ == "__main__":
    main()
