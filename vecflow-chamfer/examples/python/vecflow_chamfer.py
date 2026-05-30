#!/usr/bin/env python
"""End-to-end vecflow-chamfer demo driven by a JSON config.

Mirrors examples/cpp/src/vecflow_chamfer.cu — same pipeline, same config
shape, similar console output:

    load_dataset -> build (or deserialize) -> warmup ->
    search sweep over (search_itopk, search_topk) pairs ->
    per-stage timings, optional recall@K vs. ground truth

Ground truth is loaded from `<prefix>.ground.truth.ibin` if present (the
C++ example writes it on the first run with `output.cache_gt: true`). If
the file is missing the Python example skips recall and reports timing
only — generating exact GT is a C++-side helper that isn't exposed
through the pybind module.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# This script is named `vecflow_chamfer.py` to mirror the C++ source filename
# (examples/cpp/src/vecflow_chamfer.cu), but Python puts the script's own
# directory at sys.path[0] before resolving any import, so `import vecflow_chamfer`
# would otherwise find *this file* first and shadow the installed package.
# Strip the script's directory from sys.path so the import resolves to the
# real package in site-packages.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import vecflow_chamfer as vc


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    # Resolve a relative dataset.dir against the config file's directory,
    # matching the C++ driver's behavior.
    ds_dir = Path(cfg["dataset"]["dir"])
    if not ds_dir.is_absolute():
        ds_dir = (path.resolve().parent / ds_dir).resolve()
    cfg["dataset"]["dir"] = str(ds_dir)
    return cfg


def load_ground_truth(dataset_dir: str, prefix: str) -> tuple[list[list[int]], int] | None:
    """Read the `<prefix>.ground.truth.ibin` file written by the C++ example.

    Format: [num_queries:int32, top_k:int32] header, then a stream of
    (query_id:int32, doc_id:int32, score:float32) triplets in descending
    score order per query.

    Returns (per_query_doc_ids, top_k) or None if the file is missing.
    """
    path = Path(dataset_dir) / f"{prefix}.ground.truth.ibin"
    if not path.exists():
        return None

    raw = path.read_bytes()
    num_queries, top_k = struct.unpack_from("<ii", raw, 0)
    per_query: list[list[int]] = [[] for _ in range(num_queries)]

    triplet = struct.Struct("<iif")
    offset = 8  # past the header
    while offset + triplet.size <= len(raw):
        qid, did, _score = triplet.unpack_from(raw, offset)
        offset += triplet.size
        if 0 <= qid < num_queries:
            per_query[qid].append(did)
    return per_query, top_k


def recall_at_k(results: np.ndarray, gt_per_query: list[list[int]], k: int) -> float:
    n = min(results.shape[0], len(gt_per_query))
    if n == 0:
        return 0.0
    hit, denom = 0, 0
    for q in range(n):
        truth = set(gt_per_query[q][:k])
        if not truth:
            continue
        denom += len(truth)
        hit += sum(1 for d in results[q, :k] if int(d) in truth)
    return hit / denom if denom else 0.0


def kv(label: str, value, width: int = 17) -> None:
    print(f"      {label:<{width}}{value}")


def run(config_path: Path) -> int:
    cfg = load_config(config_path)
    dataset_dir = cfg["dataset"]["dir"]
    prefix      = cfg["dataset"]["prefix"]
    idx_cfg     = cfg["index"]
    srh_cfg     = cfg["search"]
    out_cfg     = cfg.get("output", {})

    vpq_build_enabled  = idx_cfg.get("vpq", {}).get("enabled", False)
    pq_bits            = idx_cfg.get("vpq", {}).get("pq_bits", 8) if vpq_build_enabled else 0
    pq_dim             = idx_cfg.get("vpq", {}).get("pq_dim",  32) if vpq_build_enabled else 0
    vpq_search_enabled = srh_cfg.get("vpq", {}).get("enabled", False)
    refine_rate_vpq    = srh_cfg.get("vpq", {}).get("refine_rate", 1.5)
    tokens_per_query   = cfg.get("queries", {}).get("tokens_per_query", 32)

    if vpq_search_enabled and not vpq_build_enabled:
        raise RuntimeError("search.vpq.enabled=true requires index.vpq.enabled=true")

    print("=" * 60)
    print("  vecflow-chamfer multi-vector search demo (Python)")
    print("=" * 60)
    print(f"  config:   {config_path}")
    print(f"  dataset:  {dataset_dir}/{prefix}")

    print("\n[1/4] Load dataset")
    t0 = time.perf_counter()
    dataset = vc.Dataset.load(dataset_dir, prefix)
    queries = vc.QuerySet.load(dataset_dir, prefix, dataset.embedding_dim, tokens_per_query)
    load_secs = time.perf_counter() - t0
    kv("docs:",         dataset.num_docs)
    kv("dim:",          dataset.embedding_dim)
    kv("doc tokens:",   dataset.num_doc_tokens)
    kv("queries:",      queries.num_queries)
    kv("tokens/query:", queries.num_tokens_per_query)
    kv("elapsed:",      f"{load_secs:.2f} s")

    print("\n[2/4] Build index")
    kv("params:",
       f"n_anchors={idx_cfg['n_anchors']}  graph_degree={idx_cfg['graph_degree']}"
       f"  itopk_size={idx_cfg.get('itopk_size', idx_cfg.get('build_itopk', 256))}"
       f"  n_iter={idx_cfg['n_iter']}")
    if vpq_build_enabled:
        kv("vpq:", f"pq_bits={pq_bits}  pq_dim={pq_dim}")

    vpq_tag = f".pq{pq_bits}x{pq_dim}" if vpq_build_enabled else ".novpq"
    cache_stem = f"{dataset_dir}/{prefix}.{idx_cfg['n_anchors']}{vpq_tag}"
    kv("cache stem:", cache_stem)

    cache_present = Path(f"{cache_stem}.maxivf").exists() and Path(f"{cache_stem}.meta").exists()
    force_rebuild = idx_cfg.get("force_rebuild", False)
    use_cache = cache_present and not force_rebuild

    t0 = time.perf_counter()
    if use_cache:
        index = vc.deserialize(cache_stem)
    else:
        params = vc.IndexParams(
            n_anchors    = idx_cfg["n_anchors"],
            graph_degree = idx_cfg["graph_degree"],
            itopk_size   = idx_cfg.get("itopk_size", idx_cfg.get("build_itopk", 256)),
            n_iter       = idx_cfg["n_iter"],
            pq_bits      = pq_bits,
            pq_dim       = pq_dim,
        )
        index = vc.build(dataset, params)
        vc.serialize(index, cache_stem)
    build_secs = time.perf_counter() - t0
    kv("elapsed:", f"{build_secs:.2f} s" + ("   (loaded from cache)" if use_cache else ""))

    # Truncate queries if requested.
    n_q_req = srh_cfg.get("n_queries", -1)
    actual_n_queries = queries.num_queries if n_q_req < 0 else min(n_q_req, queries.num_queries)
    if actual_n_queries != queries.num_queries:
        print(f"      (using {actual_n_queries}/{queries.num_queries} queries)")

    print("\n[3/4] Ground truth")
    gt = load_ground_truth(dataset_dir, prefix)
    if gt is None:
        print("      not found — recall will be skipped")
        print(f"      (run the C++ example once with output.cache_gt=true to populate")
        print(f"       {dataset_dir}/{prefix}.ground.truth.ibin)")
        gt_per_query, gt_top_k = None, 0
    else:
        gt_per_query, gt_top_k = gt
        print(f"      loaded {len(gt_per_query)} queries x top-{gt_top_k}")

    final_topk = int(srh_cfg["final_topk"])
    recall_k   = int(out_cfg.get("recall_k", 0)) or final_topk

    # Warmup (10 queries on the first configuration).
    warmup_n = min(10, actual_n_queries)
    if warmup_n > 0:
        warm_sp = vc.SearchParams(
            itopk            = int(srh_cfg["search_itopk"][0]),
            search_topk      = int(srh_cfg["search_topk"][0]),
            refine_rate      = float(srh_cfg.get("refine_rate_anchor", 10.0)),
            final_topk       = final_topk,
            use_vpq          = vpq_search_enabled,
            refine_rate_vpq  = refine_rate_vpq,
        )
        # QuerySet has no slice() — warmup just uses the full set; the search
        # call itself is the warmup. Acceptable: it kicks the JIT/CUDA caches.
        vc.search(index, dataset, queries, warm_sp)

    print(f"\n[4/4] Search")
    print(f"      queries:         {actual_n_queries}")
    print(f"      configurations:  {len(srh_cfg['search_itopk'])}")

    for i, (s_itopk, s_topk) in enumerate(zip(srh_cfg["search_itopk"], srh_cfg["search_topk"])):
        print(f"\n  --- config {i + 1}/{len(srh_cfg['search_itopk'])} ---")
        rerank = "anchor->vpq->full" if vpq_search_enabled else "anchor->full"
        print(f"      params:          itopk={s_itopk}  search_topk={s_topk}"
              f"  refine_rate_anchor={srh_cfg.get('refine_rate_anchor', 10.0)}"
              f"  final_topk={final_topk}  rerank={rerank}")

        sp = vc.SearchParams(
            itopk           = int(s_itopk),
            search_topk     = int(s_topk),
            refine_rate     = float(srh_cfg.get("refine_rate_anchor", 10.0)),
            final_topk      = final_topk,
            use_vpq         = vpq_search_enabled,
            refine_rate_vpq = refine_rate_vpq,
        )
        neighbors, _distances, stats = vc.search(index, dataset, queries, sp, with_stats=True)

        # E2E re-run without with_stats so the per-stage sync points don't
        # show up in the wall-clock number. The binding's H2D copy syncs the
        # raft stream; cudaDeviceSynchronize() afterwards guarantees no other
        # stream is still in flight.
        e2e_t0 = time.perf_counter()
        vc.search(index, dataset, queries, sp, with_stats=False)
        vc.cuda_sync()
        e2e_total_ms     = (time.perf_counter() - e2e_t0) * 1000.0
        e2e_per_query_ms = e2e_total_ms / actual_n_queries
        e2e_qps          = actual_n_queries * 1000.0 / e2e_total_ms

        if out_cfg.get("show_performance", True):
            def row(label, ms): print(f"        {label:<46}{ms:7.3f} ms")
            print("      per-query latency:")
            row("stage 1   CAGRA candidate search",      stats.avg_candidate_generation_ms)
            row("stage 2   hash-table dedup",            stats.avg_unique_id_finding_ms)
            row("stage 3   anchor-chamfer proxy rerank", stats.avg_anchor_chamfer_reranking_ms)
            row(f"stage 4   top-K  ({int(stats.avg_stage4_input_count)} -> "
                f"{int(stats.avg_stage4_output_count)})",
                stats.avg_first_sorting_ms)
            full_stage = 7 if vpq_search_enabled else 5
            final_stage = 8 if vpq_search_enabled else 6
            if vpq_search_enabled:
                row("stage 5   vpq-chamfer rerank", stats.avg_vpq_reranking_ms)
                row(f"stage 6   top-K  ({int(stats.avg_stage4_output_count)} -> "
                    f"{int(stats.avg_stage_vpq_output_count)})",
                    stats.avg_vpq_sorting_ms)
            row(f"stage {full_stage}   full-chamfer rerank", stats.avg_full_chamfer_reranking_ms)
            in_cnt = (int(stats.avg_stage_vpq_output_count) if vpq_search_enabled
                      else int(stats.avg_stage4_output_count))
            row(f"stage {final_stage}   top-K  ({in_cnt} -> "
                f"{int(stats.avg_stage6_output_count)})",
                stats.avg_second_sorting_ms)
            print("        " + "-" * 56)
            row("total       (sum, per-stage sync overhead)", stats.avg_per_query_ms)
            row("total e2e   (one search call, no breakdown)", e2e_per_query_ms)
            print()
            kv("throughput:",       f"{stats.throughput_qps:.1f} q/s   "
                                     f"(e2e {e2e_qps:.1f} q/s)", width=20)
            kv("candidates/query:", f"{stats.avg_candidates_per_query:.1f} (avg)", width=20)

        if gt_per_query is not None:
            r = recall_at_k(neighbors[:actual_n_queries], gt_per_query, recall_k)
            kv(f"recall@{recall_k}:", f"{r * 100.0:.2f} %", width=20)

    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="vecflow-chamfer Python end-to-end demo")
    default_cfg = Path(__file__).resolve().parent / "config.json"
    p.add_argument("config", nargs="?", default=str(default_cfg),
                   help="path to config.json (default: %(default)s)")
    args = p.parse_args(argv)
    return run(Path(args.config))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
