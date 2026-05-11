#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate `query_multi.txt` (+ optional `.spmat`) from an existing single-label
query file plus a base-label file.

The bundled SIFT1M `query.txt` has one label per query. For multi-label AND
search we need at least two AND we want the AND result set to be large
enough to be a meaningful benchmark target. This script picks a SECOND
label per query such that the intersection
        points(primary)  ∩  points(secondary)
has at least `--min-and-size` members (default 50). Queries for which no
such secondary exists are emitted with just the primary label and counted —
inspect the summary at the end and either lower the threshold or remove
those queries from the run.

The secondary is sampled *weighted by the primary-label frequency observed
in the original single-label `query.txt`* — so the multi-label query set
inherits the same skewed distribution as the single-label set, rather than
being uniform over qualifying labels.

Usage:
  python generate_multi_query.py \\
      --base-labels  ../datasets/sift1M/base.txt   \\
      --query-labels ../datasets/sift1M/query.txt  \\
      --out-txt      ../datasets/sift1M/query_multi.txt  \\
      [--out-spmat   ../datasets/sift1M/query_multi.spmat] \\
      [--min-and-size 50]  [--seed 42]
"""
import argparse
import os
import random
import struct
from collections import defaultdict


def load_txt_labels(fname):
    """Return list[list[int]], one entry per row; '-1' rows become []."""
    out = []
    with open(fname) as f:
        for line in f:
            s = line.strip()
            if not s or s == "-1":
                out.append([])
            else:
                out.append([int(x) for x in s.split(",")])
    return out


def write_txt_labels(fname, rows):
    with open(fname, "w") as f:
        for r in rows:
            if not r:
                f.write("-1\n")
            else:
                f.write(",".join(str(x) for x in r) + "\n")


def write_spmat(fname, rows, ncol):
    """CSR binary: header(nrow,ncol,nnz: int64) + indptr(int64) + indices(int32)."""
    nrow = len(rows)
    indptr = [0]
    indices = []
    for r in rows:
        indices.extend(r)
        indptr.append(indptr[-1] + len(r))
    nnz = indptr[-1]
    with open(fname, "wb") as f:
        f.write(struct.pack("<qqq", nrow, ncol, nnz))
        f.write(struct.pack(f"<{nrow + 1}q", *indptr))
        f.write(struct.pack(f"<{nnz}i", *indices))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-labels", required=True, help="base.txt — labels per dataset point")
    ap.add_argument("--query-labels", required=True, help="query.txt — labels per query")
    ap.add_argument("--out-txt", required=True, help="output query_multi.txt")
    ap.add_argument("--out-spmat", default=None, help="optional output query_multi.spmat")
    ap.add_argument("--min-and-size", type=int, default=50,
                    help="Required minimum AND result-set size (intersection of points "
                         "having BOTH labels). Default: 50.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[gen] reading base labels: {args.base_labels}")
    base_rows = load_txt_labels(args.base_labels)
    print(f"[gen] reading query labels: {args.query_labels}")
    query_rows = load_txt_labels(args.query_labels)

    # Build label → set of point indices that carry that label. We'll use these
    # sets to compute candidate-intersection sizes per (primary, secondary)
    # pair and only keep pairs with intersection ≥ args.min_and_size.
    print("[gen] indexing base label → point sets")
    label_to_points = defaultdict(set)
    for pi, labels in enumerate(base_rows):
        for l in labels:
            label_to_points[l].add(pi)

    # Co-occurrence map: candidate secondaries per primary label.
    cooc = defaultdict(set)
    for labels in base_rows:
        for li in labels:
            for lj in labels:
                if li != lj:
                    cooc[li].add(lj)

    # Compute the overall label vocabulary for the spmat header.
    max_label = 0
    for r in base_rows + query_rows:
        if r:
            max_label = max(max_label, max(r))
    ncol = max_label + 1

    # Empirical distribution of PRIMARY labels in the single-label query file —
    # we use this to weight the secondary-label sampling below so the resulting
    # multi-label queries inherit the same skew as the single-label queries.
    query_label_freq = defaultdict(int)
    for q in query_rows:
        if q:
            query_label_freq[q[0]] += 1

    # Precompute pairwise intersection sizes ONCE for every (a, b) label pair
    # that ever co-occurs. Per-query lookup is then O(1) instead of an O(|set|)
    # rescan, which is the bulk of the speedup.
    print(f"[gen] precomputing pairwise intersection sizes "
          f"(threshold ≥ {args.min_and_size}) ...")
    pair_size = {}                          # (a, b) -> |points(a) ∩ points(b)|
    seen_pairs = set()                      # de-dup work; (min,max) ordered
    n_pairs_done = 0
    for primary, cands in cooc.items():
        pts_primary = label_to_points.get(primary, set())
        for cand in cands:
            key = (primary, cand) if primary < cand else (cand, primary)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pts_cand = label_to_points.get(cand, set())
            sz = len(pts_primary & pts_cand)  # set & is C-level, fast
            pair_size[(primary, cand)] = sz
            pair_size[(cand, primary)] = sz
            n_pairs_done += 1
    print(f"[gen]   {n_pairs_done} unique label pairs cached")

    # Build the output: for each query, keep the primary and pick a secondary
    # whose AND-intersection with the primary is ≥ min_and_size. Sampling
    # is WEIGHTED by `query_label_freq` so the secondary label distribution
    # matches the primary label distribution in `query.txt`.
    print(f"[gen] picking secondaries (weighted by primary-label freq in query.txt)")
    out_rows = []
    n_no_primary = 0
    n_no_qualifying = 0
    intersection_sizes = []
    for qi, q in enumerate(query_rows):
        if not q:
            out_rows.append([])
            n_no_primary += 1
            continue
        primary = q[0]

        # Enumerate co-occurring labels and keep only those whose AND
        # intersection with the primary is large enough — O(1) lookup per
        # candidate via the precomputed `pair_size` table.
        qualifying = []  # list of (candidate, intersection_size)
        for cand in cooc.get(primary, []):
            isize = pair_size.get((primary, cand), 0)
            if isize >= args.min_and_size:
                qualifying.append((cand, isize))

        if qualifying:
            # Weighted sample by primary-label frequency observed in query.txt.
            # `+1` is a small smoothing so labels never observed as a primary
            # still have a non-zero sampling probability.
            cands_only = [c for c, _ in qualifying]
            weights = [query_label_freq.get(c, 0) + 1 for c in cands_only]
            chosen = random.choices(cands_only, weights=weights, k=1)[0]
            chosen_size = next(s for c, s in qualifying if c == chosen)
            out_rows.append([primary, chosen])
            intersection_sizes.append(chosen_size)
        else:
            # No secondary produces a large-enough intersection — emit an
            # empty row ("-1" in the .txt format). Downstream examples auto-
            # skip these so every retained query is guaranteed ≥ min_and_size
            # AND candidates.
            out_rows.append([])
            n_no_qualifying += 1

    n_pairs = sum(1 for r in out_rows if len(r) == 2)
    print(f"[gen] queries with empty label list:    {n_no_primary}")
    print(f"[gen] queries with no qualifying pair:  {n_no_qualifying}  "
          f"(no secondary gave intersection ≥ {args.min_and_size})")
    print(f"[gen] queries with 2 labels:            {n_pairs}")
    if intersection_sizes:
        a = sorted(intersection_sizes)
        print(f"[gen] AND intersection size — min: {a[0]}, median: {a[len(a)//2]}, "
              f"max: {a[-1]}, mean: {sum(a)/len(a):.1f}")

    # Side-by-side: primary-label distribution (from query.txt) vs secondary-label
    # distribution (just generated). The two should look similar if the weighted
    # sampling is honoring the skew.
    sec_freq = defaultdict(int)
    for r in out_rows:
        if len(r) >= 2:
            sec_freq[r[1]] += 1
    n_top = 10
    primary_top = sorted(query_label_freq.items(), key=lambda x: -x[1])[:n_top]
    secondary_top = sorted(sec_freq.items(),         key=lambda x: -x[1])[:n_top]
    print(f"\n[gen] Top {n_top} primary labels (from query.txt) "
          f"vs top {n_top} secondary labels (this run):")
    print(f"  {'primary':>16}     {'secondary':>16}")
    for i in range(n_top):
        p = f"label={primary_top[i][0]:>5d} ({primary_top[i][1]})" if i < len(primary_top) else ""
        s = f"label={secondary_top[i][0]:>5d} ({secondary_top[i][1]})" if i < len(secondary_top) else ""
        print(f"  {p:>16}     {s:>16}")

    print(f"[gen] writing {args.out_txt}")
    write_txt_labels(args.out_txt, out_rows)

    if args.out_spmat:
        print(f"[gen] writing {args.out_spmat}")
        write_spmat(args.out_spmat, out_rows, ncol)

    print("[gen] done.")


if __name__ == "__main__":
    main()
