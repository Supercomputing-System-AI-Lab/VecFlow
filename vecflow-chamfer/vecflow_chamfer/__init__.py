# SPDX-License-Identifier: Apache-2.0
"""vecflow_chamfer — multi-vector retrieval on GPUs with a memory hierarchy.

Two-stage pipeline: MaxIVF anchor index (CAGRA-routed) for candidate generation
and an optimized chamfer-scoring kernel for full-precision reranking. Document
embeddings stay in host RAM and are read by the GPU over the C2C link (on
Grace-Hopper) or as managed memory (on PCIe-attached GPUs) — the allocator is
selected automatically at ``Dataset.load`` time.
"""
from importlib import metadata as _md

from ._vecflow_chamfer import (
    Dataset,
    QuerySet,
    Index,
    IndexParams,
    SearchParams,
    SearchStats,
    build,
    search,
    serialize,
    deserialize,
    cuda_sync,
    ChamferKernelBench,
)

try:
    __version__ = _md.version("vecflow-chamfer") or _md.version("vecflow-chamfer-cu12")
except _md.PackageNotFoundError:  # editable install, no metadata
    __version__ = "0.0.0+dev"

__all__ = [
    "Dataset",
    "QuerySet",
    "Index",
    "IndexParams",
    "SearchParams",
    "SearchStats",
    "build",
    "search",
    "serialize",
    "deserialize",
    "cuda_sync",
    "ChamferKernelBench",
    "__version__",
]
