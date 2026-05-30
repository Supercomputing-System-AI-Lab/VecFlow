# SPDX-License-Identifier: Apache-2.0
#
# Type stubs for the pybind11-bound `vecflow_chamfer._vecflow_chamfer`
# module. Hand-written to mirror the bindings in binding/binding.cpp.

from typing import Any, Tuple, overload

import numpy as np
from numpy.typing import NDArray

class Dataset:
    """Document set (fp16 token embeddings + CSR offsets). Construct via :meth:`load`."""

    @staticmethod
    def load(dataset_dir: str, prefix: str) -> "Dataset": ...

    @property
    def num_docs(self) -> int: ...
    @property
    def num_doc_tokens(self) -> int: ...
    @property
    def embedding_dim(self) -> int: ...

class QuerySet:
    """Query set: fp16 token embeddings, fixed tokens-per-query."""

    @staticmethod
    def load(
        dataset_dir: str,
        prefix: str,
        embedding_dim: int,
        num_tokens_per_query: int = 32,
    ) -> "QuerySet": ...

    @property
    def num_queries(self) -> int: ...
    @property
    def num_tokens_per_query(self) -> int: ...
    @property
    def embedding_dim(self) -> int: ...

class IndexParams:
    """MaxIVF build-time parameters."""

    n_anchors: int
    graph_degree: int
    itopk_size: int
    n_iter: int
    pq_bits: int
    pq_dim: int

    def __init__(
        self,
        n_anchors: int,
        graph_degree: int,
        itopk_size: int,
        n_iter: int,
        pq_bits: int = 0,
        pq_dim: int = 0,
    ) -> None: ...

class SearchParams:
    """MaxIVF search-time parameters."""

    itopk: int
    search_topk: int
    refine_rate: float
    final_topk: int
    use_vpq: bool
    refine_rate_vpq: float

    def __init__(
        self,
        itopk: int,
        search_topk: int,
        refine_rate: float,
        final_topk: int,
        use_vpq: bool = False,
        refine_rate_vpq: float = 1.5,
    ) -> None: ...

class Index:
    """Opaque MaxIVF index handle. Construct via :func:`build` or :func:`deserialize`."""

class SearchStats:
    """Per-stage timing breakdown from :func:`search` with ``with_stats=True``."""

    avg_candidate_generation_ms: float
    avg_unique_id_finding_ms: float
    avg_anchor_chamfer_reranking_ms: float
    avg_first_sorting_ms: float
    avg_vpq_reranking_ms: float
    avg_vpq_sorting_ms: float
    avg_full_chamfer_reranking_ms: float
    avg_second_sorting_ms: float
    avg_candidates_per_query: float
    avg_per_query_ms: float
    total_wall_ms: float
    throughput_qps: float
    avg_stage4_input_count: float
    avg_stage4_output_count: float
    avg_stage_vpq_output_count: float
    avg_stage6_output_count: float

def build(dataset: Dataset, params: IndexParams) -> Index: ...

@overload
def search(
    index: Index,
    dataset: Dataset,
    queries: QuerySet,
    params: SearchParams,
    with_stats: bool = False,
) -> Tuple[NDArray[np.uint32], NDArray[np.float32]]: ...
@overload
def search(
    index: Index,
    dataset: Dataset,
    queries: QuerySet,
    params: SearchParams,
    with_stats: bool = True,
) -> Tuple[NDArray[np.uint32], NDArray[np.float32], SearchStats]: ...

def serialize(index: Index, stem: str) -> None: ...
def deserialize(stem: str) -> Index: ...

def cuda_sync() -> None: ...

class ChamferKernelBench:
    """Bench harness for the raw chamfer_score kernel.

    All four inputs must be GPU arrays providing ``__cuda_array_interface__``
    (cupy arrays, torch CUDA tensors, numba device arrays, ...). The
    constructor does no H→D copy — pointers are extracted from the
    interface dict and used directly. ``run()`` launches the kernel; no
    copy. ``scores()`` does one D→H to return the latest result as numpy.
    """

    def __init__(
        self,
        query: Any,        # __cuda_array_interface__ object, float16 (Tq, dim)
        dataset: Any,      # __cuda_array_interface__ object, float16 (Td, dim)
        doc_offsets: Any,  # __cuda_array_interface__ object, uint32  (n_docs+1,)
        doc_ids: Any,      # __cuda_array_interface__ object, uint32  (n_docs,)
        tokens_per_query: int,
        dim: int,
    ) -> None: ...

    def run(self) -> None: ...
    def scores(self) -> NDArray[np.float32]: ...

    @property
    def n_docs(self) -> int: ...
    @property
    def tokens_per_query(self) -> int: ...
    @property
    def dim(self) -> int: ...
