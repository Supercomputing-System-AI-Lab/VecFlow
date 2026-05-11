# SPDX-License-Identifier: Apache-2.0
#
# Type stubs for the pybind11-bound `vecflow.vecflow` module. Hand-written
# to mirror the bindings in vecflow/binding/binding.cpp. IDEs and mypy
# pick this up for completion + type-checking.

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

class VecFlow:
    """VecFlow index handle. See class docstring in the C extension."""

    def __init__(self) -> None: ...

    def build(
        self,
        dataset: NDArray[np.float32],
        data_labels: List[List[int]],
        graph_degree: int,
        specificity_threshold: int,
        graph_fname: str = ...,
        bfs_fname: str = ...,
        force_rebuild: bool = ...,
        multi_label: bool = ...,
    ) -> None:
        """Build (or reload) the dual-structured VecFlow index.

        When ``multi_label=True``, additionally prepares CSR dataset-label
        buffers so that ``search_multi(...)`` can run AND queries.
        """

    def search(
        self,
        queries: NDArray[np.float32],
        query_labels: NDArray[np.int32],
        itopk_size: int,
        topk: int = 10,
    ) -> Tuple[NDArray[np.uint32], NDArray[np.float32]]:
        """Label-aware top-k search. Returns ``(neighbors, distances)``."""

    def search_multi(
        self,
        queries: NDArray[np.float32],
        query_labels_a: NDArray[np.int32],
        query_labels_b: NDArray[np.int32],
        itopk_size: int,
        topk: int = 10,
    ) -> Tuple[NDArray[np.uint32], NDArray[np.float32]]:
        """2-label AND search. Requires ``build(... multi_label=True)``.
        Order of the two label arrays is irrelevant — the larger-frequency
        label is picked internally as the primary.
        """

    def generate_ground_truth(
        self,
        dataset: NDArray[np.float32],
        queries: NDArray[np.float32],
        data_labels: List[List[int]],
        query_labels: List[List[int]],
        topk: int,
        gt_fname: str,
    ) -> NDArray[np.uint32]:
        """Brute-force ground truth, label-gated, written to ``gt_fname``."""
