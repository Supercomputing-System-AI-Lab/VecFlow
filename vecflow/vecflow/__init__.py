# SPDX-License-Identifier: Apache-2.0
"""VecFlow — high-performance label-aware ANN search on GPUs.

Built on top of cuVS's CAGRA + IVF-Flat with a per-query label gate. See
the standalone README and the project page for usage.
"""
from importlib import metadata as _md

from .vecflow import VecFlow

try:
    __version__ = _md.version("vecflow") or _md.version("vecflow-cu12")
except _md.PackageNotFoundError:  # editable install, no metadata
    __version__ = "0.0.0+dev"

__all__ = ["VecFlow", "__version__"]
