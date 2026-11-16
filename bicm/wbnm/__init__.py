"""
Weighted Bipartite Null Models (WBNM)
=====================================

Maximum entropy null models for weighted bipartite networks, implemented with
PyTorch (they run on CPU or GPU). This subpackage is the ``wbnm`` package by
Lorenzo Buffa and Daniele Cirulli (Centro Ricerche Enrico Fermi, Rome),
distributed here under the same MIT license.

It requires PyTorch, which is not installed with ``bicm`` by default::

    pip install "bicm[weighted]"

Models
------
- BiCM: Bipartite Configuration Model (binary, degree sequence only)
- BiWCM: Bipartite Weighted Configuration Model (strength sequence)
- BiECM: Bipartite Enhanced Configuration Model (degrees and strengths)
- BiPECM: Bipartite Partial Enhanced Configuration Model (strengths and number of links)
- BiCReMA: BiCM + BiWCM two-stage approximation of the BiECM

Example
-------
>>> from bicm.wbnm import BiECM
>>> from bicm.wbnm.validation import fdr
>>>
>>> model = BiECM(biadjacency_matrix).solve()
>>> pvalues = model.pvalues()
>>> rejected, adjusted = fdr(pvalues, alpha=0.05)
>>> samples = model.sample(n_samples=1000, seed=42)
"""

try:
    import torch  # noqa: F401
except ImportError as error:  # pragma: no cover
    raise ImportError(
        'bicm.wbnm requires PyTorch. Install it with: pip install "bicm[weighted]"'
    ) from error

from bicm.wbnm.models import BiECM, BiWCM, BiPECM, BiCM, BiCReMA
from bicm.wbnm.models.base import ConvergenceWarning

__author__ = "Lorenzo Buffa, Daniele Cirulli"

__all__ = [
    "BiCM",
    "BiECM",
    "BiWCM",
    "BiPECM",
    "BiCReMA",
    "ConvergenceWarning",
]
