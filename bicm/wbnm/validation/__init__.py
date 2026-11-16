"""
Statistical validation tools for bipartite networks.

This module provides functions for computing p-values and applying
multiple testing corrections to validate network properties.
"""

from bicm.wbnm.validation.pvalues import (
    compute_pvalue,
    validate_property,
    validate_matrix_property,
    pvalues_projection,
    pvalues_projection_binary,
)
from bicm.wbnm.validation.corrections import (
    bonferroni,
    fdr,
    fdr_bh,  # backwards compatibility alias
    apply_correction,
)

__all__ = [
    # P-value computation
    "compute_pvalue",
    "validate_property",
    "validate_matrix_property",
    # Projection p-values (Monte Carlo)
    "pvalues_projection",
    "pvalues_projection_binary",
    # Multiple testing corrections
    "bonferroni",
    "fdr",
    "fdr_bh",  # backwards compatibility alias
    "apply_correction",
]
