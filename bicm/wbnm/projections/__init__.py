"""
Weighted projections of bipartite networks with statistical validation.
"""

from bicm.wbnm.projections.weighted import (
    project_rows,
    project_cols,
    validated_projection,
)

__all__ = [
    "project_rows",
    "project_cols",
    "validated_projection",
]
