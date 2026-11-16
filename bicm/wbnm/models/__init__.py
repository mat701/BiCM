"""
Null models for weighted bipartite networks.

Available Models
----------------
BiCM : Bipartite Configuration Model
    Preserves degree sequence in expectation (binary model).

BiECM : Bipartite Enhanced Configuration Model
    Preserves degree sequence and strength sequence in expectation.

BiWCM : Bipartite Weighted Configuration Model
    Preserves strength sequence in expectation.

BiPECM : Bipartite Partial Enhanced Configuration Model
    Preserves degree sequence exactly and strength sequence in expectation.

BiCReMA : Bipartite Configuration model with Random Matrix Approximation
    Approximates BiECM by decoupling topology (BiCM) and weights (BiWCM).
"""

from bicm.wbnm.models.bicm import BiCM
from bicm.wbnm.models.biecm import BiECM
from bicm.wbnm.models.biwcm import BiWCM
from bicm.wbnm.models.bipecm import BiPECM
from bicm.wbnm.models.bicrema import BiCReMA

__all__ = ["BiCM", "BiECM", "BiWCM", "BiPECM", "BiCReMA"]
