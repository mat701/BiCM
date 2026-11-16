"""
BiCM
========

BiCM is a Python package for computing the Bipartite Configuration Model
of a bipartite network and its statistical projection on one layer.
The paper of reference is at https://www.nature.com/articles/srep10595?origin=ppub

Modules:

* :mod:`bicm.graph_classes`: the :class:`BipartiteGraph` class, to solve the BiCM
  and the weighted BiWCM models and to compute the validated projections;
* :mod:`bicm.fast_projection`: validated projections of large networks directly
  from their list of edges;
* :mod:`bicm.pvalues`: accurate p-values of the co-occurrences;
* :mod:`bicm.validation`: multiple hypothesis testing;
* :mod:`bicm.wbnm`: null models for weighted networks based on PyTorch
  (optional, ``pip install "bicm[weighted]"``).

Github repository::

    https://github.com/mat701/BiCM

Author: Matteo Bruno
"""

from . import fast_projection, pvalues, validation
from ._utils import BiCMWarning
from .graph_classes import *
from .network_functions import *
from .poibin import PoiBin

__version__ = "4.0.0"
__author__ = """Matteo Bruno (matteobruno180@gmail.com)"""
