What's new
==================================

Version 4.0
--------------------------------------------

New features:

* :mod:`bicm.fast_projection`: validated projections of very large networks directly from their edges, also streamed in chunks (from the ``fast_bicm_exact`` package by Antonio Desiderio).
* :mod:`bicm.wbnm`: BiCM, BiWCM, BiECM, BiPECM and BiCReMA null models for weighted networks, implemented with PyTorch (the ``wbnm`` package by Lorenzo Buffa and Daniele Cirulli). Install it with ``pip install "bicm[weighted]"``.
* :mod:`bicm.pvalues` and :mod:`bicm.validation`: the algorithms for the p-values of the co-occurrences and for the multiple hypothesis testing, shared by all the projections.
* The projections of ``BipartiteGraph`` can be returned as adjacency matrices (``fmt='array'`` or ``fmt='sparse'``), ``get_cols_projection`` accepts ``validation_method``, and the getters update the validation without computing the p-values again.

Speed:

* The projections of ``BipartiteGraph`` are computed on the distinct degrees of the opposite layer with compiled and multithreaded code, instead of a loop on the couples of nodes in several processes. On the networks used for the tests they are from 50 to several hundred times faster, and the exact method (``poibin``) is now usable on large networks.
* The projections are multithreaded on every operating system, Windows included, and do not need the ``if __name__ == "__main__":`` guard on macOS any more. The ``progress_bar`` option is ignored and tqdm is not a dependency any more.

Numerical accuracy and fixed bugs. **Please re-run your analysis if you are affected**:

* The ``normal`` and ``rna`` methods returned the p-value of the opposite tail (the probability of at most the observed number of common neighbors). They validated the wrong couples.
* The ``poibin`` method could not return p-values smaller than about 1e-15, and the smaller ones were rounding noise. All the methods now compute the right tail directly and are accurate down to about 1e-300.
* Nodes linked to all the nodes of the other layer got a null fitness in the projections: their couples had p-value zero and were always validated, and the p-values of the other layer were underestimated. Their links are now treated as certain.
* ``get_validated_matrix`` counted the couples of nodes of one layer as the number of tests, instead of the entries of the matrix. The p-values of the links of binary networks were overwritten by those of the weighted formula.
* A projection with ``alpha >= 1`` on the columns layer used the number of rows.
* ``get_bicm_matrix`` returned nan for the nodes linked to a whole layer when the graph was not initialized with a dense matrix.
* The projections failed on perfectly nested networks, where every link is certain: they now return p-values equal to one.
* :class:`~bicm.poibin.PoiBin` computes the distribution with an exact convolution: its p-values are accurate also when extremely small.
* In :mod:`bicm.wbnm` the analytical p-values are computed in logarithms, the Monte Carlo p-values cannot be zero any more, and the multiple testing correction of the projections is applied to the distinct couples of nodes only.

Benchmark: the notebook ``benchmarks/web_of_life/web_of_life_benchmark.ipynb`` runs versions 3.5 and 4.0 on all the networks of the `Web of Life <https://www.web-of-life.es/>`_ database and compares running times, p-values and validated links.

Differences from the two packages that were merged:

* ``fast_projection.project`` returns a :class:`~bicm.fast_projection.ValidatedProjection` instead of a DataFrame (use ``to_dataframe()``), an empty one instead of None. Its defaults are those of ``BipartiteGraph`` (``correction='fdr'``, ``alpha=0.05``), the number of tests is the number of couples of columns and the normal approximations use the continuity correction. pandas, statsmodels and threadpoolctl are not required.
* ``wbnm`` is imported as ``bicm.wbnm``.
