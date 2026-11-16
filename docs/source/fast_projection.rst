Fast projection of large networks
==================================

The module :mod:`bicm.fast_projection` computes the validated projection of a bipartite network directly from the list of its edges, without creating a ``BipartiteGraph``. It is designed for networks with millions of rows: it never stores a quantity for each row, only the histogram of the degrees of the rows and the co-occurrences of the couples of columns.

The statistical model is the same of the ``BipartiteGraph`` class and the results are the same: only the path of the computation changes.

Projection from arrays of edges
--------------------------------------------

The edges are given as two aligned arrays, one entry per edge. The projection is computed on the columns layer, i.e. on the nodes of the second array. To project on the other layer, swap the two arrays.

.. code-block:: python

    from bicm.fast_projection import project

    result = project(row_ids, col_ids, method='exact', correction='fdr', alpha=0.05)

    result.to_edgelist()     # list of couples of names of nodes
    result.weight            # common neighbors of each validated couple
    result.pval              # p-values
    result.pval_corrected    # p-values adjusted for the multiple testing
    result.z_score           # z-scores
    result.to_dataframe()    # everything as a pandas DataFrame (requires pandas)

The options are:

* ``method``: 'exact' (default, the exact Poisson Binomial distribution), 'poisson', 'normal' or 'rna', see :doc:`usage`;
* ``correction``: 'fdr' (default), 'bonferroni' or 'global'. Every couple of columns counts as a test;
* ``alpha``: significance level;
* ``min_overlap``: minimum number of common neighbors of the couples that are tested;
* ``n_jobs``: number of threads of the exact method for large overlaps.

Repeated edges count once. Rows and columns linked to the whole opposite layer are removed, since their links are certain in the BiCM.

Projection from chunks of edges
--------------------------------------------

If the edges do not fit in memory they can be given in chunks, for instance one for each file of a partitioned dataset:

.. code-block:: python

    from pathlib import Path
    import pandas as pd
    from bicm.fast_projection import project_chunks

    def chunks(folder):
        for path in sorted(Path(folder).glob('*.parquet')):
            df = pd.read_parquet(path, columns=['row_id', 'col_id'])
            yield df['row_id'].to_numpy(), df['col_id'].to_numpy()

    result = project_chunks(chunks('path/to/folder'), method='exact')

.. warning::

    All the edges of a row must be in the same chunk. A column can appear in any number of chunks.

The memory used is that of a dense matrix with the co-occurrences of the columns, whatever the number of rows.

Building blocks
--------------------------------------------

The steps of the pipeline are available as separate functions:

* :func:`~bicm.fast_projection.aggregate_bipartite` and :func:`~bicm.fast_projection.aggregate_chunks` reduce the edges to a :class:`~bicm.fast_projection.BipartiteAgg`;
* :func:`~bicm.fast_projection.fit_bicm_reduced` solves the BiCM on the distinct degrees;
* :func:`bicm.pvalues.moments_grouped`, :func:`bicm.pvalues.exact_pvals_grouped` and :func:`bicm.pvalues.cooccurrence_pvalues` compute the moments and the p-values of the co-occurrences;
* :func:`~bicm.fast_projection.validate_edges` and the functions of :mod:`bicm.validation` apply the multiple testing procedure.

This module is based on the ``fast_bicm_exact`` package by Antonio Desiderio.
