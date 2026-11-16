Null models for weighted networks
==================================

The subpackage :mod:`bicm.wbnm` (Weighted Bipartite Null Models) contains maximum entropy null models for weighted bipartite networks, implemented with PyTorch. They run on CPU or GPU. It is the ``wbnm`` package by Lorenzo Buffa and Daniele Cirulli (Centro Ricerche Enrico Fermi, Rome).

PyTorch is not installed with the package by default:

.. code-block:: bash

    pip install "bicm[weighted]"

The models
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Model
     - Constraints preserved on average
   * - ``BiCM``
     - degrees (binary model, the weights are ignored)
   * - ``BiWCM``
     - strengths (sums of the weights of each node)
   * - ``BiECM``
     - degrees and strengths
   * - ``BiPECM``
     - strengths and total number of links
   * - ``BiCReMA``
     - degrees and strengths, with a two steps approximation of the BiECM that is faster on large networks

All the models require every node to have at least one link: remove the empty rows and columns before fitting.

The weighted models BiWCM_c and BiWCM_d of the ``BipartiteGraph`` class are the continuous and discrete versions of the BiWCM, solved with numba instead of PyTorch.

Usage
--------------------------------------------

All the models share the same interface:

.. code-block:: python

    import numpy as np
    from bicm.wbnm import BiECM
    from bicm.wbnm.validation import fdr

    W = np.array([
        [5, 0, 3, 1],
        [0, 2, 4, 0],
        [1, 3, 0, 2],
        [0, 1, 2, 0],
        [2, 0, 0, 3],
    ], dtype=float)

    model = BiECM(W)
    model.solve()

    expected = model.expected_matrix()
    pvalues = model.pvalues()             # P(w >= observed weight) for each entry
    rejected, adjusted = fdr(pvalues, alpha=0.05)
    samples = model.sample(n_samples=1000, seed=42)

    model_gpu = BiECM(W, device='cuda')   # optional: run on GPU

The method ``solve`` accepts the solvers 'fixed-point' (default), 'quasi-newton' and 'coordinate', and the options ``tol``, ``max_iter``, ``verbose`` and ``chunk`` (number of rows processed at a time, to limit the memory used on GPU). A :class:`~bicm.wbnm.models.base.ConvergenceWarning` is raised if the solver does not converge. The models also provide ``log_likelihood()``, ``aic()``, ``aicc()`` and ``bic()``, and can be compared with :func:`bicm.wbnm.utils.compare_models`.

The analytical p-values are the probability that the model gives a link a weight at least as large as the observed one. They are computed in logarithms, so that they stay accurate also when they are extremely small.

Validated projections
--------------------------------------------

The distribution of the weights of the projection of a weighted network has no closed form: the p-values are estimated by sampling the null model.

.. code-block:: python

    from bicm.wbnm.projections import validated_projection
    from bicm.wbnm.validation import pvalues_projection, pvalues_projection_binary

    validated, pvalues, rejected = validated_projection(model, layer='rows', n_samples=1000, alpha=0.05)

    pvals_weighted = pvalues_projection(model, n_samples=1000, rows=True)
    pvals_binary = pvalues_projection_binary(model, n_samples=1000, rows=True)

A p-value estimated from ``n`` samples is ``(c + 1) / (n + 1)``, where ``c`` is the number of samples at least as extreme as the observation: it cannot be smaller than ``1 / (n + 1)``. Use a number of samples large enough for the multiple testing correction, which is applied to the distinct couples of nodes.

For binary networks prefer the projections of ``BipartiteGraph`` or of :mod:`bicm.fast_projection`, whose p-values are analytical and exact.
