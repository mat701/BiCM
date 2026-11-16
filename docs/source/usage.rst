How to use the BiCM package
==================================

With this package you can solve the maximum entropy bipartite configuration model (BiCM) of a real network. The BiCM is used for comparing the properties of a real network with the set of networks with the same degrees. For example, your network might have a high value of assortativity, but the average assortativity of the set of networks with the same degree sequence is "similar" to the one of the real network.

The BipartiteGraph object
--------------------------------------------

To manage the real network, in the BiCM package you can create a BipartiteGraph object that will contain the information of your network. To create a BipartiteGraph object:

.. code-block:: python
    
    from bicm import BipartiteGraph
    myGraph = BipartiteGraph(biadjacency=None, edgelist=None, degree_sequences=None)
    
You can create your bipartite object without any nodes or edges, or initializing it immediately by choosing one of the options of the constructor. It can be initialized with a biadjacency matrix (numpy array, sparse matrix or list), with an edgelist (a list or numpy array of couples) or solely with degree sequences (a couple of arrays or lists that represent the degree sequences of the two layers of the bipartite network). In the BiCM package there are also functions that convert an edgelist in a biadjacency matrix or vice versa.

You can add edges to your graph with the methods

.. code-block:: python

    myGraph.set_edgelist(edgelist)
    myGraph.set_biadjacency_matrix(biadjacency_matrix)
    myGraph.set_degree_sequences(degree_sequences)

You can clear your graph by using

.. code-block:: python
    
    myGraph.clean_edges()

Note that if your network is not huge, the best way to use the BiCM package is to work with the biadjacency matrix since it will automatically take care of possible isolated nodes or fully connected nodes.

Computing the BiCM
--------------------------------------------

After you have created the BipartiteGraph object, you can compute the bicm of your network via

.. code-block:: python
    
    myGraph.solve_tool(light_mode=False, method='newton', initial_guess=None, tol=1e-8, max_steps=None, verbose=False, linsearch=True, regularise=False, print_error=True)
    myGraph.get_bicm_fitnesses()
    myGraph.get_bicm_matrix()

The method "solve_tool" will actually solve the equations and you can customize the solver with many options. The allowed methods are 'newton', 'fixed-point', 'quasinewton', 'root' and they are all performing similarly with the exception of 'root' that uses the scipy.optimize.root solver and is a bit slower. The default choice is the Newton solver.
If you use one of the get methods before solving the bicm, the solver will start with default options.
For weighted networks, the package will handle them automatically. If they are discrete it will use the biwcm_d setup, otherwise the biwcm_c.

Computing the binarised network
--------------------------------------------

You can compute the p-value of each (weighted) link in the network. To return the matrix of pvalues or the binarisation after applying a threshold (inferring the "Comparative Advantage" of nodes) you can use the two following methods:

.. code-block:: python

    myGraph.get_weighted_pvals_mat()
    myGraph.get_validated_matrix(significance=0.01, validation_method='fdr')

The validation_method variable controls the type of multiple hypothesis testing used ('fdr', 'bonferroni' or 'global'). The p-value of a link is the probability that the model gives it a weight at least as large as the observed one (for binary networks, the probability of the link); entries without a link have p-value one.

Computing the projected networks
--------------------------------------------

To compute the projection of a bipartite network on one layer (rows or columns layer), the BiCM package uses the probabilities of the model to understand if two nodes behave similarly and should be connected in the projection. This makes sure that two nodes with high degrees are not automatically linked because they share a number of common neighbors, but will first discount the information about the degrees. See https://iopscience.iop.org/article/10.1088/1367-2630/aa6b38/meta for the details of this projection method.

For the moment, only the projection of binary (unweighted) networks has been implemented in the BipartiteGraph class. Monte Carlo projections of weighted networks are available in :doc:`weighted`.

To compute the projected network and get the edgelist or the adjacency list, use one of the following:

.. code-block:: python

    myGraph.compute_projection(rows=True, alpha=0.05, approx_method='poibin', validation_method='fdr', threads_num=4)
    myGraph.get_rows_projection(fmt='edgelist')
    myGraph.get_cols_projection(fmt='adjacency_list')

You can also request the formats 'array' or 'sparse' that yield the adjacency matrix of the projection as a numpy array or scipy sparse matrix. The nodes of the matrix are in the order of the biadjacency matrix, or in the order of ``myGraph.rows_dict`` and ``myGraph.cols_dict`` if the graph was created from an edgelist or an adjacency list.

The first method is the customizable one, while the other two methods calculate the projection with default options if it has not been computed yet. They accept the same options: an option that is passed updates the projection, the others keep the values of the last computation. For instance, to validate the same p-values with a different threshold:

.. code-block:: python

    myGraph.get_rows_projection(alpha=0.01, validation_method='bonferroni')

The p-value of two nodes is the probability that the BiCM gives them at least as many common neighbors as observed. The number of common neighbors is a Poisson Binomial variable and the "approx_method" option sets how its distribution is computed:

* 'poibin' (or 'exact', default): the exact Poisson Binomial distribution;
* 'poisson': Poisson approximation, conservative, accurate when the link probabilities are small. It was the default up to version 3.5;
* 'normal': Gaussian approximation with continuity correction;
* 'rna': refined normal approximation, a Gaussian corrected with the skewness of the distribution.

The "validation_method" option sets the multiple hypothesis testing procedure: 'fdr' (False Discovery Rate, default), 'bonferroni' or 'global' (no correction). Every couple of nodes of the layer counts as a test, also the couples without common neighbors.

The p-values can be read from ``myGraph.rows_pvals`` and ``myGraph.cols_pvals``, dictionaries of the form ``{node: {neighbor: p-value}}``, or as a matrix with ``myGraph.get_projected_pvals_mat()``.

Speed and numerical accuracy of the projections
-----------------------------------------------

All the nodes with the same degree have the same fitness, so the common neighbors of two nodes are a sum over the distinct degrees of the opposite layer, not over all its nodes. The computation is compiled with numba and runs in parallel threads on every operating system ("threads_num" sets their number). Also the exact method is fast enough for large networks: there is usually no need to prefer an approximation for speed.

The p-values are right tails of a distribution and can be many orders of magnitude smaller than the machine precision (about 1e-16), in particular for the largest and most significant overlaps. They are never computed as "1 - cdf", which would turn them into zeros or rounding noise, and this matters because the thresholds of the multiple testing procedures are often very small for large networks. The exact method keeps a small relative error down to the smallest numbers a float can represent (about 1e-300): the tail is accumulated with sums of non negative terms for small overlaps, and computed with Fourier transforms of the exponentially tilted distribution for large ones. See :mod:`bicm.pvalues` for the details.

Nodes without links or linked to all the nodes of the other layer have certain links in the BiCM. Their contribution to the common neighbors is not random and it is removed before computing the p-values. If the network has such nodes, initialize the graph with the biadjacency matrix as a numpy array, so that they are treated exactly.

Warnings, logging and reproducibility
--------------------------------------------

Problems found while solving or projecting (for example a solver that did not converge) are reported as Python warnings of category ``bicm.BiCMWarning``, so they can be filtered:

.. code-block:: python

    import warnings
    import bicm
    warnings.filterwarnings("ignore", category=bicm.BiCMWarning)

Informative messages (for example "Computing the BiCM...") go to the ``bicm`` logger and are hidden by default. To show them:

.. code-block:: python

    import logging
    logging.basicConfig()
    logging.getLogger("bicm").setLevel(logging.INFO)

The sampling functions accept a ``seed`` argument (an integer or a ``numpy.random.Generator``) to get reproducible samples, e.g. ``sample_bicm(avg_mat, seed=42)``. Without it, numpy's global random state is used as in previous versions.
