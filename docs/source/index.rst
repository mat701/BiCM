BiCM documentation
=====================================

This package provides tools for the maximum entropy null models of bipartite networks:

* the Bipartite Configuration Model (BiCM) of binary networks, introduced in

    `Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints. Vallarano et al., Scientific Reports 11.1 (2021) <https://doi.org/10.1038/s41598-021-93830-4>`_,
    `Randomizing bipartite networks: the case of the World Trade Web. Saracco et al., Sci Rep 5, 10595 (2015). <https://doi.org/10.1038/srep10595>`_

* the statistically validated projection of a bipartite network on one of its layers, introduced in

    `Inferring monopartite projections of bipartite networks: an entropy-based approach, Saracco et al., 2017 New J. Phys. 19 053022 <https://doi.org/10.1088/1367-2630/aa6b38>`_ .

* the weighted models BiWCM_c and BiWCM_d, introduced in the following paper, which should also be cited when using them:

    `Inferring comparative advantage via entropy maximization, Bruno et al., J. Phys. Complex. 4, 045011 (2023), doi:10.1088/2632-072X/ad1411 <https://doi.org/10.1088/2632-072X/ad1411>`_ .

The package is made of three parts:

* the :class:`~bicm.graph_classes.BipartiteGraph` class solves the models and computes the projections of a network held in memory, see :doc:`usage`;
* :mod:`bicm.fast_projection` computes the projection of very large networks directly from their list of edges, also streamed in chunks, see :doc:`fast_projection`;
* :mod:`bicm.wbnm` contains further null models for weighted networks (BiECM, BiPECM, BiCReMA and others) implemented with PyTorch, see :doc:`weighted`.

Basic functionalities
=====================================

To install:

.. code-block:: bash

    pip install bicm

To import the module:

.. code-block:: python

    import bicm

To generate a Graph object and initialize it (with a biadjacency matrix, edgelist or degree sequences):

.. code-block:: python

    from bicm import BipartiteGraph
    myGraph = BipartiteGraph()
    myGraph.set_biadjacency_matrix(my_biadjacency_matrix)
    myGraph.set_adjacency_list(my_adjacency_list)
    myGraph.set_edgelist(my_edgelist)
    myGraph.set_degree_sequences((first_degree_sequence, second_degree_sequence))

Or alternatively, with the respective data structure as input:

.. code-block:: python

    from bicm import BipartiteGraph
    myGraph = BipartiteGraph(biadjacency=my_biadjacency_matrix, adjacency_list=my_adjacency_list, edgelist=my_edgelist, degree_sequences=((first_degree_sequence, second_degree_sequence)))

To compute the BiCM probability matrix of the graph or the relative fitnesses coefficients as dictionaries containing the nodes names as keys:

.. code-block:: python

    my_probability_matrix = myGraph.get_bicm_matrix()
    my_x, my_y = myGraph.get_bicm_fitnesses()

This will solve the bicm, or the relative biwcm model if the graph is weighted, using recommended settings for the solver.
To customize the solver you can alternatively use (in advance) the following method:

.. code-block:: python

    myGraph.solve_tool(light_mode=False, method='newton', initial_guess=None, tol=1e-8, max_steps=None, verbose=False, linsearch=True, regularise=False, print_error=True, exp=False)

You can compute the p-value of each (weighted) link in the network. To return the matrix of pvalues or the binarisation after applying a threshold:

.. code-block:: python

    myGraph.get_weighted_pvals_mat()
    myGraph.get_validated_matrix(significance=0.01, validation_method='fdr')

To get the rows or columns projection of the graph:

.. code-block:: python

    myGraph.get_rows_projection()
    myGraph.get_cols_projection()

Alternatively, to customize the projection:

.. code-block:: python

    myGraph.compute_projection(rows=True, alpha=0.05, approx_method='poibin', validation_method='fdr', threads_num=4)

Dependencies
============

This package requires Python 3.10 or later. It works on numpy arrays and it uses numba to speed up the computation. PyTorch is needed only by the weighted models of :mod:`bicm.wbnm`. Feel free to send an inquiry (and please do it!) if you find any incompatibility.

Guide
^^^^^^

.. toctree::
   :maxdepth: 2

   install
   usage
   fast_projection
   weighted
   bicm
   changelog
   FAQ
   license
   contacts



Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
