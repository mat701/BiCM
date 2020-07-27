BiCM documentation
=====================================

This package provides a tool for calculating the Bipartite Configuration Model (BiCM), as introduced in https://www.nature.com/articles/srep10595?origin=ppub , and the relative monopartite projection on one layer introduced in https://iopscience.iop.org/article/10.1088/1367-2630/aa6b38/meta . 

Basic functionalities
=====================================

To install:

.. code-block::
    
    pip install bicm

To import the module:

.. code-block:: python
    
    import bicm

To generate a Graph object and initialize it (with a biadjacency matrix, edgelist or degree sequences):

.. code-block:: python
    
    from bicm import BipartiteGraph
    myGraph = BipartiteGraph()
    myGraph.set_biadjacency_matrix(my_biadjacency_matrix)
    myGraph.set_edgelist(my_edgelist)
    myGraph.set_degree_sequences((first_degree_sequence, second_degree_sequence))

Or alternatively:

.. code-block:: python
    
    from bicm import BipartiteGraph
    myGraph = BipartiteGraph(biadjacency=my_biadjacency_matrix, edgelist=my_edgelist, degree_sequences=((first_degree_sequence, second_degree_sequence)))

To compute the BiCM probability matrix of the graph or the relative fitnesses coefficients:

.. code-block:: python
    
    my_probability_matrix = myGraph.get_bicm_matrix()
    my_x, my_y = myGraph.get_bicm_fitnesses()

This will solve the bicm using recommended settings for the solver. 
To customize the solver you can alternatively use (in advance) the following method:

.. code-block:: python
    
    myGraph.solve_bicm(light_mode=False, method='newton', initial_guess=None, tolerance=1e-8, max_steps=None, verbose=False, linsearch=True, print_error=True)

To get the rows or columns projection of the graph:

.. code-block:: python
    
    myGraph.get_rows_projection(alpha=0.05, method='poisson', threads_num=4)
    myGraph.get_rows_projection(alpha=0.05, method='poisson', threads_num=4)

Alternatively, to customize the projection:

.. code-block:: python
    
    myGraph.compute_projection(rows=True, naive=False, alpha=0.05, method='poisson', threads_num=4, progress_bar=True)

Dependencies
============

This package has been developed for Python 3.5 but it should be compatible with other versions. It works on numpy arrays and it uses numba and multiprocessing to speed up the computation. Feel free to send an inquiry (and please do it!) if you find any incompatibility.

Guide
^^^^^^

.. toctree::
   :maxdepth: 2
   :numbered:
   
   install
   usage
   functions
   FAQ
   license
   contacts



Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
