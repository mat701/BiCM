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


To generate a BiCM probability matrix of a biadjacency matrix:

.. code-block:: python

    :linenos:
    
    bicm_calculator(biad_mat)

From a degree sequence:
.. code-block:: python
    
    :linenos:
    
    bicm_light(rows_degree_sequence, cols_degree_sequence)

Dependencies
============

This package has been developed for Python 3.5 but it should be compatible with other versions. It works on numpy arrays and it uses numba and multiprocessing to speed up the computation.

Guide
^^^^^^

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents:
   
   license
   help



Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
