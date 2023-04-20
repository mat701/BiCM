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
    
    myGraph.clear_edges()

Note that if your network is not huge, the best way to use the BiCM package is to work with the biadjacency matrix since it will automatically take care of possible isolated nodes or fully connected nodes.

Computing the BiCM
--------------------------------------------

After you have created the BipartiteGraph object, you can compute the bicm of your network via

.. code-block:: python
    
    myGraph.solve_tool(light_mode=False, method='newton', initial_guess=None, tolerance=1e-8, max_steps=None, verbose=False, linsearch=True, regularise=False, print_error=True)
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

The validation_method variable controls the type of multiple hypothesis testing used.

Computing the projected networks
--------------------------------------------

To compute the projection of a bipartite network on one layer (rows or columns layer), the BiCM package uses the probabilities of the model to understand if two nodes behave similarly and should be connected in the projection. This makes sure that two nodes with high degrees are not automatically linked because they share a number of common neighbors, but will first discount the information about the degrees. See https://iopscience.iop.org/article/10.1088/1367-2630/aa6b38/meta for the details of this projection method.

For the moment, only the projection of binary (unweighted) networks has been implemented.

To compute the projected network and get the edgelist or the adjacency list, use one of the following:

.. code-block:: python
    
    myGraph.compute_projection(rows=True, alpha=0.05, method='poisson', threads_num=4, progress_bar=True)
    myGraph.get_rows_projection(fmt='edgelist')
    myGraph.get_cols_projection(fmt='adjacency_list')

You can also request the formats 'array' or 'sparse' that yield an adjacency matrix as a numpy array or scipy sparse matrix.

The first method is the customizable one, while the other two methods calculate the projection with default options if it has not been computed yet. The "method" option sets the approximation of the Poisson Binomial variable that is the number of common neighbors between nodes. Allowed options are 'poisson' (default and reliable), 'poibin' (exact but very slow, to be avoided except for small networks), 'normal' and 'rna' (to be used only in specific cases, otherwise avoid).