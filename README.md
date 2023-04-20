## BiCM package

This is a Python package for the computation of the maximum entropy bipartite configuration model (BiCM) and the projection of bipartite networks on one layer. It was developed with Python 3.5.

You can install this package via pip: 

    pip install bicm

Documentation is available at https://bipartite-configuration-model.readthedocs.io/en/latest/ .

This package is also a module of NEMtropy that you can find at https://github.com/nicoloval/NEMtropy .

For more solvers of maximum entropy configuration models visit https://meh.imtlucca.it/ .


## Basic functionalities

To install:
    
    pip install bicm

To import the module:
    
    import bicm

To generate a Graph object and initialize it (with a biadjacency matrix, edgelist or degree sequences):
    
    from bicm import BipartiteGraph
    myGraph = BipartiteGraph()
    myGraph.set_biadjacency_matrix(my_biadjacency_matrix)
    myGraph.set_adjacency_list(my_adjacency_list)
    myGraph.set_edgelist(my_edgelist)
    myGraph.set_degree_sequences((first_degree_sequence, second_degree_sequence))

Or alternatively, with the respective data structure as input:
    
    from bicm import BipartiteGraph
    myGraph = BipartiteGraph(biadjacency=my_biadjacency_matrix, adjacency_list=my_adjacency_list, edgelist=my_edgelist, degree_sequences=((first_degree_sequence, second_degree_sequence)))

To compute the BiCM probability matrix of the graph or the relative fitnesses coefficients as dictionaries containing the nodes names as keys:

    my_probability_matrix = myGraph.get_bicm_matrix()
    my_x, my_y = myGraph.get_bicm_fitnesses()

This will solve the bicm using recommended settings for the solver. 
To customize the solver you can alternatively use (in advance) the following method:
    
    myGraph.solve_tool(light_mode=False, method='newton', initial_guess=None, tolerance=1e-8, max_steps=None, verbose=False, linsearch=True, regularise=False, print_error=True, exp=False)

To get the rows or columns projection of the graph:

    myGraph.get_rows_projection()
    myGraph.get_cols_projection()

Alternatively, to customize the projection:

    myGraph.compute_projection(rows=True, alpha=0.05, method='poisson', threads_num=4, progress_bar=True)
    
Now version 3.0.0 is online, and you can use the package with weighted networks as well using the BiWCM models!

See a more detailed walkthrough in **tests/bicm_test** or **tests/biwcm_test** notebooks, or check out the API in the documentation.

## How to cite

If you use the `bicm` module, please cite its location on Github
[https://github.com/mat701/BiCM](https://github.com/mat701/BiCM) and the
original articles [Vallarano2021], [Saracco2015] and [Saracco2017].

If you use the weighted models BiWCM_c or BiMCM you might consider citing also the following paper introducing the solvers of this package:

* Bruno, M., Mazzilli, D., Patelli, A., Squartini, T., and Saracco, F. \
    *Inferring comparative advantage via entropy maximization.* \
    In preparation

### References

[Vallarano2021] [N. Vallarano, M. Bruno, E. Marchese, G. Trapani, F. Saracco, T. Squartini, G. Cimini, M. Zanon, Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints, Nature Scientific Reports](https://doi.org/10.1038/s41598-021-93830-4)

[Saracco2015] [F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini, Randomizing bipartite networks: the case of the World Trade Web, Scientific Reports 5, 10595 (2015)](http://www.nature.com/articles/srep10595).

[Saracco2017] [F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, and T. Squartini, Inferring monopartite projections of bipartite networks: an entropy-based approach, New J. Phys. 19, 053022 (2017)](http://stacks.iop.org/1367-2630/19/i=5/a=053022)


_Author_:

[Matteo Bruno](https://csl.sony.it/member/matteo-bruno/) (BiCM) (a.k.a. [mat701](https://github.com/mat701))
