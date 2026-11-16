## BiCM package

This is a Python package for the maximum entropy null models of bipartite networks: the Bipartite Configuration Model (BiCM), its weighted versions, and the statistically validated projection of bipartite networks on one layer. It requires Python 3.10 or later.

You can install this package via pip:

    pip install bicm

Documentation is available at https://bipartite-configuration-model.readthedocs.io/en/latest/ .

This package is also a module of NEMtropy that you can find at https://github.com/nicoloval/NEMtropy .

For more solvers of maximum entropy configuration models visit https://meh.imtlucca.it/ .

The package has three parts:

| Part | What it does | Requires |
|---|---|---|
| `bicm.BipartiteGraph` | Solves the BiCM and the weighted BiWCM models, validates the links and computes the projections of a network held in memory | numpy, scipy, numba |
| `bicm.fast_projection` | Validated projection of very large networks directly from their list of edges, also streamed in chunks | numpy, scipy, numba |
| `bicm.wbnm` | More null models for weighted networks (BiECM, BiPECM, BiCReMA...) with sampling and Monte Carlo projections, on CPU or GPU | PyTorch: `pip install "bicm[weighted]"` |

NOTE of the developer for version 4.0: the `normal` and `rna` projection methods returned the p-values of the wrong tail, the `poibin` method was not accurate for p-values below 1e-15, and nodes linked to all the nodes of the other layer were always validated. If you used one of these, please re-run your analysis after updating. See the [changelog](https://bipartite-configuration-model.readthedocs.io/en/latest/changelog.html).

NOTE of the developer for version 3.1: there was an error in the projection threshold, validating less links than it should have.
Please re-run your analysis after updating to the last version.


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

    myGraph.solve_tool(light_mode=False, method='newton', initial_guess=None, tol=1e-8, max_steps=None, verbose=False, linsearch=True, regularise=False, print_error=True, exp=False)

If the network is weighted, the BiWCM_d (integer weights) or BiWCM_c (continuous weights) model is solved instead. To get the p-value of each link and the links that are statistically significant:

    myGraph.get_weighted_pvals_mat()
    myGraph.get_validated_matrix(significance=0.01, validation_method='fdr')

### Projections

To get the rows or columns projection of the graph:

    myGraph.get_rows_projection()
    myGraph.get_cols_projection()

Alternatively, to customize the projection:

    myGraph.compute_projection(rows=True, alpha=0.05, approx_method='poisson', validation_method='fdr', threads_num=4)
    myGraph.get_rows_projection(fmt='edgelist')

- `approx_method` is the distribution used for the number of common neighbors of two nodes: `'poisson'` (default), `'poibin'` (the exact Poisson Binomial distribution), `'normal'` or `'rna'`.
- `validation_method` is the multiple hypothesis testing procedure: `'fdr'` (default), `'bonferroni'` or `'global'`.
- `fmt` is the format of the result: `'adjacency_list'`, `'edgelist'`, `'array'` or `'sparse'`.

The projections work on the distinct degrees of the network with compiled, multithreaded code. The p-values are computed as right tails with a small relative error down to about 1e-300, so they are not limited by the machine precision of `1 - cdf`. This matters for large networks, where the validation thresholds are very small.

### Fast projection of large networks

For networks with millions of nodes, or that do not fit in memory, the projection can be computed directly from the edges, given as two aligned arrays or as an iterable of chunks:

    from bicm.fast_projection import project, project_chunks

    result = project(row_ids, col_ids, method='exact', correction='fdr', alpha=0.05)
    result = project_chunks(chunks_of_edges, method='exact')   # each chunk is (row_ids, col_ids)

    result.to_edgelist()    # validated couples of columns
    result.pval             # their p-values; also weight, z_score, pval_corrected
    result.to_dataframe()   # requires pandas

The projection is on the columns (swap the arrays for the rows). With chunks, all the edges of a row must be in the same chunk.

### More null models for weighted networks

The subpackage `bicm.wbnm` contains the BiCM, BiWCM, BiECM, BiPECM and BiCReMA models implemented with PyTorch:

    pip install "bicm[weighted]"

    from bicm.wbnm import BiECM
    from bicm.wbnm.validation import fdr

    model = BiECM(my_weighted_matrix).solve()
    expected = model.expected_matrix()
    pvalues = model.pvalues()
    rejected, adjusted = fdr(pvalues, alpha=0.05)
    samples = model.sample(n_samples=1000, seed=42)

See the documentation for the projections of weighted networks and the comparison of the models.

### Warnings, logging and reproducibility

Problems found while solving or projecting (for example a solver that did not converge) are reported as Python warnings of category `bicm.BiCMWarning`, so they can be filtered:

    import warnings
    import bicm
    warnings.filterwarnings("ignore", category=bicm.BiCMWarning)

Informative messages (for example "Computing the BiCM...") go to the `bicm` logger and are hidden by default. To show them:

    import logging
    logging.basicConfig()
    logging.getLogger("bicm").setLevel(logging.INFO)

The sampling functions accept a `seed` argument (an integer or a `numpy.random.Generator`) to get reproducible samples, e.g. `sample_bicm(avg_mat, seed=42)`. Without it, numpy's global random state is used as in previous versions.

See a more detailed walkthrough in **tests/bicm_test** or **tests/biwcm_test** notebooks, or check out the API in the documentation.

The notebook **benchmarks/web_of_life/web_of_life_benchmark** compares versions 3.5 and 4.0 on all the networks of the [Web of Life](https://www.web-of-life.es/) database: running times, p-values and validated links.

## Development

To install the package from source together with the test dependencies and run the test suite:

    pip install -e ".[test]"
    pytest

The tests of `bicm.wbnm` run only if PyTorch is installed. The tests run automatically on GitHub Actions for every push and pull request.
A new version is published to PyPI when a GitHub release is created with a tag matching the package version (e.g. `v4.0.0`).

## How to cite

If you use the `bicm` module, please cite its location on Github
[https://github.com/mat701/BiCM](https://github.com/mat701/BiCM) and the
original articles [Vallarano2021], [Saracco2015] and [Saracco2017].

If you use the weighted models (BiWCM_c or BiWCM_d), please also cite [Bruno2023].

### References

[Vallarano2021] [N. Vallarano, M. Bruno, E. Marchese, G. Trapani, F. Saracco, T. Squartini, G. Cimini, M. Zanon, Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints, Nature Scientific Reports](https://doi.org/10.1038/s41598-021-93830-4)

[Bruno2023] [M. Bruno, D. Mazzilli, A. Patelli, T. Squartini, F. Saracco, Inferring comparative advantage via entropy maximization. J. Phys. Complex. 4, 045011 (2023), doi:10.1088/2632-072X/ad1411](https://doi.org/10.1088/2632-072X/ad1411)

[Saracco2015] [F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini, Randomizing bipartite networks: the case of the World Trade Web, Scientific Reports 5, 10595 (2015)](http://www.nature.com/articles/srep10595).

[Saracco2017] [F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, and T. Squartini, Inferring monopartite projections of bipartite networks: an entropy-based approach, New J. Phys. 19, 053022 (2017)](http://stacks.iop.org/1367-2630/19/i=5/a=053022)


_Authors_:

[Matteo Bruno](https://csl.sony.it/member/matteo-bruno/) (BiCM) (a.k.a. [mat701](https://github.com/mat701))

Antonio Desiderio (fast exact projection, from the `fast_bicm_exact` package)

Lorenzo Buffa and Daniele Cirulli, Centro Ricerche Enrico Fermi (weighted null models, from the [`wbnm`](https://github.com/lbuffa/wbnm) package)
