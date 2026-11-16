"""Automated tests for the bicm package. Run them with `pytest`."""
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

import bicm
from bicm import (
    BipartiteGraph,
    adjacency_list_from_biadjacency,
    adjacency_list_from_edgelist_bipartite,
    biadjacency_from_adjacency_list,
    biadjacency_from_edgelist,
    bicm_from_fitnesses,
    edgelist_from_adjacency_list_bipartite,
    edgelist_from_biadjacency,
    sample_bicm,
    sample_bicm_edgelist,
)
from bicm.pvalues import poibin_tail

DATA_DIR = Path(__file__).parent
TOL = 1e-5


@pytest.fixture(scope="module")
def pollinators():
    """Plant-pollinator network from Petanidou (1991), with node names."""
    table = np.loadtxt(DATA_DIR / "test_dataset.csv", delimiter=",", dtype=str)
    plants = table[1:, 0]
    pollinators = table[0, 1:]
    biad_mat = table[1:, 1:].astype(np.ubyte)
    return biad_mat, plants, pollinators


@pytest.fixture(scope="module")
def small_binary():
    """Small random binary matrix without empty rows or columns."""
    rng = np.random.default_rng(42)
    mat = (rng.random((40, 60)) < 0.15).astype(int)
    mat = mat[mat.sum(1) > 0]
    return mat[:, mat.sum(0) > 0]


@pytest.fixture(scope="module")
def weighted_matrix():
    return np.loadtxt(DATA_DIR / "test_weighted.csv", delimiter=",")


def assert_degrees_match(graph, rows_deg, cols_deg):
    avg_mat = graph.get_bicm_matrix()
    np.testing.assert_allclose(np.asarray(avg_mat).sum(1), rows_deg, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(np.asarray(avg_mat).sum(0), cols_deg, atol=TOL, rtol=TOL)


def test_version():
    assert bicm.__version__ == "4.0.0"


# Data structure conversions

def test_edgelist_biadjacency_round_trip(small_binary):
    edgelist = edgelist_from_biadjacency(small_binary)[0]
    rebuilt = biadjacency_from_edgelist(edgelist)[0]
    np.testing.assert_array_equal(rebuilt, small_binary)


@pytest.mark.parametrize("sparse_type", [scipy.sparse.csr_matrix, scipy.sparse.csr_array])
def test_edgelist_from_sparse_matches_dense(small_binary, sparse_type):
    dense = edgelist_from_biadjacency(small_binary)[0]
    sparse = edgelist_from_biadjacency(sparse_type(small_binary))[0]
    assert sorted(map(tuple, dense.tolist())) == sorted(map(tuple, sparse.tolist()))


def test_adjacency_list_round_trip(small_binary):
    adj_list, inv_adj_list, rows_deg, cols_deg = adjacency_list_from_biadjacency(small_binary)
    np.testing.assert_array_equal(rows_deg, small_binary.sum(1))
    np.testing.assert_array_equal(cols_deg, small_binary.sum(0))
    assert sum(len(v) for v in adj_list.values()) == small_binary.sum()
    assert sum(len(v) for v in inv_adj_list.values()) == small_binary.sum()
    rebuilt = biadjacency_from_adjacency_list(adj_list)
    np.testing.assert_array_equal(rebuilt, small_binary)


def test_edgelist_from_adjacency_list(small_binary):
    adj_list = adjacency_list_from_biadjacency(small_binary)[0]
    edgelist = edgelist_from_adjacency_list_bipartite(adj_list)
    assert len(edgelist) == small_binary.sum()


# BiCM solver

@pytest.mark.parametrize("input_type", ["array", "list", "csr_matrix", "csr_array"])
def test_bicm_biadjacency_inputs(small_binary, input_type):
    converters = {
        "array": lambda m: m,
        "list": lambda m: m.tolist(),
        "csr_matrix": scipy.sparse.csr_matrix,
        "csr_array": scipy.sparse.csr_array,
    }
    graph = BipartiteGraph(biadjacency=converters[input_type](small_binary))
    graph.solve_tool(print_error=False)
    assert_degrees_match(graph, small_binary.sum(1), small_binary.sum(0))


@pytest.mark.parametrize("method", ["newton", "quasinewton", "fixed-point", "root"])
def test_bicm_solver_methods(small_binary, method):
    graph = BipartiteGraph(degree_sequences=(small_binary.sum(1), small_binary.sum(0)))
    graph.solve_tool(method=method, tol=1e-10, print_error=False)
    assert_degrees_match(graph, small_binary.sum(1), small_binary.sum(0))


def test_bicm_with_names(pollinators):
    biad_mat, plants, pollinator_names = pollinators
    edgelist = edgelist_from_biadjacency(biad_mat)[0]
    plants_dict = dict(enumerate(np.unique(plants)))
    pollinators_dict = dict(enumerate(np.unique(pollinator_names)))
    edgelist_names = [(plants_dict[e[0]], pollinators_dict[e[1]]) for e in edgelist]

    from_edgelist = BipartiteGraph(edgelist=edgelist_names)
    from_edgelist.solve_tool(print_error=False)
    adj_list_names = adjacency_list_from_edgelist_bipartite(edgelist_names, convert_type=False)[0]
    from_adj_list = BipartiteGraph(adjacency_list=adj_list_names)
    from_adj_list.solve_tool(print_error=False)

    x_edgelist, y_edgelist = from_edgelist.get_bicm_fitnesses()
    x_adj_list, y_adj_list = from_adj_list.get_bicm_fitnesses()
    assert set(x_edgelist) == set(x_adj_list)
    for name in x_edgelist:
        assert x_edgelist[name] == pytest.approx(x_adj_list[name], rel=1e-4)
    for name in y_edgelist:
        assert y_edgelist[name] == pytest.approx(y_adj_list[name], rel=1e-4)


def test_solve_twice_gives_same_solution(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    graph.solve_tool(print_error=False)
    x_first = graph.x.copy()
    graph.solve_tool(light_mode=True, print_error=False)
    np.testing.assert_allclose(graph.x, x_first, rtol=1e-4)


def test_bicm_from_fitnesses(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    graph.solve_tool(print_error=False)
    avg_mat = bicm_from_fitnesses(graph.x, graph.y)
    expected = np.outer(graph.x, graph.y) / (1 + np.outer(graph.x, graph.y))
    np.testing.assert_allclose(avg_mat, expected)


# Weighted models

def test_biwcm_discrete():
    # Integer weights of moderate size. With the huge weights of test_weighted.csv
    # (up to 1e11) the discrete solver is numerically fragile and its result
    # depends on the numpy/LAPACK version.
    rng = np.random.default_rng(7)
    int_mat = rng.poisson(3.0, size=(40, 60)) * (rng.random((40, 60)) < 0.5)
    int_mat = int_mat[int_mat.sum(1) > 0]
    int_mat = int_mat[:, int_mat.sum(0) > 0]
    graph = BipartiteGraph(biadjacency=int_mat)
    assert graph.weighted and not graph.continuous_weights
    graph.solve_tool(print_error=False)
    avg_mat = graph.get_bicm_matrix()
    np.testing.assert_allclose(avg_mat.sum(1), int_mat.sum(1), rtol=1e-3)
    np.testing.assert_allclose(avg_mat.sum(0), int_mat.sum(0), rtol=1e-3)


def test_biwcm_continuous(weighted_matrix):
    graph = BipartiteGraph(biadjacency=weighted_matrix)
    assert graph.weighted and graph.continuous_weights
    graph.solve_tool(print_error=False)
    avg_mat = graph.get_bicm_matrix()
    np.testing.assert_allclose(avg_mat.sum(1), weighted_matrix.sum(1), rtol=1e-3)
    np.testing.assert_allclose(avg_mat.sum(0), weighted_matrix.sum(0), rtol=1e-3)


# Sampling

def test_sample_bicm(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    avg_mat = graph.get_bicm_matrix()
    np.random.seed(0)
    samples = [sample_bicm(avg_mat) for _ in range(200)]
    assert all(s.shape == small_binary.shape for s in samples)
    assert all(np.isin(s, [0, 1]).all() for s in samples)
    mean_links = np.mean([s.sum() for s in samples])
    assert mean_links == pytest.approx(small_binary.sum(), rel=0.05)


def test_sample_bicm_edgelist(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    graph.solve_tool(print_error=False)
    np.random.seed(0)
    edgelist = sample_bicm_edgelist(graph.x, graph.y)
    assert len(edgelist) == pytest.approx(small_binary.sum(), rel=0.2)


# Projections

def test_projection_same_for_dense_and_sparse(pollinators):
    biad_mat = pollinators[0]
    dense = BipartiteGraph(biadjacency=biad_mat)
    sparse = BipartiteGraph(biadjacency=scipy.sparse.csr_array(biad_mat))
    proj_dense = dense.get_rows_projection(alpha=0.05, progress_bar=False, threads_num=1, fmt="edgelist")
    proj_sparse = sparse.get_rows_projection(alpha=0.05, progress_bar=False, threads_num=1, fmt="edgelist")
    assert sorted(map(tuple, proj_dense)) == sorted(map(tuple, proj_sparse))
    assert len(proj_dense) > 0


@pytest.mark.parametrize("method", ["poisson", "normal", "rna"])
def test_projection_pvalues_are_probabilities(pollinators, method):
    graph = BipartiteGraph(biadjacency=pollinators[0])
    graph.compute_projection(rows=True, approx_method=method, progress_bar=False, threads_num=1)
    pvals = [p for neighbours in graph.rows_pvals.values() for p in np.atleast_1d(list(
        neighbours.values() if isinstance(neighbours, dict) else neighbours))]
    assert len(pvals) > 0
    assert all(0 <= p <= 1 + 1e-12 for p in pvals)


def test_higher_alpha_validates_more_links(pollinators):
    graph = BipartiteGraph(biadjacency=pollinators[0])
    strict = graph.get_cols_projection(alpha=0.01, progress_bar=False, threads_num=1, fmt="edgelist")
    graph.compute_projection(rows=False, alpha=0.2, progress_bar=False, threads_num=1)
    loose = graph.get_cols_projection(alpha=0.2, progress_bar=False, threads_num=1, fmt="edgelist")
    assert len(loose) >= len(strict)


# Warnings, seeds and parallel projections

def test_square_matrix_warns():
    with pytest.warns(bicm.BiCMWarning, match="square"):
        BipartiteGraph(biadjacency=np.eye(5, dtype=int))


def test_deprecated_solver_warns(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    with pytest.warns(FutureWarning, match="deprecated"):
        graph.solve_bicm(print_error=False)


def test_bicm_warnings_can_be_silenced(recwarn):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=bicm.BiCMWarning)
        warnings.filterwarnings("ignore", module="numba")
        BipartiteGraph(biadjacency=np.eye(5, dtype=int))


def test_sampling_seed_is_reproducible(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    graph.solve_tool(print_error=False)
    avg_mat = graph.get_bicm_matrix()
    np.testing.assert_array_equal(sample_bicm(avg_mat, seed=1), sample_bicm(avg_mat, seed=1))
    np.testing.assert_array_equal(sample_bicm(avg_mat, seed=1),
                                  sample_bicm(avg_mat, seed=np.random.default_rng(1)))
    assert sample_bicm_edgelist(graph.x, graph.y, seed=2) == sample_bicm_edgelist(graph.x, graph.y, seed=2)


def test_sampling_without_seed_uses_global_state(small_binary):
    avg_mat = BipartiteGraph(biadjacency=small_binary).get_bicm_matrix()
    np.random.seed(3)
    first = sample_bicm(avg_mat)
    np.random.seed(3)
    np.testing.assert_array_equal(sample_bicm(avg_mat), first)


def test_projection_progress_bar_is_ignored():
    rng = np.random.default_rng(0)
    mat = (rng.random((30, 80)) < 0.08).astype(int)
    mat[:5, :15] = 1  # a block of similar rows, so that some links get validated
    mat = mat[:, mat.sum(0) > 0]
    serial = BipartiteGraph(biadjacency=mat)
    serial.compute_projection(rows=True, approx_method="poibin", threads_num=1, progress_bar=False)
    with_bar = BipartiteGraph(biadjacency=mat)
    with_bar.compute_projection(rows=True, approx_method="poibin", threads_num=1, progress_bar=True)
    links = sorted(map(tuple, serial.get_rows_projection(fmt="edgelist")))
    assert len(links) > 0
    assert links == sorted(map(tuple, with_bar.get_rows_projection(fmt="edgelist")))


def test_parallel_projection_matches_serial(pollinators):
    serial = BipartiteGraph(biadjacency=pollinators[0])
    serial.compute_projection(rows=True, threads_num=1, progress_bar=False)
    parallel = BipartiteGraph(biadjacency=pollinators[0])
    parallel.compute_projection(rows=True, threads_num=2, progress_bar=False)
    assert sorted(map(tuple, serial.get_rows_projection(fmt="edgelist"))) == \
        sorted(map(tuple, parallel.get_rows_projection(fmt="edgelist")))


# Accuracy of the p-values of the projections

def brute_force_pvalues(graph, biad_mat, rows):
    """Exact p-values from the full probability matrix, one Bernoulli variable for each node."""
    avg_mat = graph.get_bicm_matrix() if rows else graph.get_bicm_matrix().T
    biad_mat = biad_mat if rows else biad_mat.T
    cooccurrences = biad_mat @ biad_mat.T
    return {(i, j): poibin_tail(avg_mat[i] * avg_mat[j], int(cooccurrences[i, j]))
            for i in range(len(biad_mat)) for j in range(i + 1, len(biad_mat)) if cooccurrences[i, j] > 0}


def flatten(pvals):
    return {(i, j): p for i, neighbours in pvals.items() for j, p in neighbours.items()}


@pytest.mark.parametrize("rows", [True, False])
def test_poibin_projection_is_exact(pollinators, rows):
    biad_mat = pollinators[0].astype(int)
    graph = BipartiteGraph(biadjacency=biad_mat)
    graph.solve_tool(print_error=False)
    graph.compute_projection(rows=rows, approx_method="poibin")
    pvals = flatten(graph.rows_pvals if rows else graph.cols_pvals)
    reference = brute_force_pvalues(graph, biad_mat, rows)
    assert set(pvals) == set(reference)
    for couple, pval in pvals.items():
        assert pval == pytest.approx(reference[couple], rel=1e-9)
    if rows:  # regression: the p-values used to be truncated at about 1e-15
        assert 0 < min(pvals.values()) < 1e-15


def test_approximated_pvalues_follow_the_exact_ones(pollinators):
    """Regression: the normal and rna p-values were those of the opposite tail."""
    biad_mat = pollinators[0]
    pvals = {}
    for method in ("poibin", "poisson", "normal", "rna"):
        graph = BipartiteGraph(biadjacency=biad_mat)
        graph.solve_tool(print_error=False)
        graph.compute_projection(rows=True, approx_method=method)
        flat = flatten(graph.rows_pvals)
        pvals[method] = np.array([flat[couple] for couple in sorted(flat)])
    bulk = pvals["poibin"] > 1e-3
    assert bulk.sum() > 1000
    for method, tolerance in (("poisson", 0.3), ("normal", 0.6), ("rna", 0.3)):
        log_ratio = np.log(pvals[method][bulk] / pvals["poibin"][bulk])
        assert np.median(np.abs(log_ratio)) < tolerance, method
        assert np.corrcoef(pvals[method][bulk], pvals["poibin"][bulk])[0, 1] > 0.95, method


@pytest.mark.parametrize("method", ["poibin", "poisson", "normal", "rna"])
def test_projection_with_full_and_empty_nodes(method):
    """Nodes linked to a whole layer have certain links: they do not make a couple significant.
    Regression: they got null fitness in the projection and p-values equal to zero."""
    rng = np.random.default_rng(3)
    biad_mat = (rng.random((25, 40)) < 0.25).astype(int)
    biad_mat[0, :] = 1   # full row
    biad_mat[:, 5] = 0   # empty column
    biad_mat[:, 7] = 1   # full column
    biad_mat[3, :] = 0   # row linked to the full column only
    biad_mat[3, 7] = 1
    for rows in (True, False):
        graph = BipartiteGraph(biadjacency=biad_mat)
        graph.solve_tool(print_error=False)
        graph.compute_projection(rows=rows, approx_method=method)
        pvals = flatten(graph.rows_pvals if rows else graph.cols_pvals)
        assert all(0 < p <= 1 for p in pvals.values())
        full_node = 0 if rows else 7
        for (i, j), pval in pvals.items():
            if rows and 3 in (i, j):  # its only common neighbours are certain
                assert pval == 1.0
            elif full_node in (i, j):  # the common neighbours are the degree of the other node
                assert pval > 0.3
        if method == "poibin":
            reference = brute_force_pvalues(graph, biad_mat, rows)
            for couple, pval in pvals.items():
                assert pval == pytest.approx(reference[couple], rel=1e-9)


def test_projection_same_in_light_mode():
    """Without the biadjacency matrix the degenerate nodes are found from the degrees."""
    rng = np.random.default_rng(4)
    biad_mat = (rng.random((25, 40)) < 0.25).astype(int)
    biad_mat[0, :] = 1
    biad_mat[:, 7] = 1
    full = BipartiteGraph(biadjacency=biad_mat)
    full.solve_tool(print_error=False)
    full.compute_projection(rows=True, approx_method="poibin")
    with pytest.warns(bicm.BiCMWarning, match="disconnected or connected to all"):
        light = BipartiteGraph(biadjacency=scipy.sparse.csr_array(biad_mat))
        light.solve_tool(print_error=False)
    light.compute_projection(rows=True, approx_method="poibin")
    assert not np.isnan(light.get_bicm_matrix()).any()
    np.testing.assert_allclose(light.get_bicm_matrix(), full.get_bicm_matrix(), atol=1e-6)
    full_pvals, light_pvals = flatten(full.rows_pvals), flatten(light.rows_pvals)
    assert set(full_pvals) == set(light_pvals)
    for couple in full_pvals:
        assert light_pvals[couple] == pytest.approx(full_pvals[couple], rel=1e-5)


# Interface of the projections

@pytest.fixture()
def block_graph():
    """Graph with two communities: nodes of the same community share many neighbours."""
    rng = np.random.default_rng(0)
    probs = np.full((160, 200), 0.01)
    probs[:80, :100] = 0.6
    probs[80:, 100:] = 0.6
    return BipartiteGraph(biadjacency=(rng.random(probs.shape) < probs).astype(int))


def test_projection_formats(block_graph):
    edgelist = block_graph.get_rows_projection(fmt="edgelist")
    adj_list = block_graph.get_rows_projection(fmt="adjacency_list")
    array = block_graph.get_rows_projection(fmt="array")
    sparse = block_graph.get_rows_projection(fmt="sparse")
    assert len(edgelist) > 0
    assert sorted(map(tuple, edgelist)) == sorted((i, j) for i, js in adj_list.items() for j in js)
    assert array.shape == (160, 160) and (array == array.T).all() and array.trace() == 0
    assert array.sum() == 2 * len(edgelist)
    assert all(array[i, j] == 1 for i, j in edgelist)
    np.testing.assert_array_equal(sparse.toarray(), array)
    with pytest.raises(ValueError, match="fmt"):
        block_graph.get_rows_projection(fmt="wrong")


def test_projection_getters_update_the_validation(block_graph):
    block_graph.compute_projection(rows=True, alpha=0.01, validation_method="bonferroni")
    strict = block_graph.get_rows_projection(fmt="edgelist")
    assert sorted(map(tuple, block_graph.get_rows_projection(fmt="edgelist"))) == sorted(map(tuple, strict))
    pvals = block_graph.rows_pvals
    loose = block_graph.get_rows_projection(alpha=0.5, validation_method="global", fmt="edgelist")
    assert len(loose) > len(strict)
    assert block_graph.rows_pvals is pvals  # the p-values are not computed again
    assert len(block_graph.get_rows_projection(fmt="edgelist")) == len(loose)  # last validation kept
    exact = block_graph.get_rows_projection(method="poibin", fmt="edgelist")
    assert block_graph.projection_method == "poibin"
    assert block_graph.rows_pvals is not pvals
    assert len(exact) > 0


def test_validation_methods_are_nested(block_graph):
    sizes = [len(block_graph.get_cols_projection(alpha=0.05, validation_method=method, fmt="edgelist"))
             for method in ("bonferroni", "fdr", "global")]
    assert 0 < sizes[0] <= sizes[1] <= sizes[2]
    with pytest.raises(ValueError, match="validation method"):
        block_graph.compute_projection(validation_method="wrong")
    with pytest.raises(ValueError, match="approx_method"):
        block_graph.compute_projection(approx_method="wrong")


def test_projection_with_names(pollinators):
    biad_mat, plants, _ = pollinators
    edgelist = [(plants[i], "pollinator_{}".format(j)) for i, j in zip(*np.nonzero(biad_mat))]
    named = BipartiteGraph(edgelist=edgelist)
    links = named.get_rows_projection(fmt="edgelist")
    assert len(links) > 0 and set(np.ravel(links)) <= set(plants)
    adjacency = named.get_rows_projection(fmt="array")
    inverse = {name: index for index, name in named.rows_dict.items()}
    assert all(adjacency[inverse[a], inverse[b]] == 1 for a, b in links)


def test_projected_pvalues_matrix(block_graph):
    with pytest.warns(bicm.BiCMWarning, match="First compute"):
        assert block_graph.get_projected_pvals_mat() is None
    block_graph.compute_projection(rows=False)
    matrix = block_graph.get_projected_pvals_mat()
    assert matrix.shape == (block_graph.n_cols, block_graph.n_cols)
    np.testing.assert_array_equal(matrix, matrix.T)
    for i, neighbours in block_graph.cols_pvals.items():
        for j, pval in neighbours.items():
            assert matrix[i, j] == pval
    assert (matrix == 1).sum() >= block_graph.n_cols  # diagonal and couples without common neighbours


def test_full_projection_and_clean_edges(block_graph):
    with pytest.warns(bicm.BiCMWarning, match="full projection"):
        links = block_graph.get_cols_projection(alpha=1, fmt="edgelist")
    assert len(links) == block_graph.n_cols * (block_graph.n_cols - 1) // 2
    block_graph.clean_edges()
    assert block_graph.rows_pvals is None and block_graph.cols_pvals is None
    assert not block_graph.is_initialized and not block_graph.is_cols_projected


def test_projection_needs_edges_and_binary_weights(small_binary, weighted_matrix):
    degrees_only = BipartiteGraph(degree_sequences=(small_binary.sum(1), small_binary.sum(0)))
    with pytest.warns(bicm.BiCMWarning, match="Without the edges"):
        assert degrees_only.get_rows_projection() is None
    weighted = BipartiteGraph(biadjacency=weighted_matrix)
    with pytest.warns(bicm.BiCMWarning, match="Weighted projection"):
        assert weighted.get_rows_projection() is None


# Validation of the links of the original network

def test_binary_link_pvalues(small_binary):
    graph = BipartiteGraph(biadjacency=small_binary)
    pvals = graph.get_weighted_pvals_mat()
    avg_mat = graph.get_bicm_matrix()
    np.testing.assert_allclose(pvals[small_binary == 1], avg_mat[small_binary == 1])
    assert (pvals[small_binary == 0] == 1).all()
    validated = graph.get_validated_matrix(significance=0.05, validation_method="global")
    np.testing.assert_array_equal(validated, (pvals <= 0.05).astype(np.ubyte))
    assert validated.sum() >= graph.get_validated_matrix(significance=0.05, validation_method="fdr").sum() \
        >= graph.get_validated_matrix(significance=0.05, validation_method="bonferroni").sum()


def test_weighted_link_pvalues():
    rng = np.random.default_rng(7)
    int_mat = rng.poisson(3.0, size=(40, 60)) * (rng.random((40, 60)) < 0.5)
    int_mat = int_mat[int_mat.sum(1) > 0]
    int_mat = int_mat[:, int_mat.sum(0) > 0]
    graph = BipartiteGraph(biadjacency=int_mat)
    graph.solve_tool(print_error=False)
    pvals = graph.get_weighted_pvals_mat()
    assert np.all((pvals > 0) & (pvals <= 1))
    assert (pvals[int_mat == 0] == 1).all()
    np.testing.assert_allclose(pvals, np.outer(graph.x, graph.y) ** int_mat)
    # Huge weights: the p-value underflows to zero instead of becoming nan
    huge = BipartiteGraph(biadjacency=int_mat * 10**6)
    huge.solve_tool(print_error=False)
    assert not np.isnan(huge.get_weighted_pvals_mat()).any()


def test_perfectly_nested_matrix():
    """Regression: every link is certain, nothing is solved and the projections crashed."""
    nested = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    graph = BipartiteGraph(biadjacency=nested)
    graph.solve_tool(print_error=False)
    np.testing.assert_array_equal(graph.get_bicm_matrix(), nested)
    for method in ("poisson", "poibin", "normal", "rna"):
        for rows in (True, False):
            with pytest.warns(bicm.BiCMWarning, match="No V-motifs"):
                graph.compute_projection(rows=rows, approx_method=method)
            pvals = flatten(graph.rows_pvals if rows else graph.cols_pvals)
            assert len(pvals) == 3 and all(p == 1.0 for p in pvals.values())
    assert graph.get_rows_projection() == {}
