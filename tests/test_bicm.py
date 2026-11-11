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
    assert bicm.__version__ == "3.5.0"


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


def test_poibin_projection_single_process():
    # Used to fail with a KeyError when threads_num=1 and progress_bar=False.
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
