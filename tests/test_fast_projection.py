"""Tests of the fast projection from lists of edges."""
import numpy as np
import pytest

import bicm
from bicm import BipartiteGraph
from bicm import fast_projection as bfp
from bicm import pvalues as bpv


def random_edges(n_rows=80, n_cols=12, seed=0):
    """Random network in which the first two columns share more neighbours than expected."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_rows):
        links = rng.choice(np.arange(2, n_cols), size=int(rng.integers(2, 6)), replace=False)
        if i % 3 == 0:
            links = np.union1d(links, [0, 1])
        for c in links:
            rows.append("r{}".format(i))
            cols.append("c{:02d}".format(c))
    return np.array(rows), np.array(cols)


def chunked(rows, cols, n_chunks=4):
    """Split the edges in chunks, keeping the edges of a row in the same chunk."""
    names = np.unique(rows)
    for part in np.array_split(names, n_chunks):
        mask = np.isin(rows, part)
        yield rows[mask], cols[mask]


def as_dict(result):
    return {link: (w, p) for link, w, p in zip(result.to_edgelist(), result.weight, result.pval)}


def assert_same_aggregation(first, second):
    assert (first.n_rows, first.n_cols, first.n_edges, first.n_cols_total) == \
        (second.n_rows, second.n_cols, second.n_edges, second.n_cols_total)
    np.testing.assert_array_equal(first.deg_g, second.deg_g)
    np.testing.assert_array_equal(first.n_c, second.n_c)
    for agg in (first, second):
        agg.order = np.argsort(agg.col_labels)
    np.testing.assert_array_equal(first.col_labels[first.order], second.col_labels[second.order])
    np.testing.assert_array_equal(first.col_deg[first.order], second.col_deg[second.order])

    def couples(agg):
        return {tuple(sorted((agg.col_labels[i], agg.col_labels[j]))): w
                for i, j, w in zip(agg.pairs_i, agg.pairs_j, agg.weight)}
    assert couples(first) == couples(second)


# Aggregation

def test_aggregation_counts():
    rows = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    cols = np.array(["x", "y", "x", "y", "x", "z", "z", "w"])
    agg = bfp.aggregate_bipartite(rows, cols)
    assert (agg.n_rows, agg.n_cols, agg.n_edges, agg.n_cols_total) == (4, 4, 8, 4)
    assert agg.deg_g.tolist() == [2] and agg.n_c.tolist() == [4]
    assert dict(zip(agg.col_labels.tolist(), agg.col_deg.tolist())) == {"w": 1, "x": 3, "y": 2, "z": 2}
    weights = {(agg.col_labels[i], agg.col_labels[j]): w
               for i, j, w in zip(agg.pairs_i, agg.pairs_j, agg.weight)}
    assert weights == {("x", "y"): 2, ("x", "z"): 1, ("w", "z"): 1}
    # Row "e" is linked to every column: it is removed with its co-occurrences
    full = bfp.aggregate_bipartite(np.append(rows, ["e"] * 4), np.append(cols, ["x", "y", "z", "w"]))
    assert (full.n_rows, full.n_edges) == (4, 8)
    np.testing.assert_array_equal(full.weight, agg.weight)


def test_cooccurrences_above_255():
    """The co-occurrences are not computed with small integer types, which overflow."""
    n_rows = 300
    rows = np.concatenate([np.repeat(np.arange(n_rows), 2), [n_rows]])
    cols = np.concatenate([np.tile(np.array(["a", "b"]), n_rows), ["c"]])
    for agg in (bfp.aggregate_bipartite(rows, cols), bfp.aggregate_chunks([(rows, cols)])):
        assert agg.weight.tolist() == [n_rows]


def test_repeated_edges_count_once():
    rows, cols = random_edges(seed=1)
    doubled = bfp.aggregate_bipartite(np.concatenate((rows, rows)), np.concatenate((cols, cols)))
    assert_same_aggregation(doubled, bfp.aggregate_bipartite(rows, cols))


def test_chunks_match_in_memory_aggregation():
    rows, cols = random_edges(n_rows=200, n_cols=90, seed=2)  # more than 64 columns: the buffers grow
    assert_same_aggregation(bfp.aggregate_chunks(chunked(rows, cols)), bfp.aggregate_bipartite(rows, cols))
    integers = np.unique(rows, return_inverse=True)[1], np.unique(cols, return_inverse=True)[1]
    assert bfp.aggregate_chunks(chunked(*integers)).col_labels.dtype.kind == "i"


def test_full_and_empty_nodes_are_removed_recursively():
    """A row linked to all the columns, a column linked to all the rows, and a row left
    without links once the full column is removed."""
    rows, cols = random_edges(n_rows=60, n_cols=10, seed=3)
    all_cols = np.unique(cols)
    rows = np.concatenate((rows, np.repeat("full_row", len(all_cols) + 1), np.unique(rows), ["lonely"]))
    cols = np.concatenate((cols, all_cols, ["full_col"], np.repeat("full_col", 60), ["full_col"]))
    plain = bfp.aggregate_bipartite(*random_edges(n_rows=60, n_cols=10, seed=3))
    for agg in (bfp.aggregate_bipartite(rows, cols), bfp.aggregate_chunks(chunked(rows, cols, 3))):
        assert agg.n_cols_total == plain.n_cols + 1
        agg.n_cols_total = plain.n_cols_total
        assert_same_aggregation(agg, plain)


def test_min_overlap():
    rows, cols = random_edges(seed=4)
    agg = bfp.aggregate_bipartite(rows, cols, min_overlap=3)
    assert agg.weight.min() >= 3
    full = bfp.aggregate_bipartite(rows, cols)
    assert len(agg.weight) == (full.weight >= 3).sum()
    assert bfp.aggregate_bipartite(rows, cols, min_overlap=10**6) is None


def test_degenerate_inputs():
    assert bfp.aggregate_bipartite([], []) is None
    assert bfp.aggregate_chunks([]) is None
    assert bfp.aggregate_chunks([([], [])]) is None
    assert bfp.aggregate_bipartite(["a", "b"], ["x", "x"]) is None  # only a full column
    assert len(bfp.project([], [])) == 0
    assert len(bfp.project_chunks([(["a", "b"], ["x", "x"])])) == 0
    with pytest.raises(ValueError, match="same length"):
        bfp.aggregate_bipartite(["a"], ["x", "y"])
    with pytest.raises(ValueError, match="missing"):
        bfp.aggregate_bipartite(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="missing"):
        bfp.aggregate_bipartite(np.array(["a", None], dtype=object), np.array(["x", "y"], dtype=object))


# Solver

def test_reduced_solver_reproduces_the_degrees():
    rows, cols = random_edges(seed=5)
    agg = bfp.aggregate_bipartite(rows, cols)
    x_u, y = bfp.fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    probs = bpv.link_probabilities(x_u[:, None], y[None, :])
    np.testing.assert_allclose(probs.sum(axis=1), agg.deg_g, atol=1e-6)
    np.testing.assert_allclose(agg.n_c @ probs, agg.col_deg, atol=1e-6)


# Projection

@pytest.mark.parametrize("method", ["exact", "poisson", "normal", "rna"])
def test_projection_matches_bipartite_graph(method):
    rows, cols = random_edges(n_rows=120, n_cols=14, seed=6)
    result = bfp.project(rows, cols, method=method, alpha=0.2)
    graph = BipartiteGraph(edgelist=list(zip(rows.tolist(), cols.tolist())))
    graph.solve_tool(print_error=False, tol=1e-12)
    graph_method = "poibin" if method == "exact" else method
    links = graph.get_cols_projection(alpha=0.2, method=graph_method, fmt="edgelist")
    assert len(result) > 0
    assert sorted(map(tuple, map(sorted, result.to_edgelist()))) == sorted(map(tuple, map(sorted, links)))
    assert result.n_tests == 14 * 13 // 2
    inverse = {v: k for k, v in graph.cols_dict.items()}
    for (a, b), (weight, pval) in as_dict(result).items():
        i, j = sorted((inverse[a], inverse[b]))
        assert graph.v_adj_list[i][j] == weight
        assert graph.cols_pvals[i][j] == pytest.approx(pval, rel=1e-6)


def test_exact_projection_matches_brute_force():
    rows, cols = random_edges(n_rows=70, n_cols=11, seed=7)
    result = bfp.project(rows, cols, method="exact", correction="global", alpha=1.0)
    agg = bfp.aggregate_bipartite(rows, cols)
    assert len(result) == len(agg.weight)  # with alpha = 1 every tested couple is returned
    x_u, y = bfp.fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    x = np.repeat(x_u, agg.n_c.astype(int))
    index = {label: i for i, label in enumerate(agg.col_labels)}
    for (a, b), (weight, pval) in as_dict(result).items():
        probs = bpv.link_probabilities(x, y[index[a]]) * bpv.link_probabilities(x, y[index[b]])
        assert pval == pytest.approx(bpv.poibin_tail(probs, int(weight)), rel=1e-10)


def test_project_chunks_matches_project():
    rows, cols = random_edges(n_rows=150, n_cols=16, seed=8)
    in_memory = as_dict(bfp.project(rows, cols, alpha=0.3))
    streamed = as_dict(bfp.project_chunks(chunked(rows, cols), alpha=0.3))
    assert len(in_memory) > 0
    assert {tuple(sorted(k)) for k in in_memory} == {tuple(sorted(k)) for k in streamed}
    lookup = {tuple(sorted(k)): v for k, v in streamed.items()}
    for link, (weight, pval) in in_memory.items():
        assert lookup[tuple(sorted(link))][0] == weight
        assert lookup[tuple(sorted(link))][1] == pytest.approx(pval, rel=1e-9)


def test_corrections_and_statistics():
    rows, cols = random_edges(n_rows=150, n_cols=16, seed=9)
    loose = bfp.project(rows, cols, correction="global", alpha=0.05)
    fdr = bfp.project(rows, cols, correction="fdr", alpha=0.05)
    bonferroni = bfp.project(rows, cols, correction="bonferroni", alpha=0.05)
    assert len(loose) >= len(fdr) >= len(bonferroni)
    assert set(bonferroni.to_edgelist()) <= set(fdr.to_edgelist()) <= set(loose.to_edgelist())
    assert bonferroni.threshold == pytest.approx(0.05 / (16 * 15 // 2))
    for result in (loose, fdr, bonferroni):
        assert np.all(result.pval <= result.threshold)
        assert np.all(result.pval_corrected <= 0.05 * (1 + 1e-12))
        assert np.all(result.pval_corrected >= result.pval)
        assert np.all(result.z_score > 0)
    with pytest.raises(ValueError, match="unknown method"):
        bfp.project(rows, cols, method="wrong")
    with pytest.raises(ValueError, match="validation method"):
        bfp.project(rows, cols, correction="wrong")


def test_validate_edges():
    weights = np.array([5.0, 9.0, 2.0])
    mu, var, skew = np.array([4.0, 3.0, 2.0]), np.array([2.0, 2.0, 0.0]), np.zeros(3)
    with pytest.raises(ValueError, match="exact_pvals"):
        bfp.validate_edges(weights, mu, var, skew, method="exact")
    out = bfp.validate_edges(weights, mu, var, skew, method="standard", correction_type="bonferroni", alpha=0.01)
    assert out["reject_mask"].tolist() == [False, True, False]
    assert out["z_scores"].tolist() == [pytest.approx(1 / np.sqrt(2)), pytest.approx(6 / np.sqrt(2)), 0.0]
    exact = bfp.validate_edges(weights, mu, var, skew, exact_pvals=[0.5, 1e-9, 1.0], correction_type="global")
    np.testing.assert_array_equal(exact["raw_target_pvals"], [0.5, 1e-9, 1.0])
    empty = bfp.validate_edges([], [], [], [], exact_pvals=[])
    assert empty["reject_mask"].shape == (0,)


def test_to_dataframe():
    pandas = pytest.importorskip("pandas")
    rows, cols = random_edges(n_rows=150, n_cols=16, seed=8)
    result = bfp.project(rows, cols, alpha=0.3)
    frame = result.to_dataframe()
    assert isinstance(frame, pandas.DataFrame)
    assert list(frame.columns) == ["col_i", "col_j", "weight", "z_score", "pval_raw", "pval_corrected"]
    assert len(frame) == len(result)


def test_solver_warning_is_a_bicm_warning():
    with pytest.warns(bicm.BiCMWarning, match="did not converge"):
        bfp.fit_bicm_reduced(np.array([1.0, 2.0]), np.array([5.0, 5.0]), np.array([100.0, 3.0]))
