"""Tests of the p-values of the co-occurrences: accuracy, tails far below the machine epsilon, edge cases."""
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from bicm import pvalues as bpv
from bicm.poibin import PoiBin


def exact_tail(probs, k):
    """Right tail of a Poisson-binomial variable computed with exact rational arithmetic."""
    pmf = [Fraction(1)]
    for p in probs:
        p = Fraction(float(p))
        new = [Fraction(0)] * (len(pmf) + 1)
        for v, mass in enumerate(pmf):
            new[v] += mass * (1 - p)
            new[v + 1] += mass * p
        pmf = new
    return sum(pmf[k:], Fraction(0))


def expanded_probabilities(x_u, n_c, yi, yj):
    """Probabilities of the V-motifs through every single node of the opposite layer."""
    x = np.repeat(x_u, n_c.astype(int))
    return (x * yi / (1 + x * yi)) * (x * yj / (1 + x * yj))


def rel_err(value, reference):
    return abs(value / float(reference) - 1) if reference != 0 else abs(value)


@pytest.fixture(scope="module")
def classes():
    """A small system: 6 degree classes of the opposite layer, 5 nodes of the projected layer."""
    rng = np.random.default_rng(0)
    x_u = rng.lognormal(-1.5, 1.2, 6)
    n_c = rng.integers(1, 9, 6).astype(float)
    y = rng.lognormal(-0.5, 1.2, 5)
    pairs_i, pairs_j = np.triu_indices(5, k=1)
    return x_u, n_c, y, pairs_i, pairs_j


def all_weights(classes):
    """Every couple of the fixture repeated with every possible number of co-occurrences."""
    x_u, n_c, y, pairs_i, pairs_j = classes
    weights = np.arange(0, int(n_c.sum()) + 2)
    return (np.repeat(pairs_i, len(weights)), np.repeat(pairs_j, len(weights)),
            np.tile(weights, len(pairs_i)))


# Exact p-values

@pytest.mark.parametrize("k_direct", [10**6, 0], ids=["direct", "fft"])
def test_exact_pvalues_match_rational_arithmetic(classes, k_direct):
    """Relative accuracy on the whole support, down to p-values of 1e-60 and below."""
    x_u, n_c, y, _, _ = classes
    pairs_i, pairs_j, weight = all_weights(classes)
    pvals = bpv.exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=k_direct)
    smallest = 1.0
    for i, j, k, pval in zip(pairs_i, pairs_j, weight, pvals):
        reference = exact_tail(expanded_probabilities(x_u, n_c, y[i], y[j]), int(k))
        if k == 0:  # the FFT path is never used for k = 0, the direct one returns 1
            assert pval == pytest.approx(1.0)
            continue
        assert rel_err(pval, reference) < 1e-10, (i, j, k, pval, float(reference))
        if reference > 0:
            smallest = min(smallest, float(reference))
    assert smallest < 1e-40  # the test really covers tails far below the machine epsilon


def test_tail_larger_than_support_is_zero(classes):
    x_u, n_c, y, pairs_i, pairs_j = classes
    weight = np.full(len(pairs_i), int(n_c.sum()) + 1)
    for k_direct in (10**6, 0):
        pvals = bpv.exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=k_direct)
        np.testing.assert_array_equal(pvals, 0.0)


def test_fft_matches_direct_convolution_on_large_classes():
    rng = np.random.default_rng(1)
    x_u = rng.lognormal(-4, 1.0, 60)
    n_c = rng.integers(1, 400, 60).astype(float)
    y = rng.lognormal(0.5, 1.0, 12)
    pairs_i, pairs_j = np.triu_indices(12, k=1)
    mean, var, _ = bpv.moments_grouped(x_u, n_c, y, pairs_i, pairs_j)
    weight = np.maximum(1, (mean + rng.normal(0, 4, len(mean)) * np.sqrt(var)).astype(int))
    direct = bpv.exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=10**9)
    fft = bpv.exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=0, n_jobs=1)
    threaded = bpv.exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=0, n_jobs=3)
    assert direct.min() < 1e-4 and direct.max() > 0.9
    np.testing.assert_allclose(fft, direct, rtol=1e-9)
    np.testing.assert_array_equal(threaded, fft)


@pytest.mark.parametrize("n, p", [(300, 0.5), (5000, 0.9), (10**5, 1e-3), (10**6, 0.3)])
@pytest.mark.parametrize("k_direct", [10**9, 0], ids=["direct", "fft"])
def test_single_class_is_binomial(n, p, k_direct):
    """With one degree class the variable is Binomial: compare with scipy on both tails."""
    link = np.sqrt(p)
    x_u = np.array([link / (1 - link)])
    y = np.ones(2)
    mean, std = n * p, np.sqrt(n * p * (1 - p))
    ks = np.unique(np.clip(np.array(
        [mean - 6 * std, mean - std, mean, mean + 1, mean + std, mean + 6 * std, mean + 15 * std,
         mean + 35 * std]).astype(int), 1, n))
    if k_direct > 0:
        ks = ks[ks <= 4000]  # keep the quadratic algorithm fast
    zeros = np.zeros(len(ks), dtype=int)
    pvals = bpv.exact_pvals_grouped(x_u, np.array([float(n)]), y, zeros, zeros + 1, ks, k_direct=k_direct)
    log_reference = stats.binom.logsf(ks - 1, n, p)
    representable = log_reference > -700
    np.testing.assert_allclose(np.log(pvals[representable]), log_reference[representable], atol=1e-9)
    assert np.all(pvals[~representable] < 1e-300)


def test_pvalues_below_machine_epsilon_are_not_truncated():
    """Regression: 1 - cdf cannot go below 1e-16, and returned noise or zero."""
    x_u = np.array([0.05])
    n_c = np.array([2000.0])
    y = np.array([1.0, 1.0])
    pval = bpv.exact_pvals_grouped(x_u, n_c, y, [0], [1], [60])[0]
    p_both = (0.05 / 1.05) ** 2
    assert 0 < pval < 1e-30
    assert np.log(pval) == pytest.approx(stats.binom.logsf(59, 2000, p_both), abs=1e-9)


def test_certain_and_impossible_links():
    """Null and infinite fitnesses give probabilities 0 and 1, never nan."""
    x_u = np.array([0.0, np.inf, 0.7])
    n_c = np.array([3.0, 2.0, 4.0])
    y = np.array([np.inf, 1.3, 0.0])
    np.testing.assert_array_equal(bpv.link_probabilities(x_u[:, None], y[None, :])[:, 0], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(bpv.link_probabilities(x_u[:, None], y[None, :])[:, 2], [0.0, 0.0, 0.0])
    # Couple (0, 1): the 2 nodes of infinite fitness are certain common neighbours
    probs = np.concatenate((np.zeros(3), np.ones(2), np.full(4, 0.7 * 1.3 / (1 + 0.7 * 1.3))))
    for k in range(0, 8):
        pval = bpv.cooccurrence_pvalues(x_u, n_c, y, [0], [1], [k], method="poibin")[0]
        assert pval == pytest.approx(float(exact_tail(probs, k)), rel=1e-12)
    # Couple (0, 2): no common neighbour is possible
    assert bpv.cooccurrence_pvalues(x_u, n_c, y, [0], [2], [1], method="poibin")[0] == 0.0
    mean, var, skew = bpv.moments_grouped(x_u, n_c, y, np.array([0, 0]), np.array([1, 2]))
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(var)) and np.all(np.isfinite(skew))
    assert mean[1] == 0.0 and var[1] == 0.0


def test_reference_poibin_tail(classes):
    x_u, n_c, y, _, _ = classes
    probs = expanded_probabilities(x_u, n_c, y[0], y[1])
    for k in (0, 1, 3, 10, len(probs)):
        assert rel_err(bpv.poibin_tail(probs, k), exact_tail(probs, k)) < 1e-12
    assert bpv.poibin_tail(probs, len(probs) + 1) == 0.0


# Moments and approximations

def test_moments(classes):
    x_u, n_c, y, pairs_i, pairs_j = classes
    mean, var, skew = bpv.moments_grouped(x_u, n_c, y, pairs_i, pairs_j)
    for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        probs = expanded_probabilities(x_u, n_c, y[i], y[j])
        variance = np.sum(probs * (1 - probs))
        assert mean[idx] == pytest.approx(probs.sum())
        assert var[idx] == pytest.approx(variance)
        assert skew[idx] == pytest.approx(np.sum(probs * (1 - probs) * (1 - 2 * probs)) / variance ** 1.5)


def test_approximations_are_right_tails():
    """Regression: the normal and rna p-values were the left tail P(V <= k)."""
    mean, var, skew = np.full(4, 50.0), np.full(4, 40.0), np.full(4, 0.1)
    weight = np.array([30, 50, 70, 120])
    for pvals in (bpv.poisson_pvals(weight, mean), bpv.normal_pvals(weight, mean, var),
                  bpv.rna_pvals(weight, mean, var, skew)):
        assert np.all(np.diff(pvals) < 0)
        assert pvals[0] > 0.99 and 0.4 < pvals[1] < 0.6 and pvals[3] < 1e-10
    np.testing.assert_allclose(bpv.poisson_pvals(weight, mean), stats.poisson.sf(weight - 1, 50.0))
    np.testing.assert_allclose(bpv.normal_pvals(weight, mean, var),
                               stats.norm.sf((weight - 0.5 - 50) / np.sqrt(40)))
    # The normal tail keeps its precision far from the mean (1 - cdf would return 0)
    assert 0 < bpv.normal_pvals(np.array([250.0]), mean[:1], var[:1])[0] < 1e-200


def test_approximations_are_close_to_exact_in_the_bulk():
    rng = np.random.default_rng(2)
    x_u = rng.lognormal(-2, 0.5, 30)
    n_c = rng.integers(20, 60, 30).astype(float)
    y = rng.lognormal(0, 0.3, 8)
    pairs_i, pairs_j = np.triu_indices(8, k=1)
    mean, var, _ = bpv.moments_grouped(x_u, n_c, y, pairs_i, pairs_j)
    weight = np.round(mean + np.sqrt(var)).astype(int)
    exact = bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method="poibin")
    for method, tolerance in (("rna", 0.02), ("normal", 0.1), ("poisson", 0.3)):
        approx = bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method=method)
        np.testing.assert_allclose(approx, exact, rtol=tolerance)


def test_degenerate_variance():
    np.testing.assert_array_equal(bpv.zscores([3, 2, 1], [2, 2, 2], [0, 0, 0]), [np.inf, 0, -np.inf])
    np.testing.assert_array_equal(bpv.normal_pvals([4, 1], [2, 2], [0, 0]), [0, 1])
    np.testing.assert_array_equal(bpv.rna_pvals([4, 1], [2, 2], [0, 0], [0, 0]), [0, 1])
    assert bpv.poisson_pvals([0, 1], [0.0, 0.0]).tolist() == [1.0, 0.0]


def test_cooccurrence_pvalues_interface(classes):
    x_u, n_c, y, pairs_i, pairs_j = classes
    weight = np.full(len(pairs_i), 2)
    exact = bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method="exact")
    np.testing.assert_array_equal(
        exact, bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method="poibin"))
    for method in bpv.PVALUE_METHODS:
        pvals = bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight * 0, method=method)
        np.testing.assert_array_equal(pvals, 1.0)  # at least zero common neighbours: certain
        pvals = bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method=method)
        assert np.all((pvals >= 0) & (pvals <= 1))
    with pytest.raises(ValueError, match="method"):
        bpv.cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method="wrong")
    assert len(bpv.cooccurrence_pvalues(x_u, n_c, y, [], [], [], method="poibin")) == 0


# PoiBin class

def test_poibin_distribution():
    rng = np.random.default_rng(3)
    probs = rng.random(40)
    dist = PoiBin(probs)
    ks = np.arange(41)
    assert dist.pmf(ks).sum() == pytest.approx(1.0)
    np.testing.assert_allclose(dist.cdf(ks), np.cumsum(dist.pmf(ks)))
    for k in (0, 1, 20, 39, 40):
        assert rel_err(dist.pval(k), exact_tail(probs, k)) < 1e-12
        assert dist.pval(k) == pytest.approx(bpv.poibin_tail(probs, k), rel=1e-12)
    assert dist.pval(0) == 1.0
    assert dist.pval(np.array([0, 40])).shape == (2,)


def test_poibin_small_tails_are_accurate():
    """Regression: the FFT based pmf returned noise below about 1e-15."""
    dist = PoiBin(np.full(200, 0.01))
    for k in (10, 30, 100, 150):
        assert np.log(dist.pval(k)) == pytest.approx(stats.binom.logsf(k - 1, 200, 0.01), abs=1e-9)
    assert dist.pval(200) == 0.0  # 1e-400 is not representable


def test_poibin_input_checks():
    with pytest.raises(ValueError):
        PoiBin([0.2, 1.2])
    with pytest.raises(ValueError):
        PoiBin([[0.2, 0.3]])
    dist = PoiBin([0.2, 0.3])
    with pytest.raises(ValueError):
        dist.pval(3)
    with pytest.raises(ValueError):
        dist.pmf(-1)
    with pytest.raises(TypeError):
        dist.cdf(0.5)
