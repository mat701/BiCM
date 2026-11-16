"""Tests of the multiple hypothesis testing procedures."""
import numpy as np
import pytest

from bicm.validation import adjust_pvalues, pvalue_threshold, validate_pvalues

PVALS = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])


def test_bonferroni():
    assert pvalue_threshold(PVALS, alpha=0.05, method="bonferroni") == pytest.approx(0.005)
    mask, threshold = validate_pvalues(PVALS, alpha=0.05, method="bonferroni")
    assert mask.tolist() == [True] + [False] * 9
    np.testing.assert_allclose(adjust_pvalues(PVALS, method="bonferroni"), np.minimum(PVALS * 10, 1))


def test_fdr_benjamini_hochberg():
    # The largest i with p_(i) <= i * alpha / m is i = 2
    mask, threshold = validate_pvalues(PVALS, alpha=0.05, method="fdr")
    assert threshold == pytest.approx(2 * 0.05 / 10)
    assert mask.tolist() == [True, True] + [False] * 8
    adjusted = adjust_pvalues(PVALS, method="fdr")
    assert ((adjusted <= 0.05) == mask).all()
    assert np.all(np.diff(adjusted) >= 0) and adjusted.max() <= 1
    # A p-value above its own threshold is validated if a larger one passes
    step_up = np.array([0.03, 0.03, 0.03])
    assert validate_pvalues(step_up, alpha=0.05, method="fdr")[0].all()


def test_fdr_is_invariant_to_the_order():
    rng = np.random.default_rng(0)
    order = rng.permutation(len(PVALS))
    mask = validate_pvalues(PVALS[order], alpha=0.05, method="fdr")[0]
    assert mask.tolist() == validate_pvalues(PVALS, alpha=0.05, method="fdr")[0][order].tolist()
    np.testing.assert_allclose(adjust_pvalues(PVALS[order]), adjust_pvalues(PVALS)[order])


def test_global_threshold():
    mask, threshold = validate_pvalues(PVALS, alpha=0.05, method="global")
    assert threshold == 0.05 and mask.sum() == 5
    np.testing.assert_array_equal(adjust_pvalues(PVALS, method="global"), PVALS)


def test_number_of_tests():
    """Tests without a p-value (p-value one) make the validation stricter."""
    assert pvalue_threshold(PVALS, alpha=0.05, method="bonferroni", n_tests=1000) == pytest.approx(5e-5)
    assert validate_pvalues(PVALS, alpha=0.05, method="fdr", n_tests=30)[0].sum() == 1
    explicit = np.concatenate((PVALS, np.ones(20)))
    assert pvalue_threshold(PVALS, method="fdr", n_tests=30) == pvalue_threshold(explicit, method="fdr")
    np.testing.assert_allclose(adjust_pvalues(PVALS, n_tests=30), adjust_pvalues(explicit)[:10])
    with pytest.raises(ValueError, match="n_tests"):
        pvalue_threshold(PVALS, n_tests=5)


def test_nothing_validated_and_empty_input():
    mask, threshold = validate_pvalues(np.array([0.5, 0.9]), alpha=0.05, method="fdr")
    assert threshold == 0.0 and not mask.any()
    mask, threshold = validate_pvalues(np.array([]), alpha=0.05, method="fdr")
    assert mask.shape == (0,)
    assert adjust_pvalues(np.array([])).shape == (0,)


def test_tiny_and_null_pvalues_are_validated():
    pvals = np.array([0.0, 1e-300, 5e-324, 0.5])
    for method in ("fdr", "bonferroni", "global"):
        assert validate_pvalues(pvals, alpha=0.01, method=method, n_tests=10**12)[0].tolist() == \
            [True, True, True, False]


def test_shape_is_preserved_and_aliases():
    matrix = PVALS.reshape(2, 5)
    assert validate_pvalues(matrix)[0].shape == (2, 5)
    assert adjust_pvalues(matrix).shape == (2, 5)
    assert pvalue_threshold(PVALS, method="fdr_bh") == pvalue_threshold(PVALS, method="fdr")
    assert pvalue_threshold(PVALS, method="none") == pvalue_threshold(PVALS, method="global")
    with pytest.raises(ValueError, match="validation method"):
        pvalue_threshold(PVALS, method="wrong")
