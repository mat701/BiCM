"""
Multiple hypothesis testing: validation of a set of p-values.

The functions of this module are shared by the projections of
:class:`bicm.graph_classes.BipartiteGraph` and by :mod:`bicm.fast_projection`.
"""
import numpy as np

_ALIASES = {
    'fdr': 'fdr', 'fdr_bh': 'fdr', 'bh': 'fdr',
    'bonferroni': 'bonferroni',
    'global': 'global', 'none': 'global',
}


def _method_name(method):
    try:
        return _ALIASES[method]
    except (KeyError, TypeError):
        raise ValueError('validation method must be "fdr", "bonferroni" or "global".') from None


def _tests_number(pvals, n_tests):
    if n_tests is None:
        return max(pvals.size, 1)
    if n_tests < pvals.size:
        raise ValueError('n_tests cannot be smaller than the number of p-values.')
    return n_tests


def pvalue_threshold(pvals, alpha=0.05, method='fdr', n_tests=None):
    """
    Threshold under which the p-values are statistically validated.

    :param pvals: the p-values of the performed tests.
    :type pvals: numpy.ndarray, list
    :param float alpha: significance level of the validation.
    :param str method: *fdr* for the False Discovery Rate (Benjamini-Hochberg procedure),
        *bonferroni* for the Bonferroni correction, *global* for no correction at all.
    :param int n_tests: total number of tests, if larger than the number of p-values.
        The tests whose p-value is not given are considered as not significant: this is
        the case of the couples of nodes without common neighbours in a projection,
        whose p-value is one.
    :returns: the threshold: the validated p-values are those smaller or equal to it.
        It is zero if no p-value is validated.
    :rtype: float
    """
    method = _method_name(method)
    pvals = np.asarray(pvals, dtype=np.float64).ravel()
    if method == 'global':
        return float(alpha)
    n_tests = _tests_number(pvals, n_tests)
    if method == 'bonferroni':
        return alpha / n_tests
    sorted_pvals = np.sort(pvals)
    thresholds = np.arange(1, sorted_pvals.size + 1) * (alpha / n_tests)
    validated = np.flatnonzero(sorted_pvals <= thresholds)
    if validated.size == 0:
        return 0.0
    return float(thresholds[validated[-1]])


def validate_pvalues(pvals, alpha=0.05, method='fdr', n_tests=None):
    """
    Validate a set of p-values.

    The parameters are those of :func:`pvalue_threshold`.

    :returns: a boolean array, True for the validated p-values, and the threshold used.
    :rtype: tuple
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    threshold = pvalue_threshold(pvals, alpha=alpha, method=method, n_tests=n_tests)
    if threshold == 0.0 and _method_name(method) == 'fdr':
        return np.zeros(pvals.shape, dtype=bool), threshold
    return pvals <= threshold, threshold


def adjust_pvalues(pvals, method='fdr', n_tests=None):
    """
    P-values adjusted for the multiple testing, to be compared directly with the significance level.

    :param pvals: the p-values of the performed tests.
    :type pvals: numpy.ndarray, list
    :param str method: *fdr* (Benjamini-Hochberg q-values), *bonferroni* or *global* (no adjustment).
    :param int n_tests: total number of tests, if larger than the number of p-values.
    :returns: the adjusted p-values, with the shape of the input.
    :rtype: numpy.ndarray
    """
    method = _method_name(method)
    pvals = np.asarray(pvals, dtype=np.float64)
    if method == 'global':
        return pvals.copy()
    flat = pvals.ravel()
    n_tests = _tests_number(flat, n_tests)
    if method == 'bonferroni':
        return np.minimum(pvals * n_tests, 1.0)
    order = np.argsort(flat)
    scaled = flat[order] * (n_tests / np.arange(1, flat.size + 1))
    adjusted = np.empty_like(flat)
    adjusted[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return adjusted.reshape(pvals.shape)
