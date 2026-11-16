"""
Poisson-binomial distribution: the distribution of the sum of independent
Bernoulli variables with different success probabilities.

Usage:
    Be ``p`` a list or numpy array of success probabilities of ``n`` independent
    Bernoulli random variables. Create an instance of the distribution with

    >>> pb = PoiBin(p)

    Be ``x`` an integer, or a list or numpy array of integers, between 0 and ``n``.

    >>> pb.pmf(x)   # probability mass function, P(X = x)
    >>> pb.cdf(x)   # cumulative distribution function, P(X <= x)
    >>> pb.pval(x)  # p-value, P(X >= x)

The distribution is computed with an exact convolution that only adds non
negative terms: every value of the probability mass function, and the right
tail used for the p-values, are accurate in relative terms however small they
are. Previous versions used the discrete Fourier transform of the
characteristic function [Hong2013]_, whose absolute error of about ``1e-16``
made the p-values smaller than about ``1e-14`` unreliable, and returned the
p-values as ``1 - cdf``.

The cost in time is proportional to ``n ** 2``. The projections of
:class:`bicm.graph_classes.BipartiteGraph` do not use this class but the
faster algorithms of :mod:`bicm.pvalues`.

.. [Hong2013] Yili Hong, On computing the distribution function for the Poisson
    binomial distribution, Computational Statistics & Data Analysis 59 (2013) 41-51.

Original module by Mika Straka, 2016.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _poibin_pmf(probs):
    n = len(probs)
    pmf = np.zeros(n + 1, dtype=np.float64)
    pmf[0] = 1.0
    for c in range(n):
        p = probs[c]
        q = 1.0 - p
        for v in range(c + 1, 0, -1):
            pmf[v] = pmf[v] * q + pmf[v - 1] * p
        pmf[0] *= q
    return pmf


class PoiBin:
    """Poisson-binomial distribution of the sum of Bernoulli variables of probabilities ``p``.

    :param p: success probabilities of the Bernoulli variables.
    :type p: list, numpy.ndarray
    """

    def __init__(self, p):
        self.p = np.array(p, dtype=np.float64)
        self.n = self.p.size
        self.check_input_prob()
        self.pmf_list = _poibin_pmf(self.p)
        self.cdf_list = np.minimum(np.cumsum(self.pmf_list), 1.0)
        # Right tail P(X >= k), summed from the smallest terms: no cancellations.
        self.sf_list = np.minimum(np.cumsum(self.pmf_list[::-1])[::-1], 1.0)
        self.cdf_list[-1] = 1.0
        self.sf_list[0] = 1.0

    def pmf(self, kk):
        """Probability mass function ``P(X = kk)`` for the input values kk."""
        self.check_rv_input(kk)
        return self.pmf_list[kk]

    def cdf(self, kk):
        """Cumulative distribution function ``P(X <= kk)`` for the input values kk."""
        self.check_rv_input(kk)
        return self.cdf_list[kk]

    def pval(self, kk):
        """P-value ``P(X >= kk)`` for the input values kk."""
        self.check_rv_input(kk)
        return self.sf_list[kk]

    def check_rv_input(self, kk):
        """Check that the input values kk are integers between 0 and n."""
        values = np.asarray(kk)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError('Input values must be integers.')
        if np.any(values < 0):
            raise ValueError('Input values must be non negative.')
        if np.any(values > self.n):
            raise ValueError('Input values cannot be greater than the number of probabilities, '
                             + str(self.n) + '.')
        return True

    def check_input_prob(self):
        """Check that the input probabilities are a one-dimensional array of values in [0, 1]."""
        if self.p.shape != (self.n,):
            raise ValueError('Input must be a one-dimensional array or a list.')
        if not np.all(self.p >= 0):
            raise ValueError('Input probabilities have to be non negative.')
        if not np.all(self.p <= 1):
            raise ValueError('Input probabilities have to be smaller than or equal to 1.')
