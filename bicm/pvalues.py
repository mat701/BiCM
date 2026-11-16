"""
Numerical kernels for the p-values of the co-occurrences (V-motifs) of a BiCM.

Under the BiCM the number of common neighbours of two nodes ``i`` and ``j`` of
one layer is a sum of independent Bernoulli variables, one for every node of
the opposite layer, i.e. a Poisson-binomial variable. All the nodes of the
opposite layer with the same degree share the same fitness, so the sum over
``N`` nodes reduces to a sum over ``G`` degree classes of Binomial variables
``Bin(n_g, p_g)``. Every function of this module works on this grouped
representation:

* ``x_u`` are the fitnesses of the degree classes of the opposite layer,
* ``n_c`` are the numbers of nodes in every class,
* ``y`` are the fitnesses of the nodes of the projected layer,
* ``pairs_i``, ``pairs_j`` are the indices (in ``y``) of the tested couples,
* ``weight`` are the observed co-occurrences.

The p-value of a couple is the right tail ``P(V >= weight)``.

Numerical accuracy
------------------
The right tail is never computed as ``1 - cdf``, which cannot return anything
below the machine epsilon (about ``1e-16``) and is dominated by rounding noise
long before that. Instead:

* for small co-occurrences the tail mass is accumulated together with the
  truncated distribution, using only sums of non negative terms;
* for large co-occurrences the distribution is exponentially tilted so that
  its bulk sits on the observed value, and the convolutions are computed with
  real FFTs on the tilted distribution.

Both methods are exact (no approximation of the Poisson-binomial distribution)
and keep a small relative error down to the smallest representable doubles.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import get_num_threads, njit, prange
from scipy.fft import irfft, rfft
from scipy.stats import norm, poisson

#: Largest co-occurrence handled by the direct (schoolbook) convolution.
K_DIRECT = 512
#: Maximum number of couples and of bytes of a batch of FFT convolutions.
FFT_CHUNK_PAIRS = 256
FFT_CHUNK_BYTES = 24 * 1024 * 1024

#: Methods accepted by :func:`cooccurrence_pvalues`.
PVALUE_METHODS = ('poisson', 'poibin', 'normal', 'rna')

# Switch to the Loader anchor before (1 - p)^n loses digits, not only at underflow.
_LOG_LOADER = -30.0
_LOG_UNDER = -745.0
_LN_SQRT_2PI = 0.9189385332046727  # 0.5 * log(2 * pi)
_MAX_TILT = 600.0


# ------------------------------------------------------------------------------
# Link probabilities
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _link_pq(x, y):
    """Return ``(p, 1 - p)`` with ``p = x * y / (1 + x * y)``, without cancellations.

    Infinite fitnesses (nodes linked to the whole opposite layer) give ``p = 1``
    and null fitnesses (isolated nodes) give ``p = 0``.
    """
    xy = x * y
    if xy != xy or xy <= 0.0:  # 0 * inf, or an isolated node
        return 0.0, 1.0
    if xy == np.inf:
        return 1.0, 0.0
    q = 1.0 / (1.0 + xy)
    return xy * q, q


@njit(nogil=True, cache=True)
def _vmotif_pq(x, yi, yj):
    """Probability ``p`` of a V-motif between two nodes through a node of fitness ``x``.

    Returns ``(p, 1 - p)``. The complement is computed as
    ``q_i + q_j - q_i * q_j``, which stays accurate when ``p`` approaches one.
    """
    pi, qi = _link_pq(x, yi)
    pj, qj = _link_pq(x, yj)
    return pi * pj, qi + qj - qi * qj


def link_probabilities(x, y):
    """
    Link probabilities ``x * y / (1 + x * y)`` between a node (or an array of nodes)
    of fitness ``x`` and the nodes of fitness ``y``.

    Unlike the plain formula it handles the infinite fitnesses of the nodes linked
    to the whole opposite layer (probability one) and the null fitnesses of the
    isolated nodes (probability zero) without producing ``nan``.

    :param x: fitness or fitnesses of the first layer.
    :type x: float, numpy.ndarray
    :param y: fitnesses of the second layer.
    :type y: numpy.ndarray
    :returns: the link probabilities, with the broadcast shape of ``x * y``.
    :rtype: numpy.ndarray
    """
    with np.errstate(invalid='ignore', over='ignore'):
        xy = np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64)
        probs = xy / (1.0 + xy)
    probs = np.where(np.isposinf(xy), 1.0, probs)
    return np.where(np.isnan(probs), 0.0, probs)


# ------------------------------------------------------------------------------
# Moments
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _moments_one(x_u, n_c, yi, yj):
    mean = 0.0
    var = 0.0
    third = 0.0
    for g in range(len(x_u)):
        p, q = _vmotif_pq(x_u[g], yi, yj)
        pq = p * q
        mean += n_c[g] * p
        var += n_c[g] * pq
        third += n_c[g] * pq * (q - p)
    skew = third / (var ** 1.5) if var > 0.0 else 0.0
    return mean, var, skew


@njit(parallel=True, nogil=True, cache=True)
def moments_grouped(x_u, n_c, y, pairs_i, pairs_j):
    """
    Mean, variance and skewness of the co-occurrences of every couple of nodes.

    :param numpy.ndarray x_u: fitnesses of the degree classes of the opposite layer.
    :param numpy.ndarray n_c: number of nodes of every degree class.
    :param numpy.ndarray y: fitnesses of the nodes of the projected layer.
    :param numpy.ndarray pairs_i: index of the first node of every couple.
    :param numpy.ndarray pairs_j: index of the second node of every couple.
    :returns: three arrays with the mean, the variance and the skewness.
    """
    n_pairs = len(pairs_i)
    mean = np.zeros(n_pairs, dtype=np.float64)
    var = np.zeros(n_pairs, dtype=np.float64)
    skew = np.zeros(n_pairs, dtype=np.float64)
    for idx in prange(n_pairs):
        mean[idx], var[idx], skew[idx] = _moments_one(x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]])
    return mean, var, skew


# ------------------------------------------------------------------------------
# Truncated Binomial distribution
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _bd0(x, np_):
    """Loader's deviance ``x * log(x / np_) + np_ - x``, stable when ``x`` is close to ``np_``."""
    if x == 0.0:
        return np_
    if np_ == 0.0 or x == np_:
        return 0.0
    ax = x + np_
    if abs(x - np_) < 0.1 * ax:
        d = x - np_
        v = d / ax
        s = d * v
        if s == 0.0:
            return 0.0
        ej = 2.0 * x * v
        vv = v * v
        j = 1
        while j < 1000:
            ej *= vv
            s1 = s + ej / ((j << 1) + 1)
            if s1 == s:
                return s1
            s = s1
            j += 1
        return s
    ratio = x / np_
    if np.isfinite(ratio) and ratio > 0.0:
        lg = np.log(ratio)
    else:
        lg = np.log(x) - np.log(np_)
    if x > np_:
        return x * (lg - 1.0) + np_
    return x * lg + np_ - x


@njit(nogil=True, cache=True)
def _stirlerr(x):
    """``log(x!) - log(sqrt(2 * pi * x) * (x / e)^x)`` for ``x > 0``."""
    if x <= 15.0:
        return math.lgamma(x + 1.0) - (x + 0.5) * np.log(x) + x - _LN_SQRT_2PI
    nn = x * x
    s0 = 0.083333333333333333333  # 1/12
    s1 = 0.00277777777777777777778  # 1/360
    s2 = 0.00079365079365079365079365  # 1/1260
    s3 = 0.000595238095238095238095238  # 1/1680
    s4 = 0.0008417508417508417508417508  # 1/1188
    if x > 500.0:
        return (s0 - s1 / nn) / x
    if x > 80.0:
        return (s0 - (s1 - s2 / nn) / nn) / x
    if x > 35.0:
        return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / x
    return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / x


@njit(nogil=True, cache=True)
def _binom_logpmf(n, t, p, q):
    """Logarithm of the Binomial pmf (Loader's saddle-point algorithm), with ``q = 1 - p``."""
    if t < 0.0 or t > n:
        return -np.inf
    if p <= 0.0:
        return 0.0 if t == 0.0 else -np.inf
    if q <= 0.0:
        return 0.0 if t == n else -np.inf
    if t == 0.0:
        return n * np.log(q) if q < 0.5 else n * np.log1p(-p)
    if t == n:
        return n * np.log(p) if p < 0.5 else n * np.log1p(-q)
    nt = n - t
    return (
        _stirlerr(n) - _stirlerr(t) - _stirlerr(nt)
        - _bd0(t, n * p) - _bd0(nt, n * q)
        + 0.5 * np.log(n / (2.0 * np.pi * t * nt))
    )


@njit(nogil=True, cache=True)
def _recur_up(n, ni, p, q, k, t_start, out):
    ratio = p / q
    t = t_start
    while t < k - 1 and t < ni:
        nxt = out[t] * (n - t) / (t + 1.0) * ratio
        out[t + 1] = nxt
        if nxt == 0.0:
            break
        t += 1


@njit(nogil=True, cache=True)
def _recur_down(n, p, q, t_start, out):
    inv = q / p
    t = t_start
    while t > 0:
        nxt = out[t] * t / (n - t + 1.0) * inv
        out[t - 1] = nxt
        if nxt == 0.0:
            break
        t -= 1


@njit(nogil=True, cache=True)
def _binom_pmf_trunc(n, p, q, k, out):
    """Write in ``out[:k]`` the pmf of ``Bin(n, p)`` on ``0, ..., k - 1`` (``q = 1 - p``).

    Every entry is computed with a small relative error: the recursion starts
    from the mode of the distribution (or from the last stored value), where
    the pmf is evaluated with Loader's algorithm, so that it never starts from
    an underflowed ``(1 - p)^n``.
    """
    for i in range(k):
        out[i] = 0.0
    if k <= 0:
        return
    ni = int(n)
    if ni <= 0 or p <= 0.0:
        out[0] = 1.0
        return
    if q <= 0.0:
        if ni < k:
            out[ni] = 1.0
        return
    nf = float(ni)
    logb = nf * np.log1p(-p) if p < 0.5 else nf * np.log(q)
    if logb >= _LOG_LOADER:
        out[0] = np.exp(logb)
        _recur_up(nf, ni, p, q, k, 0, out)
        return
    mode = int(np.floor((nf + 1.0) * p))
    t0 = min(mode, k - 1, ni)
    log_pmf = _binom_logpmf(nf, float(t0), p, q)
    if log_pmf < _LOG_UNDER:
        return
    out[t0] = np.exp(log_pmf)
    _recur_down(nf, p, q, t0, out)
    _recur_up(nf, ni, p, q, k, t0, out)


@njit(nogil=True, cache=True)
def _binom_sf(n, p, q, k, b):
    """Return ``P(Bin(n, p) >= k)`` given the pmf ``b[:k]`` on ``0, ..., k - 1``.

    Above the mode the tail is the sum of the pmf from ``k`` upwards, below it
    is the complement of the stored values (which is then larger than about one half).
    """
    ni = int(n)
    if k <= 0:
        return 1.0
    if k > ni or p <= 0.0:
        return 0.0
    if q <= 0.0:
        return 1.0
    nf = float(ni)
    if k <= (nf + 1.0) * p:
        left = 0.0
        for t in range(k):
            left += b[t]
        return min(1.0, max(0.0, 1.0 - left))
    log_pmf = _binom_logpmf(nf, float(k), p, q)
    if log_pmf < _LOG_UNDER:
        return 0.0
    term = np.exp(log_pmf)
    total = term
    ratio = p / q
    t = k
    while t < ni:
        step = (nf - t) / (t + 1.0) * ratio
        term *= step
        total += term
        t += 1
        # Past the mode the ratio between consecutive terms decreases:
        # the rest of the series is bounded by a geometric one.
        if step < 1.0 and term * step < 1e-18 * total * (1.0 - step):
            break
    return min(1.0, total)


# ------------------------------------------------------------------------------
# Exact p-values, direct convolution
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _pval_k1(x_u, n_c, yi, yj):
    """``P(V >= 1) = 1 - prod_g (1 - p_g)^(n_g)``, computed with ``expm1``."""
    log_q = 0.0
    for g in range(len(x_u)):
        p, q = _vmotif_pq(x_u[g], yi, yj)
        if q <= 0.0:
            if n_c[g] > 0:
                return 1.0
        elif p > 0.0:
            log_q += n_c[g] * (np.log1p(-p) if p < 0.5 else np.log(q))
    return -np.expm1(log_q)


@njit(nogil=True, cache=True)
def _exact_one_direct(x_u, n_c, yi, yj, k, dp, b, tmp):
    """``P(V >= k)`` for one couple, with the tail mass accumulated during the convolutions.

    ``dp[:k]`` holds the distribution of the partial sum on ``0, ..., k - 1`` and
    ``tail`` its mass on ``k, k + 1, ...``. Adding a class with distribution ``B``
    moves to the tail the mass ``sum_j dp[j] * P(B >= k - j)``: only non negative
    terms are added, so the result keeps its relative accuracy however small it is.
    """
    for i in range(k):
        dp[i] = 0.0
    dp[0] = 1.0
    support = 1  # dp[support:] is null
    tail = 0.0
    for g in range(len(x_u)):
        p, q = _vmotif_pq(x_u[g], yi, yj)
        ni = int(n_c[g])
        if p <= 0.0 or ni <= 0:
            continue
        _binom_pmf_trunc(n_c[g], p, q, k, b)
        surv = _binom_sf(n_c[g], p, q, k, b)  # P(B >= m), starting from m = k
        moved = 0.0
        for m in range(k, 0, -1):
            if k - m < support:
                moved += dp[k - m] * surv
            surv += b[m - 1]
        tail += moved
        b_support = min(k, ni + 1)
        new_support = min(k, support + b_support - 1)
        for j in range(new_support):
            acc = 0.0
            for t in range(max(0, j - support + 1), min(j + 1, b_support)):
                acc += dp[j - t] * b[t]
            tmp[j] = acc
        for j in range(new_support):
            dp[j] = tmp[j]
        support = new_support
    return min(1.0, tail)


@njit(parallel=True, nogil=True, cache=True)
def _exact_pvals_direct(x_u, n_c, y, pairs_i, pairs_j, weight, k_max, n_blocks):
    n_pairs = len(pairs_i)
    out = np.empty(n_pairs, dtype=np.float64)
    # The couples are split in blocks, each with its own work arrays
    for block in prange(n_blocks):
        dp = np.empty(k_max, dtype=np.float64)
        b = np.empty(k_max, dtype=np.float64)
        tmp = np.empty(k_max, dtype=np.float64)
        for idx in range(block, n_pairs, n_blocks):
            k = int(weight[idx])
            if k <= 0:
                out[idx] = 1.0
            elif k == 1:
                out[idx] = _pval_k1(x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]])
            else:
                out[idx] = _exact_one_direct(x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]], k, dp, b, tmp)
    return out


# ------------------------------------------------------------------------------
# Exact p-values, FFT convolution of the tilted distribution
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _tilt_one(x_u, n_c, yi, yj, k):
    """Choose the exponential tilting of one couple.

    Returns ``(theta, log_m, size)``:

    * if the observed value ``k`` is not above the mean, ``theta = 0`` and
      ``size = k``: the p-value is the complement of the mass on ``0, ..., k - 1``,
      which is of order one;
    * otherwise ``theta > 0`` moves the mean of the distribution on ``k`` and
      ``size`` is the number of values needed to cover the tail of the tilted
      distribution; ``log_m`` is the logarithm of its normalisation;
    * ``size = 0`` if ``k`` is larger than the maximum of the variable (null p-value).
    """
    n_groups = len(x_u)
    mean = 0.0
    n_max = 0.0
    for g in range(n_groups):
        p, q = _vmotif_pq(x_u[g], yi, yj)
        if p > 0.0:
            mean += n_c[g] * p
            n_max += n_c[g]
    if k > n_max:
        return 0.0, 0.0, 0
    if k <= mean:
        return 0.0, 0.0, k
    target = min(float(k), n_max - 0.5)
    lo = 0.0
    hi = 1.0
    while hi < _MAX_TILT:
        u = np.exp(hi)
        tilted = 0.0
        for g in range(n_groups):
            p, q = _vmotif_pq(x_u[g], yi, yj)
            if p > 0.0:
                tilted += n_c[g] * (p * u) / (q + p * u)
        if tilted >= target:
            break
        lo = hi
        hi *= 2.0
    hi = min(hi, _MAX_TILT)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        u = np.exp(mid)
        tilted = 0.0
        for g in range(n_groups):
            p, q = _vmotif_pq(x_u[g], yi, yj)
            if p > 0.0:
                tilted += n_c[g] * (p * u) / (q + p * u)
        if tilted >= target:
            hi = mid
        else:
            lo = mid
    theta = 0.5 * (lo + hi)
    u = np.exp(theta)
    um1 = np.expm1(theta)
    log_m = 0.0
    var = 0.0
    for g in range(n_groups):
        p, q = _vmotif_pq(x_u[g], yi, yj)
        if p > 0.0:
            den = q + p * u
            log_m += n_c[g] * np.log1p(p * um1)
            var += n_c[g] * (p * u / den) * (q / den)
    # Values above k needed by the tail sum_m exp(-theta * m) * pmf(k + m)
    extra = min(12.0 * np.sqrt(var) + 30.0, 45.0 / theta + 2.0, n_max - k)
    return theta, log_m, k + int(np.ceil(extra)) + 1


@njit(parallel=True, nogil=True, cache=True)
def _tilt_all(x_u, n_c, y, pairs_i, pairs_j, weight):
    n_pairs = len(weight)
    theta = np.zeros(n_pairs, dtype=np.float64)
    log_m = np.zeros(n_pairs, dtype=np.float64)
    size = np.zeros(n_pairs, dtype=np.int64)
    for idx in prange(n_pairs):
        theta[idx], log_m[idx], size[idx] = _tilt_one(
            x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]], int(weight[idx]))
    return theta, log_m, size


@njit(nogil=True, cache=True)
def _fill_binoms_class(x_g, n_g, yi, yj, theta, size, b):
    """Truncated pmf of the (tilted) Binomial of one degree class, for a batch of couples."""
    for i in range(len(size)):
        p, q = _vmotif_pq(x_g, yi[i], yj[i])
        if theta[i] > 0.0 and p > 0.0 and q > 0.0:
            u = np.exp(theta[i])
            den = q + p * u
            p, q = p * u / den, q / den
        _binom_pmf_trunc(n_g, p, q, size[i], b[i])


@njit(nogil=True, cache=True)
def _truncate_rows(dp, size):
    for i in range(dp.shape[0]):
        for j in range(size[i]):
            if dp[i, j] < 0.0:
                dp[i, j] = 0.0
        for j in range(size[i], dp.shape[1]):
            dp[i, j] = 0.0


@njit(nogil=True, cache=True)
def _tails_from_tilted(dp, weight, theta, log_m, size, out):
    for i in range(len(weight)):
        k = int(weight[i])
        if theta[i] > 0.0:
            # P(V >= k) = M * exp(-theta * k) * sum_m exp(-theta * m) * pmf_tilted(k + m)
            decay = np.exp(-theta[i])
            acc = 0.0
            for j in range(size[i] - 1, k - 1, -1):
                acc = acc * decay + dp[i, j]
            if acc > 0.0:
                out[i] = min(1.0, np.exp(log_m[i] - theta[i] * k + np.log(acc)))
            else:
                out[i] = 0.0
        else:
            left = 0.0
            for j in range(k):
                left += dp[i, j]
            out[i] = min(1.0, max(0.0, 1.0 - left))


def _nfft_for_size(size):
    """Smallest power of two that makes the circular convolution of two arrays of ``size`` linear."""
    return 1 << int(np.ceil(np.log2(max(8, 2 * int(size)))))


def _exact_fft_chunk(x_u, n_c, y, pairs_i, pairs_j, weight, theta, log_m, size):
    """Exact p-values of a batch of couples that share the size of the FFT."""
    n_pairs = len(weight)
    nfft = _nfft_for_size(size.max())
    dp = np.zeros((n_pairs, nfft), dtype=np.float64)
    b = np.zeros((n_pairs, nfft), dtype=np.float64)
    dp[:, 0] = 1.0
    yi = y[pairs_i]
    yj = y[pairs_j]
    for g in range(len(x_u)):
        if n_c[g] <= 0:
            continue
        _fill_binoms_class(x_u[g], n_c[g], yi, yj, theta, size, b)
        spectrum = rfft(dp, nfft, axis=1, workers=1) * rfft(b, nfft, axis=1, workers=1)
        dp = irfft(spectrum, nfft, axis=1, workers=1)
        _truncate_rows(dp, size)
    out = np.empty(n_pairs, dtype=np.float64)
    _tails_from_tilted(dp, weight, theta, log_m, size, out)
    return out


def _fft_index_chunks(size, idx):
    """Split the couples ``idx`` in batches with the same FFT size and a bounded memory."""
    order = np.argsort(size[idx], kind='stable')
    idx = idx[order]
    nffts = np.array([_nfft_for_size(s) for s in size[idx]], dtype=np.int64)
    chunks = []
    start = 0
    while start < len(idx):
        end_run = start + int(np.searchsorted(nffts[start:], nffts[start], side='right'))
        batch = max(1, min(FFT_CHUNK_PAIRS, FFT_CHUNK_BYTES // (16 * int(nffts[start]))))
        for pos in range(start, end_run, batch):
            chunks.append(idx[pos:min(pos + batch, end_run)])
        start = end_run
    return chunks


def _as_kernel_arrays(x_u, n_c, y, pairs_i, pairs_j, weight):
    return (np.ascontiguousarray(x_u, dtype=np.float64),
            np.ascontiguousarray(n_c, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(pairs_i, dtype=np.int64),
            np.ascontiguousarray(pairs_j, dtype=np.int64),
            np.ascontiguousarray(weight, dtype=np.int64))


def exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=K_DIRECT, n_jobs=None):
    """
    Exact Poisson-binomial p-values ``P(V >= weight)`` of the co-occurrences.

    Co-occurrences up to ``k_direct`` use a direct convolution parallelised with numba,
    larger ones use FFT convolutions of the exponentially tilted distribution, run in
    a pool of threads. Both are exact and accurate in relative terms also for
    p-values far below the machine epsilon.

    :param numpy.ndarray x_u: fitnesses of the degree classes of the opposite layer.
    :param numpy.ndarray n_c: number of nodes of every degree class.
    :param numpy.ndarray y: fitnesses of the nodes of the projected layer.
    :param numpy.ndarray pairs_i: index of the first node of every couple.
    :param numpy.ndarray pairs_j: index of the second node of every couple.
    :param numpy.ndarray weight: observed co-occurrences of every couple.
    :param int k_direct: largest co-occurrence computed with the direct convolution.
    :param int n_jobs: number of threads of the FFT convolutions, by default all the cores.
    :returns: the p-values of the couples.
    :rtype: numpy.ndarray
    """
    x_u, n_c, y, pairs_i, pairs_j, weight = _as_kernel_arrays(x_u, n_c, y, pairs_i, pairs_j, weight)
    out = np.empty(len(weight), dtype=np.float64)
    small = weight <= k_direct
    if np.any(small):
        k_max = max(int(weight[small].max()), 1)
        n_blocks = min(int(small.sum()), 8 * get_num_threads())
        out[small] = _exact_pvals_direct(x_u, n_c, y, pairs_i[small], pairs_j[small], weight[small],
                                         k_max, n_blocks)
    large = np.flatnonzero(~small)
    if large.size == 0:
        return out
    theta, log_m, size = _tilt_all(x_u, n_c, y, pairs_i[large], pairs_j[large], weight[large])
    impossible = size == 0
    out[large[impossible]] = 0.0
    local = np.flatnonzero(~impossible)
    chunks = _fft_index_chunks(size, local)

    def run(ix):
        gx = large[ix]
        return _exact_fft_chunk(x_u, n_c, y, pairs_i[gx], pairs_j[gx], weight[gx],
                                theta[ix], log_m[ix], size[ix])

    workers = (os.cpu_count() or 1) if n_jobs is None else max(1, int(n_jobs))
    workers = min(workers, len(chunks))
    if workers <= 1:
        results = [run(ix) for ix in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, chunks))
    for ix, vals in zip(chunks, results):
        out[large[ix]] = vals
    return out


# ------------------------------------------------------------------------------
# Approximations and common interface
# ------------------------------------------------------------------------------

def zscores(weight, mean, var):
    """
    z-scores ``(weight - mean) / sqrt(var)`` of the co-occurrences.

    Where the variance is null the z-score is ``+inf``, ``-inf`` or zero according
    to the sign of ``weight - mean``.
    """
    weight = np.asarray(weight, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    diff = weight - mean
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = diff / np.sqrt(var)
    degenerate = np.where(diff > 0.0, np.inf, np.where(diff < 0.0, -np.inf, 0.0))
    return np.where(var > 0.0, scores, degenerate)


def normal_pvals(weight, mean, var):
    """
    Gaussian approximation of the p-values ``P(V >= weight)``, with continuity correction.

    The right tail is computed with the survival function of the normal
    distribution, which is accurate for any value of the z-score.
    """
    return norm.sf(zscores(np.asarray(weight, dtype=np.float64) - 0.5, mean, var))


def rna_pvals(weight, mean, var, skew):
    """
    Refined normal approximation (RNA) of the p-values ``P(V >= weight)``.

    It is the Gaussian approximation with continuity correction plus the first
    Edgeworth term, proportional to the skewness. The correction can make the
    result slightly negative or larger than one: it is clipped to ``[0, 1]``.
    """
    z = zscores(np.asarray(weight, dtype=np.float64) - 0.5, mean, var)
    pvals = norm.sf(z)
    finite = np.isfinite(z)
    zf = z[finite]
    pvals[finite] += np.asarray(skew, dtype=np.float64)[finite] * (zf ** 2 - 1.0) * norm.pdf(zf) / 6.0
    return np.clip(pvals, 0.0, 1.0)


def poisson_pvals(weight, mean):
    """Poisson approximation of the p-values ``P(V >= weight)``."""
    weight = np.asarray(weight, dtype=np.float64)
    pvals = poisson.sf(weight - 1.0, np.asarray(mean, dtype=np.float64))
    return np.where(weight <= 0.0, 1.0, pvals)


def cooccurrence_pvalues(x_u, n_c, y, pairs_i, pairs_j, weight, method='poisson', n_jobs=None):
    """
    P-values ``P(V >= weight)`` of the co-occurrences of couples of nodes under the BiCM.

    :param numpy.ndarray x_u: fitnesses of the degree classes of the opposite layer.
    :param numpy.ndarray n_c: number of nodes of every degree class.
    :param numpy.ndarray y: fitnesses of the nodes of the projected layer.
    :param numpy.ndarray pairs_i: index of the first node of every couple.
    :param numpy.ndarray pairs_j: index of the second node of every couple.
    :param numpy.ndarray weight: observed co-occurrences of every couple.
    :param str method: *poibin* (or *exact*) for the exact Poisson-binomial distribution,
        *poisson*, *normal* or *rna* for its approximations.
    :param int n_jobs: number of threads of the FFT convolutions of the exact method.
    :returns: the p-values of the couples.
    :rtype: numpy.ndarray
    """
    x_u, n_c, y, pairs_i, pairs_j, weight = _as_kernel_arrays(x_u, n_c, y, pairs_i, pairs_j, weight)
    if method in ('poibin', 'exact'):
        pvals = exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, n_jobs=n_jobs)
        return np.where(weight <= 0, 1.0, pvals)
    if method not in PVALUE_METHODS:
        raise ValueError('method must be one of "poisson", "poibin", "normal" or "rna".')
    mean, var, skew = moments_grouped(x_u, n_c, y, pairs_i, pairs_j)
    if method == 'poisson':
        return poisson_pvals(weight, mean)
    if method == 'normal':
        pvals = normal_pvals(weight, mean, var)
    else:
        pvals = rna_pvals(weight, mean, var, skew)
    return np.where(weight <= 0, 1.0, pvals)


# ------------------------------------------------------------------------------
# Reference implementation
# ------------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def poibin_tail(probs, k):
    """
    Exact right tail ``P(V >= k)`` of the sum ``V`` of independent Bernoulli variables.

    Plain dynamic programming over all the variables, with a cost proportional
    to ``len(probs) * k``. It adds only non negative terms, so it is accurate
    also for extremely small tails. It is the reference used to test the
    grouped algorithms, which are much faster.

    :param numpy.ndarray probs: probabilities of the Bernoulli variables.
    :param int k: threshold of the tail.
    :rtype: float
    """
    if k <= 0:
        return 1.0
    dp = np.zeros(k, dtype=np.float64)
    dp[0] = 1.0
    tail = 0.0
    for c in range(len(probs)):
        p = probs[c]
        q = 1.0 - p
        tail += dp[k - 1] * p
        for v in range(k - 1, 0, -1):
            dp[v] = dp[v] * q + dp[v - 1] * p
        dp[0] *= q
    return min(1.0, tail)
