"""
Fast and low-memory validated projection of large bipartite networks.

This module computes the BiCM projection of a bipartite network on its columns
layer directly from the list of its edges, without building a
:class:`bicm.graph_classes.BipartiteGraph`. It is meant for networks whose rows
layer is much larger than the number of its distinct degrees:

* the edges are aggregated in the co-occurrences of the couples of columns and
  in the histogram of the degrees of the rows, either in memory
  (:func:`aggregate_bipartite`) or streaming chunks of edges
  (:func:`aggregate_chunks`);
* the BiCM is solved on the distinct degrees only (:func:`fit_bicm_reduced`);
* the p-values of the co-occurrences are computed on the degree classes of the
  rows with the algorithms of :mod:`bicm.pvalues`, by default with the exact
  Poisson-binomial distribution;
* the p-values are validated with a multiple testing procedure
  (:func:`validate_edges`).

:func:`project` and :func:`project_chunks` run the whole pipeline.

Original implementation by Antonio Desiderio (``fast_bicm_exact``).
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import bicm._utils as _bu
import bicm.models_functions as bmof
import bicm.pvalues as bpv
import bicm.solver_functions as bsof
import bicm.validation as bval

SOLVER_TOL = 1e-9
SOLVER_EPS = 1e-8

_METHODS = {'exact': 'poibin', 'poibin': 'poibin', 'poisson': 'poisson',
            'normal': 'normal', 'standard': 'normal', 'rna': 'rna'}


@dataclass
class BipartiteAgg:
    """
    Sufficient statistics of a bipartite network for the projection on its columns.

    Rows and columns without links or linked to the whole opposite layer are removed:
    their links are certain in the BiCM and carry no information.

    :ivar numpy.ndarray deg_g: distinct degrees of the rows, sorted.
    :ivar numpy.ndarray n_c: number of rows with each degree.
    :ivar numpy.ndarray col_deg: degrees of the columns.
    :ivar numpy.ndarray pairs_i: index of the first column of each couple with co-occurrences.
    :ivar numpy.ndarray pairs_j: index of the second column of each couple (``pairs_i < pairs_j``).
    :ivar numpy.ndarray weight: co-occurrences (common neighbours) of each couple.
    :ivar numpy.ndarray col_labels: names of the columns.
    :ivar int n_rows: number of rows.
    :ivar int n_cols: number of columns.
    :ivar int n_edges: number of edges.
    :ivar int n_cols_total: number of columns of the original network, removed ones included.
    """
    n_c: np.ndarray
    deg_g: np.ndarray
    col_deg: np.ndarray
    pairs_i: np.ndarray
    pairs_j: np.ndarray
    weight: np.ndarray
    col_labels: np.ndarray
    n_rows: int
    n_cols: int
    n_edges: int
    n_cols_total: int = 0

    def __post_init__(self):
        if not self.n_cols_total:
            self.n_cols_total = self.n_cols


@dataclass
class ValidatedProjection:
    """
    Validated links of the projection of a bipartite network on its columns.

    All the arrays have one entry for each validated link.

    :ivar numpy.ndarray col_i: name of the first node of the links.
    :ivar numpy.ndarray col_j: name of the second node of the links.
    :ivar numpy.ndarray weight: co-occurrences (common neighbours) of the two nodes.
    :ivar numpy.ndarray z_score: z-scores of the co-occurrences.
    :ivar numpy.ndarray pval: p-values of the co-occurrences.
    :ivar numpy.ndarray pval_corrected: p-values adjusted for the multiple testing.
    :ivar float threshold: threshold applied to the p-values.
    :ivar int n_tests: number of tests considered by the multiple testing procedure.
    """
    col_i: np.ndarray
    col_j: np.ndarray
    weight: np.ndarray
    z_score: np.ndarray
    pval: np.ndarray
    pval_corrected: np.ndarray
    threshold: float = 0.0
    n_tests: int = 0

    def __len__(self):
        return len(self.weight)

    def to_edgelist(self):
        """Return the validated links as a list of couples of names of nodes."""
        return list(zip(self.col_i.tolist(), self.col_j.tolist()))

    def to_dataframe(self):
        """Return the validated links as a ``pandas.DataFrame`` (it requires pandas)."""
        import pandas as pd
        return pd.DataFrame({
            'col_i': self.col_i, 'col_j': self.col_j, 'weight': self.weight,
            'z_score': self.z_score, 'pval_raw': self.pval, 'pval_corrected': self.pval_corrected,
        })


def _empty_projection():
    empty = np.empty(0, dtype=np.float64)
    return ValidatedProjection(col_i=np.empty(0, dtype=object), col_j=np.empty(0, dtype=object),
                               weight=np.empty(0, dtype=np.int64), z_score=empty,
                               pval=empty, pval_corrected=empty)


# ------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------

def _incidence(row_ids, col_ids):
    """Binary sparse biadjacency matrix of a list of edges and the names of its columns."""
    row_arr = np.asarray(row_ids)
    col_arr = np.asarray(col_ids)
    if row_arr.shape != col_arr.shape or row_arr.ndim != 1:
        raise ValueError('row_ids and col_ids must be 1D arrays of the same length')
    for arr in (row_arr, col_arr):
        if arr.dtype.kind == 'f' and np.isnan(arr).any():
            raise ValueError('row_ids and col_ids must not contain missing values')
        if arr.dtype == object and any(v is None or v != v for v in arr):
            raise ValueError('row_ids and col_ids must not contain missing values')
    if row_arr.size == 0:
        return None, np.empty(0, dtype=col_arr.dtype)
    row_codes = np.unique(row_arr, return_inverse=True)[1]
    col_labels, col_codes = np.unique(col_arr, return_inverse=True)
    # Co-occurrences are computed with 64 bits integers: smaller types silently overflow.
    mat = sp.csr_matrix((np.ones(row_arr.size, dtype=np.int64), (row_codes.ravel(), col_codes.ravel())),
                        shape=(int(row_codes.max()) + 1, len(col_labels)))
    mat.data[:] = 1  # repeated edges count once
    return mat, col_labels


def aggregate_bipartite(row_ids, col_ids, min_overlap=1):
    """
    Aggregate the edges of a bipartite network held in memory.

    :param row_ids: the rows nodes of the edges.
    :type row_ids: numpy.ndarray, list
    :param col_ids: the columns nodes of the edges, aligned with ``row_ids``.
    :type col_ids: numpy.ndarray, list
    :param int min_overlap: minimum number of common neighbours of the couples of columns to test.
    :returns: the aggregated network, or None if it has no couple of columns to test.
    :rtype: BipartiteAgg
    """
    mat, col_labels = _incidence(row_ids, col_ids)
    if mat is None:
        return None
    n_cols_total = mat.shape[1]
    while True:  # removing nodes can create new empty or full ones
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            return None
        row_deg = np.asarray(mat.sum(axis=1)).ravel()
        col_deg = np.asarray(mat.sum(axis=0)).ravel()
        valid_r = (row_deg > 0) & (row_deg < mat.shape[1])
        valid_c = (col_deg > 0) & (col_deg < mat.shape[0])
        if valid_r.all() and valid_c.all():
            break
        mat = mat[valid_r][:, valid_c]
        col_labels = col_labels[valid_c]
    deg_g, n_c = np.unique(row_deg, return_counts=True)
    cooc = sp.triu(mat.T @ mat, k=1).tocoo()
    keep = cooc.data >= max(1, int(min_overlap))
    if not keep.any():
        return None
    return BipartiteAgg(
        n_c=n_c.astype(np.float64), deg_g=deg_g.astype(np.int64), col_deg=col_deg.astype(np.int64),
        pairs_i=cooc.row[keep].astype(np.int64), pairs_j=cooc.col[keep].astype(np.int64),
        weight=cooc.data[keep].astype(np.int64), col_labels=col_labels,
        n_rows=int(mat.shape[0]), n_cols=int(mat.shape[1]), n_edges=int(row_deg.sum()),
        n_cols_total=n_cols_total)


def _agg_from_hist_and_cooc(deg_vals, deg_counts, col_deg, labels, cooc, min_overlap):
    """Build the aggregated network from the histogram of the rows degrees and the co-occurrences."""
    deg_vals = np.asarray(deg_vals, dtype=np.int64)
    deg_counts = np.asarray(deg_counts, dtype=np.int64)
    col_deg = np.asarray(col_deg, dtype=np.int64).copy()
    cooc = np.asarray(cooc, dtype=np.int64)
    labels = np.asarray(labels)
    n_cols_total = col_deg.size
    while True:  # removing nodes can create new empty or full ones
        n_rows = int(deg_counts.sum())
        n_cols = col_deg.size
        if n_rows == 0 or n_cols == 0:
            return None
        full_r = deg_vals == n_cols
        full_c = col_deg == n_rows
        valid_r = (deg_vals > 0) & ~full_r
        valid_c = (col_deg > 0) & ~full_c
        if valid_r.all() and valid_c.all():
            break
        # A full row is a common neighbour of every couple of columns.
        n_full_r = int(deg_counts[full_r].sum())
        deg_vals = deg_vals[valid_r] - int(full_c.sum())
        deg_counts = deg_counts[valid_r]
        col_deg = col_deg[valid_c] - n_full_r
        cooc = cooc[np.ix_(valid_c, valid_c)] - n_full_r
        labels = labels[valid_c]
    hist = Counter()
    for deg, count in zip(deg_vals.tolist(), deg_counts.tolist()):
        hist[deg] += count
    deg_g = np.array(sorted(hist), dtype=np.int64)
    n_c = np.array([hist[d] for d in deg_g.tolist()], dtype=np.float64)
    pairs_i, pairs_j = np.triu_indices(n_cols, k=1)
    weight = cooc[pairs_i, pairs_j]
    keep = weight >= max(1, int(min_overlap))
    if not keep.any():
        return None
    return BipartiteAgg(
        n_c=n_c, deg_g=deg_g, col_deg=col_deg,
        pairs_i=pairs_i[keep].astype(np.int64), pairs_j=pairs_j[keep].astype(np.int64),
        weight=weight[keep].astype(np.int64), col_labels=labels,
        n_rows=n_rows, n_cols=n_cols, n_edges=int(np.dot(deg_g, n_c)), n_cols_total=n_cols_total)


def aggregate_chunks(chunks, min_overlap=1):
    """
    Aggregate the edges of a bipartite network given in chunks.

    The edges of a row must all be in the same chunk, while a column can appear in
    any number of chunks. The memory used is that of a dense matrix of the
    co-occurrences of the columns plus the histogram of the degrees of the rows,
    whatever the number of rows.

    :param chunks: iterable of couples ``(row_ids, col_ids)`` of aligned arrays of edges.
    :param int min_overlap: minimum number of common neighbours of the couples of columns to test.
    :returns: the aggregated network, or None if it has no couple of columns to test.
    :rtype: BipartiteAgg
    """
    index = {}
    labels = []
    capacity = 64
    cooc = np.zeros((capacity, capacity), dtype=np.int64)
    col_deg = np.zeros(capacity, dtype=np.int64)
    row_hist = Counter()
    for row_ids, col_ids in chunks:
        mat, chunk_labels = _incidence(row_ids, col_ids)
        if mat is None:
            continue
        for label in chunk_labels.tolist():
            if label not in index:
                index[label] = len(labels)
                labels.append(label)
        if len(labels) > capacity:
            while capacity < len(labels):
                capacity *= 2
            pad = capacity - cooc.shape[0]
            cooc = np.pad(cooc, ((0, pad), (0, pad)))
            col_deg = np.pad(col_deg, (0, pad))
        global_idx = np.array([index[label] for label in chunk_labels.tolist()], dtype=np.int64)
        chunk_cooc = (mat.T @ mat).tocoo()
        np.add.at(cooc, (global_idx[chunk_cooc.row], global_idx[chunk_cooc.col]), chunk_cooc.data)
        col_deg[global_idx] += np.asarray(mat.sum(axis=0)).ravel()
        degs, counts = np.unique(np.asarray(mat.sum(axis=1)).ravel(), return_counts=True)
        for deg, count in zip(degs.tolist(), counts.tolist()):
            row_hist[deg] += count
    if not labels:
        return None
    n_cols = len(labels)
    return _agg_from_hist_and_cooc(list(row_hist.keys()), list(row_hist.values()), col_deg[:n_cols],
                                   np.array(labels), cooc[:n_cols, :n_cols], min_overlap)


# ------------------------------------------------------------------------------
# BiCM on the degree classes
# ------------------------------------------------------------------------------

def fit_bicm_reduced(deg_g, n_c, col_deg, tol=SOLVER_TOL):
    """
    Solve the BiCM from the distinct degrees of the rows and the degrees of the columns.

    :param numpy.ndarray deg_g: distinct degrees of the rows.
    :param numpy.ndarray n_c: number of rows with each degree.
    :param numpy.ndarray col_deg: degrees of the columns.
    :param float tol: tolerance of the solver.
    :returns: the fitnesses of the degree classes of the rows and the fitnesses of the columns.
    :rtype: tuple
    """
    deg_g = np.asarray(deg_g, dtype=np.float64)
    n_c = np.asarray(n_c, dtype=np.float64)
    col_deg = np.asarray(col_deg, dtype=np.float64)
    r_cols, invert_cols, cols_mult = np.unique(col_deg, return_inverse=True, return_counts=True)
    cols_mult = cols_mult.astype(np.float64)
    n_edges = float(np.dot(deg_g, n_c))
    if n_edges <= 0.0:
        raise ValueError('The network has no edges.')
    scale = np.sqrt(n_edges)
    x0 = np.concatenate((-np.log(np.clip(deg_g / scale, 1e-12, 1e12)),
                         -np.log(np.clip(r_cols / scale, 1e-12, 1e12))))
    args = (deg_g, r_cols, n_c, cols_mult)
    lins_args = (bmof.loglikelihood_bicm, args)
    sol = bsof.solver(
        x0,
        fun=lambda z: -bmof.loglikelihood_prime_bicm(z, args),
        fun_jac=lambda z: -bmof.loglikelihood_hessian_bicm(z, args),
        step_fun=lambda z: -bmof.loglikelihood_bicm(z, args),
        linsearch_fun=lambda z: bmof.linsearch_fun_BiCM(z, lins_args),
        hessian_regulariser=bsof.matrix_regulariser_function,
        tol=tol, eps=SOLVER_EPS, max_steps=100, method='newton',
        verbose=False, regularise=True, linsearch=True,
    )
    x_u = np.exp(-sol[:len(deg_g)])
    y_u = np.exp(-sol[len(deg_g):])
    probs = bpv.link_probabilities(x_u[:, None], y_u[None, :])
    error = max(np.max(np.abs(probs @ cols_mult / deg_g - 1.0)), np.max(np.abs(n_c @ probs / r_cols - 1.0)))
    if error > 1e-6:
        _bu.warn('The BiCM solver did not converge: maximum relative error on the degrees {:.2e}'.format(error))
    return x_u, y_u[invert_cols.ravel()]


# ------------------------------------------------------------------------------
# Validation and pipeline
# ------------------------------------------------------------------------------

def validate_edges(observed_weights, mu, var, skew, exact_pvals=None, method='exact',
                   correction_type='fdr', alpha=0.05, n_tests=None):
    """
    Validate the co-occurrences of couples of nodes given their moments or exact p-values.

    :param numpy.ndarray observed_weights: observed co-occurrences of the couples.
    :param numpy.ndarray mu: expected co-occurrences.
    :param numpy.ndarray var: variances of the co-occurrences.
    :param numpy.ndarray skew: skewnesses of the co-occurrences.
    :param numpy.ndarray exact_pvals: exact p-values, required by the method *exact*.
    :param str method: *exact* (or *poibin*), *poisson*, *normal* (or *standard*) or *rna*.
    :param str correction_type: *fdr*, *bonferroni* or *global*,
        see :func:`bicm.validation.pvalue_threshold`.
    :param float alpha: significance level of the validation.
    :param int n_tests: total number of tests, by default the number of couples given.
    :returns: a dictionary with the keys *reject_mask* (True for the validated couples),
        *corrected_pvals*, *raw_target_pvals*, *z_scores* and *threshold*.
    :rtype: dict
    """
    if method not in _METHODS:
        raise ValueError('unknown method {!r}'.format(method))
    method = _METHODS[method]
    observed_weights = np.asarray(observed_weights, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if method == 'poibin':
        if exact_pvals is None:
            raise ValueError("exact_pvals is required when method='exact'")
        pvals = np.asarray(exact_pvals, dtype=np.float64)
    elif method == 'poisson':
        pvals = bpv.poisson_pvals(observed_weights, mu)
    elif method == 'normal':
        pvals = bpv.normal_pvals(observed_weights, mu, var)
    else:
        pvals = bpv.rna_pvals(observed_weights, mu, var, skew)
    reject, threshold = bval.validate_pvalues(pvals, alpha=alpha, method=correction_type, n_tests=n_tests)
    return {
        'reject_mask': reject,
        'corrected_pvals': bval.adjust_pvalues(pvals, method=correction_type, n_tests=n_tests),
        'raw_target_pvals': pvals,
        'z_scores': bpv.zscores(observed_weights, mu, var),
        'threshold': threshold,
    }


def _project_agg(agg, method, correction, alpha, n_jobs):
    if method not in _METHODS:
        raise ValueError('unknown method {!r}'.format(method))
    if agg is None:
        return _empty_projection()
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    mu, var, skew = bpv.moments_grouped(x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j)
    exact_pvals = None
    if _METHODS[method] == 'poibin':
        exact_pvals = bpv.exact_pvals_grouped(x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j, agg.weight,
                                              n_jobs=n_jobs)
    # Every couple of columns is a test: those without common neighbours have p-value one.
    n_tests = max(agg.n_cols_total * (agg.n_cols_total - 1) // 2, len(agg.weight))
    val = validate_edges(agg.weight, mu, var, skew, exact_pvals=exact_pvals, method=method,
                         correction_type=correction, alpha=alpha, n_tests=n_tests)
    keep = val['reject_mask']
    return ValidatedProjection(
        col_i=agg.col_labels[agg.pairs_i[keep]], col_j=agg.col_labels[agg.pairs_j[keep]],
        weight=agg.weight[keep], z_score=val['z_scores'][keep],
        pval=val['raw_target_pvals'][keep], pval_corrected=val['corrected_pvals'][keep],
        threshold=val['threshold'], n_tests=n_tests)


def project(row_ids, col_ids, method='exact', correction='fdr', alpha=0.05, min_overlap=1, n_jobs=None):
    """
    Validated projection on the columns of a bipartite network given as a list of edges.

    To project on the rows, swap the two arguments. For networks that do not fit
    in memory use :func:`project_chunks`.

    :param row_ids: the rows nodes of the edges.
    :type row_ids: numpy.ndarray, list
    :param col_ids: the columns nodes of the edges, aligned with ``row_ids``.
    :type col_ids: numpy.ndarray, list
    :param str method: *exact* (or *poibin*) for the exact Poisson-binomial p-values,
        *poisson*, *normal* (or *standard*) or *rna* for its approximations.
    :param str correction: multiple testing procedure: *fdr*, *bonferroni* or *global*.
        The number of tests is the number of couples of columns of the network.
    :param float alpha: significance level of the validation.
    :param int min_overlap: minimum number of common neighbours of the couples to test.
    :param int n_jobs: number of threads of the FFT convolutions of the exact method.
    :returns: the validated links with their statistics.
    :rtype: ValidatedProjection
    """
    return _project_agg(aggregate_bipartite(row_ids, col_ids, min_overlap=min_overlap),
                        method, correction, alpha, n_jobs)


def project_chunks(chunks, method='exact', correction='fdr', alpha=0.05, min_overlap=1, n_jobs=None):
    """
    Same as :func:`project`, for a network given as an iterable of chunks of edges.

    Each chunk is a couple ``(row_ids, col_ids)`` of aligned arrays. The edges of
    a row must all be in the same chunk, see :func:`aggregate_chunks`.

    :rtype: ValidatedProjection
    """
    return _project_agg(aggregate_chunks(chunks, min_overlap=min_overlap),
                        method, correction, alpha, n_jobs)
