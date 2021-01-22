"""
This module contains several functions for bipartite networks.
"""

import numpy as np
from scipy import sparse
from numba import jit
from .PvaluesHandler import PvaluesHandler


def sample_bicm(avg_mat):
    """
    Build a biadjacency matrix sampling from the probability matrix of a BiCM.
    """
    if not isinstance(avg_mat, np.ndarray):
        avg_mat = np.array(avg_mat)
    dim1, dim2 = avg_mat.shape
    return np.array(avg_mat > np.reshape(np.random.sample(dim1 * dim2), (dim1, dim2)), dtype=int)


def sample_bicm_edgelist(x, y):
    """
    Build an edgelist sampling from the fitnesses of a BiCM.
    """
    edgelist = []
    for i in range(len(x)):
        for j in range(len(y)):
            xy = x[i] * y[j]
            if np.random.uniform() < xy / (1 + xy):
                edgelist.append((i, j))
    return edgelist


@jit(nopython=True)
def edgelist_from_biadjacency_fast(biadjacency):
    """
    Build the edgelist of a bipartite network from its biadjacency matrix.
    """
    edgelist = []
    for i in range(biadjacency.shape[0]):
        for j in range(biadjacency.shape[1]):
            if biadjacency[i, j] == 1:
                edgelist.append((i, j))
    return edgelist


def edgelist_from_biadjacency(biadjacency):
    """
    Build the edgelist of a bipartite network from its biadjacency matrix.
    Accounts for sparse matrices and returns a structured array.
    """
    if sparse.isspmatrix(biadjacency):
        coords = biadjacency.nonzero()
        if np.sum(biadjacency.data != 1) > 0:
            raise ValueError('Only binary matrices')
        return np.array(list(zip(coords[0], coords[1])), dtype=np.dtype([('rows', int), ('columns', int)])),\
               np.array(biadjacency.sum(1)).flatten(), np.array(biadjacency.sum(0)).flatten()
    else:
        if np.sum(biadjacency[biadjacency != 0] != 1) > 0:
            raise ValueError('Only binary matrices')
        return np.array(edgelist_from_biadjacency_fast(biadjacency),
                        dtype=np.dtype([('rows', int), ('columns', int)])),\
               np.sum(biadjacency, axis=1), np.sum(biadjacency, axis=0)


def biadjacency_from_edgelist(edgelist, fmt='array'):
    """
    Build the biadjacency matrix of a bipartite network from its edgelist.
    Returns a matrix of the type specified by ``fmt``, by default a numpy array.
    """
    edgelist, rows_deg, cols_deg, rows_dict, cols_dict = edgelist_from_edgelist(edgelist)
    if fmt == 'array':
        biadjacency = np.zeros((len(rows_deg), len(cols_deg)), dtype=int)
        for edge in edgelist:
            biadjacency[edge[0], edge[1]] = 1
    elif fmt == 'sparse':
        try:
            from scipy.sparse import coo_matrix
        except ImportError:
            raise ImportError('To use sparse matrices I need scipy.sparse')
        biadjacency = coo_matrix((np.ones(len(edgelist)), (edgelist['rows'], edgelist['columns'])))
    elif not isinstance(format, str):
        raise TypeError('format must be a string (either "array" or "sparse")')
    else:
        raise ValueError('format must be either "array" or "sparse"')
    return biadjacency, rows_deg, cols_deg, rows_dict, cols_dict


def edgelist_from_edgelist(edgelist):
    """
    Creates a new edgelist with the indexes of the nodes instead of the names.
    Returns also two dictionaries that keep track of the nodes.
    """
    edgelist = np.array(list(set([tuple(edge) for edge in edgelist])))
    out = np.zeros(np.shape(edgelist)[0], dtype=np.dtype([('source', object), ('target', object)]))
    out['source'] = edgelist[:, 0]
    out['target'] = edgelist[:, 1]
    edgelist = out
    unique_rows, rows_degs = np.unique(edgelist['source'], return_counts=True)
    unique_cols, cols_degs = np.unique(edgelist['target'], return_counts=True)
    rows_dict = dict(enumerate(unique_rows))
    cols_dict = dict(enumerate(unique_cols))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    edgelist_new = [(inv_rows_dict[edge[0]], inv_cols_dict[edge[1]]) for edge in edgelist]
    edgelist_new = np.array(edgelist_new, dtype=np.dtype([('rows', int), ('columns', int)]))
    return edgelist_new, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_edgelist(edgelist, convert_type=True):
    """
    Creates the adjacency list from the edgelist.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    If convert_type is True (default), then the nodes are enumerated and the adjacency list is returned as integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    if convert_type:
        edgelist, rows_degs, cols_degs, rows_dict, cols_dict = edgelist_from_edgelist(edgelist)
    adj_list = {}
    inv_adj_list = {}
    for edge in edgelist:
        adj_list.setdefault(edge[0], set()).add(edge[1])
        inv_adj_list.setdefault(edge[1], set()).add(edge[0])
    if not convert_type:
        rows_degs = np.array([len(adj_list[k]) for k in adj_list])
        rows_dict = {k: k for k in adj_list}
        cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
        cols_dict = {k: k for k in inv_adj_list}
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_adjacency_list(old_adj_list):
    """
    Creates the adjacency list from another adjacency list, convering the data type.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    Original keys are treated as rows, values as columns.
    The nodes are enumerated and the adjacency list is returned as integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    rows_dict = dict(enumerate(np.unique(list(old_adj_list.keys()))))
    cols_dict = dict(enumerate(np.unique([el for l in old_adj_list.values() for el in l])))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    adj_list = {}
    inv_adj_list = {}
    for k in old_adj_list:
        adj_list.setdefault(inv_rows_dict[k], set()).update({inv_cols_dict[val] for val in old_adj_list[k]})
        for val in old_adj_list[k]:
            inv_adj_list.setdefault(inv_cols_dict[val], set()).add(inv_rows_dict[k])
    rows_degs = np.array([len(adj_list[k]) for k in adj_list])
    cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_biadjacency(biadjacency):
    """
    Creates the adjacency list from a biadjacency matrix, given in sparse format or as a list or numpy array.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    Returns also the two degree sequences.
    """
    if sparse.isspmatrix(biadjacency):
        if np.sum(biadjacency.data != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = biadjacency.nonzero()
    else:
        biadjacency = np.array(biadjacency)
        if np.sum(biadjacency[biadjacency != 0] != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = np.where(biadjacency != 0)
    adj_list = {}
    inv_adj_list = {}
    for edge_i in range(len(coords[0])):
        adj_list.setdefault(coords[0][edge_i], set()).add(coords[1][edge_i])
        inv_adj_list.setdefault(coords[1][edge_i], set()).add(coords[0][edge_i])
    rows_degs = np.array([len(adj_list[k]) for k in adj_list])
    cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
    return adj_list, inv_adj_list, rows_degs, cols_degs


def check_sol(biad_mat, avg_bicm, return_error=False, in_place=False):
    """
    This function prints the rows sums differences between two matrices, that originally are the biadjacency matrix and its bicm average matrix.
    The intended use of this is to check if an average matrix is actually a solution for a bipartite configuration model.

    If return_error is set to True, it returns 1 if the sum of the differences is bigger than 1.

    If in_place is set to True, it checks and sums also the total error entry by entry.
    The intended use of this is to check if two solutions are the same solution.
    """
    error = 0
    if np.any(avg_bicm < 0):
        print('Negative probabilities in the average matrix! This means negative node fitnesses.')
        error = 1
    if np.any(avg_bicm > 1):
        print('Probabilities greater than 1 in the average matrix! This means negative node fitnesses.')
        error = 1
    rows_error_vec = np.abs(np.sum(biad_mat, axis=1) - np.sum(avg_bicm, axis=1))
    err_rows = np.max(rows_error_vec)
    print('max rows error =', err_rows)
    cols_error_vec = np.abs(np.sum(biad_mat, axis=0) - np.sum(avg_bicm, axis=0))
    err_cols = np.max(cols_error_vec)
    print('max columns error =', err_cols)
    tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
    print('total error =', tot_err)
    if tot_err > 1:
        error = 1
        print('WARNING total error > 1')
        if tot_err > 10:
            print('total error > 10')
    if err_rows + err_cols > 1:
        print('max error > 1')
        error = 1
        if err_rows + err_cols > 10:
            print('max error > 10')
    if in_place:
        diff_mat = np.abs(biad_mat - avg_bicm)
        print('In-place total error:', np.sum(diff_mat))
        print('In-place max error:', np.max(diff_mat))
    if return_error:
        return error
    else:
        return


def check_sol_light(x, y, rows_deg, cols_deg, return_error=False):
    """
    Light version of the check_sol function, working only on the fitnesses and the degree sequences.
    """
    error = 0
    rows_error_vec = []
    for i in range(len(x)):
        row_avgs = x[i] * y / (1 + x[i] * y)
        if error == 0:
            if np.any(row_avgs < 0):
                print('Warning: negative link probabilities')
                error = 1
            if np.any(row_avgs > 1):
                print('Warning: link probabilities > 1')
                error = 1
        rows_error_vec.append(np.sum(row_avgs) - rows_deg[i])
    rows_error_vec = np.abs(rows_error_vec)
    err_rows = np.max(rows_error_vec)
    print('max rows error =', err_rows)
    cols_error_vec = np.abs([(x * y[j] / (1 + x * y[j])).sum() - cols_deg[j] for j in range(len(y))])
    err_cols = np.max(cols_error_vec)
    print('max columns error =', err_cols)
    tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
    print('total error =', tot_err)
    if tot_err > 1:
        error = 1
        print('WARNING total error > 1')
        if tot_err > 10:
            print('total error > 10')
    if err_rows + err_cols > 1:
        print('max error > 1')
        error = 1
        if err_rows + err_cols > 10:
            print('max error > 10')
    if return_error:
        return error
    else:
        return


@jit(nopython=True)
def vmotifs_from_edgelist(edgelist, rows_num, rows_deg):
    """
    From the edgelist returns an edgelist of the rows, weighted by the couples' v-motifs number.
    """
    edgelist = edgelist[np.argsort(edgelist['rows'])]
    cols_edli = edgelist['columns']
    v_list = []
    for i in range(rows_num - 1):
        start_i = rows_deg[:i].sum()
        i_neighbors = cols_edli[start_i: start_i + rows_deg[i]]
        start_j = start_i
        for j in range(i + 1, rows_num):
            start_j += rows_deg[j - 1]
            j_neighbors = cols_edli[start_j: start_j + rows_deg[j]]
            v_ij = len(set(i_neighbors) & set(j_neighbors))
            if v_ij > 0:
                v_list.append((i, j, v_ij))
    return v_list


def vmotifs_from_adjacency_list(adj_list):
    """
    From the adjacency list returns an edgelist of the keys with the couples of nodes that share at least a common neighbor (v-motif),
    weighted by the couples' v-motifs number.
    adj_list values must be sets.
    """
    nodelist = list(adj_list.keys())
    nodes_num = len(nodelist)
    v_list = []
    for node_i in range(nodes_num - 1):
        first_node = nodelist[node_i]
        for node_j in range(node_i + 1, nodes_num):
            second_node = nodelist[node_j]
            v_num = len(adj_list[first_node] & adj_list[second_node])
            if v_num > 0:
                v_list.append((first_node, second_node, v_num))
    return v_list


def pvals_validator(pvals, rows_num, alpha=0.05):
    """
    Validate p-values given a threshold alpha.
    """
    sorted_pvals = np.sort(pvals)
    multiplier = 2 * alpha / (rows_num * (rows_num - 1))
    try:
        eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * multiplier))[0][-1]
    except IndexError:
        print('No V-motifs will be validated. Try increasing alpha')
        eff_fdr_pos = 0
    eff_fdr_th = eff_fdr_pos * multiplier
    return eff_fdr_th


def projection_calculator(biad_mat, avg_mat, alpha=0.05, rows=True, sparse_mode=True,
                          method='poisson', threads_num=4, return_pvals=False, progress_bar=True):
    """
    Calculates the projection on the rows layer (columns layers if rows is set to False).
    Returns an edge list of the indices of the vertices that share a link in the projection.

    alpha is the parameter of the FDR validation.
    method can be set to 'poibin', 'poisson', 'normal' and 'rna' according to the desired poisson binomial approximation to use.
    threads_num is the number of threads to launch when calculating the p-values.
    """
    if not rows:
        biad_mat = biad_mat.T
        avg_mat = avg_mat.T
    rows_num = biad_mat.shape[0]
    if sparse_mode:
        v_mat = (sparse.csr_matrix(biad_mat) * sparse.csr_matrix(biad_mat.T)).toarray()
    else:
        v_mat = np.dot(biad_mat, biad_mat.T)
    np.fill_diagonal(v_mat, 0)
    pval_obj = PvaluesHandler()
    pval_obj.set_avg_mat(avg_mat)
    pval_obj.compute_pvals(v_mat, method=method, threads_num=threads_num, progress_bar=progress_bar)
    pval_list = pval_obj.pval_list
    if return_pvals:
        return pval_list
    eff_fdr_th = pvals_validator([v[2] for v in pval_list], rows_num, alpha=alpha)
    return np.array([(v[0], v[1]) for v in pval_list if v[2] <= eff_fdr_th])


def projection_calculator_light(edgelist, x, y, alpha=0.05, rows=True, method='poisson',
                                threads_num=4, return_pvals=False, progress_bar=True):
    """
    Calculate the projection given only the edge list of the network, the fitnesses of the rows layer and the fitnesses of the columns layer.
    By default, the projection is calculated using a Poisson approximation. Other implemented choices are 'poibin' for the original Poisson-binomial
    distribution, 'normal' for the normal approximation and 'rna' for the refined normal approximation.
    """
    if not rows:
        edgelist = np.array([(edge[1], edge[0]) for edge in edgelist], dtype=np.dtype([('rows', object), ('columns', object)]))
    edgelist, order = np.unique(edgelist, axis=0, return_index=True)
    edgelist = edgelist[np.argsort(order)]  # np.unique does not preserve the order
    edgelist, rows_degs, cols_degs, rows_dict, cols_dict = edgelist_from_edgelist(edgelist)
    rows_num = len(rows_degs)
    v_list = vmotifs_from_edgelist(edgelist, rows_num, rows_degs)
    pval_obj = PvaluesHandler()
    if rows:
        pval_obj.set_fitnesses(x, y)
    else:
        pval_obj.set_fitnesses(y, x)
    pval_obj.compute_pvals(v_list, method=method, threads_num=threads_num, progress_bar=progress_bar)
    pval_list = pval_obj.pval_list
    if return_pvals:
        return [(rows_dict[pval[0]], rows_dict[pval[1]], pval[2]) for pval in pval_list]
    eff_fdr_th = pvals_validator([v[2] for v in pval_list], rows_num, alpha=alpha)
    return np.array([(rows_dict[v[0]], rows_dict[v[1]]) for v in pval_list if v[2] <= eff_fdr_th])


def projection_calculator_light_from_adjacency(adj_list, x, y, alpha=0.05, method='poisson', threads_num=4, return_pvals=False, progress_bar=True, change_dtype=False):
    """
    Calculate the projection given only the adjacency list of the network and its inverse, the fitnesses of the rows layer and the fitnesses of the columns layer.
    The projection is calculated on the layer of the nodes given as keys.
    By default, the projection is calculated using a Poisson approximation. Other implemented choices are 'poibin' for the original Poisson-binomial
    distribution, 'normal' for the normal approximation and 'rna' for the refined normal approximation.
    """
    adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict = adjacency_list_from_adjacency_list(adj_list)
    rows_num = len(rows_degs)
    v_list = vmotifs_from_adjacency_list(adj_list)
    pval_obj = PvaluesHandler()
    pval_obj.set_fitnesses(x, y)
    pval_obj.compute_pvals(v_list, method=method, threads_num=threads_num, progress_bar=progress_bar)
    pval_list = pval_obj.pval_list
    if return_pvals:
        if change_dtype:
            return [(rows_dict[pval[0]], rows_dict[pval[1]], pval[2]) for pval in pval_list]
        else:
            return pval_list
    eff_fdr_th = pvals_validator([v[2] for v in pval_list], rows_num, alpha=alpha)
    if change_dtype:
        return np.array([(rows_dict[v[0]], rows_dict[v[1]]) for v in pval_list if v[2] <= eff_fdr_th])
    else:
        return np.array([(v[0], v[1]) for v in pval_list if v[2] <= eff_fdr_th])
