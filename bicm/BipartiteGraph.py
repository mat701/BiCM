"""
This module contains the class BipartiteGraph that handles the graph object, and many useful functions. 

The solver functions are located here for compatibility with the numba package.
"""

import numpy as np
from numba import jit
from .functions import *
import scipy.sparse
from scipy import optimize as opt
import time


@jit(nopython=True)
def bicm_from_fitnesses(x, y):
    """
    Rebuilds the average probability matrix of the bicm from the fitnesses
    
    :param x: the fitness vector of the rows layer
    :type x: numpy.ndarray
    :param y: the fitness vector of the columns layer
    :type y: numpy.ndarray
    """
    avg_mat = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            xy = x[i] * y[j]
            avg_mat[i, j] = xy / (1 + xy)
    return avg_mat


@jit(nopython=True)
def eqs_root(xx, d_rows, d_cols, multiplier_rows, multiplier_cols, nrows, ncols, out_res):
    """
    Equations for the root solver of the reduced BiCM.
    
    :param xx: fitnesses vector
    :type xx: numpy.array
    """
    out_res -= out_res

    for i in range(nrows):
        for j in range(ncols):
            x_ij = xx[nrows + j] * xx[i]
            multiplier_ij = x_ij / (1 + x_ij)
            out_res[i] += multiplier_cols[j] * multiplier_ij
            out_res[j + nrows] += multiplier_rows[i] * multiplier_ij

    for i in range(nrows):
        out_res[i] -= d_rows[i]
    for j in range(ncols):
        out_res[j + nrows] -= d_cols[j]


def hessian_regulariser_function(B, eps=1e-8):
    """Trasform input matrix in a positive defined matrix
    by adding positive quantites to the main diagonal.

    :param np.ndarray B: Hessian matrix
    :param float eps: Magnitude of regularization
    :returns np.ndarray: Regularised Hessian
    """

    B = (B + B.transpose()) * 0.5  # symmetrization
    Bf = B + np.identity(B.shape[0]) * eps

    return Bf


def hessian_regulariser_function_eigen_based(B, eps=1e-8):
    """
    Trasform input matrix in a positive defined matrix
    
    :param numpy.ndarray B: Input matrix
    :param float eps: Magnitude of regularization
    """
    B = (B + B.transpose()) * 0.5
    l, e = np.linalg.eigh(B)
    ll = np.array([0 if li > eps else eps - li for li in l])
    Bf = np.dot(np.dot(e, (np.diag(ll) + np.diag(l))), e.transpose())
    return Bf


@jit(nopython=True)
def iterative_bicm(x0, args, exp=False):
    """
    Return the next iterative step for the Bipartite Configuration Model reduced version.

    :param numpy.ndarray x0: initial point
    :param list, tuple args: rows degree sequence, columns degree sequence, rows multipl., cols multipl.
    :param bool exp: If performing the exponential substitution
    :returns: next iteration step
    :rtype: numpy.ndarray
    """
    r_dseq_rows = args[0]
    r_dseq_cols = args[1]
    rows_multiplicity = args[2]
    cols_multiplicity = args[3]
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    x = x0[:num_rows]
    y = x0[num_rows:]
    if not exp:
        x = np.exp(- x)
        y = np.exp(- y)

    f = np.zeros(len(x0))

    for i in range(num_rows):
        rows_multiplier = rows_multiplicity[i] * x[i]
        for j in range(num_cols):
            denom = 1 + x[i] * y[j]
            f[i] += cols_multiplicity[j] * y[j] / denom
            f[j + num_rows] += rows_multiplier / denom
    tmp = np.concatenate((r_dseq_rows, r_dseq_cols))
    ff = tmp / f
    if not exp:
        ff = - np.log(ff)

    return ff


@jit(nopython=True)
def jac_root(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_j_t):
    """
    Jacobian for the root solver of the reduced BiCM.
    
    :param xx: fitnesses vector
    :type xx: numpy.array
    """
    out_j_t -= out_j_t

    for i in range(nrows):
        for j in range(ncols):
            denom_ij = (1 + xx[i] * xx[nrows + j]) ** 2
            multiplier_ij_i = xx[i] / denom_ij
            multiplier_ij_j = xx[nrows + j] / denom_ij
            out_j_t[i, i] += multiplier_cols[j] * multiplier_ij_j
            out_j_t[j + nrows, j + nrows] += multiplier_rows[i] * multiplier_ij_i
            out_j_t[i, j + nrows] += multiplier_rows[i] * multiplier_ij_j
            out_j_t[j + nrows, i] += multiplier_cols[j] * multiplier_ij_i


@jit(nopython=True)
def loglikelihood_bicm(x0, args, exp=False):
    """
    Log-likelihood function of the reduced BiCM.
    
    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list or tuple
    :param bool exp: If performing the exponential substitution
    :returns: log-likelihood of the system
    :rtype: float
    """
    r_dseq_rows = args[0]
    r_dseq_cols = args[1]
    rows_multiplicity = args[2]
    cols_multiplicity = args[3]
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    x = x0[:num_rows]
    y = x0[num_rows:]
    if not exp:
        theta_x = np.copy(x)
        theta_y = np.copy(y)
        x = np.exp(- x)
        y = np.exp(- y)
    else:
        theta_x = - np.log(x)
        theta_y = - np.log(y)
    flag = True

    f = 0
    for i in range(num_rows):
        f -= rows_multiplicity[i] * r_dseq_rows[i] * theta_x[i]
        for j in range(num_cols):
            if flag:
                f -= cols_multiplicity[j] * r_dseq_cols[j] * theta_y[j]
            f -= rows_multiplicity[i] * cols_multiplicity[j] * np.log(1 + x[i] * y[j])
        flag = False

    return f


@jit(nopython=True)
def loglikelihood_hessian_bicm(x0, args, exp=False):
    """
    Log-likelihood hessian of the reduced BiCM.
    
    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
    :param bool exp: If performing the exponential substitution
    :returns: 2D hessian matrix of the system
    :rtype: numpy.ndarray
    """
    r_dseq_rows = args[0]
    r_dseq_cols = args[1]
    rows_multiplicity = args[2]
    cols_multiplicity = args[3]
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    x = x0[:num_rows]
    y = x0[num_rows:]

    out = np.zeros((len(x0), len(x0)))
    if not exp:
        x = np.exp(- x)
        y = np.exp(- y)
    else:
        x2 = x ** 2
        y2 = y ** 2
    flag = True
    
    for h in range(num_rows):
        if exp:
            out[h, h] -= r_dseq_rows[h] / x2[h]
        for i in range(num_cols):
            denom = (1 + x[h] * y[i]) ** 2
            if not exp:
                add = cols_multiplicity[i] * rows_multiplicity[h] * x[h] * y[i] / denom
                out[h, h] -= add
                out[h, i + num_rows] = - add
                out[i + num_rows, h] = - add
                out[i + num_rows, i + num_rows] -= add
            else:
                multiplier = cols_multiplicity[i] / denom
                multiplier_h = rows_multiplicity[h] / denom
                out[h, h] += y2[i] * multiplier
                out[h, i + num_rows] = - multiplier
                out[i + num_rows, i + num_rows] += x2[h] * multiplier_h
                out[i + num_rows, h] = - multiplier_h
                if flag:
                    out[i + num_rows, i + num_rows] -= r_dseq_cols[i] / y2[i]
        flag = False

    return out


@jit(nopython=True)
def loglikelihood_hessian_diag_bicm(x0, args, exp=False):
    """
    Log-likelihood diagonal hessian of the reduced BiCM.
    
    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
    :param bool exp: If performing the exponential substitution
    :returns: 2D hessian matrix of the system
    :rtype: numpy.ndarray
    """
    r_dseq_rows = args[0]
    r_dseq_cols = args[1]
    rows_multiplicity = args[2]
    cols_multiplicity = args[3]
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    x = x0[:num_rows]
    y = x0[num_rows:]
    if not exp:
        x = np.exp(- x)
        y = np.exp(- y)
    else:
        x2 = x ** 2
        y2 = y ** 2

    f = np.zeros(num_rows + num_cols)
    flag = True

    for i in range(num_rows):
        for j in range(num_cols):
            denom = (1 + x[i] * y[j]) ** 2
            if not exp:
                add = cols_multiplicity[j] * rows_multiplicity[i] * x[i] * y[j] / denom
                f[i] -= add
                f[j + num_rows] -= add
            else:
                f[i] += cols_multiplicity[j] * y2[j] / denom
                f[j + num_rows] += rows_multiplicity[i] * x2[i] / denom
                if flag:
                    f[j + num_rows] -= r_dseq_cols[j] / y2[j]
        if exp:
            f[i] -= r_dseq_rows[i] / x2[i]
        flag = False

    return f


@jit(nopython=True)
def loglikelihood_prime_bicm(x0, args, exp=False):
    """
    Iterative function for loglikelihood gradient BiCM.
    
    :param x0: fitnesses vector
    :type x0: numpy.array
    :param args: list of arguments needed for the computation
    :type args: list or tuple
    :param bool exp: If performing the exponential substitution
    :returns: log-likelihood of the system
    :rtype: numpy.array
    """
    r_dseq_rows = args[0]
    r_dseq_cols = args[1]
    rows_multiplicity = args[2]
    cols_multiplicity = args[3]
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    x = x0[:num_rows]
    y = x0[num_rows:]
    if not exp:
        x = np.exp(- x)
        y = np.exp(- y)

    f = np.zeros(len(x0))
    flag = True

    for i in range(num_rows):
        for j in range(num_cols):
            denom = 1 + x[i] * y[j]
            if not exp:
                add = y[j] * x[i] * rows_multiplicity[i] * cols_multiplicity[j] / denom
                f[i] += add
                f[j + num_rows] += add
                if flag:
                    f[j + num_rows] -= r_dseq_cols[j] * cols_multiplicity[j]
            else:
                f[i] -= y[j] * cols_multiplicity[j] / denom
                f[j + num_rows] -= x[i] * rows_multiplicity[i] / denom
                if flag:
                    f[j + num_rows] += r_dseq_cols[j] / y[j]
        if not exp:
            f[i] -= r_dseq_rows[i] * rows_multiplicity[i]
        else:
            f[i] += r_dseq_rows[i] / x[i]
        flag = False
    
    return f


def solver(x0, fun, stop_fun, fun_jac, hessian_regulariser, tol=1e-10, eps=1e-16, regularise_eps=1e-6, max_steps=100, method='newton', verbose=False, regularise=True, full_return=False, linsearch=True):
    """
    Find roots of eq. fun = 0, using methods *newton*, *quasi-newton* or *iterative*.
    
    :param numpy.ndarray x0: Initial conditions
    :param function fun: Function to be minimized
    :param function stop_fun: Function that is used to determine whether the solver is making progress
    :param function fun_jac: Jacobian of the system, not used for the fixed-point solver
    :param function hessian_regulariser: Regularisation function for the Hessian matrix
    :param float tol: Tolerance of the solution, optional
    :param float eps: Tolerance for the regularization of the matrix, optional
    :param float regularise_eps: Value to use for the regularization, optional
    :param int max_steps: Maximum number of steps, optional
    :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is newton
    :param bool verbose: Print elapsed time, errors and iteration steps, optional
    :param bool regularise: Regularise the matrices in the computations, optional
    :param bool full_return: Return also elapsed times and errors, optional
    :param bool linsearch: Implement the linesearch when searching for roots, default is True
    :returns: Solution of the system
    :rtype: numpy.ndarray
    """
    tic_all = time.time()
    tic = time.time()
    beta = .5
    n_steps = 0
    x = x0
    f_x = fun(x)
    norm = np.linalg.norm(f_x)
    diff = 1
    if full_return:
        norm_seq = [norm]
    if verbose:
        print('\nx0 = {}'.format(x))
        print('|f(x0)| = {}'.format(norm))
        print('f(x0) = {}'.format(fun(x0)))
        print('L(x0) = {}'.format(stop_fun(x0)))
        if fun_jac is not None:
            print('B(x0) = {}'.format(fun_jac(x0)))
    toc_init = time.time() - tic
    toc_alfa = 0
    toc_update = 0
    toc_dx = 0
    toc_jacfun = 0
    tic_loop = time.time()
    while norm > tol and diff > eps and n_steps < max_steps:
        x_old = x
        tic = time.time()
        if method == 'newton':
            H = fun_jac(x)
            if regularise:
                B = hessian_regulariser(H, regularise_eps)
            else:
                B = H.__array__()
        elif method == 'quasinewton':
            B = fun_jac(x)
            if regularise:
                B = np.maximum(B, B * 0 + regularise_eps)
        toc_jacfun += time.time() - tic
        tic = time.time()
        if method == 'newton':
            dx = np.linalg.solve(B, - f_x)
        elif method == 'quasinewton':
            dx = - f_x / B
        elif method == 'fixed-point':
            dx = f_x - x
        toc_dx += time.time() - tic
        tic = time.time()
        if linsearch:
            alfa = 1
            i = 0
            stop_old = stop_fun(x)
            while (not sufficient_decrease_condition(stop_old, stop_fun(x + alfa * dx), alfa, f_x, dx)) \
                    and i < 50:
                alfa *= beta
                i += 1
        else:
            alfa = 1
        toc_alfa += time.time() - tic
        tic = time.time()
        if method in ['newton', 'quasinewton']:
            x = x + alfa * dx
        if method in ['fixed-point']:
            x = x + alfa * dx
        toc_update += time.time() - tic
        f_x = fun(x)
        norm = np.linalg.norm(f_x)
        diff = np.linalg.norm(x - x_old)
        if full_return:
            norm_seq.append(norm)
        n_steps += 1

        if verbose:
            print('step {}'.format(n_steps))
            print('alpha = {}'.format(alfa))
            print('fun = {}'.format(f_x))
            print('dx = {}'.format(dx))
            print('x = {}'.format(x))
            print('B(x) = {}'.format(B))
            print('|f(x)| = {}'.format(norm))
    toc_loop = time.time() - tic_loop
    toc_all = time.time() - tic_all
    if verbose:
        print('Number of steps for convergence = {}'.format(n_steps))
        print('toc_init = {}'.format(toc_init))
        print('toc_jacfun = {}'.format(toc_jacfun))
        print('toc_alfa = {}'.format(toc_alfa))
        print('toc_dx = {}'.format(toc_dx))
        print('toc_update = {}'.format(toc_update))
        print('toc_loop = {}'.format(toc_loop))
        print('toc_all = {}'.format(toc_all))

    if full_return:
        return x, toc_all, n_steps, np.array(norm_seq)
    else:
        return x


def sufficient_decrease_condition(f_old, f_new, alpha, grad_f, p, c1=1e-4):
    """
    Return boolean indicator if upper wolfe condition is respected.
    """
    sup = f_old + c1 * alpha * np.dot(grad_f, p.T)
    return bool(f_new < sup)


class BipartiteGraph:
    """Bipartite Graph class for undirected binary bipartite networks.

    This class handles the bipartite graph object to compute the 
    Bipartite Configuration Model (BiCM), which can be used as a null model 
    for the analysis of undirected and binary bipartite networks. 
    The class provides methods for calculating the probabilities and matrices
    of the null model and for projecting the bipartite network on its layers.
    The object can be initialized passing one of the parameters, or the nodes and
    edges can be passed later.
    
    :param biadjacency: binary input matrix describing the biadjacency matrix
            of a bipartite graph with the nodes of one layer along the rows
            and the nodes of the other layer along the columns.
    :type biadjacency: numpy.array, scipy.sparse, list, optional
    :param adjacency_list: dictionary that contains the adjacency list of nodes.
        The keys are considered as the nodes on the rows layer and the values,
        to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
    :type adjacency_list: dict, optional
    :param edgelist: list of edges containing couples (row_node, col_node) of
        nodes forming an edge. each element in the couples must belong to
        the respective layer.
    :type edgelist: list, numpy.array, optional
    :param degree_sequences: couple of lists describing the degree sequences
        of both layers.
    :type degree_sequences: list, numpy.array, tuple, optional
    """
    
    def __init__(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
        self.n_rows = None
        self.n_cols = None
        self.r_n_rows = None
        self.r_n_cols = None
        self.shape = None
        self.n_edges = None
        self.r_n_edges = None
        self.n_nodes = None
        self.r_n_nodes = None
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.inv_adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.r_rows_deg = None
        self.r_cols_deg = None
        self.rows_dict = None
        self.cols_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_reduced = False
        self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist, degree_sequences=degree_sequences)
        self.avg_mat = None
        self.x = None
        self.y = None
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        self.dict_x = None
        self.dict_y = None
        self.theta_x = None
        self.theta_y = None
        self.r_theta_x = None
        self.r_theta_y = None
        self.r_theta_xy = None
        self.projected_rows_edgelist = None
        self.projected_cols_edgelist = None
        self.rows_pvals = None
        self.cols_pvals = None
        self.is_rows_projected = False
        self.is_cols_projected = False
        self.initial_guess = None
        self.method = None
        self.rows_multiplicity = None
        self.cols_multiplicity = None
        self.r_invert_rows_deg = None
        self.r_invert_cols_deg = None
        self.r_dim = None
        self.verbose = False
        self.linsearch = True
        self.regularise = True
        self.nonfixed_rows = None
        self.fixed_rows = None
        self.full_rows_num = None
        self.nonfixed_cols = None
        self.fixed_cols = None
        self.full_cols_num = None
        self.J_T = None
        self.residuals = None
        self.full_rows_num = None
        self.full_rows_num = None
        self.solution_converged = None

    def _initialize_graph(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
        """
        Internal method for the initialization of the graph.
        Use the setter methods instead.

        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list, optional
        :param adjacency_list: dictionary that contains the adjacency list of nodes.
            The keys are considered as the nodes on the rows layer and the values,
            to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
        :type adjacency_list: dict, optional
        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array, optional
        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple, optional
        """
        if biadjacency is not None:
            if not isinstance(biadjacency, (list, np.ndarray)) and not scipy.sparse.isspmatrix(biadjacency):
                raise TypeError(
                    'The biadjacency matrix must be passed as a list or numpy array or scipy sparse matrix')
            else:
                if isinstance(biadjacency, list):
                    self.biadjacency = np.array(biadjacency)
                else:
                    self.biadjacency = biadjacency
                if self.biadjacency.shape[0] == self.biadjacency.shape[1]:
                    print(
                        'Your matrix is square. Please remember that it is treated as a biadjacency matrix, not an adjacency matrix.')
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg = adjacency_list_from_biadjacency(
                    self.biadjacency)
                self.n_rows = len(self.rows_deg)
                self.n_cols = len(self.cols_deg)
                self._initialize_node_dictionaries()
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError('The edgelist must be passed as a list or numpy array')
            elif len(edgelist[0]) != 2:
                raise ValueError(
                    'This is not an edgelist. An edgelist must be a vector of couples of nodes. Try passing a biadjacency matrix')
            else:
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = adjacency_list_from_edgelist(
                    edgelist)
                #                 self.edgelist, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = edgelist_from_edgelist(edgelist)
                self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
                self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
                self.is_initialized = True

        elif adjacency_list is not None:
            if not isinstance(adjacency_list, dict):
                raise TypeError('The adjacency list must be passed as a dictionary')
            else:
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
                    adjacency_list_from_adjacency_list(adjacency_list)
                self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
                self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
                self.is_initialized = True

        elif degree_sequences is not None:
            if not isinstance(degree_sequences, (list, np.ndarray, tuple)):
                raise TypeError('The degree sequences must be passed as a list, tuple or numpy array')
            elif len(degree_sequences) != 2:
                raise TypeError('degree_sequences must contain two vectors, the two layers degree sequences')
            elif not isinstance(degree_sequences[0], (list, np.ndarray)) or not isinstance(degree_sequences[1],
                                                                                           (list, np.ndarray)):
                raise TypeError('The two degree sequences must be lists or numpy arrays')
            elif np.sum(degree_sequences[0]) != np.sum(degree_sequences[1]):
                raise ValueError('The two degree sequences must have the same total sum.')
            elif (np.array(degree_sequences[0]) < 0).sum() + (np.array(degree_sequences[1]) < 0).sum() > 0:
                raise ValueError('A degree cannot be negative.')
            else:
                self.rows_deg = degree_sequences[0]
                self.cols_deg = degree_sequences[1]
                self.n_rows = len(self.rows_deg)
                self.n_cols = len(self.cols_deg)
                self._initialize_node_dictionaries()
                self.is_initialized = True

        if self.is_initialized:
            self.n_rows = len(self.rows_deg)
            self.n_cols = len(self.cols_deg)
            self.shape = [self.n_rows, self.n_cols]
            self.n_edges = np.sum(self.rows_deg)
            self.n_nodes = self.n_rows + self.n_cols

    def _initialize_node_dictionaries(self):
        self.rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))
        self.inv_rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.inv_cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))

    def degree_reduction(self, rows_deg=None, cols_deg=None):
        """Reduce the degree sequences to contain no repetition of the degrees.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """
        if rows_deg is None:
            rows_deg = self.rows_deg
        else:
            cols_deg -= self.full_rows_num
        if cols_deg is None:
            cols_deg = self.cols_deg
        else:
            rows_deg -= self.full_cols_num
        self.r_rows_deg, self.r_invert_rows_deg, self.rows_multiplicity \
            = np.unique(rows_deg, return_index=False, return_inverse=True, return_counts=True)
        self.r_cols_deg, self.r_invert_cols_deg, self.cols_multiplicity \
            = np.unique(cols_deg, return_index=False, return_inverse=True, return_counts=True)
        self.r_n_rows = self.r_rows_deg.size
        self.r_n_cols = self.r_cols_deg.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.r_n_edges = np.sum(rows_deg)
        self.is_reduced = True

    def _set_initial_guess(self):
        """
        Internal method to set the initial point of the solver.
        """
        if self.initial_guess is None:
            self.r_x = self.r_rows_deg / (
                        np.sqrt(self.r_n_edges))  # This +1 increases the stability of the solutions.
            self.r_y = self.r_cols_deg / (np.sqrt(self.r_n_edges))
        elif self.initial_guess == 'random':
            self.r_x = np.random.rand(self.r_n_rows).astype(np.float64)
            self.r_y = np.random.rand(self.r_n_cols).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.r_x = np.ones(self.r_n_rows, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.r_y = np.ones(self.r_n_cols, dtype=np.float64)
        elif self.initial_guess == 'degrees':
            self.r_x = self.r_rows_deg.astype(np.float64)
            self.r_y = self.r_cols_deg.astype(np.float64)
        if not self.exp:
            self.r_theta_x = - np.log(self.r_x)
            self.r_theta_y = - np.log(self.r_y)
            self.x0 = np.concatenate((self.r_theta_x, self.r_theta_y))
        else:
            self.x0 = np.concatenate((self.r_x, self.r_y))

    def initialize_avg_mat(self):
        """
        Reduces the matrix eliminating empty or full rows or columns.
        It repeats the process on the so reduced matrix until no more reductions are possible.
        For instance, a perfectly nested matrix will be reduced until all entries are set to 0 or 1.
        """
        self.avg_mat = np.zeros_like(self.biadjacency, dtype=float)
        r_biad_mat = np.copy(self.biadjacency)
        rows_num, cols_num = self.biadjacency.shape
        rows_degs = self.biadjacency.sum(1)
        cols_degs = self.biadjacency.sum(0)
        good_rows = np.arange(rows_num)
        good_cols = np.arange(cols_num)
        zero_rows = np.where(rows_degs == 0)[0]
        zero_cols = np.where(cols_degs == 0)[0]
        full_rows = np.where(rows_degs == cols_num)[0]
        full_cols = np.where(cols_degs == rows_num)[0]
        self.full_rows_num = 0
        self.full_cols_num = 0
        while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), zero_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), zero_cols)]
            good_rows = np.delete(good_rows, zero_rows)
            good_cols = np.delete(good_cols, zero_cols)
            full_rows = np.where(r_biad_mat.sum(1) == r_biad_mat.shape[1])[0]
            full_cols = np.where(r_biad_mat.sum(0) == r_biad_mat.shape[0])[0]
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)
            self.avg_mat[good_rows[full_rows][:, None], good_cols] = 1
            self.avg_mat[good_rows[:, None], good_cols[full_cols]] = 1
            good_rows = np.delete(good_rows, full_rows)
            good_cols = np.delete(good_cols, full_cols)
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), full_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), full_cols)]
            zero_rows = np.where(r_biad_mat.sum(1) == 0)[0]
            zero_cols = np.where(r_biad_mat.sum(0) == 0)[0]

        self.nonfixed_rows = good_rows
        self.fixed_rows = np.delete(np.arange(rows_num), good_rows)
        self.nonfixed_cols = good_cols
        self.fixed_cols = np.delete(np.arange(cols_num), good_cols)
        return r_biad_mat

    def _initialize_fitnesses(self):
        """
        Internal method to initialize the fitnesses of the BiCM. 
        If there are empty rows, the corresponding fitnesses are set to 0, 
        while for full rows the corresponding columns are set to numpy.inf.
        """
        if not self.exp:
            self.theta_x = np.zeros(self.n_rows, dtype=float)
            self.theta_y = np.zeros(self.n_cols, dtype=float)
        self.x = np.zeros(self.n_rows, dtype=float)
        self.y = np.zeros(self.n_cols, dtype=float)
        good_rows = np.arange(self.n_rows)
        good_cols = np.arange(self.n_cols)
        bad_rows = np.array([])
        bad_cols = np.array([])
        self.full_rows_num = 0
        self.full_cols_num = 0
        if np.any(np.isin(self.rows_deg, (0, self.n_cols))) or np.any(np.isin(self.cols_deg, (0, self.n_rows))):
            print('''
                      WARNING: this system has at least a node that is disconnected or connected to all nodes of the opposite layer. 
                      This may cause some convergence issues.
                      Please use the full mode providing a biadjacency matrix or an edgelist, or clean your data from these nodes. 
                      ''')
            zero_rows = np.where(self.rows_deg == 0)[0]
            zero_cols = np.where(self.cols_deg == 0)[0]
            full_rows = np.where(self.rows_deg == self.n_cols)[0]
            full_cols = np.where(self.cols_deg == self.n_rows)[0]
            self.x[full_rows] = np.inf
            self.y[full_cols] = np.inf
            if not self.exp:
                self.theta_x[full_rows] = - np.inf
                self.theta_y[full_rows] = - np.inf
                self.theta_x[zero_rows] = np.inf
                self.theta_y[zero_cols] = np.inf
            bad_rows = np.concatenate((zero_rows, full_rows))
            bad_cols = np.concatenate((zero_cols, full_cols))
            good_rows = np.delete(np.arange(self.n_rows), bad_rows)
            good_cols = np.delete(np.arange(self.n_cols), bad_cols)
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)

        self.nonfixed_rows = good_rows
        self.fixed_rows = bad_rows
        self.nonfixed_cols = good_cols
        self.fixed_cols = bad_cols

    def _initialize_problem(self, rows_deg=None, cols_deg=None):
        """
        Initializes the solver reducing the degree sequences, setting the initial guess and setting the functions for the solver.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """
        if ~self.is_reduced:
            self.degree_reduction(rows_deg=rows_deg, cols_deg=cols_deg)
        self._set_initial_guess()
        if self.method == 'root':
            self.J_T = np.zeros((self.r_dim, self.r_dim), dtype=np.float64)
            self.residuals = np.zeros(self.r_dim, dtype=np.float64)
        else:
            self.args = (self.r_rows_deg, self.r_cols_deg, self.rows_multiplicity, self.cols_multiplicity)
            d_fun = {
                'newton': lambda x: - loglikelihood_prime_bicm(x, self.args, exp=self.exp),
                'quasinewton': lambda x: - loglikelihood_prime_bicm(x, self.args, exp=self.exp),
                'fixed-point': lambda x: iterative_bicm(x, self.args, exp=self.exp),
            }
            d_fun_jac = {
                'newton': lambda x: - loglikelihood_hessian_bicm(x, self.args, exp=self.exp),
                'quasinewton': lambda x: - loglikelihood_hessian_diag_bicm(x, self.args, exp=self.exp),
                'fixed-point': None,
            }
            d_fun_stop = {
                'newton': lambda x: - loglikelihood_bicm(x, self.args, exp=self.exp),
                'quasinewton': lambda x: - loglikelihood_bicm(x, self.args, exp=self.exp),
                'fixed-point': lambda x: - loglikelihood_bicm(x, self.args, exp=self.exp),
            }
            self.hess_reg = hessian_regulariser_function
            try:
                self.fun = d_fun[self.method]
                self.fun_jac = d_fun_jac[self.method]
                self.stop_fun = d_fun_stop[self.method]
            except (TypeError, KeyError):
                raise ValueError('Method must be "newton","quasinewton", "fixed-point" or "root".')

    def _equations_root(self, x):
        """
        Equations for the *root* solver
        """
        eqs_root(x, self.r_rows_deg, self.r_cols_deg,
                 self.rows_multiplicity, self.cols_multiplicity,
                 self.r_n_rows, self.r_n_cols, self.residuals)

    def _jacobian_root(self, x):
        """
        Jacobian for the *root* solver
        """
        jac_root(x, self.rows_multiplicity, self.cols_multiplicity,
                 self.r_n_rows, self.r_n_cols, self.J_T)

    def _residuals_jacobian(self, x):
        """
        Residuals and jacobian for the *root* solver
        """
        self._equations_root(x)
        self._jacobian_root(x)
        return self.residuals, self.J_T

    def _clean_root(self):
        """
        Clean variables used for the *root* solver
        """
        self.J_T = None
        self.residuals = None

    def _solve_root(self):
        """
        Internal *root* solver
        """
        x0 = self.x0
        opz = {'col_deriv': True, 'diag': None}
        res = opt.root(self._residuals_jacobian, x0,
                       method='hybr', jac=True, options=opz)
        self._clean_root()
        return res.x

    def _check_solution(self, return_error=False, in_place=False):
        """
        Check if the solution of the BiCM is compatible with the degree sequences of the graph.
        
        :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
        :param bool in_place: check also the error in the single entries of the matrices. 
            Always False unless comparing two different solutions.
        """
        if self.biadjacency is not None and self.avg_mat is not None:
            return check_sol(self.biadjacency, self.avg_mat, return_error=return_error, in_place=in_place)
        else:
            return check_sol_light(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols],
                                   self.rows_deg[self.nonfixed_rows] - self.full_cols_num,
                                   self.cols_deg[self.nonfixed_cols] - self.full_rows_num,
                                   return_error=return_error)

    def _set_solved_problem(self, solution):
        """
        Sets the solution of the problem.
        
        :param numpy.ndarray solution: A numpy array containing that reduced fitnesses of the two layers, consecutively.
        """
        if not self.exp:
            if self.theta_x is None:
                self.theta_x = np.zeros(self.n_rows)
            if self.theta_y is None:
                self.theta_y = np.zeros(self.n_cols)
            self.r_theta_xy = solution
            self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
            self.r_theta_y = self.r_theta_xy[self.r_n_rows:]
            self.r_xy = np.exp(- self.r_theta_xy)
            self.r_x = np.exp(- self.r_theta_x)
            self.r_y = np.exp(- self.r_theta_y)
            self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
            self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
        else:
            self.r_xy = solution
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
        if self.x is None:
            self.x = np.zeros(self.n_rows)
        if self.y is None:
            self.y = np.zeros(self.n_cols)
        self.x[self.nonfixed_rows] = self.r_x[self.r_invert_rows_deg]
        self.y[self.nonfixed_cols] = self.r_y[self.r_invert_cols_deg]
        self.dict_x = dict([(self.rows_dict[i], self.x[i]) for i in range(len(self.x))])
        self.dict_y = dict([(self.cols_dict[j], self.y[j]) for j in range(len(self.y))])

    def _solve_bicm_full(self):
        """
        Internal method for computing the solution of the BiCM via matrices.
        """
        r_biadjacency = self.initialize_avg_mat()
        if len(r_biadjacency) > 0:  # Every time the matrix is not perfectly nested
            rows_deg = self.rows_deg[self.nonfixed_rows]
            cols_deg = self.cols_deg[self.nonfixed_cols]
            self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
            if self.method == 'root':
                sol = self._solve_root()
            else:
                sol = solver(self.x0, self.fun, self.stop_fun, self.fun_jac, self.hess_reg, method=self.method,
                             regularise=self.regularise, tol=self.tolerance, max_steps=self.max_steps,
                             verbose=self.verbose, linsearch=self.linsearch)
            self._set_solved_problem(sol)
            r_avg_mat = bicm_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
            self.avg_mat[self.nonfixed_rows[:, None], self.nonfixed_cols] = np.copy(r_avg_mat)

    def _solve_bicm_light(self):
        """
        Internal method for computing the solution of the BiCM via degree sequences.
        """
        self._initialize_fitnesses()
        rows_deg = self.rows_deg[self.nonfixed_rows]
        cols_deg = self.cols_deg[self.nonfixed_cols]
        self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
        if self.method == 'root':
            sol = self._solve_root()
        else:
            sol = solver(self.x0, self.fun, self.stop_fun, self.fun_jac, self.hess_reg, method=self.method,
                         regularise=self.regularise, tol=self.tolerance, max_steps=self.max_steps,
                         verbose=self.verbose, linsearch=self.linsearch)
        self._set_solved_problem(sol)

    def _set_parameters(self, method, initial_guess, tolerance, regularise, max_steps, verbose, linsearch, exp):
        """
        Internal method for setting the parameters of the solver.
        """
        self.method = method
        self.initial_guess = initial_guess
        self.tolerance = tolerance
        self.verbose = verbose
        self.linsearch = linsearch
        self.regularise = regularise
        self.exp = exp
        if max_steps is None:
            if method == 'fixed-point':
                self.max_steps = 2000
            else:
                self.max_steps = 1000
        else:
            self.max_steps = max_steps

    def solve_bicm(self, light_mode=None, method='newton', initial_guess=None, tolerance=1e-10, max_steps=None,
                   verbose=False, linsearch=True, regularise=None, print_error=True, exp=False):
        """Solve the BiCM of the graph.
        It does not return the solution, use the getter methods instead.

        :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
            If the graph has been initialized without the matrix, the light mode is used regardless.
        :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is newton
        :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*, default is None
        :param float tolerance: Tolerance of the solution, optional
        :param int max_steps: Maximum number of steps, optional
        :param bool, optional verbose: Print elapsed time, errors and iteration steps, optional
        :param bool linsearch: Implement the linesearch when searching for roots, default is True
        :param bool regularise: Regularise the matrices in the computations, optional
        :param bool print_error: Print the final error of the solution
        :param bool exp: if this is set to true the solver works with the reparameterization $x_i = e^{-\theta_i}$,
            $y_\alpha = e^{-\theta_\alpha}$. It might be slightly faster but also might not converge.
        """
        if not self.is_initialized:
            print('Graph is not initialized. I can\'t compute the BiCM.')
            return
        if regularise is None:
            if exp:
                regularise = True
            else:
                regularise = False
        if regularise and not exp:
            print('Warning: regularise is only recommended in exp mode.')
        if method == 'root':
            exp = True
        self._set_parameters(method=method, initial_guess=initial_guess, tolerance=tolerance,
                             regularise=regularise, max_steps=max_steps, verbose=verbose, linsearch=linsearch, exp=exp)
        if self.biadjacency is not None and (light_mode is None or not light_mode):
            self._solve_bicm_full()
        else:
            if light_mode is False:
                print('''
                I cannot work with the full mode without the biadjacency matrix.
                This will not account for disconnected or fully connected nodes.
                Solving in light mode...
                ''')
            self._solve_bicm_light()
        if print_error:
            self.solution_converged = not bool(self._check_solution(return_error=True))
            if self.solution_converged:
                print('Solver converged.')
            else:
                print('Solver did not converge.')
        self.is_randomized = True

    def get_bicm_matrix(self):
        """Get the matrix of probabilities of the BiCM.
        If the BiCM has not been computed, it also computes it with standard settings.
        
        :returns: The average matrix of the BiCM
        :rtype: numpy.ndarray
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
        elif not self.is_randomized:
            self.solve_bicm()
        if self.avg_mat is not None:
            return self.avg_mat
        else:
            self.avg_mat = bicm_from_fitnesses(self.x, self.y)
            return self.avg_mat

    def get_bicm_fitnesses(self):
        """Get the fitnesses of the BiCM.
        If the BiCM has not been computed, it also computes it with standard settings.

        :returns: The fitnesses of the BiCM in the format **rows fitnesses dictionary, columns fitnesses dictionary**
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
        elif not self.is_randomized:
            print('Computing the BiCM...')
            self.solve_bicm()
        return self.dict_x, self.dict_y

    def compute_projection(self, rows=True, alpha=0.05, method='poisson', threads_num=4, progress_bar=True):
        """Compute the projection of the network on the rows or columns layer.
        If the BiCM has not been computed, it also computes it with standard settings.
        This is the most customizable method for the pvalues computation.

        :param bool rows: True if requesting the rows projection.
        :param float alpha: Threshold for the FDR validation.
        :param str method: Method for the approximation of the pvalues computation.
            Implemented methods are *poisson*, *poibin*, *normal*, *rna*.
        :param threads_num: Number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :param bool progress_bar: Show progress bar of the pvalues computation.
        """
        if self.adj_list is None:
            print('''
            Without the edges I can't compute the projection. 
            Use set_biadjacency_matrix, set_adjacency_list or set_edgelist to add edges.
            ''')
            return
        else:
            if not self.is_randomized:
                print('First I have to compute the BiCM. Computing...')
                self.solve_bicm()
            if rows:
                if self.avg_mat is not None and self.biadjacency is not None:
                    self.rows_pvals = projection_calculator(self.biadjacency, self.avg_mat,
                                                            rows=True, alpha=alpha, method=method,
                                                            threads_num=threads_num, return_pvals=True,
                                                            progress_bar=progress_bar)
                else:
                    self.rows_pvals = projection_calculator_light_from_adjacency(self.adj_list, self.x, self.y,
                                                                                 #                     self.rows_pvals = projection_calculator_light(self.edgelist, self.x, self.y,
                                                                                 alpha=alpha, method=method,
                                                                                 threads_num=threads_num,
                                                                                 return_pvals=True,
                                                                                 progress_bar=progress_bar)
                self.projected_rows_edgelist = self._projection_from_pvals(rows=True, alpha=alpha)
                self.is_rows_projected = True
            else:
                if self.avg_mat is not None and self.biadjacency is not None:
                    self.cols_pvals = projection_calculator(self.biadjacency, self.avg_mat,
                                                            rows=False, alpha=alpha, method=method,
                                                            threads_num=threads_num, return_pvals=True,
                                                            progress_bar=progress_bar)
                else:
                    self.cols_pvals = projection_calculator_light_from_adjacency(self.inv_adj_list, self.y, self.x,
                                                                                 #                     self.cols_pvals = projection_calculator_light(self.edgelist, self.x, self.y,
                                                                                 alpha=alpha, method=method,
                                                                                 threads_num=threads_num,
                                                                                 return_pvals=True,
                                                                                 progress_bar=progress_bar)
                self.projected_cols_edgelist = self._projection_from_pvals(rows=False, alpha=alpha)
                self.is_cols_projected = True

    def _projection_from_pvals(self, rows=True, alpha=0.05):
        """Internal method to build the projected network from pvalues.
        
        :param float alpha:  Threshold for the FDR validation. 
        """
        if rows:
            eff_fdr_th = pvals_validator([v[2] for v in self.rows_pvals], self.n_rows, alpha=alpha)
            return np.array([(v[0], v[1]) for v in self.rows_pvals if v[2] <= eff_fdr_th])
        else:
            eff_fdr_th = pvals_validator([v[2] for v in self.cols_pvals], self.n_cols, alpha=alpha)
            return np.array([(v[0], v[1]) for v in self.cols_pvals if v[2] <= eff_fdr_th])

    def get_rows_projection(self, alpha=0.05, method='poisson', threads_num=4, progress_bar=True):
        """Get the projected network on the rows layer of the graph.
        
        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.a
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :returns: edgelist of the projected network on the rows layer
        :rtype: numpy.array
        """
        if not self.is_rows_projected:
            self.compute_projection(rows=True, alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.rows_dict is None:
            return np.array(self.projected_rows_edgelist)
        else:
            return np.array(
                [(self.rows_dict[edge[0]], self.rows_dict[edge[1]]) for edge in self.projected_rows_edgelist])

    def get_cols_projection(self, alpha=0.05, method='poisson', threads_num=4, progress_bar=True):
        """Get the projected network on the columns layer of the graph.
        
        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :returns: edgelist of the projected network on the columns layer
        :rtype: numpy.array
        """
        if not self.is_cols_projected:
            self.compute_projection(rows=False, alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.cols_dict is None:
            return np.array(self.projected_cols_edgelist)
        else:
            return np.array(
                [(self.cols_dict[edge[0]], self.cols_dict[edge[1]]) for edge in self.projected_cols_edgelist])

    def set_biadjacency_matrix(self, biadjacency):
        """Set the biadjacency matrix of the graph.
        
        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(biadjacency=biadjacency)

    def set_adjacency_list(self, adj_list):
        """Set the adjacency list of the graph.

        :param adj_list: a dictionary containing the adjacency list
                of a bipartite graph with the nodes of one layer as keys
                and lists of neighbor nodes of the other layer as values.
        :type adj_list: dict
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(adjacency_list=adj_list)

    def set_edgelist(self, edgelist):
        """Set the edgelist of the graph.
        
        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequences):
        """Set the degree sequence of the graph.
        
        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(degree_sequences=degree_sequences)

    def clean_edges(self):
        """Clean the edges of the graph.
        """
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.is_initialized = False
