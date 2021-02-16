"""
This module contains the class BipartiteGraph that handles the graph object, and many useful functions.

The solver functions are located here for compatibility with the numba package.
"""

import numpy as np
import scipy.sparse
import scipy
from scipy.stats import norm, poisson
from . import models_functions as mof
from . import solver_functions as sof
from . import network_functions as nef
from tqdm import tqdm
from . import poibin as pb
from functools import partial
from platform import system

if system() != 'Windows':
    from multiprocessing import Pool


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
        self.rows_projection = None
        self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist,
                               degree_sequences=degree_sequences)
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
        self.projected_rows_adj_list = None
        self.projected_cols_adj_list = None
        self.v_adj_list = None
        self.projection_method = 'poisson'
        self.threads_num = 1
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
        self.full_return = False
        self.linsearch = True
        self.regularise = True
        self.tol = None
        self.eps = None
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
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg = \
                    nef.adjacency_list_from_biadjacency(self.biadjacency)
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
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
                    nef.adjacency_list_from_edgelist_bipartite(edgelist)
                self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
                self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
                self.is_initialized = True

        elif adjacency_list is not None:
            if not isinstance(adjacency_list, dict):
                raise TypeError('The adjacency list must be passed as a dictionary')
            else:
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
                    nef.adjacency_list_from_adjacency_list_bipartite(adjacency_list)
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
        Initializes the solver reducing the degree sequences,
        setting the initial guess and setting the functions for the solver.
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
                'newton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
                'fixed-point': lambda x: mof.iterative_bicm(x, self.args),
                'newton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
                'fixed-point_exp': lambda x: mof.iterative_bicm_exp(x, self.args),
            }
            d_fun_jac = {
                'newton': lambda x: - mof.loglikelihood_hessian_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_hessian_diag_bicm(x, self.args),
                'fixed-point': None,
                'newton_exp': lambda x: - mof.loglikelihood_hessian_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_hessian_diag_bicm_exp(x, self.args),
                'fixed-point_exp': None,
            }
            d_fun_step = {
                'newton': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'fixed-point': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'newton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
                'fixed-point_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
            }

            if self.exp:
                self.hessian_regulariser = sof.matrix_regulariser_function_eigen_based
            else:
                self.hessian_regulariser = sof.matrix_regulariser_function

            if self.exp:
                lins_args = (mof.loglikelihood_bicm_exp, self.args)
            else:
                lins_args = (mof.loglikelihood_bicm, self.args)
            lins_fun = {
                'newton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
                'quasinewton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
                'fixed-point': lambda x: mof.linsearch_fun_BiCM_fixed(x),
                'newton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
                'quasinewton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
                'fixed-point_exp': lambda x: mof.linsearch_fun_BiCM_exp_fixed(x),
            }
            if self.exp:
                method = self.method + '_exp'
            else:
                method = self.method
            try:
                self.fun = d_fun[method]
                self.fun_jac = d_fun_jac[method]
                self.step_fun = d_fun_step[method]
                self.fun_linsearch = lins_fun[method]
            except (TypeError, KeyError):
                raise ValueError('Method must be "newton","quasinewton", "fixed-point" or "root".')

    def _equations_root(self, x):
        """
        Equations for the *root* solver
        """
        mof.eqs_root(x, self.r_rows_deg, self.r_cols_deg,
                     self.rows_multiplicity, self.cols_multiplicity,
                     self.r_n_rows, self.r_n_cols, self.residuals)

    def _jacobian_root(self, x):
        """
        Jacobian for the *root* solver
        """
        mof.jac_root(x, self.rows_multiplicity, self.cols_multiplicity,
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
        res = scipy.optimize.root(self._residuals_jacobian, x0,
                                  method='hybr', jac=True, options=opz)
        self._clean_root()
        return res.x

    def check_sol(self, biad_mat, avg_bicm, return_error=False, in_place=False):
        """
        This function prints the rows sums differences between two matrices, that originally are the biadjacency matrix and its bicm2 average matrix.
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

    def check_sol_light(self, x, y, rows_deg, cols_deg, return_error=False):
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

    def _check_solution(self, return_error=False, in_place=False):
        """
        Check if the solution of the BiCM is compatible with the degree sequences of the graph.

        :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
        :param bool in_place: check also the error in the single entries of the matrices.
            Always False unless comparing two different solutions.
        """
        if self.biadjacency is not None and self.avg_mat is not None:
            return self.check_sol(self.biadjacency, self.avg_mat, return_error=return_error, in_place=in_place)
        else:
            return self.check_sol_light(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols],
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
        if self.full_return:
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        # Reset solver lambda functions for multiprocessing compatibility
        self.hessian_regulariser = None
        self.fun = None
        self.fun_jac = None
        self.step_fun = None
        self.fun_linsearch = None

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
                x0 = self.x0
                sol = sof.solver(
                    x0,
                    fun=self.fun,
                    fun_jac=self.fun_jac,
                    step_fun=self.step_fun,
                    linsearch_fun=self.fun_linsearch,
                    hessian_regulariser=self.hessian_regulariser,
                    tol=self.tol,
                    eps=self.eps,
                    max_steps=self.max_steps,
                    method=self.method,
                    verbose=self.verbose,
                    regularise=self.regularise,
                    full_return=self.full_return,
                    linsearch=self.linsearch,
                )
            self._set_solved_problem(sol)
            r_avg_mat = nef.bicm_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
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
            x0 = self.x0
            sol = sof.solver(
                x0,
                fun=self.fun,
                fun_jac=self.fun_jac,
                step_fun=self.step_fun,
                linsearch_fun=self.fun_linsearch,
                hessian_regulariser=self.hessian_regulariser,
                tol=self.tol,
                eps=self.eps,
                max_steps=self.max_steps,
                method=self.method,
                verbose=self.verbose,
                regularise=self.regularise,
                full_return=self.full_return,
                linsearch=self.linsearch,
            )
        self._set_solved_problem(sol)

    def _set_parameters(self, method, initial_guess, tol, eps, regularise, max_steps, verbose, linsearch, exp,
                        full_return):
        """
        Internal method for setting the parameters of the solver.
        """
        self.method = method
        self.initial_guess = initial_guess
        self.tol = tol
        self.eps = eps
        self.verbose = verbose
        self.linsearch = linsearch
        self.regularise = regularise
        self.exp = exp
        self.full_return = full_return
        if max_steps is None:
            if method == 'fixed-point':
                self.max_steps = 200
            else:
                self.max_steps = 100
        else:
            self.max_steps = max_steps

    def solve_tool(
            self,
            method='newton',
            initial_guess=None,
            light_mode=None,
            tol=1e-8,
            eps=1e-8,
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=None,
            print_error=True,
            full_return=False,
            exp=False):
        """Solve the BiCM of the graph.
        It does not return the solution, use the getter methods instead.

        :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
            If the graph has been initialized without the matrix, the light mode is used regardless.
        :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is newton
        :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*, default is None
        :param float tol: Tolerance of the solution, optional
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
                regularise = False
            else:
                regularise = True
        if regularise and exp:
            print('Warning: regularise is only recommended in non-exp mode.')
        if method == 'root':
            exp = True
        self._set_parameters(method=method, initial_guess=initial_guess, tol=tol, eps=eps, regularise=regularise,
                             max_steps=max_steps, verbose=verbose, linsearch=linsearch, exp=exp,
                             full_return=full_return)
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

    def solve_bicm(
            self,
            method='newton',
            initial_guess=None,
            light_mode=None,
            tolerance=None,
            tol=1e-8,
            eps=1e-8,
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=None,
            print_error=True,
            full_return=False,
            exp=False):
        """
        Deprecated method, replaced by solve_tool
        """
        if tolerance is not None:
            tol = tolerance
        print('solve_bicm has been deprecated calling solve_tool instead')
        self.solve_tool(
            method=method,
            initial_guess=initial_guess,
            light_mode=light_mode,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            verbose=verbose,
            linsearch=linsearch,
            regularise=regularise,
            print_error=print_error,
            full_return=full_return,
            exp=exp)

    def get_bicm_matrix(self):
        """Get the matrix of probabilities of the BiCM.
        If the BiCM has not been computed, it also computes it with standard settings.

        :returns: The average matrix of the BiCM
        :rtype: numpy.ndarray
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
        elif not self.is_randomized:
            self.solve_tool()
        if self.avg_mat is not None:
            return self.avg_mat
        else:
            self.avg_mat = nef.bicm_from_fitnesses(self.x, self.y)
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
            self.solve_tool()
        return self.dict_x, self.dict_y

    def pval_calculator(self, v_list_key, x, y):
        """
        Calculate the p-values of the v-motifs numbers of one vertices and all its neighbours.

        :param int v_list_key: the key of the node to consider for the adjacency list of the v-motifs.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a dictionary containing as keys the nodes that form v-motifs with the considered node, and as values the corresponding p-values.
        """
        node_xy = x[v_list_key] * y
        temp_pvals_dict = {}
        for neighbor in self.v_adj_list[v_list_key]:
            neighbor_xy = x[neighbor] * y
            probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
            avg_v = np.sum(probs)
            if self.projection_method == 'poisson':
                temp_pvals_dict[neighbor] = \
                    poisson.sf(k=self.v_adj_list[v_list_key][neighbor] - 1, mu=avg_v)
            elif self.projection_method == 'normal':
                sigma_v = np.sqrt(np.sum(probs * (1 - probs)))
                temp_pvals_dict[neighbor] = \
                    norm.cdf((self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v)
            elif self.projection_method == 'rna':
                var_v_arr = probs * (1 - probs)
                sigma_v = np.sqrt(np.sum(var_v_arr))
                gamma_v = (sigma_v ** (-3)) * np.sum(var_v_arr * (1 - 2 * probs))
                eval_x = (self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v
                pval = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
                temp_pvals_dict[neighbor] = max(min(pval, 1), 0)
        return temp_pvals_dict

    def pval_calculator_poibin(self, deg_couple, deg_dict, degs, x, y):
        """
        Calculate the p-values of the v-motifs numbers of all nodes with a given couple degrees.

        :param tuple deg_couple: the couple of degrees considered.
        :param tuple deg_couple: the couple of degrees considered.
        :param tuple deg_couple: the couple of degrees considered.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a list containing 3-tuples with the two nodes considered and their p-value.
        """
        node_xy = x[deg_dict[deg_couple[0]][0]] * y
        neighbor_xy = x[deg_dict[deg_couple[1]][0]] * y
        probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
        pb_obj = pb.PoiBin(probs)
        pval_dict = dict()
        for node1 in deg_dict[deg_couple[0]]:
            for node2 in deg_dict[deg_couple[1]]:
                if node2 in self.v_adj_list[node1]:
                    if node1 not in pval_dict:
                        pval_dict[node1] = dict()
                    pval_dict[node1][node2] = pb_obj.pval(int(self.v_adj_list[node1][node2]))
                elif node1 in self.v_adj_list[node2]:
                    if node2 not in pval_dict:
                        pval_dict[node2] = dict()
                    pval_dict[node2][node1] = pb_obj.pval(int(self.v_adj_list[node2][node1]))
        return pval_dict

    def _projection_calculator(self):
        if self.rows_projection:
            adj_list = self.adj_list
            inv_adj_list = self.inv_adj_list
            x = self.x
            y = self.y
        else:
            adj_list = self.inv_adj_list
            inv_adj_list = self.adj_list
            x = self.y
            y = self.x

        self.v_adj_list = {k: dict() for k in list(adj_list.keys())}
        for node_i in np.sort(list(adj_list.keys())):
            for neighbor in adj_list[node_i]:
                for node_j in inv_adj_list[neighbor]:
                    if node_j > node_i:
                        self.v_adj_list[node_i][node_j] = self.v_adj_list[node_i].get(node_j, 0) + 1
        if self.projection_method != 'poibin':
            v_list_keys = list(self.v_adj_list.keys())
            pval_adj_list = dict()
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as pool:
                    partial_function = partial(self.pval_calculator, x=x, y=y)
                    if self.progress_bar:
                        pvals_dicts = pool.map(partial_function, tqdm(v_list_keys))
                    else:
                        pvals_dicts = pool.map(partial_function, v_list_keys)
                for k_i in range(len(v_list_keys)):
                    k = v_list_keys[k_i]
                    pval_adj_list[k] = pvals_dicts[k_i]
            else:
                if self.progress_bar:
                    for k in tqdm(v_list_keys):
                        pval_adj_list[k] = self.pval_calculator(k, x=x, y=y)
                else:
                    for k in v_list_keys:
                        pval_adj_list[k] = self.pval_calculator(k, x=x, y=y)
        else:
            if self.rows_projection:
                degs = self.rows_deg
            else:
                degs = self.cols_deg
            unique_degs = np.unique(degs)
            deg_dict = {k: [] for k in unique_degs}
            for k_i in range(len(degs)):
                deg_dict[degs[k_i]].append(k_i)

            v_list_coupled = []
            deg_couples = [(unique_degs[deg_i], unique_degs[deg_j])
                           for deg_i in range(len(unique_degs))
                           for deg_j in range(deg_i, len(unique_degs))]
            if self.progress_bar:
                print('Calculating p-values...')
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as pool:
                    partial_function = partial(self.pval_calculator_poibin, deg_dict=deg_dict, degs=degs, x=x, y=y)
                    if self.progress_bar:
                        pvals_dicts = pool.map(partial_function, tqdm(deg_couples))
                    else:
                        pvals_dicts = pool.map(partial_function, deg_couples)
            else:
                pvals_dicts = []
                if self.progress_bar:
                    for deg_couple in tqdm(deg_couples):
                        pvals_dicts.append(
                            self.pval_calculator_poibin(deg_couple, deg_dict=deg_dict, degs=degs, x=x, y=y))
                else:
                    for deg_couple in v_list_coupled:
                        pvals_dicts.append(
                            self.pval_calculator_poibin(deg_couple, deg_dict=deg_dict, degs=degs, x=x, y=y))
            pval_adj_list = {k: dict() for k in self.v_adj_list}
            for pvals_dict in pvals_dicts:
                for node in pvals_dict:
                    pval_adj_list[node].update(pvals_dict[node])
        return pval_adj_list

    def compute_projection(self, rows=True, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
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
        self.rows_projection = rows
        self.projection_method = method
        self.progress_bar = progress_bar
        if threads_num is None:
            if system() == 'Windows':
                threads_num = 1
            else:
                threads_num = 4
        else:
            if system() == 'Windows' and threads_num != 1:
                threads_num = 1
                print('Parallel processes not yet implemented on Windows, computing anyway...')
        self.threads_num = threads_num
        if self.adj_list is None:
            print('''
            Without the edges I can't compute the projection. 
            Use set_biadjacency_matrix, set_adjacency_list or set_edgelist to add edges.
            ''')
            return
        else:
            if not self.is_randomized:
                print('First I have to compute the BiCM. Computing...')
                self.solve_tool()
            if rows:
                self.rows_pvals = self._projection_calculator()
                self.projected_rows_adj_list = self._projection_from_pvals(alpha=alpha)
                self.is_rows_projected = True
            else:
                self.cols_pvals = self._projection_calculator()
                self.projected_cols_adj_list = self._projection_from_pvals(alpha=alpha)
                self.is_cols_projected = True

    def _pvals_validator(self, pval_list, alpha=0.05):
        sorted_pvals = np.sort(pval_list)
        if self.rows_projection:
            multiplier = 2 * alpha / (self.n_rows * (self.n_rows - 1))
        else:
            multiplier = 2 * alpha / (self.n_cols * (self.n_cols - 1))
        try:
            eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * alpha * multiplier))[0][-1]
        except:
            print('No V-motifs will be validated. Try increasing alpha')
            eff_fdr_pos = 0
        eff_fdr_th = eff_fdr_pos * multiplier
        return eff_fdr_th

    def _projection_from_pvals(self, alpha=0.05):
        """Internal method to build the projected network from pvalues.

        :param float alpha:  Threshold for the FDR validation.
        """
        pval_list = []
        if self.rows_projection:
            pvals_adj_list = self.rows_pvals
        else:
            pvals_adj_list = self.cols_pvals
        for node in pvals_adj_list:
            for neighbor in pvals_adj_list[node]:
                pval_list.append(pvals_adj_list[node][neighbor])
        eff_fdr_th = self._pvals_validator(pval_list, alpha=alpha)
        projected_adj_list = dict([])
        for node in self.v_adj_list:
            for neighbor in self.v_adj_list[node]:
                if pvals_adj_list[node][neighbor] <= eff_fdr_th:
                    if node not in projected_adj_list.keys():
                        projected_adj_list[node] = []
                    projected_adj_list[node].append(neighbor)
        return projected_adj_list
        #     return np.array([(v[0], v[1]) for v in self.rows_pvals if v[2] <= eff_fdr_th])
        # else:
        #     eff_fdr_th = nef.pvals_validator([v[2] for v in self.cols_pvals], self.n_cols, alpha=alpha)
        #     return np.array([(v[0], v[1]) for v in self.cols_pvals if v[2] <= eff_fdr_th])

    def get_rows_projection(self, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
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
            self.compute_projection(rows=True, alpha=alpha, method=method, threads_num=threads_num,
                                    progress_bar=progress_bar)
        if self.rows_dict is None:
            return self.projected_rows_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_rows_adj_list:
                adj_list_to_return[self.rows_dict[node]] = []
                for neighbor in self.projected_rows_adj_list[node]:
                    adj_list_to_return[self.rows_dict[node]].append(self.rows_dict[neighbor])
            return adj_list_to_return

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
            self.compute_projection(rows=False,
                                    alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.cols_dict is None:
            return self.projected_cols_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_cols_adj_list:
                adj_list_to_return[self.cols_dict[node]] = []
                for neighbor in self.projected_cols_adj_list[node]:
                    adj_list_to_return[self.cols_dict[node]].append(self.cols_dict[neighbor])
            return adj_list_to_return

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

# """
# This module contains the class BipartiteGraph that handles the graph object, and many useful functions.
#
# The solver functions are located here for compatibility with the numba package.
# """
#
# import numpy as np
# import scipy.sparse
# import scipy
# from scipy.stats import norm, poisson
# from . import models_functions as mof
# from . import solver_functions as sof
# from . import network_functions as nef
# # Stops Numba Warning for experimental feature
# from numba.core.errors import NumbaExperimentalFeatureWarning
# from tqdm import tqdm
# import warnings
# from platform import system
# from .poibin import PoiBin
# from functools import partial
# if system() != 'Windows':
#     from multiprocessing import Pool, Manager, Process
#
# warnings.simplefilter(
#     action='ignore',
#     category=NumbaExperimentalFeatureWarning)
#
#
# class BipartiteGraph:
#     """Bipartite Graph class for undirected binary bipartite networks.
#
#     This class handles the bipartite graph object to compute the
#     Bipartite Configuration Model (BiCM), which can be used as a null model
#     for the analysis of undirected and binary bipartite networks.
#     The class provides methods for calculating the probabilities and matrices
#     of the null model and for projecting the bipartite network on its layers.
#     The object can be initialized passing one of the parameters, or the nodes and
#     edges can be passed later.
#
#     :param biadjacency: binary input matrix describing the biadjacency matrix
#             of a bipartite graph with the nodes of one layer along the rows
#             and the nodes of the other layer along the columns.
#     :type biadjacency: numpy.array, scipy.sparse, list, optional
#     :param adjacency_list: dictionary that contains the adjacency list of nodes.
#         The keys are considered as the nodes on the rows layer and the values,
#         to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
#     :type adjacency_list: dict, optional
#     :param edgelist: list of edges containing couples (row_node, col_node) of
#         nodes forming an edge. each element in the couples must belong to
#         the respective layer.
#     :type edgelist: list, numpy.array, optional
#     :param degree_sequences: couple of lists describing the degree sequences
#         of both layers.
#     :type degree_sequences: list, numpy.array, tuple, optional
#     """
#
#     def __init__(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
#         self.n_rows = None
#         self.n_cols = None
#         self.r_n_rows = None
#         self.r_n_cols = None
#         self.shape = None
#         self.n_edges = None
#         self.r_n_edges = None
#         self.n_nodes = None
#         self.r_n_nodes = None
#         self.biadjacency = None
#         self.edgelist = None
#         self.adj_list = None
#         self.inv_adj_list = None
#         self.rows_deg = None
#         self.cols_deg = None
#         self.r_rows_deg = None
#         self.r_cols_deg = None
#         self.rows_dict = None
#         self.cols_dict = None
#         self.is_initialized = False
#         self.is_randomized = False
#         self.is_reduced = False
#         self.rows_projection = None
#         self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist, degree_sequences=degree_sequences)
#         self.avg_mat = None
#         self.x = None
#         self.y = None
#         self.r_x = None
#         self.r_y = None
#         self.r_xy = None
#         self.dict_x = None
#         self.dict_y = None
#         self.theta_x = None
#         self.theta_y = None
#         self.r_theta_x = None
#         self.r_theta_y = None
#         self.r_theta_xy = None
#         self.projected_rows_adj_list = None
#         self.projected_cols_adj_list = None
#         self.projection_method = 'poisson'
#         self.threads_num = 1
#         self.rows_pvals = None
#         self.cols_pvals = None
#         self.is_rows_projected = False
#         self.is_cols_projected = False
#         self.initial_guess = None
#         self.method = None
#         self.rows_multiplicity = None
#         self.cols_multiplicity = None
#         self.r_invert_rows_deg = None
#         self.r_invert_cols_deg = None
#         self.r_dim = None
#         self.verbose = False
#         self.full_return = False
#         self.linsearch = True
#         self.regularise = True
#         self.tol = None
#         self.eps = None
#         self.nonfixed_rows = None
#         self.fixed_rows = None
#         self.full_rows_num = None
#         self.nonfixed_cols = None
#         self.fixed_cols = None
#         self.full_cols_num = None
#         self.J_T = None
#         self.residuals = None
#         self.full_rows_num = None
#         self.full_rows_num = None
#         self.solution_converged = None
#
#     def _initialize_graph(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
#         """
#         Internal method for the initialization of the graph.
#         Use the setter methods instead.
#
#         :param biadjacency: binary input matrix describing the biadjacency matrix
#                 of a bipartite graph with the nodes of one layer along the rows
#                 and the nodes of the other layer along the columns.
#         :type biadjacency: numpy.array, scipy.sparse, list, optional
#         :param adjacency_list: dictionary that contains the adjacency list of nodes.
#             The keys are considered as the nodes on the rows layer and the values,
#             to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
#         :type adjacency_list: dict, optional
#         :param edgelist: list of edges containing couples (row_node, col_node) of
#             nodes forming an edge. each element in the couples must belong to
#             the respective layer.
#         :type edgelist: list, numpy.array, optional
#         :param degree_sequences: couple of lists describing the degree sequences
#             of both layers.
#         :type degree_sequences: list, numpy.array, tuple, optional
#         """
#         if biadjacency is not None:
#             if not isinstance(biadjacency, (list, np.ndarray)) and not scipy.sparse.isspmatrix(biadjacency):
#                 raise TypeError(
#                     'The biadjacency matrix must be passed as a list or numpy array or scipy sparse matrix')
#             else:
#                 if isinstance(biadjacency, list):
#                     self.biadjacency = np.array(biadjacency)
#                 else:
#                     self.biadjacency = biadjacency
#                 if self.biadjacency.shape[0] == self.biadjacency.shape[1]:
#                     print(
#                         'Your matrix is square. Please remember that it is treated as a biadjacency matrix, not an adjacency matrix.')
#                 self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg = \
#                     nef.adjacency_list_from_biadjacency(self.biadjacency)
#                 self.n_rows = len(self.rows_deg)
#                 self.n_cols = len(self.cols_deg)
#                 self._initialize_node_dictionaries()
#                 self.is_initialized = True
#
#         elif edgelist is not None:
#             if not isinstance(edgelist, (list, np.ndarray)):
#                 raise TypeError('The edgelist must be passed as a list or numpy array')
#             elif len(edgelist[0]) != 2:
#                 raise ValueError(
#                     'This is not an edgelist. An edgelist must be a vector of couples of nodes. Try passing a biadjacency matrix')
#             else:
#                 self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
#                     nef.adjacency_list_from_edgelist_bipartite(edgelist)
#                 #                 self.edgelist, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = edgelist_from_edgelist_bipartite(edgelist)
#                 self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
#                 self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
#                 self.is_initialized = True
#
#         elif adjacency_list is not None:
#             if not isinstance(adjacency_list, dict):
#                 raise TypeError('The adjacency list must be passed as a dictionary')
#             else:
#                 self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
#                     nef.adjacency_list_from_adjacency_list_bipartite(adjacency_list)
#                 self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
#                 self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
#                 self.is_initialized = True
#
#         elif degree_sequences is not None:
#             if not isinstance(degree_sequences, (list, np.ndarray, tuple)):
#                 raise TypeError('The degree sequences must be passed as a list, tuple or numpy array')
#             elif len(degree_sequences) != 2:
#                 raise TypeError('degree_sequences must contain two vectors, the two layers degree sequences')
#             elif not isinstance(degree_sequences[0], (list, np.ndarray)) or not isinstance(degree_sequences[1],
#                                                                                            (list, np.ndarray)):
#                 raise TypeError('The two degree sequences must be lists or numpy arrays')
#             elif np.sum(degree_sequences[0]) != np.sum(degree_sequences[1]):
#                 raise ValueError('The two degree sequences must have the same total sum.')
#             elif (np.array(degree_sequences[0]) < 0).sum() + (np.array(degree_sequences[1]) < 0).sum() > 0:
#                 raise ValueError('A degree cannot be negative.')
#             else:
#                 self.rows_deg = degree_sequences[0]
#                 self.cols_deg = degree_sequences[1]
#                 self.n_rows = len(self.rows_deg)
#                 self.n_cols = len(self.cols_deg)
#                 self._initialize_node_dictionaries()
#                 self.is_initialized = True
#
#         if self.is_initialized:
#             self.n_rows = len(self.rows_deg)
#             self.n_cols = len(self.cols_deg)
#             self.shape = [self.n_rows, self.n_cols]
#             self.n_edges = np.sum(self.rows_deg)
#             self.n_nodes = self.n_rows + self.n_cols
#
#     def _initialize_node_dictionaries(self):
#         self.rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
#         self.cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))
#         self.inv_rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
#         self.inv_cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))
#
#     def degree_reduction(self, rows_deg=None, cols_deg=None):
#         """Reduce the degree sequences to contain no repetition of the degrees.
#         The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
#         """
#         if rows_deg is None:
#             rows_deg = self.rows_deg
#         else:
#             cols_deg -= self.full_rows_num
#         if cols_deg is None:
#             cols_deg = self.cols_deg
#         else:
#             rows_deg -= self.full_cols_num
#         self.r_rows_deg, self.r_invert_rows_deg, self.rows_multiplicity \
#             = np.unique(rows_deg, return_index=False, return_inverse=True, return_counts=True)
#         self.r_cols_deg, self.r_invert_cols_deg, self.cols_multiplicity \
#             = np.unique(cols_deg, return_index=False, return_inverse=True, return_counts=True)
#         self.r_n_rows = self.r_rows_deg.size
#         self.r_n_cols = self.r_cols_deg.size
#         self.r_dim = self.r_n_rows + self.r_n_cols
#         self.r_n_edges = np.sum(rows_deg)
#         self.is_reduced = True
#
#     def _set_initial_guess(self):
#         """
#         Internal method to set the initial point of the solver.
#         """
#         if self.initial_guess is None:
#             self.r_x = self.r_rows_deg / (
#                         np.sqrt(self.r_n_edges))  # This +1 increases the stability of the solutions.
#             self.r_y = self.r_cols_deg / (np.sqrt(self.r_n_edges))
#         elif self.initial_guess == 'random':
#             self.r_x = np.random.rand(self.r_n_rows).astype(np.float64)
#             self.r_y = np.random.rand(self.r_n_cols).astype(np.float64)
#         elif self.initial_guess == 'uniform':
#             self.r_x = np.ones(self.r_n_rows, dtype=np.float64)  # All probabilities will be 1/2 initially
#             self.r_y = np.ones(self.r_n_cols, dtype=np.float64)
#         elif self.initial_guess == 'degrees':
#             self.r_x = self.r_rows_deg.astype(np.float64)
#             self.r_y = self.r_cols_deg.astype(np.float64)
#         if not self.exp:
#             self.r_theta_x = - np.log(self.r_x)
#             self.r_theta_y = - np.log(self.r_y)
#             self.x0 = np.concatenate((self.r_theta_x, self.r_theta_y))
#         else:
#             self.x0 = np.concatenate((self.r_x, self.r_y))
#
#     def initialize_avg_mat(self):
#         """
#         Reduces the matrix eliminating empty or full rows or columns.
#         It repeats the process on the so reduced matrix until no more reductions are possible.
#         For instance, a perfectly nested matrix will be reduced until all entries are set to 0 or 1.
#         """
#         self.avg_mat = np.zeros_like(self.biadjacency, dtype=float)
#         r_biad_mat = np.copy(self.biadjacency)
#         rows_num, cols_num = self.biadjacency.shape
#         rows_degs = self.biadjacency.sum(1)
#         cols_degs = self.biadjacency.sum(0)
#         good_rows = np.arange(rows_num)
#         good_cols = np.arange(cols_num)
#         zero_rows = np.where(rows_degs == 0)[0]
#         zero_cols = np.where(cols_degs == 0)[0]
#         full_rows = np.where(rows_degs == cols_num)[0]
#         full_cols = np.where(cols_degs == rows_num)[0]
#         self.full_rows_num = 0
#         self.full_cols_num = 0
#         while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
#             r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), zero_rows), :]
#             r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), zero_cols)]
#             good_rows = np.delete(good_rows, zero_rows)
#             good_cols = np.delete(good_cols, zero_cols)
#             full_rows = np.where(r_biad_mat.sum(1) == r_biad_mat.shape[1])[0]
#             full_cols = np.where(r_biad_mat.sum(0) == r_biad_mat.shape[0])[0]
#             self.full_rows_num += len(full_rows)
#             self.full_cols_num += len(full_cols)
#             self.avg_mat[good_rows[full_rows][:, None], good_cols] = 1
#             self.avg_mat[good_rows[:, None], good_cols[full_cols]] = 1
#             good_rows = np.delete(good_rows, full_rows)
#             good_cols = np.delete(good_cols, full_cols)
#             r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), full_rows), :]
#             r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), full_cols)]
#             zero_rows = np.where(r_biad_mat.sum(1) == 0)[0]
#             zero_cols = np.where(r_biad_mat.sum(0) == 0)[0]
#
#         self.nonfixed_rows = good_rows
#         self.fixed_rows = np.delete(np.arange(rows_num), good_rows)
#         self.nonfixed_cols = good_cols
#         self.fixed_cols = np.delete(np.arange(cols_num), good_cols)
#         return r_biad_mat
#
#     def _initialize_fitnesses(self):
#         """
#         Internal method to initialize the fitnesses of the BiCM.
#         If there are empty rows, the corresponding fitnesses are set to 0,
#         while for full rows the corresponding columns are set to numpy.inf.
#         """
#         if not self.exp:
#             self.theta_x = np.zeros(self.n_rows, dtype=float)
#             self.theta_y = np.zeros(self.n_cols, dtype=float)
#         self.x = np.zeros(self.n_rows, dtype=float)
#         self.y = np.zeros(self.n_cols, dtype=float)
#         good_rows = np.arange(self.n_rows)
#         good_cols = np.arange(self.n_cols)
#         bad_rows = np.array([])
#         bad_cols = np.array([])
#         self.full_rows_num = 0
#         self.full_cols_num = 0
#         if np.any(np.isin(self.rows_deg, (0, self.n_cols))) or np.any(np.isin(self.cols_deg, (0, self.n_rows))):
#             print('''
#                       WARNING: this system has at least a node that is disconnected or connected to all nodes of the opposite layer.
#                       This may cause some convergence issues.
#                       Please use the full mode providing a biadjacency matrix or an edgelist, or clean your data from these nodes.
#                       ''')
#             zero_rows = np.where(self.rows_deg == 0)[0]
#             zero_cols = np.where(self.cols_deg == 0)[0]
#             full_rows = np.where(self.rows_deg == self.n_cols)[0]
#             full_cols = np.where(self.cols_deg == self.n_rows)[0]
#             self.x[full_rows] = np.inf
#             self.y[full_cols] = np.inf
#             if not self.exp:
#                 self.theta_x[full_rows] = - np.inf
#                 self.theta_y[full_rows] = - np.inf
#                 self.theta_x[zero_rows] = np.inf
#                 self.theta_y[zero_cols] = np.inf
#             bad_rows = np.concatenate((zero_rows, full_rows))
#             bad_cols = np.concatenate((zero_cols, full_cols))
#             good_rows = np.delete(np.arange(self.n_rows), bad_rows)
#             good_cols = np.delete(np.arange(self.n_cols), bad_cols)
#             self.full_rows_num += len(full_rows)
#             self.full_cols_num += len(full_cols)
#
#         self.nonfixed_rows = good_rows
#         self.fixed_rows = bad_rows
#         self.nonfixed_cols = good_cols
#         self.fixed_cols = bad_cols
#
#     def _initialize_problem(self, rows_deg=None, cols_deg=None):
#         """
#         Initializes the solver reducing the degree sequences, setting the initial guess and setting the functions for the solver.
#         The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
#         """
#         if ~self.is_reduced:
#             self.degree_reduction(rows_deg=rows_deg, cols_deg=cols_deg)
#         self._set_initial_guess()
#         if self.method == 'root':
#             self.J_T = np.zeros((self.r_dim, self.r_dim), dtype=np.float64)
#             self.residuals = np.zeros(self.r_dim, dtype=np.float64)
#         else:
#             self.args = (self.r_rows_deg, self.r_cols_deg, self.rows_multiplicity, self.cols_multiplicity)
#             d_fun = {
#                 'newton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
#                 'quasinewton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
#                 'fixed-point': lambda x: mof.iterative_bicm(x, self.args),
#                 'newton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
#                 'quasinewton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
#                 'fixed-point_exp': lambda x: mof.iterative_bicm_exp(x, self.args),
#             }
#             d_fun_jac = {
#                 'newton': lambda x: - mof.loglikelihood_hessian_bicm(x, self.args),
#                 'quasinewton': lambda x: - mof.loglikelihood_hessian_diag_bicm(x, self.args),
#                 'fixed-point': None,
#                 'newton_exp': lambda x: - mof.loglikelihood_hessian_bicm_exp(x, self.args),
#                 'quasinewton_exp': lambda x: - mof.loglikelihood_hessian_diag_bicm_exp(x, self.args),
#                 'fixed-point_exp': None,
#             }
#             d_fun_step = {
#                 'newton': lambda x: - mof.loglikelihood_bicm(x, self.args),
#                 'quasinewton': lambda x: - mof.loglikelihood_bicm(x, self.args),
#                 'fixed-point': lambda x: - mof.loglikelihood_bicm(x, self.args),
#                 'newton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
#                 'quasinewton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
#                 'fixed-point_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
#             }
#
#             if self.exp:
#                 self.hessian_regulariser = sof.matrix_regulariser_function_eigen_based
#             else:
#                 self.hessian_regulariser = sof.matrix_regulariser_function
#
#             if self.exp:
#                 lins_args = (mof.loglikelihood_bicm_exp, self.args)
#             else:
#                 lins_args = (mof.loglikelihood_bicm, self.args)
#             lins_fun = {
#                 'newton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
#                 'quasinewton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
#                 'fixed-point': lambda x: mof.linsearch_fun_BiCM_fixed(x),
#                 'newton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
#                 'quasinewton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
#                 'fixed-point_exp': lambda x: mof.linsearch_fun_BiCM_exp_fixed(x),
#             }
#             if self.exp:
#                 method = self.method + '_exp'
#             else:
#                 method = self.method
#             try:
#                 self.fun = d_fun[method]
#                 self.fun_jac = d_fun_jac[method]
#                 self.step_fun = d_fun_step[method]
#                 self.fun_linsearch = lins_fun[method]
#             except (TypeError, KeyError):
#                 raise ValueError('Method must be "newton","quasinewton", "fixed-point" or "root".')
#
#     def _equations_root(self, x):
#         """
#         Equations for the *root* solver
#         """
#         mof.eqs_root(x, self.r_rows_deg, self.r_cols_deg,
#                  self.rows_multiplicity, self.cols_multiplicity,
#                  self.r_n_rows, self.r_n_cols, self.residuals)
#
#     def _jacobian_root(self, x):
#         """
#         Jacobian for the *root* solver
#         """
#         mof.jac_root(x, self.rows_multiplicity, self.cols_multiplicity,
#                  self.r_n_rows, self.r_n_cols, self.J_T)
#
#     def _residuals_jacobian(self, x):
#         """
#         Residuals and jacobian for the *root* solver
#         """
#         self._equations_root(x)
#         self._jacobian_root(x)
#         return self.residuals, self.J_T
#
#     def _clean_root(self):
#         """
#         Clean variables used for the *root* solver
#         """
#         self.J_T = None
#         self.residuals = None
#
#     def _solve_root(self):
#         """
#         Internal *root* solver
#         """
#         x0 = self.x0
#         opz = {'col_deriv': True, 'diag': None}
#         res = scipy.optimize.root(self._residuals_jacobian, x0,
#                                   method='hybr', jac=True, options=opz)
#         self._clean_root()
#         return res.x
#
#     def _check_solution(self, return_error=False, in_place=False):
#         """
#         Check if the solution of the BiCM is compatible with the degree sequences of the graph.
#
#         :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
#         :param bool in_place: check also the error in the single entries of the matrices.
#             Always False unless comparing two different solutions.
#         """
#         if self.biadjacency is not None and self.avg_mat is not None:
#             return nef.check_sol(self.biadjacency, self.avg_mat, return_error=return_error, in_place=in_place)
#         else:
#             return nef.check_sol_light(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols],
#                                    self.rows_deg[self.nonfixed_rows] - self.full_cols_num,
#                                    self.cols_deg[self.nonfixed_cols] - self.full_rows_num,
#                                    return_error=return_error)
#
#     def _set_solved_problem(self, solution):
#         """
#         Sets the solution of the problem.
#
#         :param numpy.ndarray solution: A numpy array containing that reduced fitnesses of the two layers, consecutively.
#         """
#         if not self.exp:
#             if self.theta_x is None:
#                 self.theta_x = np.zeros(self.n_rows)
#             if self.theta_y is None:
#                 self.theta_y = np.zeros(self.n_cols)
#             self.r_theta_xy = solution
#             self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
#             self.r_theta_y = self.r_theta_xy[self.r_n_rows:]
#             self.r_xy = np.exp(- self.r_theta_xy)
#             self.r_x = np.exp(- self.r_theta_x)
#             self.r_y = np.exp(- self.r_theta_y)
#             self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
#             self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
#         else:
#             self.r_xy = solution
#             self.r_x = self.r_xy[:self.r_n_rows]
#             self.r_y = self.r_xy[self.r_n_rows:]
#         if self.x is None:
#             self.x = np.zeros(self.n_rows)
#         if self.y is None:
#             self.y = np.zeros(self.n_cols)
#         self.x[self.nonfixed_rows] = self.r_x[self.r_invert_rows_deg]
#         self.y[self.nonfixed_cols] = self.r_y[self.r_invert_cols_deg]
#         self.dict_x = dict([(self.rows_dict[i], self.x[i]) for i in range(len(self.x))])
#         self.dict_y = dict([(self.cols_dict[j], self.y[j]) for j in range(len(self.y))])
#         if self.full_return:
#             self.comput_time = solution[1]
#             self.n_steps = solution[2]
#             self.norm_seq = solution[3]
#             self.diff_seq = solution[4]
#             self.alfa_seq = solution[5]
#
#     def _solve_bicm_full(self):
#         """
#         Internal method for computing the solution of the BiCM via matrices.
#         """
#         r_biadjacency = self.initialize_avg_mat()
#         if len(r_biadjacency) > 0:  # Every time the matrix is not perfectly nested
#             rows_deg = self.rows_deg[self.nonfixed_rows]
#             cols_deg = self.cols_deg[self.nonfixed_cols]
#             self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
#             if self.method == 'root':
#                 sol = self._solve_root()
#             else:
#                 x0 = self.x0
#                 sol = sof.solver(
#                     x0,
#                     fun=self.fun,
#                     fun_jac=self.fun_jac,
#                     step_fun=self.step_fun,
#                     linsearch_fun=self.fun_linsearch,
#                     hessian_regulariser=self.hessian_regulariser,
#                     tol=self.tol,
#                     eps=self.eps,
#                     max_steps=self.max_steps,
#                     method=self.method,
#                     verbose=self.verbose,
#                     regularise=self.regularise,
#                     full_return=self.full_return,
#                     linsearch=self.linsearch,
#                 )
#             self._set_solved_problem(sol)
#             r_avg_mat = nef.bicm_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
#             self.avg_mat[self.nonfixed_rows[:, None], self.nonfixed_cols] = np.copy(r_avg_mat)
#
#     def _solve_bicm_light(self):
#         """
#         Internal method for computing the solution of the BiCM via degree sequences.
#         """
#         self._initialize_fitnesses()
#         rows_deg = self.rows_deg[self.nonfixed_rows]
#         cols_deg = self.cols_deg[self.nonfixed_cols]
#         self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
#         if self.method == 'root':
#             sol = self._solve_root()
#         else:
#             x0 = self.x0
#             sol = sof.solver(
#                 x0,
#                 fun=self.fun,
#                 fun_jac=self.fun_jac,
#                 step_fun=self.step_fun,
#                 linsearch_fun=self.fun_linsearch,
#                 hessian_regulariser=self.hessian_regulariser,
#                 tol=self.tol,
#                 eps=self.eps,
#                 max_steps=self.max_steps,
#                 method=self.method,
#                 verbose=self.verbose,
#                 regularise=self.regularise,
#                 full_return=self.full_return,
#                 linsearch=self.linsearch,
#             )
#         self._set_solved_problem(sol)
#
#     def _set_parameters(self, method, initial_guess, tol, eps, regularise, max_steps, verbose, linsearch, exp, full_return):
#         """
#         Internal method for setting the parameters of the solver.
#         """
#         self.method = method
#         self.initial_guess = initial_guess
#         self.tol = tol
#         self.eps = eps
#         self.verbose = verbose
#         self.linsearch = linsearch
#         self.regularise = regularise
#         self.exp = exp
#         self.full_return = full_return
#         if max_steps is None:
#             if method == 'fixed-point':
#                 self.max_steps = 200
#             else:
#                 self.max_steps = 100
#         else:
#             self.max_steps = max_steps
#
#     def solve_tool(
#             self,
#             method='newton',
#             initial_guess=None,
#             light_mode=None,
#             tol=1e-8,
#             eps=1e-8,
#             max_steps=None,
#             verbose=False,
#             linsearch=True,
#             regularise=None,
#             print_error=True,
#             full_return=False,
#             exp=False):
#         """Solve the BiCM of the graph.
#         It does not return the solution, use the getter methods instead.
#
#         :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
#             If the graph has been initialized without the matrix, the light mode is used regardless.
#         :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is newton
#         :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*, default is None
#         :param float tol: Tolerance of the solution, optional
#         :param int max_steps: Maximum number of steps, optional
#         :param bool, optional verbose: Print elapsed time, errors and iteration steps, optional
#         :param bool linsearch: Implement the linesearch when searching for roots, default is True
#         :param bool regularise: Regularise the matrices in the computations, optional
#         :param bool print_error: Print the final error of the solution
#         :param bool exp: if this is set to true the solver works with the reparameterization $x_i = e^{-\theta_i}$,
#             $y_\alpha = e^{-\theta_\alpha}$. It might be slightly faster but also might not converge.
#         """
#         if not self.is_initialized:
#             print('Graph is not initialized. I can\'t compute the BiCM.')
#             return
#         if regularise is None:
#             if exp:
#                 regularise = False
#             else:
#                 regularise = True
#         if regularise and exp:
#             print('Warning: regularise is only recommended in non-exp mode.')
#         if method == 'root':
#             exp = True
#         self._set_parameters(method=method, initial_guess=initial_guess, tol=tol, eps=eps, regularise=regularise,
#                              max_steps=max_steps, verbose=verbose, linsearch=linsearch, exp=exp, full_return=full_return)
#         if self.biadjacency is not None and (light_mode is None or not light_mode):
#             self._solve_bicm_full()
#         else:
#             if light_mode is False:
#                 print('''
#                 I cannot work with the full mode without the biadjacency matrix.
#                 This will not account for disconnected or fully connected nodes.
#                 Solving in light mode...
#                 ''')
#             self._solve_bicm_light()
#         if print_error:
#             self.solution_converged = not bool(self._check_solution(return_error=True))
#             if self.solution_converged:
#                 print('Solver converged.')
#             else:
#                 print('Solver did not converge.')
#         self.is_randomized = True
#
#     def get_bicm_matrix(self):
#         """Get the matrix of probabilities of the BiCM.
#         If the BiCM has not been computed, it also computes it with standard settings.
#
#         :returns: The average matrix of the BiCM
#         :rtype: numpy.ndarray
#         """
#         if not self.is_initialized:
#             raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
#         elif not self.is_randomized:
#             self.solve_tool()
#         if self.avg_mat is not None:
#             return self.avg_mat
#         else:
#             self.avg_mat = nef.bicm_from_fitnesses(self.x, self.y)
#             return self.avg_mat
#
#     def get_bicm_fitnesses(self):
#         """Get the fitnesses of the BiCM.
#         If the BiCM has not been computed, it also computes it with standard settings.
#
#         :returns: The fitnesses of the BiCM in the format **rows fitnesses dictionary, columns fitnesses dictionary**
#         """
#         if not self.is_initialized:
#             raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
#         elif not self.is_randomized:
#             print('Computing the BiCM...')
#             self.solve_tool()
#         return self.dict_x, self.dict_y
#
#     def _pval_calculator(self, v_list_key):
#         if self.rows_projection:
#             x = self.x
#             y = self.y
#         else:
#             x = self.y
#             y = self.x
#         node_xy = x[v_list_key] * y
#         temp_pvals_dict = {}
#         for neighbor in self.v_adj_list[v_list_key]:
#             neighbor_xy = x[neighbor] * y
#             probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
#             avg_v = np.sum(probs)
#             if self.projection_method == 'poisson':
#                 temp_pvals_dict[neighbor] = \
#                     poisson.sf(k=self.v_adj_list[v_list_key][neighbor] - 1, mu=avg_v)
#             elif self.projection_method == 'normal':
#                 sigma_v = np.sqrt(np.sum(probs * (1 - probs)))
#                 temp_pvals_dict[neighbor] = \
#                     norm.cdf((self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v)
#             elif self.projection_method == 'rna':
#                 var_v_arr = probs * (1 - probs)
#                 sigma_v = np.sqrt(np.sum(var_v_arr))
#                 gamma_v = (sigma_v ** (-3)) * np.sum(var_v_arr * (1 - 2 * probs))
#                 eval_x = (self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v
#                 pval = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
#                 temp_pvals_dict[neighbor] = max(min(pval, 1), 0)
#         return temp_pvals_dict
#
#     def _pval_calculator_poibin(self, deg_couple, deg_dict, degs):
#         if self.rows_projection:
#             x = self.x
#             y = self.y
#         else:
#             x = self.y
#             y = self.x
#
#         node_xy = x[deg_dict[degs[deg_couple[0]]][0]] * y
#         neighbor_xy = x[deg_dict[degs[deg_couple[1]]][0]] * y
#         probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
#         pb_obj = PoiBin(probs)
#         pval_list = [(node1, node2, pb_obj.pval(int(self.v_adj_list[node1][node2])))
#                      for node1 in deg_dict[degs[deg_couple[0]]]
#                      for node2 in deg_dict[degs[deg_couple[1]]]
#                      if node2 in self.v_adj_list[node1]]
#         return pval_list
#
#     def _projection_calculator(self):
#         if self.rows_projection:
#             adj_list = self.adj_list
#             inv_adj_list = self.inv_adj_list
#         else:
#             adj_list = self.inv_adj_list
#             inv_adj_list = self.adj_list
#
#         print(adj_list)
#         self.v_adj_list = {k: dict() for k in list(adj_list.keys())}
#         for node_i in adj_list:
#             for neighbor in adj_list[node_i]:
#                 for node_j in inv_adj_list[neighbor]:
#                     if node_j != node_i:
#                         self.v_adj_list[node_i][node_j] = self.v_adj_list[node_i].get(node_j, 0) + 1
#         if self.projection_method != 'poibin':
#             v_list_keys = list(self.v_adj_list.keys())
#             pval_adj_list = dict()
#             if self.threads_num > 1:
#                 with Pool(processes=self.threads_num) as pool:
#                     if self.progress_bar:
#                         pvals_dicts = pool.map(self._pval_calculator, tqdm(v_list_keys))
#                     else:
#                         pvals_dicts = pool.map(self._pval_calculator, v_list_keys)
#                 for k_i in range(len(v_list_keys)):
#                     k = v_list_keys[k_i]
#                     pval_adj_list[k] = pvals_dicts[k_i]
#             else:
#                 if self.progress_bar:
#                     for k in tqdm(v_list_keys):
#                         pval_adj_list[k] = self._pval_calculator(k)
#                 else:
#                     for k in v_list_keys:
#                         pval_adj_list[k] = self._pval_calculator(k)
#         else:
#             if self.rows_projection:
#                 degs = self.rows_deg
#             else:
#                 degs = self.cols_deg
#             deg_dict = {k: [] for k in degs}
#             for k_i in range(len(degs)):
#                 deg_dict[degs[k_i]].append(k_i)
#
#             v_list_coupled = []
#             deg_couples = [(deg_i, deg_j) for deg_i in range(len(degs)) for deg_j in range(deg_i, len(degs))]
#             if self.progress_bar:
#                 print('Calculating p-values...')
#             if self.threads_num > 1:
#                 with Pool(processes=self.threads_num) as pool:
#                     if self.progress_bar:
#                         pval_list_coupled = pool.map(partial(self._pval_calculator_poibin, deg_dict, degs),
#                                                      tqdm(deg_couples))
#                     else:
#                         pval_list_coupled = pool.map(partial(self._pval_calculator_poibin, deg_dict, degs),
#                                                      deg_couples)
#             else:
#                 pval_list_coupled = []
#                 if self.progress_bar:
#                     for deg_couple in tqdm(deg_couples):
#                         pval_list_coupled.append(self._pval_calculator_poibin(deg_couple, deg_dict, degs))
#                 else:
#                     for deg_couple in v_list_coupled:
#                         pval_list_coupled.append(self._pval_calculator_poibin(deg_couple, deg_dict, degs))
#
#             # for deg_i in range(len(degs)):
#             #     for deg_j in range(deg_i, len(degs)):
#             #         red_v_list = [(node1, node2, self.v_adj_list[node1][node2])
#             #                       for node1 in deg_dict[degs[deg_i]]
#             #                       for node2 in deg_dict[degs[deg_j]]
#             #                       if node2 in self.v_adj_list[node1]]
#             #         if len(red_v_list) != 0:
#             #             v_list_coupled.append(red_v_list)
#             # if self.progress_bar:
#             #     print('Calculating p-values...')
#             # if self.threads_num > 1:
#             #     with Pool(processes=self.threads_num) as p:
#             #         if self.progress_bar:
#             #             pval_list_coupled = p.map(self._pval_calculator_poibin, tqdm(v_list_coupled))
#             #         else:
#             #             pval_list_coupled = p.map(self._pval_calculator_poibin, v_list_coupled)
#             # else:
#             #     pval_list_coupled = []
#             #     if self.progress_bar:
#             #         for v_couple in tqdm(v_list_coupled):
#             #             pval_list_coupled.append(self._pval_calculator_poibin(v_couple))
#             #     else:
#             #         for v_couple in v_list_coupled:
#             #             pval_list_coupled.append(self._pval_calculator_poibin(v_couple))
#             pval_adj_list = {k: dict() for k in self.v_adj_list}
#             for red_pval_list in pval_list_coupled:
#                 for p_couple in red_pval_list:
#                     pval_adj_list[p_couple[0]][p_couple[1]] = p_couple[2]
#         return pval_adj_list
#
#     def compute_projection(self, rows=True, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
#         """Compute the projection of the network on the rows or columns layer.
#         If the BiCM has not been computed, it also computes it with standard settings.
#         This is the most customizable method for the pvalues computation.
#
#         :param bool rows: True if requesting the rows projection.
#         :param float alpha: Threshold for the FDR validation.
#         :param str method: Method for the approximation of the pvalues computation.
#             Implemented methods are *poisson*, *poibin*, *normal*, *rna*.
#         :param threads_num: Number of threads to use for the parallelization. If it is set to 1,
#             the computation is not parallelized.
#         :param bool progress_bar: Show progress bar of the pvalues computation.
#         """
#         self.rows_projection = rows
#         self.projection_method = method
#         self.progress_bar = progress_bar
#         if threads_num is None:
#             if system() == 'Windows':
#                 threads_num = 1
#             else:
#                 threads_num = 4
#         else:
#             if system() == 'Windows' and threads_num != 1:
#                 threads_num = 1
#                 print('Parallel processes not yet implemented on Windows, computing anyway...')
#         self.threads_num = threads_num
#         if self.adj_list is None:
#             print('''
#             Without the edges I can't compute the projection.
#             Use set_biadjacency_matrix, set_adjacency_list or set_edgelist to add edges.
#             ''')
#             return
#         else:
#             if not self.is_randomized:
#                 print('First I have to compute the BiCM. Computing...')
#                 self.solve_tool()
#             if rows:
#                 self.rows_pvals = None
#                 self.rows_pvals = self._projection_calculator()
#                 self.projected_rows_adj_list = self._projection_from_pvals(alpha=alpha)
#                 self.is_rows_projected = True
#             else:
#                 self.cols_pvals = None
#                 self.cols_pvals = self._projection_calculator()
#                 self.projected_cols_adj_list = self._projection_from_pvals(alpha=alpha)
#                 self.is_cols_projected = True
#
#     def _pvals_validator(self, pval_list, alpha=0.05):
#         sorted_pvals = np.sort(pval_list)
#         if self.rows_projection:
#             multiplier = 2 * alpha / (self.n_rows * (self.n_rows - 1))
#         else:
#             multiplier = 2 * alpha / (self.n_cols * (self.n_cols - 1))
#         try:
#             eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * alpha * multiplier))[0][-1]
#         except:
#             print('No V-motifs will be validated. Try increasing alpha')
#             eff_fdr_pos = 0
#         eff_fdr_th = eff_fdr_pos * multiplier
#         return eff_fdr_th
#
#     def _projection_from_pvals(self, alpha=0.05):
#         """Internal method to build the projected network from pvalues.
#
#         :param float alpha:  Threshold for the FDR validation.
#         """
#         pval_list = []
#         if self.rows_projection:
#             pvals_adj_list = self.rows_pvals
#         else:
#             pvals_adj_list = self.cols_pvals
#         for node in pvals_adj_list:
#             for neighbor in pvals_adj_list[node]:
#                 pval_list.append(pvals_adj_list[node][neighbor])
#         eff_fdr_th = self._pvals_validator(pval_list, alpha=alpha)
#         projected_adj_list = dict([])
#         for node in self.v_adj_list:
#             for neighbor in self.v_adj_list[node]:
#                 if pvals_adj_list[node][neighbor] <= eff_fdr_th:
#                     if node not in projected_adj_list.keys():
#                         projected_adj_list[node] = []
#                     projected_adj_list[node].append(neighbor)
#         return projected_adj_list
#         #     return np.array([(v[0], v[1]) for v in self.rows_pvals if v[2] <= eff_fdr_th])
#         # else:
#         #     eff_fdr_th = nef.pvals_validator([v[2] for v in self.cols_pvals], self.n_cols, alpha=alpha)
#         #     return np.array([(v[0], v[1]) for v in self.cols_pvals if v[2] <= eff_fdr_th])
#
#     def get_rows_projection(self, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
#         """Get the projected network on the rows layer of the graph.
#
#         :param alpha: threshold for the validation of the projected edges.
#         :type alpha: float, optional
#         :param method: approximation method for the calculation of the p-values.a
#             Implemented choices are: poisson, poibin, normal, rna
#         :type method: str, optional
#         :param threads_num: number of threads to use for the parallelization. If it is set to 1,
#             the computation is not parallelized.
#         :type threads_num: int, optional
#         :param bool progress_bar: Show the progress bar
#         :returns: edgelist of the projected network on the rows layer
#         :rtype: numpy.array
#         """
#         if not self.is_rows_projected:
#             self.compute_projection(rows=True, alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
#         if self.rows_dict is None:
#             return self.projected_rows_adj_list
#         else:
#             adj_list_to_return = {}
#             for node in self.projected_rows_adj_list:
#                 adj_list_to_return[self.rows_dict[node]] = []
#                 for neighbor in self.projected_rows_adj_list[node]:
#                     adj_list_to_return[self.rows_dict[node]].append(self.rows_dict[neighbor])
#             return adj_list_to_return
#
#     def get_cols_projection(self, alpha=0.05, method='poisson', threads_num=4, progress_bar=True):
#         """Get the projected network on the columns layer of the graph.
#
#         :param alpha: threshold for the validation of the projected edges.
#         :type alpha: float, optional
#         :param method: approximation method for the calculation of the p-values.
#             Implemented choices are: poisson, poibin, normal, rna
#         :type method: str, optional
#         :param threads_num: number of threads to use for the parallelization. If it is set to 1,
#             the computation is not parallelized.
#         :type threads_num: int, optional
#         :param bool progress_bar: Show the progress bar
#         :returns: edgelist of the projected network on the columns layer
#         :rtype: numpy.array
#         """
#         if not self.is_cols_projected:
#             self.compute_projection(rows=False,
#                                     alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
#         if self.cols_dict is None:
#             return self.projected_cols_adj_list
#         else:
#             adj_list_to_return = {}
#             for node in self.projected_cols_adj_list:
#                 adj_list_to_return[self.cols_dict[node]] = []
#                 for neighbor in self.projected_cols_adj_list[node]:
#                     adj_list_to_return[self.cols_dict[node]].append(self.cols_dict[neighbor])
#             return adj_list_to_return
#
#     def set_biadjacency_matrix(self, biadjacency):
#         """Set the biadjacency matrix of the graph.
#
#         :param biadjacency: binary input matrix describing the biadjacency matrix
#                 of a bipartite graph with the nodes of one layer along the rows
#                 and the nodes of the other layer along the columns.
#         :type biadjacency: numpy.array, scipy.sparse, list
#         """
#         if self.is_initialized:
#             print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
#         else:
#             self._initialize_graph(biadjacency=biadjacency)
#
#     def set_adjacency_list(self, adj_list):
#         """Set the adjacency list of the graph.
#
#         :param adj_list: a dictionary containing the adjacency list
#                 of a bipartite graph with the nodes of one layer as keys
#                 and lists of neighbor nodes of the other layer as values.
#         :type adj_list: dict
#         """
#         if self.is_initialized:
#             print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
#         else:
#             self._initialize_graph(adjacency_list=adj_list)
#
#     def set_edgelist(self, edgelist):
#         """Set the edgelist of the graph.
#
#         :param edgelist: list of edges containing couples (row_node, col_node) of
#             nodes forming an edge. each element in the couples must belong to
#             the respective layer.
#         :type edgelist: list, numpy.array
#         """
#         if self.is_initialized:
#             print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
#         else:
#             self._initialize_graph(edgelist=edgelist)
#
#     def set_degree_sequences(self, degree_sequences):
#         """Set the degree sequence of the graph.
#
#         :param degree_sequences: couple of lists describing the degree sequences
#             of both layers.
#         :type degree_sequences: list, numpy.array, tuple
#         """
#         if self.is_initialized:
#             print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
#         else:
#             self._initialize_graph(degree_sequences=degree_sequences)
#
#     def clean_edges(self):
#         """Clean the edges of the graph.
#         """
#         self.biadjacency = None
#         self.edgelist = None
#         self.adj_list = None
#         self.rows_deg = None
#         self.cols_deg = None
#         self.is_initialized = False
