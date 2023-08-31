"""
This module contains the class BipartiteGraph that handles the graph object, and many useful functions.

The solver functions are located here for compatibility with the numba package.
"""

import numpy as np
import scipy.sparse
import scipy
from scipy.stats import norm, poisson
import bicm.models_functions as mof
import bicm.solver_functions as sof
import bicm.network_functions as nef
from tqdm import tqdm
import bicm.poibin as pb
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
        self.avg_mat = None
        self.x = None
        self.y = None
        self.r_x = None
        self.r_y = None
        self.solution_array = None
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
        self.rows_pvals_mat = None
        self.cols_pvals_mat = None
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
        self.solution_converged = None
        self.loglikelihood = None
        self.progress_bar = None
        self.weighted = False
        self.continuous_weights = False
        self.rows_seq = None
        self.cols_seq = None
        self.r_rows_seq = None
        self.r_cols_seq = None
        self.pvals_mat = None
        self.exp = False
        self.error = None
        self.sparse = None
        self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist,
                               degree_sequences=degree_sequences)

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
            if not isinstance(biadjacency, (list, np.ndarray)):
                if scipy.sparse.isspmatrix(biadjacency):
                    self.sparse = True
                else:
                    raise TypeError(
                        'The biadjacency matrix must be passed as a list or numpy array or scipy sparse matrix')
            else:
                self.sparse = False
            if isinstance(biadjacency, list):
                self.biadjacency = np.array(biadjacency)
            else:
                self.biadjacency = biadjacency
            if self.biadjacency.shape[0] == self.biadjacency.shape[1]:
                print(
                    'Your matrix is square. Please remember that it \
                    is treated as a biadjacency matrix, not an adjacency matrix.')
            if self.sparse:
                self.continuous_weights = not np.all(np.equal(np.mod(self.biadjacency.data, 1), 0))
            else:
                self.continuous_weights = not np.all(np.equal(np.mod(self.biadjacency, 1), 0))
            if self.continuous_weights or np.max(self.biadjacency) > 1:
                self.weighted = True
                self.rows_seq = self.biadjacency.sum(1)
                self.cols_seq = self.biadjacency.sum(0)
                self.rows_deg = (self.biadjacency != 0).sum(1)
                self.cols_deg = (self.biadjacency != 0).sum(0)
                if self.continuous_weights:
                    print('Continuous weighted model: BiWCM_c')
                else:
                    print('Discrete weighted model: BiWCM_d')
            else:
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
                    'This is not an edgelist. An edgelist must be a vector of couples of nodes. Try passing a '
                    'biadjacency matrix')
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
        if self.weighted:
            self.r_rows_seq = np.copy(self.r_rows_deg)
            self.r_cols_seq = np.copy(self.r_cols_deg)
        self.r_n_rows = self.r_rows_deg.size
        self.r_n_cols = self.r_cols_deg.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.r_n_edges = np.sum(rows_deg)
        self.is_reduced = True

    def _set_initial_guess(self):
        """
        Internal method to set the initial point of the solver.
        """
        if self.initial_guess is None or self.initial_guess == 'chung_lu':  # Chung-Lu approximation
            self.r_x = self.r_rows_deg / (np.sqrt(self.r_n_edges))
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
        elif isinstance(self.initial_guess, (tuple, list, np.ndarray)):
            if len(self.initial_guess) == 2 and type(self.initial_guess[0]) in [tuple, list, np.ndarray]:
                if len(self.initial_guess[0]) == self.r_n_rows and len(self.initial_guess[1]) == self.r_n_cols:
                    self.r_x = np.array(self.initial_guess[0])
                    self.r_y = np.array(self.initial_guess[1])
                elif len(self.initial_guess[0]) == self.n_rows and len(self.initial_guess[1]) == self.n_cols:
                    self.r_x[self.r_invert_rows_deg] = self.initial_guess[0][self.nonfixed_rows]
                    self.r_y[self.r_invert_cols_deg] = self.initial_guess[1][self.nonfixed_cols]
                else:
                    raise ValueError('The size of the given initial condition is not the same of the input graph')
            else:
                if len(self.initial_guess) == self.r_n_rows + self.r_n_cols:
                    self.r_x = np.array(self.initial_guess[:self.r_n_rows])
                    self.r_y = np.array(self.initial_guess[self.r_n_rows:])
                elif len(self.initial_guess[0]) == self.n_rows and len(self.initial_guess[1]) == self.n_cols:
                    self.r_x[self.r_invert_rows_deg] = self.initial_guess[:self.r_n_rows][self.nonfixed_rows]
                    self.r_y[self.r_invert_cols_deg] = self.initial_guess[self.r_n_rows:][self.nonfixed_cols]
                else:
                    raise ValueError('The size of the given initial condition is not the same of the input graph')
        else:
            raise ValueError('Initial_guess must be None, "chung_lu", "random", "uniform", "degrees" or an array or a tuple of 2 lists/numpy arrays')
        if not self.exp:
            if self.weighted:  # Avoid negative thetas
                normalization = max(np.max(self.r_x), np.max(self.r_y))
                if normalization >= 1:
                    self.r_x /= 2 * normalization
                    self.r_y /= 2 * normalization
            self.r_theta_x = - np.log(self.r_x)
            self.r_theta_y = - np.log(self.r_y)
            self.x0 = np.concatenate((self.r_theta_x, self.r_theta_y))
            # if self.weighted: # Avoid thetas' products = 1
            #     self.x0 /= 2 * np.max(np.abs(self.x0))
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
        if not self.weighted:
            full_rows = np.where(rows_degs == cols_num)[0]
            full_cols = np.where(cols_degs == rows_num)[0]
        else:
            full_rows = np.array([])
            full_cols = np.array([])
        self.full_rows_num = 0
        self.full_cols_num = 0
        while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), zero_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), zero_cols)]
            good_rows = np.delete(good_rows, zero_rows)
            good_cols = np.delete(good_cols, zero_cols)
            if not self.weighted:
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
                      WARNING: this system has at least a node that is disconnected or connected to all nodes
                       of the opposite layer. This may cause some convergence issues.
                      Please use the full mode providing a biadjacency matrix or an edgelist,
                       or clean your data from these nodes. 
                      ''')
            zero_rows = np.where(self.rows_deg == 0)[0]
            zero_cols = np.where(self.cols_deg == 0)[0]
            if not self.exp:
                self.theta_x[zero_rows] = np.inf
                self.theta_y[zero_cols] = np.inf
            if not self.weighted:
                full_rows = np.where(self.rows_deg == self.n_cols)[0]
                full_cols = np.where(self.cols_deg == self.n_rows)[0]
                self.x[full_rows] = np.inf
                self.y[full_cols] = np.inf
                if not self.exp:
                    self.theta_x[full_rows] = - np.inf
                    self.theta_y[full_rows] = - np.inf
            else:
                full_rows = np.array([])
                full_cols = np.array([])
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
            self.degree_reduction(rows_deg=rows_deg, cols_deg=cols_deg)  # if weighted rows_deg=rows_seq
        self._set_initial_guess()
        if self.method == 'root':
            self.J_T = np.zeros((self.r_dim, self.r_dim), dtype=np.float64)
            self.residuals = np.zeros(self.r_dim, dtype=np.float64)
        else:
            self.args = (self.r_rows_deg, self.r_cols_deg, self.rows_multiplicity, self.cols_multiplicity)
            if self.continuous_weights:
                d_fun = {
                    'newton': lambda x: - mof.loglikelihood_prime_biwcm_c(x, self.args),
                    'quasinewton': lambda x: - mof.loglikelihood_prime_biwcm_c(x, self.args),
                    'fixed-point': lambda x: mof.iterative_biwcm_c(x, self.args),
                    'newton_exp': lambda x: - mof.loglikelihood_prime_biwcm_c_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_prime_biwcm_c_exp(x, self.args),
                    'fixed-point_exp': lambda x: mof.iterative_biwcm_c_exp(x, self.args),
                }
                d_fun_jac = {
                    'newton': lambda x: - mof.loglikelihood_hessian_biwcm_c(x, self.args),
                    'quasinewton': lambda x: - mof.loglikelihood_hessian_diag_biwcm_c(x, self.args),
                    'fixed-point': None,
                    'newton_exp': lambda x: - mof.loglikelihood_hessian_biwcm_c_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_hessian_diag_biwcm_c_exp(x, self.args),
                    'fixed-point_exp': None,
                }
                d_fun_step = {
                    # 'newton': lambda x: - mof.loglikelihood_biwcm_c(x, self.args),
                    # 'quasinewton': lambda x: - mof.loglikelihood_biwcm_c(x, self.args),
                    # 'fixed-point': lambda x: - mof.loglikelihood_biwcm_c(x, self.args),
                    'newton': lambda x: mof.mrse_biwcm_c(x, self.args),
                    'quasinewton': lambda x: mof.mrse_biwcm_c(x, self.args),
                    'fixed-point': lambda x: mof.mrse_biwcm_c(x, self.args),
                    'newton_exp': lambda x: - mof.loglikelihood_biwcm_c_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_biwcm_c_exp(x, self.args),
                    'fixed-point_exp': lambda x: - mof.loglikelihood_biwcm_c_exp(x, self.args),
                }
            elif self.weighted:
                d_fun = {
                    'newton': lambda x: - mof.loglikelihood_prime_biwcm_d(x, self.args),
                    'quasinewton': lambda x: - mof.loglikelihood_prime_biwcm_d(x, self.args),
                    'fixed-point': lambda x: mof.iterative_biwcm_d(x, self.args),
                    'newton_exp': lambda x: - mof.loglikelihood_prime_biwcm_d_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_prime_biwcm_d_exp(x, self.args),
                    'fixed-point_exp': lambda x: mof.iterative_biwcm_d_exp(x, self.args),
                }
                d_fun_jac = {
                    'newton': lambda x: - mof.loglikelihood_hessian_biwcm_d(x, self.args),
                    'quasinewton': lambda x: - mof.loglikelihood_hessian_diag_biwcm_d(x, self.args),
                    'fixed-point': None,
                    'newton_exp': lambda x: - mof.loglikelihood_hessian_biwcm_d_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_hessian_diag_biwcm_d_exp(x, self.args),
                    'fixed-point_exp': None,
                }
                d_fun_step = {
                    # 'newton': lambda x: - mof.loglikelihood_biwcm_d(x, self.args),
                    # 'quasinewton': lambda x: - mof.loglikelihood_biwcm_d(x, self.args),
                    # 'fixed-point': lambda x: - mof.loglikelihood_biwcm_d(x, self.args),
                    'newton': lambda x: mof.mrse_biwcm_d(x, self.args),
                    'quasinewton': lambda x: mof.mrse_biwcm_d(x, self.args),
                    'fixed-point': lambda x: mof.mrse_biwcm_d(x, self.args),
                    'newton_exp': lambda x: - mof.loglikelihood_biwcm_d_exp(x, self.args),
                    'quasinewton_exp': lambda x: - mof.loglikelihood_biwcm_d_exp(x, self.args),
                    'fixed-point_exp': lambda x: - mof.loglikelihood_biwcm_d_exp(x, self.args),
                }
            else:
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
                method = self.method + '_exp'
            else:
                method = self.method
            
            # lins_args = (d_fun_step[method], self.args)
            if self.continuous_weights:
                if self.exp:
                    lins_args = (mof.loglikelihood_biwcm_c_exp, self.args)
                else:
                    lins_args = (mof.loglikelihood_biwcm_c, self.args)
            elif self.weighted:
                if self.exp:
                    lins_args = (mof.loglikelihood_biwcm_d_exp, self.args)
                else:
                    lins_args = (mof.loglikelihood_biwcm_d, self.args)
            else:
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

    @staticmethod
    def check_sol(biad_mat, avg_bicm, return_error=False, in_place=False, mrse=None):
        """
        Static method.
        This function prints the rows sums differences between two matrices,
        that originally are the biadjacency matrix and its bicm average matrix.
        The intended use of this is to check if an average matrix is actually a solution
         for a bipartite configuration model.

        If return_error is set to True, it returns 1 if the sum of the differences is bigger than 1.

        If in_place is set to True, it checks and sums also the total error entry by entry.
        The intended use of this is to check if two solutions are the same solution.
        """
        error = 0
        weighted = np.max(biad_mat) > 1
        if np.any(avg_bicm < 0):
            print('Negative probabilities in the average matrix!')
            error = 1
        if not weighted:
            if np.any(avg_bicm > 1):
                print('Probabilities greater than 1 in the average matrix!')
                error = 1
        if mrse is None:
            if weighted:
                mrse = True
            else:
                mrse = False
        rows_sums = np.sum(biad_mat, axis=1)
        cols_sums = np.sum(biad_mat, axis=0)
        if mrse:
            rows_error_vec = np.zeros(len(rows_sums))
            cols_error_vec = np.zeros(len(cols_sums))
            nonzero_rows = np.where(rows_sums != 0)[0]
            nonzero_cols = np.where(cols_sums != 0)[0]
            rows_error_vec[nonzero_rows] = \
                np.abs(rows_sums - np.sum(avg_bicm, axis=1))[nonzero_rows] / rows_sums[nonzero_rows]
            cols_error_vec[nonzero_cols] = \
                np.abs(cols_sums - np.sum(avg_bicm, axis=0))[nonzero_cols] / cols_sums[nonzero_cols]
        else:
            rows_error_vec = np.abs(rows_sums - np.sum(avg_bicm, axis=1))
            cols_error_vec = np.abs(cols_sums - np.sum(avg_bicm, axis=0))    
        err_rows = np.max(rows_error_vec)
        err_cols = np.max(cols_error_vec)
        print('max rows error =', err_rows)
        print('max columns error =', err_cols)
        tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
        print('total error =', tot_err)
        max_err = np.max([err_rows, err_cols])
        if max_err > 1e-3:
            # error = 1
            print('WARNING max error > 0.001')
            # if tot_err > 10:
            #     print('total error > 10')
        # if err_rows + err_cols > 1:
        #     print('max error > 1')
        #     error = 1
        #     if err_rows + err_cols > 10:
        #         print('max error > 10')
        if in_place:
            diff_mat = np.abs(biad_mat - avg_bicm)
            print('In-place total error:', np.sum(diff_mat))
            print('In-place max error:', np.max(diff_mat))
        if return_error:
            if error == 1:
                return error
            else:
                return max_err
        else:
            return

    def check_sol_light(self, return_error=False):
        """
        Light version of the check_sol function, working only on the fitnesses and the degree sequences.
        """
        error = 0
        exp_rows_sum = np.zeros(self.r_n_rows)
        exp_cols_sum = np.zeros(self.r_n_cols)
    
        for i in range(self.r_n_rows):
            for j in range(self.r_n_cols):
                if self.continuous_weights:
                    multiplier = 1 / (self.r_theta_x[i] + self.r_theta_y[j])
                elif self.weighted:
                    xy = self.r_x[i] * self.r_y[j]
                    multiplier = xy / (1 - xy)
                else:
                    xy = self.r_x[i] * self.r_y[j]
                    multiplier = xy / (1 + xy)
                exp_rows_sum[i] += self.cols_multiplicity[j] * multiplier
                exp_cols_sum[j] += self.rows_multiplicity[i] * multiplier
                if error == 0:
                    if multiplier < 0:
                        print('Warning: negative link probabilities')
                        error = 1
                    if not self.weighted:
                        if multiplier > 1:
                            print('Warning: link probabilities > 1')
                            error = 1
        if self.weighted:
            # use mrse
            rows_error_vec = np.abs((exp_rows_sum - self.r_rows_seq) / self.r_rows_seq)
            cols_error_vec = np.abs((exp_cols_sum - self.r_cols_seq) / self.r_cols_seq)
        else:
            # use made
            rows_error_vec = np.abs((exp_rows_sum - self.r_rows_deg))
            cols_error_vec = np.abs((exp_cols_sum - self.r_cols_deg))
        err_rows = np.max(rows_error_vec)
        print('max rows error =', err_rows)
        err_cols = np.max(cols_error_vec)
        print('max columns error =', err_cols)
        tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
        print('total error =', tot_err)
        max_err = np.max([err_rows, err_cols])
        if max_err > 1e-3:
            # error = 1
            print('WARNING max error > 0.001')
            # if tot_err > 10:
            #     print('total error > 10')
        # if err_rows + err_cols > 1:
        #     print('max error > 1')
        #     error = 1
        #     if err_rows + err_cols > 10:
        #         print('max error > 10')
        if return_error:
            if error == 1:
                return error
            else:
                return max_err
        else:
            return

    def _check_solution(self, return_error=False, in_place=False):
        """
        Check if the solution of the BiCM is compatible with the degree sequences of the graph.

        :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
        :param bool in_place: check also the error in the single entries of the matrices.
            Always False unless comparing two different solutions.
        """
        if self.biadjacency is not None and self.avg_mat is not None and not self.sparse:
            return self.check_sol(self.biadjacency, self.avg_mat, return_error=return_error, in_place=in_place)
        else:
            return self.check_sol_light(return_error=return_error)

    def _set_solved_problem(self, solution):
        """
        Sets the solution of the problem.

        :param numpy.ndarray solution: A numpy array containing that reduced fitnesses of the two layers, consecutively.
        """
        if not self.exp:
            if self.theta_x is None:
                self.theta_x = np.zeros(self.n_rows)
                self.theta_x[:] = np.inf
            if self.theta_y is None:
                self.theta_y = np.zeros(self.n_cols)
                self.theta_y[:] = np.inf
            self.r_theta_xy = solution
            self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
            self.r_theta_y = self.r_theta_xy[self.r_n_rows:]
            self.solution_array = np.exp(- self.r_theta_xy)
            self.r_x = np.exp(- self.r_theta_x)
            self.r_y = np.exp(- self.r_theta_y)
            self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
            self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
        else:
            self.solution_array = solution
            self.r_x = self.solution_array[:self.r_n_rows]
            self.r_y = self.solution_array[self.r_n_rows:]
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
        if self.method != 'root':
            if self.continuous_weights:
                if self.exp:
                    self.loglikelihood = mof.loglikelihood_biwcm_c_exp(self.solution_array, self.args)
                else:
                    self.loglikelihood = mof.loglikelihood_biwcm_c(self.solution_array, self.args)
            elif self.weighted:
                if self.exp:
                    self.loglikelihood = mof.loglikelihood_biwcm_d_exp(self.solution_array, self.args)
                else:
                    self.loglikelihood = mof.loglikelihood_biwcm_d(self.solution_array, self.args)
            else:
                self.loglikelihood = self.step_fun(self.solution_array)
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
            if self.weighted:
                rows_seq = self.rows_seq[self.nonfixed_rows]
                cols_seq = self.cols_seq[self.nonfixed_cols]
                self._initialize_problem(rows_deg=rows_seq, cols_deg=cols_seq)
            else:
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
            if self.continuous_weights:
                r_avg_mat = nef.biwcm_c_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
            elif self.weighted:
                r_avg_mat = nef.biwcm_d_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
            else:
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
        if tol is None:
            self.tol = 1e-8
            if self.weighted:
                self.tol *= 1e-10
        else:
            self.tol = tol
        if eps is None:
            self.eps = 1e-8
            if self.weighted:
                self.eps *= 1e-10
        else:
            self.eps = tol
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
            if self.weighted:
                self.max_steps *= 1000
        else:
            self.max_steps = max_steps

    def solve_tool(
            self,
            method=None,
            initial_guess=None,
            light_mode=None,
            tol=None,
            eps=None,
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=None,
            print_error=True,
            full_return=False,
            exp=False,
            model=None):
        """Solve the BiCM of the graph.
        It does not return the solution, use the getter methods instead.

        :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
            If the graph has been initialized without the matrix or in sparse mode, the light mode is used regardless.
        :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is set by the model
        solved
        :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*,
        default is None
        :param float tol: Tolerance of the solution, optional
        :param float eps: Tolerance of the difference between consecutive solutions, optional
        :param int max_steps: Maximum number of steps, optional
        :param bool, optional verbose: Print elapsed time, errors and iteration steps, optional
        :param bool linsearch: Implement the linesearch when searching for roots, default is True
        :param bool regularise: Regularise the matrices in the computations, optional
        :param bool print_error: Print the final error of the solution
        :param bool full_return: If True, the solver returns some more insights on the convergence. Default False.
        :param bool exp: if this is set to true the solver works with the reparameterization $x_i = e^{-\theta_i}$,
            $y_\alpha = e^{-\theta_\alpha}$. It might be slightly faster but also might not converge.
        :param str model: Model to be used, to be passed only if the user wants to use a different model
        than the recognized one.
        """
        if model == 'biwcm_c':
            self.continuous_weights = True
            self.weighted = True
        elif model == 'biwcm_d':
            self.continuous_weights = False
            self.weighted = True
        elif model == 'bicm':
            self.weighted = False
            self.continuous_weights = False
        if method is None:
            if self.continuous_weights:
                method = 'fixed-point'
            # elif self.weighted:
            #     method = 'quasinewton'
            else:
                method = 'newton'
        if not self.is_initialized:
            print('Graph is not initialized. I can\'t compute the BiCM.')
            return
        if regularise is None:
            if self.weighted or exp:
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
        if self.biadjacency is not None and not self.sparse and (light_mode is None or not light_mode):
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
            max_err = self._check_solution(return_error=True)
            if max_err >= 0.001:
                self.solution_converged = False
            else:
                self.solution_converged = True
            if self.solution_converged:
                print('Solver converged.')
            else:
                if max_err >= 1:
                    print('Solver did not converge: error', max_err)
                else:
                    print('Solver converged, the maximum relative error is {:.2f}%'.format(max_err * 100))
            self.error = max_err
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
        print('solve_bicm has been deprecated, calling solve_tool instead')
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

    def get_fitnesses(self):
        """See get_bicm_fitnesses."""
        self.get_bicm_fitnesses()

    def pval_calculator(self, v_list_key, x, y):
        """
        Calculate the p-values of the v-motifs numbers of one vertex and all its neighbours.

        :param int v_list_key: the key of the node to consider for the adjacency list of the v-motifs.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a dictionary containing as keys the nodes that form v-motifs with the
        considered node, and as values the corresponding p-values.
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

    def pval_calculator_poibin(self, deg_couple, deg_dict, x, y):
        """
        Calculate the p-values of the v-motifs numbers of all nodes with a given couple degrees.

        :param tuple deg_couple: the couple of degrees considered.
        :param dict deg_dict: node-degrees dictionary.
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
                    partial_function = partial(self.pval_calculator_poibin, deg_dict=deg_dict, x=x, y=y)
                    if self.progress_bar:
                        pvals_dicts = pool.map(partial_function, tqdm(deg_couples))
                    else:
                        pvals_dicts = pool.map(partial_function, deg_couples)
            else:
                pvals_dicts = []
                if self.progress_bar:
                    for deg_couple in tqdm(deg_couples):
                        pvals_dicts.append(
                            self.pval_calculator_poibin(deg_couple, deg_dict=deg_dict, x=x, y=y))
                else:
                    for deg_couple in v_list_coupled:
                        pvals_dicts.append(
                            self.pval_calculator_poibin(deg_couple, deg_dict=deg_dict, x=x, y=y))
            pval_adj_list = {k: dict() for k in self.v_adj_list}
            for pvals_dict in pvals_dicts:
                for node in pvals_dict:
                    pval_adj_list[node].update(pvals_dict[node])
        return pval_adj_list

    def compute_projection(self, rows=True, alpha=0.05, approx_method=None, method=None,
                           threads_num=None, progress_bar=True, validation_method='fdr'):
        """Compute the projection of the network on the rows or columns layer.
        If the BiCM has not been computed, it also computes it with standard settings.
        This is the most customizable method for the pvalues computation.

        :param bool rows: True if requesting the rows' projection.
        :param float alpha: Threshold for the p-values validation.
        :param str approx_method: Method for the approximation of the pvalues computation.
            Implemented methods are *poisson*, *poibin*, *normal*, *rna*.
        :param str method: Deprecated, same as approx_method.
        :param threads_num: Number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :param bool progress_bar: Show progress bar of the pvalues computation.
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if approx_method is None:
            if method is not None:
                print('"method" is deprecated, use approx_method instead')
                approx_method = method
            else:
                approx_method = 'poisson'
        self.rows_projection = rows
        self.projection_method = approx_method
        self.progress_bar = progress_bar
        if self.weighted:
            print('Weighted projection not yet implemented.')
            return
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
        if self.adj_list is None and self.biadjacency is None:
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
                self.projected_rows_adj_list = self._projection_from_pvals(alpha=alpha,
                                                                           validation_method=validation_method)
                self.is_rows_projected = True
            else:
                self.cols_pvals = self._projection_calculator()
                self.projected_cols_adj_list = self._projection_from_pvals(alpha=alpha,
                                                                           validation_method=validation_method)
                self.is_cols_projected = True
                
    def compute_weighted_pvals_mat(self):
        """
        Compute the pvalues matrix representing the significance of the original matrix.
        """
        if self.biadjacency is None:
            assert self.edgelist is not None or self.adj_list is not None, 'Graph links must be given in some format!'
            print('Computing biadjacency matrix...')
            if self.edgelist is not None:
                self.biadjacency = nef.biadjacency_from_edgelist(self.edgelist)[0]
            elif self.adj_list is not None:
                self.biadjacency = nef.biadjacency_from_adjacency_list(self.adj_list)
        if not self.is_randomized:
            print('First I have to compute the BiCM. Computing...')
            self.solve_tool()
        if not self.weighted:
            self.pvals_mat = self.avg_mat ** self.biadjacency
        self.pvals_mat = (np.tile(self.x, (len(self.y), 1)).T * np.tile(self.y, (len(self.x), 1))) ** self.biadjacency

    def get_weighted_pvals_mat(self):
        """
        Return the pvalues matrix representing the significance of the original matrix.
        """
        if self.pvals_mat is None:
            self.compute_weighted_pvals_mat()
        return self.pvals_mat

    def get_validated_matrix(self, significance=0.01, validation_method=None):
        """
        Extract a backbone of the original network keeping only the most significant links.
        At the moment this method only applies a global significance level for any link.

        :param float significance:  Threshold for the pvalues significance.
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if self.pvals_mat is None:
            self.compute_weighted_pvals_mat()
        if validation_method is None:
            print('FDR validated matrix will be returned by default. '
                  'Set validation_method to "global" or "bonferroni" for alternatives or use .get_weighted_pvals_mat.')
            validation_method = 'fdr'
        assert validation_method in ['bonferroni', 'fdr', 'global'], 'validation_method must be a valid string'
        pvals_array = self.pvals_mat.flatten()
        val_threshold = self._pvals_validator(pvals_array, alpha=significance, validation_method=validation_method)
        return (self.pvals_mat < val_threshold).astype(np.ubyte)

    def _pvals_validator(self, pval_list, alpha=0.05, validation_method='fdr'):
        sorted_pvals = np.sort(pval_list)
        if self.rows_projection:
            multiplier = 2 * alpha / (self.n_rows * (self.n_rows - 1))
        else:
            multiplier = 2 * alpha / (self.n_cols * (self.n_cols - 1))
        eff_fdr_th = alpha
        if validation_method == 'bonferroni':
            eff_fdr_th = multiplier
            if sorted_pvals[0] > eff_fdr_th:
                print('No V-motifs will be validated. Try increasing alpha')
        elif validation_method == 'fdr':
            try:
                eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * alpha * multiplier))[0][-1]
            except IndexError:
                print('No V-motifs will be validated. Try increasing alpha')
                eff_fdr_pos = 0
            eff_fdr_th = (eff_fdr_pos + 1) * multiplier  # +1 because of Python numbering: our pvals are ordered 1,...,n
        return eff_fdr_th

    def _projection_from_pvals(self, alpha=0.05, validation_method='fdr'):
        """Internal method to build the projected network from pvalues.

        :param float alpha:  Threshold for the validation.
        :param str validation_method:  The type of validation to apply.
        """
        pval_list = []
        if self.rows_projection:
            pvals_adj_list = self.rows_pvals
        else:
            pvals_adj_list = self.cols_pvals
        for node in pvals_adj_list:
            for neighbor in pvals_adj_list[node]:
                pval_list.append(pvals_adj_list[node][neighbor])
        eff_fdr_th = self._pvals_validator(pval_list, alpha=alpha, validation_method=validation_method)
        projected_adj_list = dict([])
        for node in self.v_adj_list:
            for neighbor in self.v_adj_list[node]:
                if pvals_adj_list[node][neighbor] <= eff_fdr_th:
                    if node not in projected_adj_list.keys():
                        projected_adj_list[node] = []
                    projected_adj_list[node].append(neighbor)
        return projected_adj_list

    def get_rows_projection(self,
                            alpha=0.05,
                            method='poisson',
                            threads_num=None,
                            progress_bar=True,
                            fmt='adjacency_list',
                            validation_method='fdr'):
        """Get the projected network on the rows layer of the graph.

        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :param str fmt: the desired format for the output: adjacency_list (default) or edgelist
        :returns: the projected network on the rows layer, in the format specified by fmt
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if not self.is_rows_projected:
            self.compute_projection(rows=True, alpha=alpha, approx_method=method, threads_num=threads_num,
                                    progress_bar=progress_bar, validation_method=validation_method)

        if self.rows_dict is None:
            adj_list_to_return = self.projected_rows_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_rows_adj_list:
                adj_list_to_return[self.rows_dict[node]] = []
                for neighbor in self.projected_rows_adj_list[node]:
                    adj_list_to_return[self.rows_dict[node]].append(self.rows_dict[neighbor])
        if fmt == 'adjacency_list':
            return adj_list_to_return
        elif fmt == 'edgelist':
            return nef.edgelist_from_adjacency_list_bipartite(adj_list_to_return)

    def get_cols_projection(self,
                            alpha=0.05,
                            method='poisson',
                            threads_num=None,
                            progress_bar=True,
                            fmt='adjacency_list'):
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
        :param str fmt: the desired format for the output: adjacency_list (default) or edgelist
        :returns: the projected network on the columns layer, in the format specified by fmt
        """
        if not self.is_cols_projected:
            self.compute_projection(rows=False,
                                    alpha=alpha, approx_method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.cols_dict is None:
            return self.projected_cols_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_cols_adj_list:
                adj_list_to_return[self.cols_dict[node]] = []
                for neighbor in self.projected_cols_adj_list[node]:
                    adj_list_to_return[self.cols_dict[node]].append(self.cols_dict[neighbor])
        if fmt == 'adjacency_list':
            return adj_list_to_return
        elif fmt == 'edgelist':
            return nef.edgelist_from_adjacency_list_bipartite(adj_list_to_return)
        
    def _compute_projected_pvals_mat(self, layer='rows'):
        """
        Compute the pvalues matrix representing the significance of the original matrix.
        """
        if layer == 'rows':
            n_dim = self.n_rows
            pval_adjlist = self.rows_pvals
        elif layer == 'columns':
            n_dim = self.n_cols
            pval_adjlist = self.cols_pvals
        pval_mat = np.ones((n_dim, n_dim))
        for v in pval_adjlist:
            for w in pval_adjlist[v]:
                pval_mat[v,w] = pval_adjlist[v][w]
                pval_mat[w,v] = pval_mat[v,w]
        if layer == 'rows':
            self.rows_pvals_mat = pval_mat
        elif layer == 'columns':
            self.cols_pvals_mat = pval_mat

    def get_projected_pvals_mat(self, layer=None):
        """
        Return the pvalues matrix of the projection if it has been computed.
        
        :param layer: the layer to return.
        :type layer: string, optional
        """
        if layer is None:
            if not self.is_rows_projected and not self.is_cols_projected:
                print('First compute a projection.')
                return None
            elif self.is_rows_projected and self.is_cols_projected:
                print("Please specify the layer with layer='rows' or layer='columns'.")
                return None
            elif self.is_rows_projected:
                layer = 'rows'
            elif self.is_cols_projected:
                layer = 'columns'
        if layer == 'rows':
            if self.rows_pvals_mat is None:
                self._compute_projected_pvals_mat(layer=layer)
            return self.rows_pvals_mat
        elif layer == 'columns':
            if self.cols_pvals_mat is None:
                self._compute_projected_pvals_mat(layer=layer)
            return self.cols_pvals_mat
        else:
            raise ValueError("layer must be either 'rows' or 'columns' or None")

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

    def set_to_continuous(self):
        self.continuous_weights = True

    def clean_edges(self):
        """Clean the edges of the graph.
        """
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.is_initialized = False

    def model_loglikelihood(self):
        """Returns the loglikelihood of the solution of last model executed.
        """
        return self.loglikelihood
