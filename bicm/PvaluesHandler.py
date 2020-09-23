import numpy as np
from numba import jit
from multiprocessing import Pool
from tqdm import tqdm_notebook
from .poibin import PoiBin
from scipy.stats import poisson, norm


@jit(nopython=True)
def v_probs_from_fitnesses(x_i, x_j, y):
    return x_i * x_j * (y ** 2) / ((1 + x_i * y) * (1 + x_j * y))


@jit(nopython=True)
def v_list_from_v_mat(v_mat):
    n_rows = len(v_mat)
    v_list = []
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if v_mat[i, j] > 0:
                v_list.append((i, j, v_mat[i, j]))
    return v_list


class PvalClass:
    def __init__(self):
        # Full problem parameters
        self.x = None
        self.y = None
        self.avg_mat = None
        self.avg_v_mat = None
        self.n_rows = None
        self.n_cols = None
        self.pval_list = None
        self.light_mode = None
        self.method = None
        self.threads_num = None
        self.progress_bar = None
    
    def set_fitnesses(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n_rows = len(x)
        self.n_cols = len(y)
        self.light_mode = True
        self.avg_mat = None
    
    def set_avg_mat(self, avg_mat):
        self.avg_mat = np.array(avg_mat, dtype=float)
        self.n_rows, self.n_cols = self.avg_mat.shape
        self.light_mode = False
        self.x = None
        self.y = None
    
    def pval_calculator(self, v):
        i = v[0]
        j = v[1]
        if self.method == 'poisson':
            if self.light_mode:
                avg_v = np.sum(v_probs_from_fitnesses(self.x[i], self.x[j], self.y))
            else:
                avg_v = self.avg_v_mat[i, j]
            return i, j, poisson.sf(k=v[2] - 1, mu=avg_v)
        elif self.method == 'normal':
            if self.light_mode:
                probs = v_probs_from_fitnesses(self.x[i], self.x[j], self.y)
            else:
                probs = self.avg_mat[i] * self.avg_mat[j]
            avg_v = np.sum(probs)
            sigma_v = np.sqrt(np.sum(probs * (1 - probs)))
            return i, j, norm.cdf((v[2] + 0.5 - avg_v) / sigma_v)
        elif self.method == 'rna':
            if self.light_mode:
                probs = v_probs_from_fitnesses(self.x[i], self.x[j], self.y)
            else:
                probs = self.avg_mat[i] * self.avg_mat[j]
            avg_v = np.sum(probs)
            var_v_arr = probs * (1 - probs)
            sigma_v = np.sqrt(np.sum(var_v_arr))
            gamma_v = (sigma_v ** (-3)) * np.sum(var_v_arr * (1 - 2 * probs))
            eval_x = (v[2] + 0.5 - avg_v) / sigma_v
            pval_temp = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
            if pval_temp < 0:
                return i, j, 0
            elif pval_temp > 1:
                return i, j, 1
            else:
                return i, j, pval_temp
    
    def pval_calculator_poibin(self, v_couple):
        if self.light_mode:
            probs = v_probs_from_fitnesses(self.x[v_couple[0][0]], self.x[v_couple[0][1]], self.y)
        else:
            probs = self.avg_mat[v_couple[0][0]] * self.avg_mat[v_couple[0][1]]
        pb_obj = PoiBin(probs)
        try:
            pval_list = [(v[0], v[1], pb_obj.pval(int(v[2]))) for v in v_couple]
        except:
            pval_list = v_couple
        return pval_list
    
    def _calculate_pvals(self, v_list):
        """
        Internal method for calculating pvalues given an overlap list.
        v_list contains triplets of two nodes and their number of v-motifs (common neighbors).
        The parallelization is different from poibin solver and other types of solvers.
        """
        if self.method != 'poibin':
            if self.method == 'poisson' and not self.light_mode:
                self.avg_v_mat = np.dot(self.avg_mat, self.avg_mat.T)
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as p:
                    if self.progress_bar:
                        self.pval_list = p.map(self.pval_calculator, tqdm_notebook(v_list))
                    else:
                        self.pval_list = p.map(self.pval_calculator, v_list)
            else:
                self.pval_list = []
                if self.progress_bar:
                    for v in tqdm_notebook(v_list):
                        self.pval_list.append(self.pval_calculator(v))
                else:
                    for v in v_list:
                        self.pval_list.append(self.pval_calculator(v))
        else:
            if self.progress_bar:
                print('Building the V-motifs list...')
            if self.light_mode:
                r_x, r_invert = np.unique(self.x, return_inverse=True)
                r_n_rows = len(r_x)
            else:
                r_deg, r_invert = np.unique(self.avg_mat.sum(1), return_inverse=True)
                r_n_rows = len(r_deg)
            v_list_coupled = []
            for i in range(r_n_rows):
                pos_i = np.where(r_invert == i)[0]
                for j in range(i, r_n_rows):
                    pos_j = np.where(r_invert == j)[0]
                    red_v_list = [v for v in v_list if (v[0] in pos_i and v[1] in pos_j) or (v[0] in pos_j and v[1] in pos_i)]
                    if len(red_v_list) > 0:
                        v_list_coupled.append(red_v_list)
            if self.progress_bar:
                print('Calculating p-values...')
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as p:
                    if self.progress_bar:
                        pval_list_coupled = p.map(self.pval_calculator_poibin, tqdm_notebook(v_list_coupled))
                    else:
                        pval_list_coupled = p.map(self.pval_calculator_poibin, v_list_coupled)
            else:
                pval_list_coupled = []
                if self.progress_bar:
                    for v_couple in tqdm_notebook(v_list_coupled):
                        pval_list_coupled.append(self.pval_calculator_poibin(v_couple))
                else:
                    for v_couple in v_list_coupled:
                        pval_list_coupled.append(self.pval_calculator_poibin(v_couple))
            self.pval_list = [pval for pvals_set in pval_list_coupled for pval in pvals_set]
        return
    
    def compute_pvals(self, observed_net, method='poisson', threads_num=1, progress_bar=True):
        self.method = method
        self.threads_num = threads_num
        self.progress_bar = progress_bar
        if self.light_mode:
            self._calculate_pvals(observed_net)
        else:
            v_list = v_list_from_v_mat(observed_net)
            self._calculate_pvals(v_list)
