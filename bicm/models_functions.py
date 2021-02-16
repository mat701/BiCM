import numpy as np
from numba import jit
from . import solver_functions as sof


# BiCM functions
# --------------


@jit(nopython=True)
def linsearch_fun_BiCM(xx, args):
    """Linsearch function for BiCM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    # eps2 = 1e-2
    # alfa0 = (eps2 - 1) * x / dx
    # for a in alfa0:
    #     if a >= 0:
    #         alfa = min(alfa, a)

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sof.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        is False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_BiCM_fixed(xx):
    """Linsearch function for BiCM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type xx: (numpy.ndarray, numpy.ndarray, func, float, float, int)
    :return: Working alpha.
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    # eps2 = 1e-2
    # alfa0 = (eps2 - 1) * x / dx
    # for a in alfa0:
    #     if a >= 0:
    #         alfa = min(alfa, a)

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            cond is False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


@jit(nopython=True)
def linsearch_fun_BiCM_exp(xx, args):
    """Linsearch function for BiCM newton and quasinewton methods.
    This is the linesearch function in the exponential mode.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sof.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        is False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_BiCM_exp_fixed(xx):
    """Linsearch function for BiCM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on BiCM exponential version.
    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
    :rtype: float
    """
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa * dx) < np.linalg.norm(dx_old)
        while(
            not cond
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


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


@jit(nopython=True)
def iterative_bicm(x0, args):
    """
    Return the next iterative step for the Bipartite Configuration Model reduced version.

    :param numpy.ndarray x0: initial point
    :param list, tuple args: rows degree sequence, columns degree sequence, rows multipl., cols multipl.
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
    ff = - np.log(ff)

    return ff


@jit(nopython=True)
def iterative_bicm_exp(x0, args):
    """
    Return the next iterative step for the Bipartite Configuration Model reduced version.

    :param numpy.ndarray x0: initial point
    :param list, tuple args: rows degree sequence, columns degree sequence, rows multipl., cols multipl.
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

    f = np.zeros(len(x0))

    for i in range(num_rows):
        rows_multiplier = rows_multiplicity[i] * x[i]
        for j in range(num_cols):
            denom = 1 + x[i] * y[j]
            f[i] += cols_multiplicity[j] * y[j] / denom
            f[j + num_rows] += rows_multiplier / denom
    tmp = np.concatenate((r_dseq_rows, r_dseq_cols))
    ff = tmp / f

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
def loglikelihood_bicm(x0, args):
    """
    Log-likelihood function of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list or tuple
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
    theta_x = np.copy(x)
    theta_y = np.copy(y)
    x = np.exp(- x)
    y = np.exp(- y)

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
def loglikelihood_bicm_exp(x0, args):
    """
    Log-likelihood function of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list or tuple
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
def loglikelihood_hessian_bicm(x0, args):
    """
    Log-likelihood hessian of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
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
    x = np.exp(- x)
    y = np.exp(- y)

    for h in range(num_rows):
        for i in range(num_cols):
            denom = (1 + x[h] * y[i]) ** 2
            add = cols_multiplicity[i] * rows_multiplicity[h] * x[h] * y[i] / denom
            out[h, h] -= add
            out[h, i + num_rows] = - add
            out[i + num_rows, h] = - add
            out[i + num_rows, i + num_rows] -= add

    return out


@jit(nopython=True)
def loglikelihood_hessian_bicm_exp(x0, args):
    """
    Log-likelihood hessian of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
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
    x2 = x ** 2
    y2 = y ** 2
    flag = True

    for h in range(num_rows):
        out[h, h] -= r_dseq_rows[h] / x2[h]
        for i in range(num_cols):
            denom = (1 + x[h] * y[i]) ** 2
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
def loglikelihood_hessian_diag_bicm(x0, args):
    """
    Log-likelihood diagonal hessian of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
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
    x = np.exp(- x)
    y = np.exp(- y)

    f = np.zeros(num_rows + num_cols)

    for i in range(num_rows):
        for j in range(num_cols):
            denom = (1 + x[i] * y[j]) ** 2
            add = cols_multiplicity[j] * rows_multiplicity[i] * x[i] * y[j] / denom
            f[i] -= add
            f[j + num_rows] -= add

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_bicm_exp(x0, args):
    """
    Log-likelihood diagonal hessian of the reduced BiCM.

    :param numpy.ndarray x0: 1D fitnesses vector
    :param args: list of arguments needed for the computation
    :type args: list, tuple
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
    x2 = x ** 2
    y2 = y ** 2

    f = np.zeros(num_rows + num_cols)
    flag = True

    for i in range(num_rows):
        for j in range(num_cols):
            denom = (1 + x[i] * y[j]) ** 2
            f[i] += cols_multiplicity[j] * y2[j] / denom
            f[j + num_rows] += rows_multiplicity[i] * x2[i] / denom
            if flag:
                f[j + num_rows] -= r_dseq_cols[j] / y2[j]
        f[i] -= r_dseq_rows[i] / x2[i]
        flag = False

    return f


@jit(nopython=True)
def loglikelihood_prime_bicm(x0, args):
    """
    Iterative function for loglikelihood gradient BiCM.

    :param x0: fitnesses vector
    :type x0: numpy.array
    :param args: list of arguments needed for the computation
    :type args: list or tuple
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
    x = np.exp(- x)
    y = np.exp(- y)

    f = np.zeros(len(x0))
    flag = True

    for i in range(num_rows):
        for j in range(num_cols):
            denom = 1 + x[i] * y[j]
            add = y[j] * x[i] * rows_multiplicity[i] * cols_multiplicity[j] / denom
            f[i] += add
            f[j + num_rows] += add
            if flag:
                f[j + num_rows] -= r_dseq_cols[j] * cols_multiplicity[j]
        f[i] -= r_dseq_rows[i] * rows_multiplicity[i]
        flag = False

    return f

@jit(nopython=True)
def loglikelihood_prime_bicm_exp(x0, args):
    """
    Iterative function for loglikelihood gradient BiCM.

    :param x0: fitnesses vector
    :type x0: numpy.array
    :param args: list of arguments needed for the computation
    :type args: list or tuple
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

    f = np.zeros(len(x0))
    flag = True

    for i in range(num_rows):
        for j in range(num_cols):
            denom = 1 + x[i] * y[j]
            f[i] -= y[j] * cols_multiplicity[j] / denom
            f[j + num_rows] -= x[i] * rows_multiplicity[i] / denom
            if flag:
                f[j + num_rows] += r_dseq_cols[j] / y[j]
        f[i] += r_dseq_rows[i] / x[i]
        flag = False

    return f