"""
Bipartite Enhanced Configuration Model (BiECM).

The BiECM preserves both the degree sequence and strength sequence
of the bipartite network in expectation.
"""

from typing import Optional, Literal, Tuple
import math
import time
import warnings
import torch
from torch.distributions import Exponential

from bicm.wbnm.models.base import BipartiteModel, ArrayLike, ConvergenceWarning


class BiECM(BipartiteModel):
    """
    Bipartite Enhanced Configuration Model.

    The BiECM is a maximum entropy null model that preserves:
    - Expected degree sequence (number of links per node)
    - Expected strength sequence (sum of weights per node)

    The model assigns to each node i a pair of fitness parameters (x_i, theta_i)
    such that the probability of a link and the expected weight are determined
    by these parameters.

    Parameters
    ----------
    biadjacency : ArrayLike
        The observed biadjacency matrix of shape (n_rows, n_cols).
    device : str, optional
        Device for computation. Auto-detects GPU if None.

    Attributes
    ----------
    x_rows : torch.Tensor
        Fitness parameters for degrees of row nodes (after solving).
    x_cols : torch.Tensor
        Fitness parameters for degrees of column nodes (after solving).
    theta_rows : torch.Tensor
        Fitness parameters for strengths of row nodes (after solving).
    theta_cols : torch.Tensor
        Fitness parameters for strengths of column nodes (after solving).

    Examples
    --------
    >>> import torch
    >>> from bicm.wbnm.models import BiECM
    >>>
    >>> # Create a random weighted bipartite network
    >>> W = torch.poisson(torch.full((10, 20), 5.0)) * (torch.rand(10, 20) > 0.5).float()
    >>>
    >>> # Fit the model
    >>> model = BiECM(W)
    >>> model.solve()
    >>>
    >>> # Generate samples
    >>> samples = model.sample(n_samples=100)
    >>>
    >>> # Check that expected properties match observed
    >>> expected = model.expected_matrix()
    >>> print(f"Observed total weight: {W.sum():.2f}")
    >>> print(f"Expected total weight: {expected.sum():.2f}")

    References
    ----------
    - Saracco et al., "Randomizing bipartite networks: the case of the
           World Trade Web", Scientific Reports (2015)
    - Bruno et al., "Inferring comparative advantage via entropy maximization",
           Journal of Physics: Complexity (2023)
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        super().__init__(biadjacency, device=device)

        # Fitness parameters (set after solving)
        self.x_rows: Optional[torch.Tensor] = None
        self.x_cols: Optional[torch.Tensor] = None
        self.theta_rows: Optional[torch.Tensor] = None
        self.theta_cols: Optional[torch.Tensor] = None

    @torch.no_grad()
    def solve(
        self,
        method: Literal["fixed-point", "quasi-newton", "coordinate"] = "fixed-point",
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: Optional[float] = None,
        beta: float = 0.9,
        beta_schedule: Optional[Literal["warmup", "linear"]] = None,
        warmup_steps: int = 500,
        chunk: Optional[int] = None,
        **kwargs
    ) -> "BiECM":
        """
        Solve for the fitness parameters.

        Parameters
        ----------
        method : {"fixed-point", "quasi-newton"}, default="fixed-point"
            Optimization method:
            - "fixed-point": Fixed-point iteration with momentum (stable)
            - "quasi-newton": Diagonal quasi-Newton method (can be faster)
        x0 : torch.Tensor, optional
            Initial guess for parameters. If None, uses default initialization.
        tol : float, default=1e-8
            Convergence tolerance (max relative error).
        max_iter : int, default=10000
            Maximum number of iterations.
        verbose : bool, default=False
            Whether to print convergence information.
        alpha : float, optional
            Step size for fixed-point (default 1.0) and quasi-Newton (default 0.5).
            Ignored by coordinate solver.
        beta : float, default=0.9
            Momentum coefficient (only for fixed-point method).
        beta_schedule : {None, "warmup", "linear"}, default=None
            Schedule for momentum coefficient (only for fixed-point method):
            - None: beta stays fixed throughout optimization.
            - "warmup": starts at beta, jumps to 0.999 after warmup_steps.
            - "linear": linearly interpolates from beta to 0.999 over warmup_steps.
        warmup_steps : int, default=500
            Number of steps for beta schedule (only used if beta_schedule is set).
        **kwargs
            Additional arguments passed to the optimizer.

        Returns
        -------
        self : BiECM
            Returns self for method chaining.

        Raises
        ------
        ConvergenceWarning
            If the optimization does not converge.
        """
        if method != "fixed-point" and (
            beta != 0.9 or beta_schedule is not None or warmup_steps != 500
        ):
            warnings.warn(
                f"beta, beta_schedule, and warmup_steps are only used with "
                f"method='fixed-point', not '{method}'.",
                UserWarning,
                stacklevel=2,
            )

        # Get observed sequences
        k_rows = self.degree_sequence_rows.to(torch.float64)
        k_cols = self.degree_sequence_cols.to(torch.float64)
        s_rows = self.strength_sequence_rows.to(torch.float64)
        s_cols = self.strength_sequence_cols.to(torch.float64)

        # Initialize parameters
        if x0 is None:
            x0 = self._initial_guess(k_rows, k_cols, s_rows, s_cols, method=method)
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self.device)

        # Method-specific alpha defaults
        _alpha_fp = alpha if alpha is not None else 1.0
        _alpha_qn = alpha if alpha is not None else 0.1

        # Solve based on method
        t0 = time.perf_counter()

        if method == "fixed-point":
            params, info = self._solve_fixed_point(
                x0, k_rows, k_cols, s_rows, s_cols,
                tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_fp, beta=beta,
                beta_schedule=beta_schedule, warmup_steps=warmup_steps,
                chunk=chunk, **kwargs
            )
        elif method == "quasi-newton":
            params, info = self._solve_quasi_newton(
                x0, k_rows, k_cols, s_rows, s_cols,
                tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_qn, chunk=chunk, **kwargs
            )
        elif method == "coordinate":
            params, info = self._solve_coordinate(
                k_rows, k_cols, s_rows, s_cols, x0,
                tol=tol, max_iter=max_iter, verbose=verbose, chunk=chunk
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.time_s = time.perf_counter() - t0
        self.converged = info["converged"]
        self.n_iter = info["n_iter"]
        self.final_residual = info["final_residual"]

        # Unpack and store parameters
        # Order: [theta_rows, theta_cols, x_rows, x_cols]
        n_r, n_c = self.n_rows, self.n_cols
        self.theta_rows = params[:n_r]
        self.theta_cols = params[n_r:n_r + n_c]
        self.x_rows = params[n_r + n_c:2*n_r + n_c]
        self.x_cols = params[2*n_r + n_c:]

        self.is_solved = True
        self._params = {
            "x_rows": self.x_rows,
            "x_cols": self.x_cols,
            "theta_rows": self.theta_rows,
            "theta_cols": self.theta_cols,
        }

        return self

    def _initial_guess(
        self,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        method: str = "fixed-point",
    ) -> torch.Tensor:
        """Generate initial guess for parameters."""
        n_r, n_c = self.n_rows, self.n_cols

        # Avoid zeros
        k_rows = torch.clamp(k_rows, min=1e-10)
        k_cols = torch.clamp(k_cols, min=1e-10)
        s_rows = torch.clamp(s_rows, min=1e-10)
        s_cols = torch.clamp(s_cols, min=1e-10)

        k_tot = k_rows.sum()

        if method == "coordinate":
            theta_rows = 1.0 / s_rows
            theta_cols = 1.0 / s_cols
        else:
            theta_rows = torch.sqrt(k_rows / s_rows)
            theta_cols = torch.sqrt(k_cols / s_cols)

        # Initialize x (degree multipliers) based on degree
        x_rows = k_rows / torch.sqrt(k_tot)
        x_cols = k_cols / torch.sqrt(k_tot)

        # Order: [theta_rows, theta_cols, x_rows, x_cols]
        return torch.cat([theta_rows, theta_cols, x_rows, x_cols])

    def _compute_expected_matrices(
        self,
        theta_rows: torch.Tensor,
        theta_cols: torch.Tensor,
        x_rows: torch.Tensor,
        x_cols: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expected connection probability and weight matrices.

        Returns
        -------
        p_matrix : torch.Tensor
            Connection probability matrix P(A_ij = 1).
        w_matrix : torch.Tensor
            Expected weight matrix E[w_ij].
        """
        # Compute outer products
        x_outer = x_rows[:, None] * x_cols[None, :]
        theta_sum = theta_rows[:, None] + theta_cols[None, :]

        # Connection probability: p_ij = x_i * x_j / (theta_i + theta_j + x_i * x_j)
        p_matrix = x_outer / (theta_sum + x_outer)

        # Expected weight: E[w_ij] = p_ij / (theta_i + theta_j)
        w_matrix = p_matrix / theta_sum

        return p_matrix, w_matrix

    def _solve_fixed_point(
        self,
        x0: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        beta: float = 0.9,
        beta_schedule: Optional[Literal["warmup", "linear"]] = None,
        warmup_steps: int = 500,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using fixed-point iteration with momentum.

        Uses Nesterov-like momentum for accelerated convergence.
        Supports optional beta scheduling for difficult networks.
        """
        # Build beta schedule function
        if beta_schedule is None:
            get_beta = lambda t: beta
        elif beta_schedule == "warmup":
            get_beta = lambda t: beta if t < warmup_steps else 0.999
        elif beta_schedule == "linear":
            get_beta = lambda t: min(0.999, beta + (0.999 - beta) * t / warmup_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        n_r, n_c = self.n_rows, self.n_cols
        params = x0.clone()
        converged = False

        # Split parameters
        theta_rows = params[:n_r].clone()
        theta_cols = params[n_r:n_r + n_c].clone()
        x_rows = params[n_r + n_c:2*n_r + n_c].clone()
        x_cols = params[2*n_r + n_c:].clone()

        # Combine for easier handling
        theta_mults = torch.cat([theta_rows, theta_cols])
        x_mults = torch.cat([x_rows, x_cols])
        strength = torch.cat([s_rows, s_cols])
        degree = torch.cat([k_rows, k_cols])

        # Initialize momentum
        v_theta = torch.zeros_like(theta_mults)
        v_x = torch.zeros_like(x_mults)

        w = len(str(max_iter))
        for iteration in range(max_iter):
            # Compute expected matrices
            tr, tc = theta_mults[:n_r], theta_mults[n_r:]
            xr, xc = x_mults[:n_r], x_mults[n_r:]
            if chunk is None:
                p_matrix, w_matrix = self._compute_expected_matrices(tr, tc, xr, xc)
                e_degree = torch.cat([p_matrix.sum(dim=1), p_matrix.sum(dim=0)])
                e_strength = torch.cat([w_matrix.sum(dim=1), w_matrix.sum(dim=0)])
            else:
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = xr[i0:i1, None] * xc[None, :]
                    ts = tr[i0:i1, None] + tc[None, :]
                    p = xo / (ts + xo)
                    w = p / ts
                    e_kr[i0:i1] = p.sum(1)
                    e_kc += p.sum(0)
                    e_sr[i0:i1] = w.sum(1)
                    e_sc += w.sum(0)
                e_degree = torch.cat([e_kr, e_kc])
                e_strength = torch.cat([e_sr, e_sc])

            # Compute relative errors
            rel_err_degree = torch.abs(degree - e_degree) / torch.clamp(degree, min=1e-10)
            rel_err_strength = torch.abs(strength - e_strength) / torch.clamp(strength, min=1e-10)

            max_err_degree = rel_err_degree.max().item()
            max_err_strength = rel_err_strength.max().item()
            max_error = max(max_err_degree, max_err_strength)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_degree={max_err_degree:.2e}  max_err_strength={max_err_strength:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"Converged at iter {iteration}")
                converged = True
                break

            # Compute gradients (directions of update)
            delta_theta = e_strength / (strength + 1e-12) - 1
            delta_x = degree / (e_degree + 1e-12) - 1

            # Get current beta value from schedule
            beta_t = get_beta(iteration)

            # Update momentum
            v_theta = beta_t * v_theta + (1 - beta_t) * delta_theta
            v_x = beta_t * v_x + (1 - beta_t) * delta_x

            # Update parameters with momentum
            theta_mults = theta_mults * (1 + alpha * v_theta)
            x_mults = x_mults * (1 + alpha * v_x)

        else:
            warnings.warn(
                f"Fixed-point solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        # Reconstruct params array
        return torch.cat([theta_mults, x_mults]), {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_quasi_newton(
        self,
        x0: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.05,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using diagonal quasi-Newton method.

        The log-likelihood gradient is dL/dtheta_i = E[s_i] - s_i and
        dL/dx_i = k_i/x_i - sum_j p_ij/x_i (simplified via the diagonal
        Hessian approximation). Newton step for maximising L:

            params += gradient / hessian_diag

        All parameters (theta and x) must stay positive; params are clamped to 1e-30.
        """
        n_r, n_c = self.n_rows, self.n_cols

        # Split initial guess; all parameters must be positive
        theta_r = x0[:n_r].clamp(min=1e-30)
        theta_c = x0[n_r:n_r + n_c].clamp(min=1e-30)
        x_r = x0[n_r + n_c:2 * n_r + n_c].clamp(min=1e-30)
        x_c = x0[2 * n_r + n_c:].clamp(min=1e-30)

        constraints = torch.cat([s_rows, s_cols, k_rows, k_cols])
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            if chunk is None:
                p_matrix, w_matrix = self._compute_expected_matrices(
                    theta_r, theta_c, x_r, x_c
                )
                e_sr = w_matrix.sum(dim=1)
                e_sc = w_matrix.sum(dim=0)
                e_kr = p_matrix.sum(dim=1)
                e_kc = p_matrix.sum(dim=0)
                cov_s = (2 - p_matrix) * w_matrix ** 2 / torch.clamp(p_matrix, min=1e-10)
                cov_k = p_matrix * (1 - p_matrix)
                hs_r = cov_s.sum(dim=1); hs_c = cov_s.sum(dim=0)
                hk_r = cov_k.sum(dim=1); hk_c = cov_k.sum(dim=0)
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                hs_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hs_c = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                hk_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hk_c = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = x_r[i0:i1, None] * x_c[None, :]
                    ts = theta_r[i0:i1, None] + theta_c[None, :]
                    p = xo / (ts + xo)
                    w = p / ts
                    e_sr[i0:i1] = w.sum(1); e_sc += w.sum(0)
                    e_kr[i0:i1] = p.sum(1); e_kc += p.sum(0)
                    cs = (2 - p) * w ** 2 / torch.clamp(p, min=1e-10)
                    ck = p * (1 - p)
                    hs_r[i0:i1] = cs.sum(1); hs_c += cs.sum(0)
                    hk_r[i0:i1] = ck.sum(1); hk_c += ck.sum(0)

            # --- theta: log-space Newton step ---
            # dL/d(log theta_i) = (E[s_i] - s_i) * theta_i
            # -H_ii(log) ≈ sum_j (2-p)*w^2/p * theta_i^2
            # step = (E[s]-s) / (hs * theta)
            theta_r = torch.clamp(
                theta_r * torch.exp(
                    torch.clamp(alpha * (e_sr - s_rows) / (hs_r * theta_r + 1e-10), -1., 1.)
                ), min=1e-30
            )
            theta_c = torch.clamp(
                theta_c * torch.exp(
                    torch.clamp(alpha * (e_sc - s_cols) / (hs_c * theta_c + 1e-10), -1., 1.)
                ), min=1e-30
            )

            # --- x: log-space Newton step ---
            # dL/d(log x_i) = k_i - E[k_i];  -H_ii = sum_j p*(1-p)
            # Clamp exponent to [-10, 10] to prevent overflow on first iterations
            x_r = torch.clamp(
                x_r * torch.exp(
                    torch.clamp(alpha * (k_rows - e_kr) / (hk_r + 1e-10), -1., 1.)
                ), min=1e-30
            )
            x_c = torch.clamp(
                x_c * torch.exp(
                    torch.clamp(alpha * (k_cols - e_kc) / (hk_c + 1e-10), -1., 1.)
                ), min=1e-30
            )

            # Convergence: max relative error on all four constraints
            max_err_degree = max(
                (torch.abs(e_kr - k_rows) / k_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(e_kc - k_cols) / k_cols.clamp(min=1e-10)).max().item(),
            )
            max_err_strength = max(
                (torch.abs(e_sr - s_rows) / s_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(e_sc - s_cols) / s_cols.clamp(min=1e-10)).max().item(),
            )
            max_error = max(max_err_degree, max_err_strength)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_degree={max_err_degree:.2e}  max_err_strength={max_err_strength:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"Converged at iter {iteration}")
                converged = True
                break
        else:
            warnings.warn(
                f"Quasi-Newton solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return torch.cat([theta_r, theta_c, x_r, x_c]), {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_coordinate(
        self,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        x0: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        chunk: Optional[int] = None,
        n_bisect: int = 60,
    ) -> torch.Tensor:
        """
        Coordinate descent with bisection in PyTorch.

        Each coordinate group is solved via 60-step bisection in log-space
        (~1e-18 precision). The bisection bounds (lo/hi) are tensors so all
        nodes in a group are updated in parallel, but the matrix sums are
        computed in sequential chunks for memory efficiency.
        theta_rows/theta_cols are decreasing functions; x_rows/x_cols are increasing.
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk
        tr = x0[:n_r].clamp(min=1e-30)
        tc = x0[n_r:n_r + n_c].clamp(min=1e-30)
        xr = x0[n_r + n_c:2 * n_r + n_c].clamp(min=1e-30)
        xc = x0[2 * n_r + n_c:].clamp(min=1e-30)

        max_rel = float("inf")
        converged = False
        w = len(str(max_iter))
        for outer in range(max_iter):
            # --- Bisect theta_rows (f is decreasing: f > target → lo = mid) ---
            lo = tr.new_full((n_r,), -30.)
            hi = tr.new_full((n_r,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                tr_mid = log_mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr[i0:i1, None] * xc[None, :]
                    ts = tr_mid[i0:i1, None] + tc[None, :]
                    f[i0:i1] = (xo / (ts * (ts + xo))).sum(1)
                above = f > s_rows
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            tr = ((lo + hi) / 2).exp()

            # --- Bisect theta_cols (decreasing) ---
            lo = tc.new_full((n_c,), -30.)
            hi = tc.new_full((n_c,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                tc_mid = log_mid.exp()
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr[i0:i1, None] * xc[None, :]
                    ts = tr[i0:i1, None] + tc_mid[None, :]
                    f += (xo / (ts * (ts + xo))).sum(0)
                above = f > s_cols
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            tc = ((lo + hi) / 2).exp()

            # --- Bisect x_rows (f is increasing: f > target → hi = mid) ---
            lo = xr.new_full((n_r,), -30.)
            hi = xr.new_full((n_r,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                xr_mid = log_mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr_mid[i0:i1, None] * xc[None, :]
                    ts = tr[i0:i1, None] + tc[None, :]
                    f[i0:i1] = (xo / (ts + xo)).sum(1)
                above = f > k_rows
                hi = torch.where(above, log_mid, hi)
                lo = torch.where(above, lo, log_mid)
            xr = ((lo + hi) / 2).exp()

            # --- Bisect x_cols (increasing) ---
            lo = xc.new_full((n_c,), -30.)
            hi = xc.new_full((n_c,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                xc_mid = log_mid.exp()
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr[i0:i1, None] * xc_mid[None, :]
                    ts = tr[i0:i1, None] + tc[None, :]
                    f += (xo / (ts + xo)).sum(0)
                above = f > k_cols
                hi = torch.where(above, log_mid, hi)
                lo = torch.where(above, lo, log_mid)
            xc = ((lo + hi) / 2).exp()

            # --- Convergence check ---
            e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_kc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                xo = xr[i0:i1, None] * xc[None, :]
                ts = tr[i0:i1, None] + tc[None, :]
                p = xo / (ts + xo)
                w = p / ts
                e_sr[i0:i1] = w.sum(1)
                e_sc += w.sum(0)
                e_kr[i0:i1] = p.sum(1)
                e_kc += p.sum(0)

            max_err_degree = max(
                (torch.abs(k_rows - e_kr) / k_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(k_cols - e_kc) / k_cols.clamp(min=1e-10)).max().item(),
            )
            max_err_strength = max(
                (torch.abs(s_rows - e_sr) / s_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(s_cols - e_sc) / s_cols.clamp(min=1e-10)).max().item(),
            )
            max_rel = max(max_err_degree, max_err_strength)
            if verbose:
                print(f"Iter {outer:{w}d}  max_err_degree={max_err_degree:.2e}  max_err_strength={max_err_strength:.2e}")
            if max_rel < tol:
                if verbose:
                    print(f"Converged at iter {outer}")
                converged = True
                break
        else:
            warnings.warn(
                f"Coordinate solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_rel:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return torch.cat([tr, tc, xr, xc]), {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel}

    def connection_probability(self) -> torch.Tensor:
        """
        Compute the matrix of connection probabilities.

        Returns
        -------
        prob : torch.Tensor
            Matrix where entry (i,j) is P(A_ij = 1).
        """
        self._check_solved()

        x_outer = self.x_rows[:, None] * self.x_cols[None, :]
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        return x_outer / (theta_sum + x_outer)

    def expected_matrix(self) -> torch.Tensor:
        """
        Compute expected biadjacency matrix.

        Returns
        -------
        expected : torch.Tensor
            Matrix where entry (i,j) is the expected weight E[w_ij].
        """
        self._check_solved()

        p_matrix = self.connection_probability()
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        return p_matrix / theta_sum

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate random samples from the BiECM ensemble.

        For each edge (i,j):
        1. Sample topology: A_ij ~ Bernoulli(p_ij)
        2. If A_ij = 1, sample weight: w_ij ~ Exponential(theta_i + theta_j)

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples, n_rows, n_cols).
        """
        self._check_solved()

        if seed is not None:
            torch.manual_seed(seed)

        p_matrix = self.connection_probability()
        # Rate for exponential distribution
        rate_matrix = self.theta_rows[:, None] + self.theta_cols[None, :]

        samples = torch.zeros(n_samples, self.n_rows, self.n_cols,
                              dtype=torch.float64, device=self.device)

        for s in range(n_samples):
            # Sample topology
            topology = (torch.rand(self.n_rows, self.n_cols,
                                   dtype=torch.float64, device=self.device) < p_matrix).to(torch.float64)

            # Sample weights (exponential distribution)
            weights = Exponential(rate_matrix).sample()

            # Combine: weight only where there's a link
            samples[s] = topology * weights

        return samples

    @torch.no_grad()
    def sample_topology(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample only the binary topology (useful for some validations).

        Returns
        -------
        samples : torch.Tensor
            Binary matrices of shape (n_samples, n_rows, n_cols).
        """
        self._check_solved()

        if seed is not None:
            torch.manual_seed(seed)

        p = self.connection_probability()

        samples = torch.zeros(n_samples, self.n_rows, self.n_cols,
                              dtype=torch.int8, device=self.device)
        for s in range(n_samples):
            samples[s] = (torch.rand(self.n_rows, self.n_cols,
                                     dtype=torch.float64, device=self.device) < p).to(torch.int8)

        return samples

    def log_likelihood(self) -> float:
        """
        Compute the log-likelihood of the model.

        Returns
        -------
        ll : float
            Log-likelihood value.
        """
        self._check_solved()

        k_rows = self.degree_sequence_rows
        k_cols = self.degree_sequence_cols
        s_rows = self.strength_sequence_rows
        s_cols = self.strength_sequence_cols

        # Hamiltonian terms
        hamiltonian = (
            -(self.theta_rows * s_rows).sum()
            - (self.theta_cols * s_cols).sum()
            - (-torch.log(self.x_rows) * k_rows).sum()
            - (-torch.log(self.x_cols) * k_cols).sum()
        )

        # Partition function
        x_outer = self.x_rows[:, None] * self.x_cols[None, :]
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        partition = -torch.log(1 + x_outer / theta_sum).sum()

        return (hamiltonian + partition).item()

    def n_parameters(self) -> int:
        """Return the number of free parameters in the model."""
        # 4 sets of parameters: x_rows, x_cols, theta_rows, theta_cols
        return 2 * (self.n_rows + self.n_cols)

    def aic(self) -> float:
        """
        Compute the Akaike Information Criterion (AIC).

        AIC = -2 * log_likelihood + 2 * n_parameters

        Lower AIC indicates a better model (balancing fit and complexity).

        Returns
        -------
        aic : float
            The AIC value.
        """
        return -2 * self.log_likelihood() + 2 * self.n_parameters()

    def aicc(self) -> float:
        """
        Compute the corrected AIC (AICc) for small sample sizes.

        AICc = AIC + (2k(k+1)) / (n - k - 1)

        where k is the number of parameters and n is the sample size
        (number of matrix entries).

        Returns
        -------
        aicc : float
            The AICc value.
        """
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        aic = self.aic()
        correction = (2 * k * (k + 1)) / (n - k - 1)
        return aic + correction

    def bic(self) -> float:
        """
        Compute the Bayesian Information Criterion (BIC).

        BIC = -2 * log_likelihood + k * log(n)

        where k is the number of parameters and n is the sample size.

        Returns
        -------
        bic : float
            The BIC value.
        """
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        return -2 * self.log_likelihood() + k * math.log(n)

    def pvalues(self, observed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute analytical p-values for the observed weights.

        Under BiECM, the p-value combines the probability of having an edge
        and the probability of observing that weight or more given an edge.

        P(W_ij >= w) = p_ij * exp(-w / E[W_ij | A_ij=1])

        where p_ij is the connection probability and E[W_ij | A_ij=1] is
        the expected weight conditional on having a link.

        Parameters
        ----------
        observed : torch.Tensor, optional
            The observed weight matrix. If None, uses the matrix passed
            to the constructor.

        Returns
        -------
        pvalues : torch.Tensor
            Matrix of p-values, same shape as observed.
            Entry (i,j) is P(W_ij >= observed_ij) under the null model.

        Notes
        -----
        These are one-sided p-values for the "greater" alternative.
        Low p-values indicate weights that are significantly larger
        than expected under the null model.
        """
        self._check_solved()

        if observed is None:
            observed = self.biadjacency
        observed = torch.as_tensor(observed, dtype=torch.float64, device=self.device)

        # Connection probability
        p_matrix = self.connection_probability()

        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        # P(W >= w) = p_ij * exp(-w * (theta_i + theta_j)) where w > 0, 1 where w = 0.
        # The product is computed in logarithms: the two factors can underflow
        # separately even when the p-value is representable.
        linked = observed > 0
        log_pvalues = torch.log(p_matrix) - observed * theta_sum
        pvalues = torch.where(linked, torch.exp(log_pvalues), torch.ones_like(observed))

        return pvalues
