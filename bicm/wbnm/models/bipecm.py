"""
Bipartite Partial Enhanced Configuration Model (BiPECM).

Also known as PCM (Partial Configuration Model). Preserves the degree
sequence exactly and the strength sequence in expectation.
"""

from typing import Optional, Literal, Tuple
import math
import time
import warnings
import torch
from torch.distributions import Exponential

from bicm.wbnm.models.base import BipartiteModel, ArrayLike, ConvergenceWarning


class BiPECM(BipartiteModel):
    """
    Bipartite Partial Enhanced Configuration Model.

    The BiPECM (also called PCM) is a maximum entropy null model that:
    - Preserves the degree sequence exactly (hard constraint)
    - Preserves the strength sequence in expectation (soft constraint)

    This makes it intermediate between BiCM (only degrees) and BiECM
    (both degrees and strengths in expectation).

    Parameters
    ----------
    biadjacency : ArrayLike
        The observed biadjacency matrix of shape (n_rows, n_cols).
    device : str, optional
        Device for computation. Auto-detects GPU if None.

    Attributes
    ----------
    theta_rows : torch.Tensor
        Fitness parameters for row nodes (after solving).
    theta_cols : torch.Tensor
        Fitness parameters for column nodes (after solving).
    density_param : float
        Global density parameter (after solving).

    Examples
    --------
    >>> import torch
    >>> from bicm.wbnm.models import BiPECM
    >>>
    >>> W = torch.poisson(torch.full((10, 20), 5.0)) * (torch.rand(10, 20) > 0.3).float()
    >>> model = BiPECM(W)
    >>> model.solve()
    >>> samples = model.sample(n_samples=100)
    >>>
    >>> # Verify exact degree preservation
    >>> for s in samples:
    ...     assert torch.allclose((s > 0).sum(dim=1).float(), (W > 0).sum(dim=1).float())

    Notes
    -----
    The key difference from BiECM is that BiPECM conditions on the observed
    topology (adjacency structure), so sampling only redistributes weights
    on existing edges. This is useful when the topology itself is considered
    meaningful and should not be randomized.

    References
    ----------
    - Parisi et al., "A faster horse on a safer trail: generalized
           inference for the efficient reconstruction of weighted networks",
           New Journal of Physics (2020)
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        super().__init__(biadjacency, device=device)

        if (self.biadjacency < 0).any():
            raise ValueError("biadjacency contains negative values. BiPECM requires non-negative weights.")

        # Store the binary topology (fixed under this model)
        self.topology = (self.biadjacency > 0).to(torch.float64)

        self.theta_rows: Optional[torch.Tensor] = None
        self.theta_cols: Optional[torch.Tensor] = None
        self.density_param: Optional[float] = None

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
        use_momentum: bool = True,
        chunk: Optional[int] = None,
        **kwargs
    ) -> "BiPECM":
        """
        Solve for the fitness parameters.

        Parameters
        ----------
        method : {"fixed-point", "quasi-newton"}, default="fixed-point"
            Optimization method.
        x0 : torch.Tensor, optional
            Initial guess for parameters.
        tol : float, default=1e-8
            Convergence tolerance.
        max_iter : int, default=10000
            Maximum number of iterations.
        verbose : bool, default=False
            Whether to print convergence information.
        alpha : float, optional
            Step size for fixed-point (default 1.0) and quasi-Newton (default 0.5).
            Ignored by coordinate solver.
        beta : float, default=0.9
            Momentum coefficient (only for fixed-point with momentum).
        beta_schedule : {None, "warmup", "linear"}, default=None
            Schedule for momentum coefficient (only for fixed-point method):
            - None: beta stays fixed throughout optimization.
            - "warmup": starts at beta, jumps to 0.999 after warmup_steps.
            - "linear": linearly interpolates from beta to 0.999 over warmup_steps.
        warmup_steps : int, default=500
            Number of steps for beta schedule (only used if beta_schedule is set).
        use_momentum : bool, default=True
            Whether to use momentum in fixed-point iteration.

        Returns
        -------
        self : BiPECM
        """
        s_rows = self.strength_sequence_rows.to(torch.float64)
        s_cols = self.strength_sequence_cols.to(torch.float64)
        density = self.topology.sum().item()

        if x0 is None:
            x0 = self._initial_guess(s_rows, s_cols, density, method=method)
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self.device)

        if method != "fixed-point" and (
            beta != 0.9 or beta_schedule is not None
            or warmup_steps != 500 or not use_momentum
        ):
            warnings.warn(
                f"beta, beta_schedule, warmup_steps, and use_momentum are only "
                f"used with method='fixed-point', not '{method}'.",
                UserWarning,
                stacklevel=2,
            )

        # Method-specific alpha defaults
        _alpha_fp = alpha if alpha is not None else 1.0
        _alpha_qn = alpha if alpha is not None else 0.5

        t0 = time.perf_counter()

        if method == "fixed-point":
            params, info = self._solve_fixed_point(
                x0, s_rows, s_cols, density,
                tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_fp, beta=beta,
                beta_schedule=beta_schedule, warmup_steps=warmup_steps,
                use_momentum=use_momentum, chunk=chunk
            )
        elif method == "quasi-newton":
            params, info = self._solve_quasi_newton(
                x0, s_rows, s_cols, density,
                tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_qn, chunk=chunk
            )
        elif method == "coordinate":
            params, info = self._solve_coordinate(
                s_rows, s_cols, density, x0,
                tol=tol, max_iter=max_iter, verbose=verbose, chunk=chunk
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.time_s = time.perf_counter() - t0
        self.converged = info["converged"]
        self.n_iter = info["n_iter"]
        self.final_residual = info["final_residual"]

        self.theta_rows = params[:self.n_rows]
        self.theta_cols = params[self.n_rows:-1]
        self.density_param = params[-1].item()

        self.is_solved = True
        self._params = {
            "theta_rows": self.theta_rows,
            "theta_cols": self.theta_cols,
            "density_param": self.density_param,
        }

        return self

    def _initial_guess(
        self,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        density: float,
        method: str = "fixed-point",
    ) -> torch.Tensor:
        """Generate initial guess for parameters."""
        if method == "coordinate":
            theta_rows = 1.0 / torch.clamp(s_rows, min=1e-10)
            theta_cols = 1.0 / torch.clamp(s_cols, min=1e-10)
        else:
            k_rows = self.degree_sequence_rows.to(torch.float64)
            k_cols = self.degree_sequence_cols.to(torch.float64)
            theta_rows = torch.sqrt(torch.clamp(k_rows, min=1e-10) / torch.clamp(s_rows, min=1e-10))
            theta_cols = torch.sqrt(torch.clamp(k_cols, min=1e-10) / torch.clamp(s_cols, min=1e-10))

        # Initialize density parameter consistently with the chosen theta values.
        # From D ≈ n*m * gamma / (theta_bar + gamma) with gamma = exp(-mu):
        theta_bar = theta_rows.mean().item() + theta_cols.mean().item()
        density_mult = math.log((self.n_rows * self.n_cols - density) / (density * theta_bar))

        return torch.cat([theta_rows, theta_cols,
                          torch.tensor([density_mult], dtype=torch.float64, device=self.device)])

    def _compute_expected_matrices(
        self,
        theta_rows: torch.Tensor,
        theta_cols: torch.Tensor,
        density_mult: float
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
        theta_sum = theta_rows[:, None] + theta_cols[None, :]

        # Connection probability: p_ij = exp(-density_mult) / (theta_i + theta_j + exp(-density_mult))
        exp_neg_density = math.exp(-density_mult)
        p_matrix = exp_neg_density / (theta_sum + exp_neg_density)

        # Expected weight: E[w_ij] = p_ij / (theta_i + theta_j)
        w_matrix = p_matrix / theta_sum

        return p_matrix, w_matrix

    def _solve_fixed_point(
        self,
        x0: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        density: float,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        beta: float = 0.9,
        beta_schedule: Optional[Literal["warmup", "linear"]] = None,
        warmup_steps: int = 500,
        use_momentum: bool = True,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using fixed-point iteration with optional momentum.
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

        theta_rows = params[:n_r].clone()
        theta_cols = params[n_r:n_r + n_c].clone()
        density_mult = params[-1].item()

        theta_mults = torch.cat([theta_rows, theta_cols])
        strength = torch.cat([s_rows, s_cols])

        if not use_momentum:
            beta = 0.0
            get_beta = lambda t: 0.0

        v_theta = torch.zeros_like(theta_mults)
        v_density = 0.0

        w = len(str(max_iter))
        for iteration in range(max_iter):
            tr, tc = theta_mults[:n_r], theta_mults[n_r:]
            exp_neg_mu = math.exp(-density_mult)
            if chunk is None:
                p_matrix, w_matrix = self._compute_expected_matrices(tr, tc, density_mult)
                e_strength = torch.cat([w_matrix.sum(dim=1), w_matrix.sum(dim=0)])
                e_density = p_matrix.sum().item()
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                e_density = 0.0
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    ts = tr[i0:i1, None] + tc[None, :]
                    p = exp_neg_mu / (ts + exp_neg_mu)
                    w = p / ts
                    e_sr[i0:i1] = w.sum(1)
                    e_sc += w.sum(0)
                    e_density += p.sum().item()
                e_strength = torch.cat([e_sr, e_sc])

            # Compute relative errors
            rel_err_strength = torch.abs(strength - e_strength) / torch.clamp(strength, min=1e-10)
            rel_err_density = abs(density - e_density) / max(density, 1e-10)

            max_err_strength = rel_err_strength.max().item()
            max_error = max(max_err_strength, rel_err_density)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_strength={max_err_strength:.2e}  max_err_density={rel_err_density:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"Converged at iter {iteration}")
                converged = True
                break

            delta_theta = e_strength / (strength + 1e-12) - 1
            delta_density = e_density / (density + 1e-12) - 1

            beta_t = get_beta(iteration)

            v_theta = beta_t * v_theta + (1 - beta_t) * delta_theta
            v_density = beta_t * v_density + (1 - beta_t) * delta_density

            theta_mults = theta_mults * (1 + alpha * v_theta)
            density_mult += math.log(1 + alpha * v_density)

        else:
            warnings.warn(
                f"Fixed-point solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return (
            torch.cat([theta_mults, torch.tensor([density_mult], dtype=torch.float64, device=self.device)]),
            {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error},
        )

    def _solve_quasi_newton(
        self,
        x0: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        density: float,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.5,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using diagonal quasi-Newton method.

        dL/dtheta_i = E[s_i] - s_i  and  dL/dmu = E[density] - density.
        Newton step for maximising L: params += gradient / hessian_diag.
        theta must stay positive (clamped to 1e-30); mu is unconstrained.
        """
        n_r, n_c = self.n_rows, self.n_cols

        # theta must be positive; mu is real-valued
        theta_r = x0[:n_r].clamp(min=1e-30)
        theta_c = x0[n_r:n_r + n_c].clamp(min=1e-30)
        mu = x0[-1].item()

        constraints = torch.cat([s_rows, s_cols,
                                 torch.tensor([density], dtype=torch.float64, device=self.device)])
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            exp_neg_mu = math.exp(-mu)
            if chunk is None:
                p_matrix, w_matrix = self._compute_expected_matrices(theta_r, theta_c, mu)
                e_sr = w_matrix.sum(dim=1)
                e_sc = w_matrix.sum(dim=0)
                e_d = p_matrix.sum().item()
                cov_s = (2 - p_matrix) * w_matrix ** 2 / torch.clamp(p_matrix, min=1e-10)
                cov_k = p_matrix * (1 - p_matrix)
                hs_r = cov_s.sum(dim=1); hs_c = cov_s.sum(dim=0)
                hk_sum = cov_k.sum().item()
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                e_d = 0.0
                hs_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hs_c = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                hk_sum = 0.0
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    ts = theta_r[i0:i1, None] + theta_c[None, :]
                    p = exp_neg_mu / (ts + exp_neg_mu)
                    w = p / ts
                    e_sr[i0:i1] = w.sum(1); e_sc += w.sum(0)
                    e_d += p.sum().item()
                    cs = (2 - p) * w ** 2 / torch.clamp(p, min=1e-10)
                    hs_r[i0:i1] = cs.sum(1); hs_c += cs.sum(0)
                    hk_sum += (p * (1 - p)).sum().item()

            # --- theta: log-space Newton step  dL/d(log theta) = (E[s]-s)*theta ---
            theta_r = torch.clamp(
                theta_r * torch.exp(
                    torch.clamp(alpha * (e_sr - s_rows) / (hs_r * theta_r + 1e-10), -10., 10.)
                ), min=1e-30
            )
            theta_c = torch.clamp(
                theta_c * torch.exp(
                    torch.clamp(alpha * (e_sc - s_cols) / (hs_c * theta_c + 1e-10), -10., 10.)
                ), min=1e-30
            )

            # --- mu: direct-space Newton step  dL/dmu = E[density] - density ---
            mu_step = max(-5., min(5., alpha * (e_d - density) / (hk_sum + 1e-10)))
            mu = max(-50., min(50., mu + mu_step))

            # Convergence check
            gradient = torch.cat([
                (e_sr - s_rows), (e_sc - s_cols),
                torch.tensor([e_d - density], dtype=torch.float64, device=self.device)
            ])
            rel_error = torch.abs(gradient) / torch.clamp(constraints, min=1e-10)
            max_err_strength = rel_error[:-1].max().item()
            max_err_density = rel_error[-1].item()
            max_error = max(max_err_strength, max_err_density)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_strength={max_err_strength:.2e}  max_err_density={max_err_density:.2e}")

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

        return (
            torch.cat([theta_r, theta_c, torch.tensor([mu], dtype=torch.float64, device=self.device)]),
            {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error},
        )

    def _solve_coordinate(
        self,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        density: float,
        x0: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        chunk: Optional[int] = None,
        n_bisect: int = 60,
    ) -> torch.Tensor:
        """
        Coordinate descent with bisection in PyTorch.

        Each coordinate group is solved via 60-step bisection. The bisection
        bounds (lo/hi) are tensors so all nodes in a group are updated in
        parallel, but the matrix sums are computed in sequential chunks for
        memory efficiency.
        theta_rows/theta_cols are bisected in log-space (decreasing functions).
        mu is bisected in real space [-50, 50] (density is decreasing in mu).
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk
        tr = x0[:n_r].clamp(min=1e-30)
        tc = x0[n_r:n_r + n_c].clamp(min=1e-30)
        mu = x0[-1].item()

        max_rel = float("inf")
        converged = False
        w = len(str(max_iter))
        for outer in range(max_iter):
            # --- Bisect theta_rows (decreasing: f > target → lo = mid) ---
            lo = tr.new_full((n_r,), -30.)
            hi = tr.new_full((n_r,), 30.)
            exp_neg_mu = math.exp(-mu)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                tr_mid = log_mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    ts = tr_mid[i0:i1, None] + tc[None, :]
                    f[i0:i1] = (exp_neg_mu / (ts * (ts + exp_neg_mu))).sum(1)
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
                    ts = tr[i0:i1, None] + tc_mid[None, :]
                    f += (exp_neg_mu / (ts * (ts + exp_neg_mu))).sum(0)
                above = f > s_cols
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            tc = ((lo + hi) / 2).exp()

            # --- Bisect mu in real space (density decreasing in mu) ---
            mu_lo, mu_hi = -50.0, 50.0
            for _ in range(n_bisect):
                mu_mid = (mu_lo + mu_hi) / 2
                e = math.exp(-mu_mid)
                f_d = 0.0
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    ts = tr[i0:i1, None] + tc[None, :]
                    f_d += (e / (ts + e)).sum().item()
                if f_d > density:
                    mu_lo = mu_mid
                else:
                    mu_hi = mu_mid
            mu = (mu_lo + mu_hi) / 2

            # --- Convergence check ---
            exp_neg_mu = math.exp(-mu)
            e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            e_d = 0.0
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                ts = tr[i0:i1, None] + tc[None, :]
                p = exp_neg_mu / (ts + exp_neg_mu)
                w = p / ts
                e_sr[i0:i1] = w.sum(1)
                e_sc += w.sum(0)
                e_d += p.sum().item()

            max_err_strength = max(
                (torch.abs(s_rows - e_sr) / s_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(s_cols - e_sc) / s_cols.clamp(min=1e-10)).max().item(),
            )
            max_err_density = abs(density - e_d) / max(density, 1e-10)
            max_rel = max(max_err_strength, max_err_density)
            if verbose:
                print(f"Iter {outer:{w}d}  max_err_strength={max_err_strength:.2e}  max_err_density={max_err_density:.2e}")
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

        return (
            torch.cat([tr, tc, torch.tensor([mu], dtype=torch.float64, device=self.device)]),
            {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel},
        )

    def connection_probability(self) -> torch.Tensor:
        """
        Compute the matrix of connection probabilities.

        Returns
        -------
        prob : torch.Tensor
            Matrix where entry (i,j) is P(A_ij = 1).
        """
        self._check_solved()

        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]
        exp_neg_density = math.exp(-self.density_param)

        return exp_neg_density / (theta_sum + exp_neg_density)

    def expected_matrix(self) -> torch.Tensor:
        """
        Compute expected biadjacency matrix.

        Note: Only entries corresponding to existing edges will be non-zero
        when conditioning on topology.

        Returns
        -------
        expected : torch.Tensor
            Matrix where entry (i,j) is the expected weight E[w_ij].
        """
        self._check_solved()

        p_matrix = self.connection_probability()
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        return p_matrix / theta_sum

    def expected_matrix_conditional(self) -> torch.Tensor:
        """
        Compute expected weight matrix conditional on topology.

        This gives E[w_ij | A_ij = 1] for existing edges.

        Returns
        -------
        expected : torch.Tensor
            Matrix where entry (i,j) is E[w_ij | A_ij = 1] * A_ij.
        """
        self._check_solved()

        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]

        # E[w_ij | A_ij = 1] = 1 / (theta_i + theta_j)
        expected_weight = 1.0 / theta_sum

        # Only on existing edges
        return expected_weight * self.topology

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None,
        conditional: bool = True
    ) -> torch.Tensor:
        """
        Generate random samples from the BiPECM ensemble.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility.
        conditional : bool, default=True
            If True, preserve exact topology (sample weights only on existing edges).
            If False, sample both topology and weights.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples, n_rows, n_cols).
            Each sample has the same sparsity pattern as the original if conditional=True.
        """
        self._check_solved()

        if seed is not None:
            torch.manual_seed(seed)

        samples = torch.zeros(n_samples, self.n_rows, self.n_cols,
                              dtype=torch.float64, device=self.device)

        # Rate for exponential distribution
        rate_matrix = self.theta_rows[:, None] + self.theta_cols[None, :]

        if conditional:
            # Sample weights only on existing edges (preserve topology)
            for s in range(n_samples):
                weights = Exponential(rate_matrix).sample()
                samples[s] = weights * self.topology
        else:
            # Sample both topology and weights
            p_matrix = self.connection_probability()
            for s in range(n_samples):
                topology = (torch.rand(self.n_rows, self.n_cols,
                                       dtype=torch.float64, device=self.device) < p_matrix).to(torch.float64)
                weights = Exponential(rate_matrix).sample()
                samples[s] = topology * weights

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

        s_rows = self.strength_sequence_rows
        s_cols = self.strength_sequence_cols
        density = self.topology.sum()

        # Hamiltonian terms
        hamiltonian = (
            -(self.theta_rows * s_rows).sum()
            - (self.theta_cols * s_cols).sum()
            - self.density_param * density
        )

        # Partition function
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]
        exp_neg_density = math.exp(-self.density_param)

        partition = -torch.log(1 + exp_neg_density / theta_sum).sum()

        return (hamiltonian + partition).item()

    def n_parameters(self) -> int:
        """Return the number of free parameters in the model."""
        # theta_rows, theta_cols, and one density parameter
        return self.n_rows + self.n_cols + 1

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

        Under BiPECM, the p-value combines the probability of having an edge
        and the probability of observing that weight or more given an edge.

        P(W_ij >= w) = p_ij * exp(-w / E[W_ij | A_ij=1])

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
