"""
Bipartite Configuration model with Random Matrix Approximation (BiCReMA).

BiCReMA is an approximation of the BiECM that decouples the topological
information from the weights by solving BiCM and BiWCM independently
and in sequence.
"""

from typing import Optional, Literal, Tuple
import math
import time
import warnings
import torch
from torch.distributions import Exponential

from bicm.wbnm.models.base import BipartiteModel, ArrayLike, ConvergenceWarning


class BiCReMA(BipartiteModel):
    """
    Bipartite Configuration model with Random Matrix Approximation.

    BiCReMA approximates the BiECM by separating topology from weights.
    It solves two independent sub-problems in sequence:

    1. **BiCM stage**: find x_rows, x_cols to reproduce the degree sequence.
       Connection probability: p_ij = x_i * x_j / (1 + x_i * x_j)

    2. **BiWCM stage**: using the BiCM probabilities as a fixed topology,
       find H_rows, H_cols to reproduce the strength sequence.
       Expected weight: E[w_ij] = p_ij^BiCM / (H_i + H_j)

    Compared to the BiECM, this model uses the same number of parameters
    (2*(n_rows + n_cols)) but its parameters are decoupled: x controls the
    topology and H controls the weights.

    Parameters
    ----------
    biadjacency : ArrayLike
        The observed biadjacency matrix of shape (n_rows, n_cols).
    device : str, optional
        Device for computation. Auto-detects GPU if None.

    Attributes
    ----------
    x_rows : torch.Tensor
        BiCM fitness parameters for row nodes (after solving).
    x_cols : torch.Tensor
        BiCM fitness parameters for column nodes (after solving).
    H_rows : torch.Tensor
        BiWCM fitness parameters for row nodes (after solving).
    H_cols : torch.Tensor
        BiWCM fitness parameters for column nodes (after solving).

    Examples
    --------
    >>> import torch
    >>> from bicm.wbnm.models import BiCReMA
    >>>
    >>> W = torch.poisson(torch.full((10, 20), 5.0)) * (torch.rand(10, 20) > 0.5).float()
    >>> model = BiCReMA(W)
    >>> model.solve()
    >>> samples = model.sample(n_samples=100)

    Notes
    -----
    BiCReMA is not a proper maximum-entropy model: the two sub-problems are
    solved independently. As a consequence, the log-likelihood factorises into
    a BiCM topology term and a conditional exponential-weight term.

    References
    ----------
    - Garuccio et al., "Multiscale network renormalization: scale-invariance
           beyond localization", Physical Review Research (2023)
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        super().__init__(biadjacency, device=device)

        if (self.biadjacency < 0).any():
            raise ValueError("biadjacency contains negative values. BiCReMA requires non-negative weights.")

        self.x_rows: Optional[torch.Tensor] = None
        self.x_cols: Optional[torch.Tensor] = None
        self.H_rows: Optional[torch.Tensor] = None
        self.H_cols: Optional[torch.Tensor] = None

        self.converged_bicm: Optional[bool] = None
        self.n_iter_bicm: Optional[int] = None
        self.time_s_bicm: Optional[float] = None
        self.final_residual_bicm: Optional[float] = None
        self.converged_biwcm: Optional[bool] = None
        self.n_iter_biwcm: Optional[int] = None
        self.time_s_biwcm: Optional[float] = None
        self.final_residual_biwcm: Optional[float] = None

    # ------------------------------------------------------------------
    # Public solve
    # ------------------------------------------------------------------

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
        **kwargs,
    ) -> "BiCReMA":
        """
        Solve for the BiCReMA parameters via two-stage iteration.

        **Stage 1 (BiCM)**: find x_rows, x_cols matching the degree sequence.
        **Stage 2 (BiWCM conditioned)**: with p^BiCM fixed, find H_rows, H_cols
        matching the strength sequence.

        Parameters
        ----------
        method : {"fixed-point", "quasi-newton", "coordinate"}, default="fixed-point"
            Optimisation method applied to both stages:

            - "fixed-point": multiplicative update (BiCM) / momentum update (BiWCM).
              Robust, good default.
            - "quasi-newton": diagonal Newton step in log-space (BiCM) and
              direct space (BiWCM). Faster near the solution.
            - "coordinate": independent bisection per node (~1e-18 precision).
              Most robust for extreme networks.

        x0 : torch.Tensor, optional
            Initial guess of shape (2*(n_rows + n_cols),) ordered as
            [H_rows, H_cols, x_rows, x_cols] (same convention as BiECM).
            If None, uses default initialisation.
        tol : float, default=1e-8
            Convergence tolerance (max relative error on each sequence).
        max_iter : int, default=10000
            Maximum number of iterations for each stage.
        verbose : bool, default=False
            Print convergence information per stage.
        alpha : float, optional
            Step size. Defaults: 1.0 (fixed-point), 0.5 (quasi-newton).
            Ignored by coordinate solver.
        beta : float, default=0.9
            Momentum coefficient for the BiWCM fixed-point stage.
        beta_schedule : {None, "warmup", "linear"}, default=None
            Schedule for beta (BiWCM fixed-point only):
            - None: beta stays fixed.
            - "warmup": beta until warmup_steps, then 0.999.
            - "linear": ramps from beta to 0.999 over warmup_steps.
        warmup_steps : int, default=500
            Steps for beta schedule.
        chunk : int, default=256
            Row chunk size for memory-efficient matrix operations
            (coordinate solver only).

        Returns
        -------
        self : BiCReMA

        Raises
        ------
        ConvergenceWarning
            If either stage does not converge within max_iter.
        ValueError
            If an unknown method is requested.
        """
        if method != "fixed-point" and (
            beta != 0.9 or beta_schedule is not None or warmup_steps != 500
        ):
            warnings.warn(
                f"beta, beta_schedule, and warmup_steps are only used with "
                f"method='fixed-point' (Stage 2 BiWCM), not '{method}'.",
                UserWarning,
                stacklevel=2,
            )

        k_rows = self.degree_sequence_rows.to(torch.float64)
        k_cols = self.degree_sequence_cols.to(torch.float64)
        s_rows = self.strength_sequence_rows.to(torch.float64)
        s_cols = self.strength_sequence_cols.to(torch.float64)

        if x0 is None:
            x0 = self._initial_guess(k_rows, k_cols, s_rows, s_cols)
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self.device)

        n_r, n_c = self.n_rows, self.n_cols
        # Layout: [H_rows, H_cols, x_rows, x_cols]
        H_rows = x0[:n_r].clone()
        H_cols = x0[n_r:n_r + n_c].clone()
        x_rows = x0[n_r + n_c:2 * n_r + n_c].clone()
        x_cols = x0[2 * n_r + n_c:].clone()

        _alpha_fp = alpha if alpha is not None else 1.0
        _alpha_qn = alpha if alpha is not None else 0.5

        t_start = time.perf_counter()

        # ---- Stage 1: BiCM ----
        t0 = time.perf_counter()
        if method == "fixed-point":
            x_rows, x_cols, info_bicm = self._solve_bicm_fixed_point(
                x_rows, x_cols, k_rows, k_cols,
                tol=tol, max_iter=max_iter, verbose=verbose, alpha=_alpha_fp,
                chunk=chunk,
            )
        elif method == "quasi-newton":
            x_rows, x_cols, info_bicm = self._solve_bicm_quasi_newton(
                x_rows, x_cols, k_rows, k_cols,
                tol=tol, max_iter=max_iter, verbose=verbose, alpha=_alpha_qn,
                chunk=chunk,
            )
        elif method == "coordinate":
            x_rows, x_cols, info_bicm = self._solve_bicm_coordinate(
                x_rows, x_cols, k_rows, k_cols,
                tol=tol, max_iter=max_iter, verbose=verbose, chunk=chunk,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}. "
                             "Choose from 'fixed-point', 'quasi-newton', 'coordinate'.")
        self.time_s_bicm = time.perf_counter() - t0
        self.converged_bicm = info_bicm["converged"]
        self.n_iter_bicm = info_bicm["n_iter"]
        self.final_residual_bicm = info_bicm["final_residual"]

        # ---- Stage 2: BiWCM conditioned on BiCM probabilities ----
        # When chunk is None, materialise p_bicm once for efficiency.
        # When chunk is set, pass x_rows/x_cols so stage 2 computes p_bicm on the fly.
        if chunk is None:
            x_outer = x_rows[:, None] * x_cols[None, :]
            p_bicm = x_outer / (1.0 + x_outer)
        else:
            p_bicm = None  # computed on-the-fly inside stage-2 solvers

        t0 = time.perf_counter()
        if method == "fixed-point":
            H_rows, H_cols, info_biwcm = self._solve_biwcm_fixed_point(
                H_rows, H_cols, s_rows, s_cols, p_bicm,
                tol=tol, max_iter=max_iter, verbose=verbose, alpha=_alpha_fp,
                beta=beta, beta_schedule=beta_schedule, warmup_steps=warmup_steps,
                chunk=chunk, x_rows=x_rows, x_cols=x_cols,
            )
        elif method == "quasi-newton":
            H_rows, H_cols, info_biwcm = self._solve_biwcm_quasi_newton(
                H_rows, H_cols, s_rows, s_cols, p_bicm,
                tol=tol, max_iter=max_iter, verbose=verbose, alpha=_alpha_qn,
                chunk=chunk, x_rows=x_rows, x_cols=x_cols,
            )
        elif method == "coordinate":
            H_rows, H_cols, info_biwcm = self._solve_biwcm_coordinate(
                H_rows, H_cols, s_rows, s_cols, p_bicm,
                tol=tol, max_iter=max_iter, verbose=verbose, chunk=chunk,
                x_rows=x_rows, x_cols=x_cols,
            )
        self.time_s_biwcm = time.perf_counter() - t0
        self.converged_biwcm = info_biwcm["converged"]
        self.n_iter_biwcm = info_biwcm["n_iter"]
        self.final_residual_biwcm = info_biwcm["final_residual"]

        self.time_s = time.perf_counter() - t_start
        self.converged = self.converged_bicm and self.converged_biwcm
        self.n_iter = self.n_iter_bicm + self.n_iter_biwcm
        self.final_residual = max(self.final_residual_bicm, self.final_residual_biwcm)

        self.x_rows = x_rows
        self.x_cols = x_cols
        self.H_rows = H_rows
        self.H_cols = H_cols

        self.is_solved = True
        self._params = {
            "x_rows": self.x_rows,
            "x_cols": self.x_cols,
            "H_rows": self.H_rows,
            "H_cols": self.H_cols,
        }

        return self

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initial_guess(
        self,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default initial parameters.

        Layout: [H_rows, H_cols, x_rows, x_cols]
        x via the classical degree heuristic; H via k/s (accounts for
        the p_ij weighting in Stage 2: E[s_i] = sum_j p_ij/(H_i+H_j),
        where sum_j p_ij ≈ k_i after BiCM convergence).
        """
        k_rows = torch.clamp(k_rows, min=1e-10)
        k_cols = torch.clamp(k_cols, min=1e-10)
        s_rows = torch.clamp(s_rows, min=1e-10)
        s_cols = torch.clamp(s_cols, min=1e-10)

        k_tot = k_rows.sum()
        x_rows = k_rows / torch.sqrt(k_tot)
        x_cols = k_cols / torch.sqrt(k_tot)
        H_rows = k_rows / s_rows
        H_cols = k_cols / s_cols
        return torch.cat([H_rows, H_cols, x_rows, x_cols])

    # ------------------------------------------------------------------
    # Stage 1 — BiCM solvers  (parametrisation: x, p_ij = x_i x_j / (1+x_i x_j))
    # ------------------------------------------------------------------

    def _solve_bicm_fixed_point(
        self,
        x_rows: torch.Tensor,
        x_cols: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        chunk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiplicative fixed-point for the BiCM stage.

        Update rule: x_i *= (k_i / E[k_i])^alpha
        """
        n_r = self.n_rows
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            if chunk is None:
                xo = x_rows[:, None] * x_cols[None, :]
                p = xo / (1.0 + xo)
                e_kr = p.sum(dim=1)
                e_kc = p.sum(dim=0)
            else:
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros_like(x_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = x_rows[i0:i1, None] * x_cols[None, :]
                    p = xo / (1.0 + xo)
                    e_kr[i0:i1] = p.sum(1)
                    e_kc += p.sum(0)

            err_r = torch.abs(k_rows - e_kr) / torch.clamp(k_rows, min=1e-10)
            err_c = torch.abs(k_cols - e_kc) / torch.clamp(k_cols, min=1e-10)
            max_error = max(err_r.max().item(), err_c.max().item())

            if verbose and iteration % 100 == 0:
                print(f"  [BiCM fp]  iter {iteration:{w}d}  max_err_degree={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"  [BiCM fp]  converged at iter {iteration}")
                converged = True
                break

            x_rows = x_rows * (k_rows / torch.clamp(e_kr, min=1e-12)) ** alpha
            x_cols = x_cols * (k_cols / torch.clamp(e_kc, min=1e-12)) ** alpha

        else:
            warnings.warn(
                f"BiCM fixed-point did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return x_rows, x_cols, {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_bicm_quasi_newton(
        self,
        x_rows: torch.Tensor,
        x_cols: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.5,
        chunk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagonal quasi-Newton for the BiCM stage, working in log-x space.

        Log-likelihood gradient: dL/d(log x_i) = k_i - E[k_i]
        Diagonal negative-Hessian: sum_j p_ij * (1 - p_ij)
        Newton step (log-space): log(x_i) += alpha * (k_i - E[k_i]) / D_ii
        """
        n_r = self.n_rows
        x_rows = x_rows.clamp(min=1e-30)
        x_cols = x_cols.clamp(min=1e-30)
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            if chunk is None:
                xo = x_rows[:, None] * x_cols[None, :]
                p = xo / (1.0 + xo)
                e_kr = p.sum(dim=1)
                e_kc = p.sum(dim=0)
                pq = p * (1.0 - p)
                hess_r = pq.sum(dim=1)
                hess_c = pq.sum(dim=0)
            else:
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros_like(x_cols)
                hess_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hess_c = torch.zeros_like(x_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = x_rows[i0:i1, None] * x_cols[None, :]
                    p = xo / (1.0 + xo)
                    e_kr[i0:i1] = p.sum(1)
                    e_kc += p.sum(0)
                    pq = p * (1.0 - p)
                    hess_r[i0:i1] = pq.sum(1)
                    hess_c += pq.sum(0)

            err_r = torch.abs(k_rows - e_kr) / torch.clamp(k_rows, min=1e-10)
            err_c = torch.abs(k_cols - e_kc) / torch.clamp(k_cols, min=1e-10)
            max_error = max(err_r.max().item(), err_c.max().item())

            if verbose and iteration % 100 == 0:
                print(f"  [BiCM qn]  iter {iteration:{w}d}  max_err_degree={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"  [BiCM qn]  converged at iter {iteration}")
                converged = True
                break

            x_rows = torch.clamp(
                x_rows * torch.exp(
                    torch.clamp(alpha * (k_rows - e_kr) / (hess_r + 1e-10), -10., 10.)
                ), min=1e-30,
            )
            x_cols = torch.clamp(
                x_cols * torch.exp(
                    torch.clamp(alpha * (k_cols - e_kc) / (hess_c + 1e-10), -10., 10.)
                ), min=1e-30,
            )

        else:
            warnings.warn(
                f"BiCM quasi-Newton did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return x_rows, x_cols, {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_bicm_coordinate(
        self,
        x_rows: torch.Tensor,
        x_cols: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        chunk: Optional[int] = None,
        n_bisect: int = 60,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coordinate descent with bisection for the BiCM stage.

        f(log x_i) = sum_j x_i*x_j / (1 + x_i*x_j) is strictly increasing.
        Bisect over [-30, 30] in log-x space: ~1e-18 precision in 60 steps.
        All nodes in a group are updated in parallel; matrix sums are computed
        in row-chunks for memory efficiency.
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk
        # work in log-x space
        log_xr = torch.log(x_rows.clamp(min=1e-30))
        log_xc = torch.log(x_cols.clamp(min=1e-30))
        max_rel = float("inf")
        converged = False

        w = len(str(max_iter))
        for outer in range(max_iter):
            # --- bisect log_xr (f increasing: f > k → hi = mid) ---
            lo = log_xr.new_full((n_r,), -30.)
            hi = log_xr.new_full((n_r,), 30.)
            xc = log_xc.exp()   # materialise once; log_xc is fixed during this bisection
            for _ in range(n_bisect):
                mid = (lo + hi) / 2
                xr_mid = mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr_mid[i0:i1, None] * xc[None, :]
                    f[i0:i1] = (xo / (1.0 + xo)).sum(1)
                above = f > k_rows
                hi = torch.where(above, mid, hi)
                lo = torch.where(above, lo, mid)
            log_xr = (lo + hi) / 2

            # --- bisect log_xc (f increasing: f > k → hi = mid) ---
            lo = log_xc.new_full((n_c,), -30.)
            hi = log_xc.new_full((n_c,), 30.)
            xr = log_xr.exp()
            for _ in range(n_bisect):
                mid = (lo + hi) / 2
                xc_mid = mid.exp()
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    xo = xr[i0:i1, None] * xc_mid[None, :]
                    f += (xo / (1.0 + xo)).sum(0)
                above = f > k_cols
                hi = torch.where(above, mid, hi)
                lo = torch.where(above, lo, mid)
            log_xc = (lo + hi) / 2

            # --- convergence check ---
            xr = log_xr.exp()
            xc = log_xc.exp()
            e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_kc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                xo = xr[i0:i1, None] * xc[None, :]
                p = xo / (1.0 + xo)
                e_kr[i0:i1] = p.sum(1)
                e_kc += p.sum(0)

            max_rel = max(
                (torch.abs(k_rows - e_kr) / k_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(k_cols - e_kc) / k_cols.clamp(min=1e-10)).max().item(),
            )
            if verbose:
                print(f"  [BiCM cd]  iter {outer:{w}d}  max_err_degree={max_rel:.2e}")
            if max_rel < tol:
                if verbose:
                    print(f"  [BiCM cd]  converged at iter {outer}")
                converged = True
                break
        else:
            warnings.warn(
                f"BiCM coordinate solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_rel:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return log_xr.exp(), log_xc.exp(), {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel}

    # ------------------------------------------------------------------
    # Stage 2 — BiWCM conditioned solvers
    # (p_bicm fixed; E[s_i] = sum_j p_ij / (H_i + H_j))
    # ------------------------------------------------------------------

    def _solve_biwcm_fixed_point(
        self,
        H_rows: torch.Tensor,
        H_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        p_bicm: Optional[torch.Tensor],
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        beta: float = 0.9,
        beta_schedule: Optional[Literal["warmup", "linear"]] = None,
        warmup_steps: int = 500,
        chunk: Optional[int] = None,
        x_rows: Optional[torch.Tensor] = None,
        x_cols: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Momentum fixed-point for the BiWCM stage.

        Update: H_i *= (1 + alpha * v_i)
        where v accumulates E[s_i]/s_i - 1 with momentum beta.
        When chunk is set, p_bicm is computed on-the-fly from x_rows/x_cols
        to avoid materialising the full n_rows x n_cols probability matrix.
        """
        if beta_schedule is None:
            get_beta = lambda t: beta
        elif beta_schedule == "warmup":
            get_beta = lambda t: beta if t < warmup_steps else 0.999
        elif beta_schedule == "linear":
            get_beta = lambda t: min(0.999, beta + (0.999 - beta) * t / warmup_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        n_r = self.n_rows
        H = torch.cat([H_rows, H_cols])
        v = torch.zeros_like(H)
        strength = torch.cat([s_rows, s_cols])
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            Hr, Hc = H[:n_r], H[n_r:]
            if chunk is None:
                H_sum = Hr[:, None] + Hc[None, :]
                e_sr = (p_bicm / H_sum).sum(dim=1)
                e_sc = (p_bicm / H_sum).sum(dim=0)
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros_like(Hc)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = x_rows[i0:i1, None] * x_cols[None, :]
                    p_chunk = xo / (1.0 + xo)
                    H_sum = Hr[i0:i1, None] + Hc[None, :]
                    w = p_chunk / H_sum
                    e_sr[i0:i1] = w.sum(1)
                    e_sc += w.sum(0)
            e_s = torch.cat([e_sr, e_sc])

            err = torch.abs(strength - e_s) / torch.clamp(strength, min=1e-10)
            max_error = err.max().item()

            if verbose and iteration % 100 == 0:
                print(f"  [BiWCM fp] iter {iteration:{w}d}  max_err_strength={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"  [BiWCM fp] converged at iter {iteration}")
                converged = True
                break

            delta = e_s / (strength + 1e-12) - 1.0
            beta_t = get_beta(iteration)
            v = beta_t * v + (1.0 - beta_t) * delta
            H = H * (1.0 + alpha * v)

        else:
            warnings.warn(
                f"BiWCM fixed-point did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return H[:n_r], H[n_r:], {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_biwcm_quasi_newton(
        self,
        H_rows: torch.Tensor,
        H_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        p_bicm: Optional[torch.Tensor],
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.5,
        chunk: Optional[int] = None,
        x_rows: Optional[torch.Tensor] = None,
        x_cols: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagonal quasi-Newton for the BiWCM stage.

        Root equation: f(H_i) = E[s_i] - s_i = 0
        Derivative:    df/dH_i = -sum_j p_ij / (H_i + H_j)^2
        Newton step:   H_i += alpha * (E[s_i] - s_i) / sum_j p_ij / (H_i + H_j)^2

        H must stay positive; clamped to 1e-30.
        When chunk is set, p_bicm is computed on-the-fly from x_rows/x_cols.
        """
        n_r = self.n_rows
        H_rows = H_rows.clamp(min=1e-30)
        H_cols = H_cols.clamp(min=1e-30)
        max_error = float("inf")
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            if chunk is None:
                H_sum = H_rows[:, None] + H_cols[None, :]
                w = p_bicm / H_sum
                e_sr = w.sum(dim=1)
                e_sc = w.sum(dim=0)
                hess_r = (p_bicm / H_sum ** 2).sum(dim=1)
                hess_c = (p_bicm / H_sum ** 2).sum(dim=0)
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros_like(H_cols)
                hess_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hess_c = torch.zeros_like(H_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    xo = x_rows[i0:i1, None] * x_cols[None, :]
                    p_chunk = xo / (1.0 + xo)
                    H_sum = H_rows[i0:i1, None] + H_cols[None, :]
                    w = p_chunk / H_sum
                    e_sr[i0:i1] = w.sum(1)
                    e_sc += w.sum(0)
                    hess_r[i0:i1] = (p_chunk / H_sum ** 2).sum(1)
                    hess_c += (p_chunk / H_sum ** 2).sum(0)

            err_r = torch.abs(s_rows - e_sr) / torch.clamp(s_rows, min=1e-10)
            err_c = torch.abs(s_cols - e_sc) / torch.clamp(s_cols, min=1e-10)
            max_error = max(err_r.max().item(), err_c.max().item())

            if verbose and iteration % 100 == 0:
                print(f"  [BiWCM qn] iter {iteration:{w}d}  max_err_strength={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"  [BiWCM qn] converged at iter {iteration}")
                converged = True
                break

            H_rows = torch.clamp(
                H_rows * torch.exp(
                    torch.clamp(alpha * (e_sr - s_rows) / (hess_r * H_rows + 1e-10), -10., 10.)
                ), min=1e-30
            )
            H_cols = torch.clamp(
                H_cols * torch.exp(
                    torch.clamp(alpha * (e_sc - s_cols) / (hess_c * H_cols + 1e-10), -10., 10.)
                ), min=1e-30
            )

        else:
            warnings.warn(
                f"BiWCM quasi-Newton did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return H_rows, H_cols, {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_biwcm_coordinate(
        self,
        H_rows: torch.Tensor,
        H_cols: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        p_bicm: Optional[torch.Tensor],
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        chunk: Optional[int] = None,
        n_bisect: int = 60,
        x_rows: Optional[torch.Tensor] = None,
        x_cols: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coordinate descent with bisection for the BiWCM stage.

        f(log H_i) = sum_j p_ij / (H_i + H_j) is strictly decreasing in H_i.
        Bisect over [-30, 30] in log-H space: ~1e-18 precision in 60 steps.
        All nodes in a group are updated in parallel; matrix sums are computed
        in row-chunks for memory efficiency.
        When chunk is set, p_bicm is computed on-the-fly from x_rows/x_cols.
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk
        H_r = H_rows.clamp(min=1e-30)
        H_c = H_cols.clamp(min=1e-30)
        max_rel = float("inf")
        converged = False

        def _p_chunk(i0, i1):
            """Return the p_bicm slice for rows [i0:i1]."""
            if p_bicm is not None:
                return p_bicm[i0:i1]
            xo = x_rows[i0:i1, None] * x_cols[None, :]
            return xo / (1.0 + xo)

        w = len(str(max_iter))
        for outer in range(max_iter):
            # --- bisect log H_rows (f decreasing: f > s → H too small → lo = mid) ---
            lo = H_r.new_full((n_r,), -30.)
            hi = H_r.new_full((n_r,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                H_r_mid = log_mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    H_sum = H_r_mid[i0:i1, None] + H_c[None, :]
                    f[i0:i1] = (_p_chunk(i0, i1) / H_sum).sum(1)
                above = f > s_rows
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            H_r = ((lo + hi) / 2).exp()

            # --- bisect log H_cols (f decreasing: f > s → H too small → lo = mid) ---
            lo = H_c.new_full((n_c,), -30.)
            hi = H_c.new_full((n_c,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                H_c_mid = log_mid.exp()
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    H_sum = H_r[i0:i1, None] + H_c_mid[None, :]
                    f += (_p_chunk(i0, i1) / H_sum).sum(0)
                above = f > s_cols
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            H_c = ((lo + hi) / 2).exp()

            # --- convergence check ---
            e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                H_sum = H_r[i0:i1, None] + H_c[None, :]
                w = _p_chunk(i0, i1) / H_sum
                e_sr[i0:i1] = w.sum(1)
                e_sc += w.sum(0)

            max_rel = max(
                (torch.abs(s_rows - e_sr) / s_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(s_cols - e_sc) / s_cols.clamp(min=1e-10)).max().item(),
            )
            if verbose:
                print(f"  [BiWCM cd] iter {outer:{w}d}  max_err_strength={max_rel:.2e}")
            if max_rel < tol:
                if verbose:
                    print(f"  [BiWCM cd] converged at iter {outer}")
                converged = True
                break
        else:
            warnings.warn(
                f"BiWCM coordinate solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_rel:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return H_r, H_c, {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connection_probability(self) -> torch.Tensor:
        """
        Compute the BiCM connection-probability matrix.

        p_ij = x_i * x_j / (1 + x_i * x_j)

        Returns
        -------
        p : torch.Tensor of shape (n_rows, n_cols)
        """
        self._check_solved()
        x_outer = self.x_rows[:, None] * self.x_cols[None, :]
        return x_outer / (1.0 + x_outer)

    def expected_matrix(self) -> torch.Tensor:
        """
        Compute the expected biadjacency matrix.

        E[w_ij] = p_ij / (H_i + H_j)

        Returns
        -------
        expected : torch.Tensor of shape (n_rows, n_cols)
        """
        self._check_solved()
        p = self.connection_probability()
        H_sum = self.H_rows[:, None] + self.H_cols[None, :]
        return p / H_sum

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate random samples from the BiCReMA ensemble.

        For each edge (i,j):
        1. Sample topology: A_ij ~ Bernoulli(p_ij^BiCM)
        2. If A_ij = 1, sample weight: w_ij ~ Exponential(H_i + H_j)

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
        rate_matrix = self.H_rows[:, None] + self.H_cols[None, :]

        samples = torch.zeros(
            n_samples, self.n_rows, self.n_cols,
            dtype=torch.float64, device=self.device,
        )
        for s in range(n_samples):
            topology = (
                torch.rand(self.n_rows, self.n_cols,
                           dtype=torch.float64, device=self.device) < p_matrix
            ).to(torch.float64)
            samples[s] = topology * Exponential(rate_matrix).sample()

        return samples

    def log_likelihood(self) -> float:
        """
        Approximate log-likelihood of the BiCReMA model.

        Factorises as:

            L = L_BiCM(topology | x) + L_weight(weights | topology, H)

        where::

            L_BiCM   = sum_i k_i * log(x_i^r) + sum_j k_j * log(x_j^c)
                       - sum_ij log(1 + x_i * x_j)

            L_weight = sum_{ij: A_ij=1} [log(H_i + H_j) - (H_i + H_j) * w_ij]

        Note: since BiCReMA is not a proper maximum-entropy model this is
        only an approximation to the full ensemble log-likelihood.

        Returns
        -------
        ll : float
        """
        self._check_solved()

        k_rows = self.degree_sequence_rows.to(torch.float64)
        k_cols = self.degree_sequence_cols.to(torch.float64)

        ll_bicm = (
            (torch.log(self.x_rows) * k_rows).sum()
            + (torch.log(self.x_cols) * k_cols).sum()
            - torch.log1p(self.x_rows[:, None] * self.x_cols[None, :]).sum()
        )

        A = (self.biadjacency > 0).to(torch.float64)
        H_sum = self.H_rows[:, None] + self.H_cols[None, :]
        ll_weight = (A * (torch.log(H_sum) - H_sum * self.biadjacency)).sum()

        return (ll_bicm + ll_weight).item()

    def n_parameters(self) -> int:
        """Return the number of free parameters (same as BiECM)."""
        return 2 * (self.n_rows + self.n_cols)

    def aic(self) -> float:
        """AIC = -2*L + 2*k  (based on approximate log-likelihood)."""
        return -2 * self.log_likelihood() + 2 * self.n_parameters()

    def aicc(self) -> float:
        """Corrected AIC for small sample sizes."""
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        return self.aic() + (2 * k * (k + 1)) / (n - k - 1)

    def bic(self) -> float:
        """BIC = -2*L + k*log(n)  (based on approximate log-likelihood)."""
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        return -2 * self.log_likelihood() + k * math.log(n)

    def pvalues(self, observed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Analytical p-values under the BiCReMA null model.

        P(W_ij >= w) = p_ij^BiCM^{A_ij} * exp(-w * (H_i + H_j))

        Zero weights yield p-value = 1.

        Parameters
        ----------
        observed : torch.Tensor, optional
            Weight matrix. If None, uses the matrix passed at construction.

        Returns
        -------
        pvalues : torch.Tensor of shape (n_rows, n_cols)
        """
        self._check_solved()

        if observed is None:
            observed = self.biadjacency
        observed = torch.as_tensor(observed, dtype=torch.float64, device=self.device)

        p = self.connection_probability()
        H_sum = self.H_rows[:, None] + self.H_cols[None, :]
        # Product computed in logarithms, see BiECM.pvalues
        linked = observed > 0
        return torch.where(linked, torch.exp(torch.log(p) - observed * H_sum), torch.ones_like(observed))
