"""
Bipartite Weighted Configuration Model (BiWCM).

The BiWCM preserves the strength sequence of the bipartite network
in expectation, without constraining the topology.
"""

from typing import Optional, Literal
import math
import time
import warnings
import torch
from torch.distributions import Exponential

from bicm.wbnm.models.base import BipartiteModel, ArrayLike, ConvergenceWarning


class BiWCM(BipartiteModel):
    """
    Bipartite Weighted Configuration Model.

    The BiWCM is a maximum entropy null model that preserves:
    - Expected strength sequence (sum of weights per node)

    Unlike BiECM, it does not separately constrain the degree sequence,
    making it a simpler model suitable when only total weight matters.

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

    Examples
    --------
    >>> import torch
    >>> from bicm.wbnm.models import BiWCM
    >>>
    >>> W = torch.distributions.Exponential(0.2).sample((10, 20))
    >>> model = BiWCM(W)
    >>> model.solve()
    >>> samples = model.sample(n_samples=100)

    References
    ----------
    - Squartini et al., "Analytical maximum-likelihood method to detect
           patterns in real networks", New Journal of Physics (2011)
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        super().__init__(biadjacency, device=device)

        if (self.biadjacency < 0).any():
            raise ValueError("biadjacency contains negative values. BiWCM requires non-negative weights.")

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
    ) -> "BiWCM":
        """
        Solve for the fitness parameters.

        Parameters
        ----------
        method : {"fixed-point", "quasi-newton", "coordinate"}, default="fixed-point"
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
            Step size. Defaults: 1.0 (fixed-point), 0.5 (quasi-newton).
            Ignored by coordinate solver.
        beta : float
            Ignored. BiWCM does not use momentum. Accepted for API symmetry
            with other models.
        beta_schedule : {"warmup", "linear"}, optional
            Ignored. BiWCM does not use momentum.
        warmup_steps : int
            Ignored. BiWCM does not use momentum.

        Returns
        -------
        self : BiWCM
        """
        if beta != 0.9 or beta_schedule is not None or warmup_steps != 500:
            warnings.warn(
                "BiWCM does not use momentum: beta, beta_schedule, and "
                "warmup_steps are ignored.",
                UserWarning,
                stacklevel=2,
            )
        s_rows = self.strength_sequence_rows.to(torch.float64)
        s_cols = self.strength_sequence_cols.to(torch.float64)

        if x0 is None:
            x0 = self._initial_guess(s_rows, s_cols)
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self.device)

        _alpha_fp = alpha if alpha is not None else 1.0
        _alpha_qn = alpha if alpha is not None else 0.5

        t0 = time.perf_counter()

        if method == "fixed-point":
            params, info = self._solve_fixed_point(
                x0, s_rows, s_cols, tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_fp, chunk=chunk
            )
        elif method == "quasi-newton":
            params, info = self._solve_quasi_newton(
                x0, s_rows, s_cols, tol=tol, max_iter=max_iter, verbose=verbose,
                alpha=_alpha_qn, chunk=chunk
            )
        elif method == "coordinate":
            params, info = self._solve_coordinate(
                s_rows, s_cols, x0, tol=tol, max_iter=max_iter, verbose=verbose, chunk=chunk
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.time_s = time.perf_counter() - t0
        self.converged = info["converged"]
        self.n_iter = info["n_iter"]
        self.final_residual = info["final_residual"]

        self.theta_rows = params[:self.n_rows]
        self.theta_cols = params[self.n_rows:]

        self.is_solved = True
        self._params = {
            "theta_rows": self.theta_rows,
            "theta_cols": self.theta_cols,
        }

        return self

    def _initial_guess(
        self,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor
    ) -> torch.Tensor:
        """Generate initial guess for parameters."""
        theta_rows = 1.0 / torch.clamp(s_rows, min=1e-10)
        theta_cols = 1.0 / torch.clamp(s_cols, min=1e-10)
        return torch.cat([theta_rows, theta_cols])

    def _compute_expected_strengths(self, theta_rows: torch.Tensor, theta_cols: torch.Tensor):
        """Compute expected strength sequences from current parameters."""
        expected_matrix = 1.0 / (theta_rows[:, None] + theta_cols[None, :])
        e_s_rows = expected_matrix.sum(dim=1)
        e_s_cols = expected_matrix.sum(dim=0)
        return e_s_rows, e_s_cols, expected_matrix

    def _solve_fixed_point(
        self,
        x0: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using fixed-point iteration.

        The update rule is: theta_new = theta * (expected_strength / observed_strength)
        """
        params = x0.clone()
        n_r = self.n_rows
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            theta_rows = params[:n_r]
            theta_cols = params[n_r:]

            if chunk is None:
                e_s_rows, e_s_cols, _ = self._compute_expected_strengths(theta_rows, theta_cols)
            else:
                e_s_rows = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_s_cols = torch.zeros_like(theta_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    w = 1.0 / (theta_rows[i0:i1, None] + theta_cols[None, :])
                    e_s_rows[i0:i1] = w.sum(1)
                    e_s_cols += w.sum(0)

            # Compute relative errors
            rel_error_rows = torch.abs(s_rows - e_s_rows) / torch.clamp(s_rows, min=1e-10)
            rel_error_cols = torch.abs(s_cols - e_s_cols) / torch.clamp(s_cols, min=1e-10)
            max_error = max(rel_error_rows.max().item(), rel_error_cols.max().item())

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_strength={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"Converged at iter {iteration}")
                converged = True
                break

            # Fixed-point update: theta *= (1 + alpha * (E[s]/s - 1))
            # At alpha=1 this reduces to theta *= E[s]/s (standard multiplicative step).
            # For alpha != 1 this is a proper linear interpolation (consistent with BiECM).
            delta_rows = e_s_rows / torch.clamp(s_rows, min=1e-10) - 1
            delta_cols = e_s_cols / torch.clamp(s_cols, min=1e-10) - 1

            params[:n_r] *= 1 + alpha * delta_rows
            params[n_r:] *= 1 + alpha * delta_cols
        else:
            warnings.warn(
                f"Fixed-point solver did not converge after {max_iter} iterations. "
                f"Final max relative error: {max_error:.2e}. "
                "Try a different method or increase max_iter.",
                ConvergenceWarning,
            )

        return params, {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_quasi_newton(
        self,
        x0: torch.Tensor,
        s_rows: torch.Tensor,
        s_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.5,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using diagonal quasi-Newton method.

        The log-likelihood gradient is dL/d(theta_i) = E[s_i] - s_i and the
        diagonal of the negative-Hessian is sum_j E[w_ij]^2 > 0, so the
        Newton step for maximising L is:

            theta += (E[s] - s) / sum_j E[w_ij]^2

        theta must stay positive (domain constraint); params are clamped to 1e-30.
        """
        params = x0.clone().clamp(min=1e-30)
        n_r = self.n_rows
        constraints = torch.cat([s_rows, s_cols])
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            theta_rows = params[:n_r]
            theta_cols = params[n_r:]

            if chunk is None:
                e_s_rows, e_s_cols, expected_matrix = self._compute_expected_strengths(
                    theta_rows, theta_cols
                )
                hessian_diag = torch.cat([
                    (expected_matrix ** 2).sum(dim=1),
                    (expected_matrix ** 2).sum(dim=0),
                ])
            else:
                e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_sc = torch.zeros_like(theta_cols)
                hess_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hess_c = torch.zeros_like(theta_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    w = 1.0 / (theta_rows[i0:i1, None] + theta_cols[None, :])
                    e_sr[i0:i1] = w.sum(1)
                    e_sc += w.sum(0)
                    hess_r[i0:i1] = (w ** 2).sum(1)
                    hess_c += (w ** 2).sum(0)
                e_s_rows, e_s_cols = e_sr, e_sc
                hessian_diag = torch.cat([hess_r, hess_c])

            gradient = torch.cat([-s_rows + e_s_rows, -s_cols + e_s_cols])

            # Log-space Newton step to keep theta positive:
            # step = (E[s]-s) / (H_diag * theta)
            delta = gradient / (hessian_diag * params + 1e-10)
            params = torch.clamp(
                params * torch.exp(torch.clamp(alpha * delta, -10., 10.)),
                min=1e-30
            )

            # Check convergence
            rel_error = torch.abs(gradient) / torch.clamp(constraints, min=1e-10)
            max_error = rel_error.max().item()

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_strength={max_error:.2e}")

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

        return params, {"n_iter": iteration + 1, "converged": converged, "final_residual": max_error}

    def _solve_coordinate(
        self,
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

        Each coordinate group is solved via 60-step bisection in log-space.
        The bisection bounds (lo/hi) are tensors so all nodes in a group are
        updated in parallel, but the matrix sums are computed in sequential
        chunks for memory efficiency.

        n_bisect=60 gives ~1e-18 precision over a [-30, 30] log-range.
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk
        tr = x0[:n_r].clamp(min=1e-30)
        tc = x0[n_r:].clamp(min=1e-30)

        max_rel = float("inf")
        converged = False
        w = len(str(max_iter))
        for outer in range(max_iter):

            # --- Bisect for theta_rows ---
            # f(tr_i) = sum_j 1/(tr_i + tc_j) is strictly decreasing in log(tr_i)
            # f > target  →  tr_i too small  →  raise lower bound
            lo = tr.new_full((n_r,), -30.)
            hi = tr.new_full((n_r,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                tr_mid = log_mid.exp()
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    f[i0:i1] = (1. / (tr_mid[i0:i1, None] + tc[None, :])).sum(1)
                above = f > s_rows
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            tr = ((lo + hi) / 2).exp()

            # --- Bisect for theta_cols ---
            lo = tc.new_full((n_c,), -30.)
            hi = tc.new_full((n_c,), 30.)
            for _ in range(n_bisect):
                log_mid = (lo + hi) / 2
                tc_mid = log_mid.exp()
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    f += (1. / (tr[i0:i1, None] + tc_mid[None, :])).sum(0)
                above = f > s_cols
                lo = torch.where(above, log_mid, lo)
                hi = torch.where(above, hi, log_mid)
            tc = ((lo + hi) / 2).exp()

            # --- Convergence check ---
            e_sr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_sc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                w = 1. / (tr[i0:i1, None] + tc[None, :])
                e_sr[i0:i1] = w.sum(1)
                e_sc += w.sum(0)

            max_rel = max(
                (torch.abs(s_rows - e_sr) / s_rows.clamp(min=1e-10)).max().item(),
                (torch.abs(s_cols - e_sc) / s_cols.clamp(min=1e-10)).max().item(),
            )
            if verbose:
                print(f"Iter {outer:{w}d}  max_err_strength={max_rel:.2e}")
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

        return torch.cat([tr, tc]), {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel}

    def expected_matrix(self) -> torch.Tensor:
        """
        Compute expected biadjacency matrix.

        Returns
        -------
        expected : torch.Tensor
            Matrix where entry (i,j) is the expected weight E[w_ij] = 1/(theta_i + theta_j).
        """
        self._check_solved()
        return 1.0 / (self.theta_rows[:, None] + self.theta_cols[None, :])

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate random samples from the BiWCM ensemble.

        Weights are sampled from exponential distributions with
        rate parameter (theta_i + theta_j).

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

        # Rate parameter for exponential = theta_i + theta_j
        rate_matrix = self.theta_rows[:, None] + self.theta_cols[None, :]

        samples = torch.zeros(n_samples, self.n_rows, self.n_cols,
                              dtype=torch.float64, device=self.device)
        for s in range(n_samples):
            samples[s] = Exponential(rate_matrix).sample()

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

        # Hamiltonian term: -sum(theta * s)
        hamiltonian = -(self.theta_rows * s_rows).sum() - (self.theta_cols * s_cols).sum()

        # Partition function term: sum(log(theta_i + theta_j))
        partition = torch.log(self.theta_rows[:, None] + self.theta_cols[None, :]).sum()

        return (hamiltonian + partition).item()

    def n_parameters(self) -> int:
        """Return the number of free parameters in the model."""
        return self.n_rows + self.n_cols

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

        Under BiWCM, weights follow an exponential distribution with
        rate (theta_i + theta_j). The p-value for observing w_ij or more
        is P(W >= w_ij) = exp(-w_ij * (theta_i + theta_j)).

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

        # P(W >= w) = exp(-w * (theta_i + theta_j)) for the exponential distribution
        theta_sum = self.theta_rows[:, None] + self.theta_cols[None, :]
        return torch.exp(-observed * theta_sum)
