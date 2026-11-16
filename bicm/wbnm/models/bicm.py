"""
Bipartite Configuration Model (BiCM).

The BiCM preserves only the degree sequence
of the bipartite network in expectation.
"""

from typing import Optional, Literal
import math
import time
import warnings
import torch

from bicm.wbnm.models.base import BipartiteModel, ArrayLike, ConvergenceWarning


class BiCM(BipartiteModel):
    """
    Bipartite Configuration Model.

    The BiCM is a maximum entropy null model that preserves:
    - Expected degree sequence (number of links per node)

    Canonical parametrization:

        p_ij = 1 / (1 + exp(theta_i + theta_j))

    Parameters
    ----------
    biadjacency : ArrayLike
        The observed biadjacency matrix of shape (n_rows, n_cols).
        If weighted, it is binarized internally.
    device : str, optional
        Device for computation. Auto-detects GPU if None.
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        super().__init__(biadjacency, device=device)

        # Binarize input (BiCM is binary model)
        if not torch.all((self.biadjacency == 0) | (self.biadjacency == 1)):
            warnings.warn(
                "BiCM received a weighted matrix; it will be binarized "
                "(non-zero entries → 1). Use BiECM or BiPECM to preserve weights.",
                UserWarning,
                stacklevel=2,
            )
        self.topology = (self.biadjacency > 0).to(torch.float64)

        # Canonical parameters (set after solving)
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
    ) -> "BiCM":
        """
        Solve for the canonical parameters.

        Parameters
        ----------
        method : {"fixed-point", "quasi-newton", "coordinate"}
            Optimization method:
            - "fixed-point": multiplicative update in theta-space (default, robust).
            - "quasi-newton": diagonal Newton step, faster near the solution.
            - "coordinate": bisection over each coordinate group, ~1e-18 precision.
        x0 : torch.Tensor, optional
            Initial guess for parameters.
        tol : float
            Convergence tolerance (max relative error on degree sequence).
        max_iter : int
            Maximum number of iterations.
        verbose : bool
            Print convergence information.
        alpha : float, optional
            Step size for fixed-point (default 1.0) and quasi-Newton (default 0.5).
            Ignored by coordinate solver.
        beta : float
            Ignored. BiCM does not use momentum. Accepted for API symmetry
            with other models.
        beta_schedule : {"warmup", "linear"}, optional
            Ignored. BiCM does not use momentum.
        warmup_steps : int
            Ignored. BiCM does not use momentum.
        """
        if beta != 0.9 or beta_schedule is not None or warmup_steps != 500:
            warnings.warn(
                "BiCM does not use momentum: beta, beta_schedule, and "
                "warmup_steps are ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Observed degree sequences
        k_rows = self.degree_sequence_rows.to(torch.float64)
        k_cols = self.degree_sequence_cols.to(torch.float64)

        # Initialize parameters
        if x0 is None:
            x0 = self._initial_guess(k_rows, k_cols)
        x0 = torch.as_tensor(x0, dtype=torch.float64, device=self.device)

        # Method-specific alpha defaults
        _alpha_fp = alpha if alpha is not None else 1.0
        _alpha_qn = alpha if alpha is not None else 0.5

        # Dispatch solver
        t0 = time.perf_counter()

        if method == "fixed-point":
            params, info = self._solve_fixed_point(
                x0, k_rows, k_cols,
                tol=tol, max_iter=max_iter,
                verbose=verbose, alpha=_alpha_fp,
                chunk=chunk, **kwargs
            )

        elif method == "quasi-newton":
            params, info = self._solve_quasi_newton(
                x0, k_rows, k_cols,
                tol=tol, max_iter=max_iter,
                verbose=verbose, alpha=_alpha_qn,
                chunk=chunk, **kwargs
            )

        elif method == "coordinate":
            params, info = self._solve_coordinate(
                k_rows, k_cols, x0,
                tol=tol, max_iter=max_iter,
                verbose=verbose, chunk=chunk
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        self.time_s = time.perf_counter() - t0
        self.converged = info["converged"]
        self.n_iter = info["n_iter"]
        self.final_residual = info["final_residual"]

        # Unpack and store parameters
        n_r = self.n_rows
        self.theta_rows = params[:n_r]
        self.theta_cols = params[n_r:]

        self.is_solved = True
        self._params = {
            "theta_rows": self.theta_rows,
            "theta_cols": self.theta_cols,
        }

        return self

    def _initial_guess(
        self,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor
    ) -> torch.Tensor:
        """Generate initial guess for parameters."""

        k_rows = torch.clamp(k_rows, min=1e-10)
        k_cols = torch.clamp(k_cols, min=1e-10)

        k_tot = k_rows.sum()

        # Initialize via classical x-heuristic, then map to theta
        x_rows = k_rows / torch.sqrt(k_tot)
        x_cols = k_cols / torch.sqrt(k_tot)

        theta_rows = -torch.log(x_rows)
        theta_cols = -torch.log(x_cols)

        return torch.cat([theta_rows, theta_cols])

    def _solve_fixed_point(
        self,
        x0: torch.Tensor,
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 1.0,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using multiplicative fixed-point iteration
        in canonical (theta) space.
        """

        n_r = self.n_rows
        params = x0.clone()
        converged = False

        degree = torch.cat([k_rows, k_cols])

        w = len(str(max_iter))
        for iteration in range(max_iter):

            theta_rows = params[:n_r]
            theta_cols = params[n_r:]

            if chunk is None:
                p_matrix = torch.sigmoid(-(theta_rows[:, None] + theta_cols[None, :]))
                e_degree = torch.cat([p_matrix.sum(dim=1), p_matrix.sum(dim=0)])
            else:
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros_like(theta_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    p = torch.sigmoid(-(theta_rows[i0:i1, None] + theta_cols[None, :]))
                    e_kr[i0:i1] = p.sum(1)
                    e_kc += p.sum(0)
                e_degree = torch.cat([e_kr, e_kc])

            # Zero-degree nodes are trivially solved (theta→+∞, clamped);
            # mask them out so their 0/1e-10 ratio does not dominate max_error.
            rel_error = torch.abs(degree - e_degree) / degree.clamp(min=1e-10)
            max_error = torch.where(degree > 0, rel_error, degree.new_zeros(1)).max().item()

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_degree={max_error:.2e}")

            if max_error < tol:
                if verbose:
                    print(f"Converged at iter {iteration}")
                converged = True
                break

            delta = torch.clamp(e_degree / (degree + 1e-12), min=1e-12)
            params = params + alpha * torch.log(delta)

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
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        alpha: float = 0.5,
        chunk: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve using diagonal quasi-Newton method.

        The log-likelihood gradient is dL/d(theta_i) = E[k_i] - k_i and the
        diagonal of the negative-Hessian is sum_j p_ij*(1-p_ij) > 0, so the
        Newton step for maximising L is:

            theta += (E[k] - k) / sum_j p*(1-p)
        """
        n_r = self.n_rows
        params = x0.clone()
        constraints = torch.cat([k_rows, k_cols])
        converged = False

        w = len(str(max_iter))
        for iteration in range(max_iter):
            theta_rows = params[:n_r]
            theta_cols = params[n_r:]

            if chunk is None:
                p_matrix = torch.sigmoid(-(theta_rows[:, None] + theta_cols[None, :]))
                e_k_rows = p_matrix.sum(dim=1)
                e_k_cols = p_matrix.sum(dim=0)
                pq = p_matrix * (1 - p_matrix)
                hessian_diag = torch.cat([pq.sum(dim=1), pq.sum(dim=0)])
            else:
                e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                e_kc = torch.zeros_like(theta_cols)
                hess_r = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                hess_c = torch.zeros_like(theta_cols)
                for i0 in range(0, n_r, chunk):
                    i1 = min(i0 + chunk, n_r)
                    p = torch.sigmoid(-(theta_rows[i0:i1, None] + theta_cols[None, :]))
                    e_kr[i0:i1] = p.sum(1)
                    e_kc += p.sum(0)
                    pq = p * (1 - p)
                    hess_r[i0:i1] = pq.sum(1)
                    hess_c += pq.sum(0)
                e_k_rows, e_k_cols = e_kr, e_kc
                hessian_diag = torch.cat([hess_r, hess_c])

            gradient = torch.cat([e_k_rows - k_rows, e_k_cols - k_cols])

            delta = gradient / (hessian_diag + 1e-10)
            params = torch.clamp(params + alpha * delta, -30., 30.)

            rel_error = torch.abs(gradient) / constraints.clamp(min=1e-10)
            max_error = torch.where(constraints > 0, rel_error, constraints.new_zeros(1)).max().item()

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:{w}d}  max_err_degree={max_error:.2e}")

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
        k_rows: torch.Tensor,
        k_cols: torch.Tensor,
        x0: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False,
        chunk: Optional[int] = None,
        n_bisect: int = 60,
    ) -> torch.Tensor:
        """
        Coordinate descent with bisection.

        Works in u = -theta space where f(u_i) = sum_j sigmoid(u_i + u_j) is
        strictly increasing. Each group of coordinates is solved in parallel
        via 60-step bisection over [-30, 30], giving ~1e-18 precision.
        Matrix sums are computed in sequential chunks for memory efficiency.
        """
        n_r, n_c = self.n_rows, self.n_cols
        _chunk = n_r if chunk is None else chunk

        # u = -theta; initial values from x0
        ur = -x0[:n_r]
        uc = -x0[n_r:]

        max_rel = float("inf")
        converged = False
        w = len(str(max_iter))
        for outer in range(max_iter):

            # --- Bisect u_rows ---
            # f(u_i) = sum_j sigmoid(u_i + u_j) is increasing in u_i
            # f > k_i  →  u_i too large  →  hi = mid
            lo = ur.new_full((n_r,), -30.)
            hi = ur.new_full((n_r,), 30.)
            for _ in range(n_bisect):
                mid = (lo + hi) / 2
                f = torch.zeros(n_r, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    f[i0:i1] = torch.sigmoid(mid[i0:i1, None] + uc[None, :]).sum(1)
                above = f > k_rows
                hi = torch.where(above, mid, hi)
                lo = torch.where(above, lo, mid)
            ur = (lo + hi) / 2

            # --- Bisect u_cols ---
            lo = uc.new_full((n_c,), -30.)
            hi = uc.new_full((n_c,), 30.)
            for _ in range(n_bisect):
                mid = (lo + hi) / 2
                f = torch.zeros(n_c, dtype=torch.float64, device=self.device)
                for i0 in range(0, n_r, _chunk):
                    i1 = min(i0 + _chunk, n_r)
                    f += torch.sigmoid(ur[i0:i1, None] + mid[None, :]).sum(0)
                above = f > k_cols
                hi = torch.where(above, mid, hi)
                lo = torch.where(above, lo, mid)
            uc = (lo + hi) / 2

            # --- Convergence check ---
            e_kr = torch.zeros(n_r, dtype=torch.float64, device=self.device)
            e_kc = torch.zeros(n_c, dtype=torch.float64, device=self.device)
            for i0 in range(0, n_r, _chunk):
                i1 = min(i0 + _chunk, n_r)
                p = torch.sigmoid(ur[i0:i1, None] + uc[None, :])
                e_kr[i0:i1] = p.sum(1)
                e_kc += p.sum(0)

            rel_r = torch.abs(k_rows - e_kr) / k_rows.clamp(min=1e-10)
            rel_c = torch.abs(k_cols - e_kc) / k_cols.clamp(min=1e-10)
            max_rel = max(
                torch.where(k_rows > 0, rel_r, k_rows.new_zeros(1)).max().item(),
                torch.where(k_cols > 0, rel_c, k_cols.new_zeros(1)).max().item(),
            )
            if verbose:
                print(f"Iter {outer:{w}d}  max_err_degree={max_rel:.2e}")
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

        # Convert back: theta = -u
        return torch.cat([-ur, -uc]), {"n_iter": outer + 1, "converged": converged, "final_residual": max_rel}

    def connection_probability(self) -> torch.Tensor:
        """Compute connection probability matrix."""
        self._check_solved()

        logits = self.theta_rows[:, None] + self.theta_cols[None, :]
        return torch.sigmoid(-logits)

    def expected_matrix(self) -> torch.Tensor:
        """For binary model, expected matrix equals connection probability."""
        return self.connection_probability()

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate random binary samples from the BiCM ensemble.
        """

        self._check_solved()

        if seed is not None:
            torch.manual_seed(seed)

        p_matrix = self.connection_probability()

        samples = torch.zeros(
            n_samples, self.n_rows, self.n_cols,
            dtype=torch.float64, device=self.device
        )

        for s in range(n_samples):
            samples[s] = (
                torch.rand(self.n_rows, self.n_cols,
                           dtype=torch.float64, device=self.device)
                < p_matrix
            ).to(torch.float64)

        return samples

    def log_likelihood(self) -> float:
        """Compute log-likelihood in canonical form."""

        self._check_solved()

        k_rows = self.degree_sequence_rows
        k_cols = self.degree_sequence_cols

        thetas = torch.cat([self.theta_rows, self.theta_cols])
        constraints = torch.cat([k_rows, k_cols])

        hamiltonian = -torch.sum(thetas * constraints)

        partition = -torch.sum(
            torch.log1p(
                torch.exp(
                    -self.theta_rows[:, None]
                    -self.theta_cols[None, :]
                )
            )
        )

        return (hamiltonian + partition).item()

    def pvalues(self, observed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute one-sided p-values for the observed binary links.

        Under BiCM each entry is Bernoulli(p_ij), so the p-value for
        observing a link at (i, j) is P(A_ij = 1) = p_ij.
        Entries with no observed link receive p-value = 1 (no test).

        Parameters
        ----------
        observed : torch.Tensor, optional
            Binary (or weighted) matrix of shape (n_rows, n_cols).
            If weighted, it is binarized internally.
            If None, uses the matrix passed to the constructor.

        Returns
        -------
        pvalues : torch.Tensor
            Matrix of p-values, same shape as observed.
            Low p-values indicate links significantly more likely than
            expected under the null model.
        """
        self._check_solved()

        if observed is None:
            observed = self.biadjacency
        observed = torch.as_tensor(observed, dtype=torch.float64, device=self.device)

        p_matrix = self.connection_probability()

        # P(A_ij >= a_ij): p_ij if a_ij=1, 1 if a_ij=0
        return torch.where(observed > 0, p_matrix, torch.ones_like(p_matrix))

    def n_parameters(self) -> int:
        return self.n_rows + self.n_cols

    def aic(self) -> float:
        return -2 * self.log_likelihood() + 2 * self.n_parameters()

    def aicc(self) -> float:
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        return self.aic() + (2 * k * (k + 1)) / (n - k - 1)

    def bic(self) -> float:
        k = self.n_parameters()
        n = self.n_rows * self.n_cols
        return -2 * self.log_likelihood() + k * math.log(n)