"""
Base class for bipartite null models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import warnings
import torch

# Type alias for array-like inputs
ArrayLike = Union[torch.Tensor, "np.ndarray", list]


class ConvergenceWarning(RuntimeWarning):
    """Warning raised when a solver does not converge within the maximum number of iterations."""
    pass


class BipartiteModel(ABC):
    """
    Abstract base class for bipartite null models.

    All null models inherit from this class and must implement:
    - solve(): Find the model parameters (fitness/Lagrange multipliers)
    - sample(): Generate random samples from the null model
    - expected_matrix(): Return the expected biadjacency matrix

    Parameters
    ----------
    biadjacency : ArrayLike
        The observed biadjacency matrix of shape (n_rows, n_cols).
        Can be numpy array, torch tensor, or list.
    device : str, optional
        Device to use for computation ('cpu', 'cuda', 'cuda:0', etc.).
        If None, auto-detects GPU availability.

    Attributes
    ----------
    biadjacency : torch.Tensor
        The observed biadjacency matrix.
    n_rows : int
        Number of row nodes.
    n_cols : int
        Number of column nodes.
    is_solved : bool
        Whether the model has been solved.
    device : torch.device
        The device used for computation.
    """

    def __init__(self, biadjacency: ArrayLike, device: Optional[str] = None):
        if device is not None:
            self.device = torch.device(device)
        elif isinstance(biadjacency, torch.Tensor):
            self.device = biadjacency.device
        else:
            self.device = torch.device("cpu")

        # np.matrix is a subclass of np.ndarray but has quirks; normalize to plain ndarray first
        try:
            import numpy as np
            if isinstance(biadjacency, np.matrix):
                biadjacency = np.asarray(biadjacency)
        except ImportError:
            pass

        self.biadjacency = torch.as_tensor(biadjacency, dtype=torch.float64, device=self.device)

        if self.biadjacency.ndim != 2:
            raise ValueError(f"biadjacency must be 2D, got {self.biadjacency.ndim}D")

        if self.biadjacency.numel() == 0:
            raise ValueError("biadjacency matrix is empty (zero rows or zero columns).")

        if torch.isnan(self.biadjacency).any():
            raise ValueError("biadjacency contains NaN values.")

        if torch.isinf(self.biadjacency).any():
            raise ValueError("biadjacency contains infinite values.")

        zero_rows = (self.biadjacency > 0).sum(dim=1) == 0
        zero_cols = (self.biadjacency > 0).sum(dim=0) == 0
        if zero_rows.any():
            n = int(zero_rows.sum().item())
            warnings.warn(
                f"{n} row node(s) have no links and cannot be fitted. "
                "Consider removing zero-degree rows before solving "
                "(e.g. W = W[W.sum(1) > 0]).",
                UserWarning,
                stacklevel=3,
            )
        if zero_cols.any():
            n = int(zero_cols.sum().item())
            warnings.warn(
                f"{n} column node(s) have no links and cannot be fitted. "
                "Consider removing zero-degree columns before solving "
                "(e.g. W = W[:, W.sum(0) > 0]).",
                UserWarning,
                stacklevel=3,
            )

        self.n_rows, self.n_cols = self.biadjacency.shape
        self.is_solved = False
        self.converged: Optional[bool] = None
        self.n_iter: Optional[int] = None
        self.time_s: Optional[float] = None
        self.final_residual: Optional[float] = None

        # To be set by subclasses after solving
        self._params: Optional[dict] = None

    @property
    def degree_sequence_rows(self) -> torch.Tensor:
        """Degree sequence of row nodes."""
        return (self.biadjacency > 0).sum(dim=1)

    @property
    def degree_sequence_cols(self) -> torch.Tensor:
        """Degree sequence of column nodes."""
        return (self.biadjacency > 0).sum(dim=0)

    @property
    def strength_sequence_rows(self) -> torch.Tensor:
        """Strength (weighted degree) sequence of row nodes."""
        return self.biadjacency.sum(dim=1)

    @property
    def strength_sequence_cols(self) -> torch.Tensor:
        """Strength (weighted degree) sequence of column nodes."""
        return self.biadjacency.sum(dim=0)

    @abstractmethod
    def solve(self, **kwargs) -> "BipartiteModel":
        """
        Solve for the model parameters.

        Returns
        -------
        self : BipartiteModel
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random samples from the null model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples, n_rows, n_cols) containing
            sampled biadjacency matrices.
        """
        pass

    @abstractmethod
    def expected_matrix(self) -> torch.Tensor:
        """
        Compute the expected biadjacency matrix under the null model.

        Returns
        -------
        expected : torch.Tensor
            Expected biadjacency matrix of shape (n_rows, n_cols).
        """
        pass

    def _check_solved(self):
        """Raise error if model has not been solved."""
        if not self.is_solved:
            raise RuntimeError(
                "Model must be solved before this operation. "
                "Call model.solve() first."
            )

    def __repr__(self) -> str:
        status = "solved" if self.is_solved else "not solved"
        return (
            f"{self.__class__.__name__}("
            f"n_rows={self.n_rows}, n_cols={self.n_cols}, {status})"
        )
