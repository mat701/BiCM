"""
Example datasets for testing and demonstration.
"""

from typing import Dict, Optional, Tuple
import torch


AVAILABLE_DATASETS = {
    "toy": "Small toy network for quick testing",
    "random_dense": "Random dense weighted bipartite network",
    "random_sparse": "Random sparse weighted bipartite network",
}


def list_datasets() -> Dict[str, str]:
    """
    List available example datasets.

    Returns
    -------
    datasets : dict
        Dictionary mapping dataset names to descriptions.
    """
    return AVAILABLE_DATASETS.copy()


def load_example(
    name: str = "toy",
    seed: Optional[int] = 42,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Load an example dataset.

    Parameters
    ----------
    name : str, default="toy"
        Name of the dataset. Use list_datasets() to see available options.
    seed : int, optional
        Random seed for reproducible random datasets.
    device : str, optional
        Device for the returned tensor. If None, uses CPU.

    Returns
    -------
    biadjacency : torch.Tensor
        The biadjacency matrix.
    info : dict
        Dictionary with metadata about the dataset.

    Examples
    --------
    >>> from bicm.wbnm.data import load_example, list_datasets
    >>> print(list_datasets())
    >>> W, info = load_example("toy")
    >>> print(f"Shape: {W.shape}")
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device) if device else torch.device("cpu")

    if name == "toy":
        # Small toy network
        biadjacency = torch.tensor([
            [5, 0, 3, 1],
            [0, 2, 4, 0],
            [1, 3, 0, 2],
            [0, 1, 2, 0],
            [2, 0, 0, 3],
        ], dtype=torch.float64, device=dev)

        info = {
            "name": "toy",
            "description": "Small toy weighted bipartite network",
            "n_rows": 5,
            "n_cols": 4,
            "n_edges": 12,
            "total_weight": biadjacency.sum().item(),
        }

    elif name == "random_dense":
        # Dense random network
        n_rows, n_cols = 20, 30
        biadjacency = torch.distributions.Exponential(1.0 / 5.0).sample((n_rows, n_cols)).to(dtype=torch.float64, device=dev)
        # Sparsify slightly
        biadjacency = biadjacency * (torch.rand(n_rows, n_cols, device=dev) > 0.1).to(torch.float64)

        info = {
            "name": "random_dense",
            "description": "Random dense weighted bipartite network",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_edges": int((biadjacency > 0).sum().item()),
            "total_weight": biadjacency.sum().item(),
            "seed": seed,
        }

    elif name == "random_sparse":
        # Sparse random network
        n_rows, n_cols = 50, 100
        biadjacency = torch.distributions.Exponential(1.0 / 3.0).sample((n_rows, n_cols)).to(dtype=torch.float64, device=dev)
        # Make sparse
        biadjacency = biadjacency * (torch.rand(n_rows, n_cols, device=dev) > 0.8).to(torch.float64)

        info = {
            "name": "random_sparse",
            "description": "Random sparse weighted bipartite network",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_edges": int((biadjacency > 0).sum().item()),
            "total_weight": biadjacency.sum().item(),
            "seed": seed,
        }

    else:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {available}"
        )

    return biadjacency, info
