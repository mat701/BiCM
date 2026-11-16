"""
Network utility functions.
"""

from typing import Tuple, List, Optional
import torch


def biadjacency_to_edgelist(
    biadjacency: torch.Tensor,
    weighted: bool = True
) -> List[Tuple]:
    """
    Convert biadjacency matrix to edge list.

    Parameters
    ----------
    biadjacency : torch.Tensor
        Biadjacency matrix of shape (n_rows, n_cols).
    weighted : bool, default=True
        If True, include weights as third element.

    Returns
    -------
    edges : list of tuples
        List of (row_idx, col_idx) or (row_idx, col_idx, weight) tuples.
    """
    W = torch.as_tensor(biadjacency, dtype=torch.float64)
    nonzero_indices = torch.nonzero(W)
    rows = nonzero_indices[:, 0]
    cols = nonzero_indices[:, 1]

    if weighted:
        weights = W[rows, cols]
        return list(zip(rows.tolist(), cols.tolist(), weights.tolist()))
    else:
        return list(zip(rows.tolist(), cols.tolist()))


def edgelist_to_biadjacency(
    edges: List[Tuple],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    weighted: bool = True,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Convert edge list to biadjacency matrix.

    Parameters
    ----------
    edges : list of tuples
        List of (row_idx, col_idx) or (row_idx, col_idx, weight) tuples.
    n_rows : int, optional
        Number of row nodes. If None, inferred from edges.
    n_cols : int, optional
        Number of column nodes. If None, inferred from edges.
    weighted : bool, default=True
        If True, expect weights in edges. If False, create binary matrix.
    device : str, optional
        Device for the returned tensor.

    Returns
    -------
    biadjacency : torch.Tensor
        Biadjacency matrix of shape (n_rows, n_cols).
    """
    dev = torch.device(device) if device else torch.device("cpu")

    if len(edges) == 0:
        n_rows = n_rows or 0
        n_cols = n_cols or 0
        return torch.zeros(n_rows, n_cols, dtype=torch.float64, device=dev)

    # Parse edges
    if weighted and len(edges[0]) == 3:
        rows, cols, weights = zip(*edges)
        weights = torch.tensor(weights, dtype=torch.float64, device=dev)
    else:
        if len(edges[0]) == 3:
            rows, cols, _ = zip(*edges)
        else:
            rows, cols = zip(*edges)
        weights = torch.ones(len(edges), dtype=torch.float64, device=dev)

    rows = torch.tensor(rows, dtype=torch.long, device=dev)
    cols = torch.tensor(cols, dtype=torch.long, device=dev)

    # Determine dimensions
    n_rows = n_rows or (rows.max().item() + 1)
    n_cols = n_cols or (cols.max().item() + 1)

    # Create matrix
    biadjacency = torch.zeros(n_rows, n_cols, dtype=torch.float64, device=dev)
    biadjacency[rows, cols] = weights

    return biadjacency


def density(biadjacency: torch.Tensor) -> float:
    """
    Compute the density (connectance) of a bipartite network.

    Density = L / (n_rows * n_cols)

    where L is the number of edges.

    Parameters
    ----------
    biadjacency : torch.Tensor
        Biadjacency matrix.

    Returns
    -------
    density : float
        Value between 0 and 1.
    """
    W = torch.as_tensor(biadjacency, dtype=torch.float64)
    n_rows, n_cols = W.shape
    n_edges = (W > 0).sum().item()
    return float(n_edges) / (n_rows * n_cols)


# Alias for density
connectance = density


def rca(
    biadjacency: torch.Tensor,
    threshold: float = 1.0
) -> torch.Tensor:
    """
    Compute the Revealed Comparative Advantage (RCA) binarization.

    RCA_ij = (w_ij / s_i) / (d_j / W_tot)

    where s_i = sum_j w_ij (row strength), d_j = sum_i w_ij (col strength),
    and W_tot = sum_ij w_ij (total weight).

    A binary link is placed where RCA_ij >= threshold (default 1).

    Parameters
    ----------
    biadjacency : torch.Tensor
        Weighted biadjacency matrix of shape (n_rows, n_cols).
    threshold : float, default=1.0
        Threshold above which a link is considered present.

    Returns
    -------
    binary : torch.Tensor
        Binary biadjacency matrix of shape (n_rows, n_cols), dtype float64.
    """
    W = torch.as_tensor(biadjacency, dtype=torch.float64)
    row_strengths = W.sum(dim=1, keepdim=True)      # (n_rows, 1)
    col_strengths = W.sum(dim=0, keepdim=True)      # (1, n_cols)
    total = W.sum()

    # Avoid division by zero for empty rows/cols
    row_strengths = row_strengths.clamp(min=1e-300)
    col_strengths = col_strengths.clamp(min=1e-300)

    rca_matrix = (W / row_strengths) / (col_strengths / total)
    return (rca_matrix >= threshold).to(torch.float64)
