"""
Weighted projections of bipartite networks.
"""

from typing import Literal, Optional, Tuple
import torch

from bicm.wbnm.models.base import BipartiteModel
from bicm.wbnm.validation.pvalues import validate_property
from bicm.wbnm.validation.corrections import apply_correction


def project_rows(
    biadjacency: torch.Tensor,
    method: Literal["sum", "jaccard", "cosine", "hyperbolic"] = "sum"
) -> torch.Tensor:
    """
    Project bipartite network onto row nodes.

    Creates a weighted unipartite network where row nodes are connected
    if they share column nodes in common.

    Parameters
    ----------
    biadjacency : torch.Tensor
        Biadjacency matrix of shape (n_rows, n_cols).
    method : {"sum", "jaccard", "cosine", "hyperbolic"}, default="sum"
        Projection method:
        - "sum": Sum of products (W @ W.T)
        - "jaccard": Jaccard-weighted
        - "cosine": Cosine similarity weighted
        - "hyperbolic": Hyperbolic tangent weighted

    Returns
    -------
    projection : torch.Tensor
        Symmetric matrix of shape (n_rows, n_rows).

    Examples
    --------
    >>> W = torch.tensor([[1, 0, 2], [0, 1, 1], [1, 1, 0]], dtype=torch.float64)
    >>> P = project_rows(W, method="sum")
    >>> print(P.shape)
    torch.Size([3, 3])
    """
    W = torch.as_tensor(biadjacency, dtype=torch.float64)

    if method == "sum":
        # Simple matrix multiplication: P_ij = sum_k W_ik * W_jk
        projection = W @ W.T

    elif method == "jaccard":
        # Jaccard-weighted projection
        # J_ij = |intersection| / |union|
        binary = (W > 0).to(torch.float64)
        intersection = binary @ binary.T

        # Union: k_i + k_j - intersection
        degrees = binary.sum(dim=1)
        union = degrees[:, None] + degrees[None, :] - intersection

        # Avoid division by zero
        projection = torch.where(union > 0, intersection / union,
                                 torch.zeros_like(intersection))

    elif method == "cosine":
        # Cosine similarity weighted
        norms = torch.linalg.norm(W, dim=1)
        normalized = W / norms[:, None]
        normalized = torch.nan_to_num(normalized, nan=0.0)
        projection = normalized @ normalized.T

    elif method == "hyperbolic":
        # Hyperbolic weighting: tanh(W @ W.T)
        projection = torch.tanh(W @ W.T)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove self-loops
    projection.fill_diagonal_(0)

    return projection


def project_cols(
    biadjacency: torch.Tensor,
    method: Literal["sum", "jaccard", "cosine", "hyperbolic"] = "sum"
) -> torch.Tensor:
    """
    Project bipartite network onto column nodes.

    Parameters
    ----------
    biadjacency : torch.Tensor
        Biadjacency matrix of shape (n_rows, n_cols).
    method : {"sum", "jaccard", "cosine", "hyperbolic"}, default="sum"
        Projection method (see project_rows for details).

    Returns
    -------
    projection : torch.Tensor
        Symmetric matrix of shape (n_cols, n_cols).
    """
    return project_rows(biadjacency.T, method=method)


def validated_projection(
    model: BipartiteModel,
    layer: Literal["rows", "cols"] = "rows",
    method: Literal["sum", "jaccard", "cosine", "hyperbolic"] = "sum",
    n_samples: int = 1000,
    alpha: float = 0.05,
    correction: Literal["bonferroni", "fdr_bh", "none"] = "fdr_bh",
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute statistically validated projection.

    Projects the bipartite network and validates each edge against
    the null model ensemble.

    Parameters
    ----------
    model : BipartiteModel
        A solved null model (BiECM, BiWCM, or BiPECM).
    layer : {"rows", "cols"}, default="rows"
        Which layer to project onto.
    method : {"sum", "jaccard", "cosine", "hyperbolic"}, default="sum"
        Projection method.
    n_samples : int, default=1000
        Number of null model samples for p-value estimation.
    alpha : float, default=0.05
        Significance level.
    correction : {"bonferroni", "fdr_bh", "none"}, default="fdr_bh"
        Multiple testing correction method.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        Alternative hypothesis. "greater" tests for edges stronger
        than expected by chance.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    validated : torch.Tensor
        The validated projection (significant edges only).
    pvalues : torch.Tensor
        Matrix of p-values.
    rejected : torch.Tensor
        Boolean matrix indicating significant edges.

    Examples
    --------
    >>> from bicm.wbnm.models import BiECM
    >>> from bicm.wbnm.projections import validated_projection
    >>>
    >>> model = BiECM(W).solve()
    >>> validated, pvalues, significant = validated_projection(
    ...     model, layer="rows", n_samples=1000
    ... )
    >>>
    >>> # Count significant edges
    >>> print(f"Significant edges: {significant.sum() // 2}")
    """
    model._check_solved()

    # Choose projection function
    project_func = project_rows if layer == "rows" else project_cols

    # Compute observed projection
    observed_projection = project_func(model.biadjacency, method=method)

    # Generate samples and compute projections
    # For BiPECM, use conditional=False to allow topology variation
    from bicm.wbnm.models.bipecm import BiPECM
    if isinstance(model, BiPECM):
        samples = model.sample(n_samples, seed=seed, conditional=False)
    else:
        samples = model.sample(n_samples, seed=seed)
    sample_projections = torch.stack([project_func(s, method=method) for s in samples])

    # Compute p-values element-wise
    pvalues = validate_property(observed_projection, sample_projections, alternative)
    pvalues.fill_diagonal_(1.0)

    # Apply the multiple testing correction to the distinct couples of nodes only:
    # the matrix is symmetric and its diagonal is not tested.
    n = pvalues.shape[0]
    upper = torch.triu_indices(n, n, offset=1, device=pvalues.device)
    rejected_upper, _ = apply_correction(pvalues[upper[0], upper[1]], method=correction, alpha=alpha)
    rejected = torch.zeros_like(pvalues, dtype=torch.bool)
    rejected[upper[0], upper[1]] = rejected_upper
    rejected = rejected | rejected.T

    # Create validated projection (keep only significant edges)
    validated = torch.where(rejected, torch.ones_like(observed_projection),
                            torch.zeros_like(observed_projection))

    return validated, pvalues, rejected
