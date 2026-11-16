"""
P-value computation for statistical validation.
"""

from typing import Callable, Optional, Literal, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from bicm.wbnm.models.base import BipartiteModel


def compute_pvalue(
    observed: float,
    samples: torch.Tensor,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> float:
    """
    Compute empirical p-value from samples.

    Parameters
    ----------
    observed : float
        The observed value of the statistic.
    samples : torch.Tensor
        Tensor of sampled values from the null distribution.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        The alternative hypothesis:
        - "greater": observed is significantly greater than null
        - "less": observed is significantly less than null
        - "two-sided": observed is significantly different from null

    Returns
    -------
    pvalue : float
        The empirical p-value.

    Examples
    --------
    >>> samples = torch.randn(1000)
    >>> compute_pvalue(2.5, samples, alternative="greater")
    0.006  # approximately
    """
    samples = torch.as_tensor(samples, dtype=torch.float64)
    n = len(samples)

    if alternative == "greater":
        # P(X >= observed)
        pvalue = ((samples >= observed).sum().item() + 1) / (n + 1)
    elif alternative == "less":
        # P(X <= observed)
        pvalue = ((samples <= observed).sum().item() + 1) / (n + 1)
    elif alternative == "two-sided":
        # Two-tailed test
        mean = samples.mean().item()
        deviation = abs(observed - mean)
        extreme_count = (torch.abs(samples - mean) >= deviation).sum().item()
        pvalue = (extreme_count + 1) / (n + 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(pvalue)


def validate_property(
    observed: Union[float, torch.Tensor],
    samples: torch.Tensor,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Union[float, torch.Tensor]:
    """
    Validate a scalar or array property against null model samples.

    Parameters
    ----------
    observed : float or torch.Tensor
        The observed value(s) of the property.
    samples : torch.Tensor
        Samples from null model. Shape should be (n_samples,) for scalar
        properties or ``(n_samples, *observed.shape)`` for array properties.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        The alternative hypothesis.

    Returns
    -------
    pvalues : float or torch.Tensor
        P-value(s) matching the shape of observed.

    Examples
    --------
    >>> # Scalar property
    >>> observed_clustering = 0.45
    >>> null_clustering = torch.tensor([model.sample() for _ in range(1000)])
    >>> pvalue = validate_property(observed_clustering, null_clustering)
    >>>
    >>> # Array property (e.g., node-level)
    >>> observed_degrees = torch.tensor([5, 10, 3, 8])
    >>> null_degrees = torch.randn(1000, 4) + 6
    >>> pvalues = validate_property(observed_degrees, null_degrees)
    """
    observed = torch.as_tensor(observed, dtype=torch.float64)
    samples = torch.as_tensor(samples, dtype=torch.float64)

    if observed.ndim == 0:
        # Scalar case
        return compute_pvalue(float(observed), samples.flatten(), alternative)

    # Array case: same estimator as compute_pvalue, vectorised over the entries
    n = samples.shape[0]
    if alternative == "greater":
        counts = (samples >= observed).sum(dim=0)
    elif alternative == "less":
        counts = (samples <= observed).sum(dim=0)
    elif alternative == "two-sided":
        mean = samples.mean(dim=0)
        counts = (torch.abs(samples - mean) >= torch.abs(observed - mean)).sum(dim=0)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return (counts.to(torch.float64) + 1) / (n + 1)


def validate_matrix_property(
    observed_matrix: torch.Tensor,
    sample_matrices: torch.Tensor,
    property_func: Callable[[torch.Tensor], Union[float, torch.Tensor]],
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Union[float, torch.Tensor]:
    """
    Validate a property computed from matrices.

    This is useful when the property is computed from the full matrix
    (e.g., clustering coefficient, modularity) rather than being
    element-wise.

    Parameters
    ----------
    observed_matrix : torch.Tensor
        The observed biadjacency matrix.
    sample_matrices : torch.Tensor
        Tensor of sampled matrices, shape (n_samples, n_rows, n_cols).
    property_func : callable
        Function that takes a matrix and returns the property value(s).
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        The alternative hypothesis.

    Returns
    -------
    pvalues : float or torch.Tensor
        P-value(s) for the property.

    Examples
    --------
    >>> def total_weight(W):
    ...     return W.sum()
    >>>
    >>> pvalue = validate_matrix_property(
    ...     observed_matrix,
    ...     model.sample(1000),
    ...     total_weight,
    ...     alternative="greater"
    ... )
    """
    # Compute observed property
    observed = property_func(observed_matrix)

    # Compute property for all samples
    sample_values = torch.stack([torch.as_tensor(property_func(m), dtype=torch.float64) for m in sample_matrices])

    return validate_property(observed, sample_values, alternative)


def pvalues_projection(
    model: "BipartiteModel",
    n_samples: int = 1000,
    rows: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute p-values for the projected network using Monte Carlo sampling.

    The projection of a bipartite network onto one set of nodes creates
    a weighted unipartite network where edge weights represent shared
    connections. This function computes p-values for each entry in the
    projection matrix.

    Parameters
    ----------
    model : BipartiteModel
        A solved bipartite null model (BiWCM, BiECM, or BiPECM).
    n_samples : int, default=1000
        Number of Monte Carlo samples to generate.
    rows : bool, default=True
        If True, project onto row nodes (W @ W.T).
        If False, project onto column nodes (W.T @ W).
    verbose : bool, default=False
        If True, show progress bar.

    Returns
    -------
    pvalues : torch.Tensor
        Matrix of p-values for the projection.
        Shape is (n_rows, n_rows) if rows=True, (n_cols, n_cols) otherwise.
        Diagonal entries are set to 1.0.

    Examples
    --------
    >>> from bicm.wbnm.models import BiECM
    >>> from bicm.wbnm.validation import pvalues_projection
    >>>
    >>> model = BiECM(W)
    >>> model.solve()
    >>>
    >>> # P-values for row projection
    >>> pvals_rows = pvalues_projection(model, n_samples=1000, rows=True)
    >>>
    >>> # Significant pairs (FDR corrected)
    >>> from bicm.wbnm.validation import fdr_bh
    >>> significant = fdr_bh(pvals_rows, alpha=0.05)

    Notes
    -----
    This function uses Monte Carlo sampling because the distribution of
    projection weights is complex and does not have a simple closed form.
    The p-value for entry (i,j) is estimated as (c + 1) / (n_samples + 1),
    where c is the number of samples whose projection weight is >= the
    observed one. The estimate cannot be smaller than 1 / (n_samples + 1):
    use enough samples for the multiple testing correction that follows.
    """
    model._check_solved()

    # Get original matrix
    original_matrix = model.biadjacency

    # Compute observed projection
    if rows:
        observed_projection = original_matrix @ original_matrix.T
        n = model.n_rows
    else:
        observed_projection = original_matrix.T @ original_matrix
        n = model.n_cols

    # Initialize p-value counter
    pvalue_counts = torch.zeros(n, n, dtype=torch.float64, device=model.device)

    # Monte Carlo sampling
    if verbose:
        try:
            from tqdm import trange
            iterator = trange(n_samples, desc="Computing projection p-values")
        except ImportError:
            iterator = range(n_samples)
            print(f"Computing projection p-values ({n_samples} samples)...")
    else:
        iterator = range(n_samples)

    for _ in iterator:
        # Generate sample
        sample = model.sample(n_samples=1)[0]

        # Compute projection
        if rows:
            sample_projection = sample @ sample.T
        else:
            sample_projection = sample.T @ sample

        # Count where sample >= observed
        pvalue_counts += (sample_projection >= observed_projection).to(torch.float64)

    # Compute p-values. The observed network counts as one more sample: the
    # estimate is never zero, its smallest value is 1 / (n_samples + 1).
    pvalues = (pvalue_counts + 1) / (n_samples + 1)

    # Set diagonal to 1 (self-connections are not meaningful)
    pvalues.fill_diagonal_(1.0)

    return pvalues


def pvalues_projection_binary(
    model: "BipartiteModel",
    n_samples: int = 1000,
    rows: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute p-values for the binary projected network.

    Similar to pvalues_projection, but uses the binary (unweighted)
    projection where edge weights count the number of shared neighbors.

    Parameters
    ----------
    model : BipartiteModel
        A solved bipartite null model (BiECM or BiPECM).
        Must have a sample_topology method.
    n_samples : int, default=1000
        Number of Monte Carlo samples to generate.
    rows : bool, default=True
        If True, project onto row nodes.
        If False, project onto column nodes.
    verbose : bool, default=False
        If True, show progress bar.

    Returns
    -------
    pvalues : torch.Tensor
        Matrix of p-values for the binary projection.
        Diagonal entries are set to 1.0.
    """
    model._check_solved()

    # Get binary topology
    original_binary = (model.biadjacency > 0).to(torch.float64)

    # Compute observed projection
    if rows:
        observed_projection = original_binary @ original_binary.T
        n = model.n_rows
    else:
        observed_projection = original_binary.T @ original_binary
        n = model.n_cols

    # Initialize p-value counter
    pvalue_counts = torch.zeros(n, n, dtype=torch.float64, device=model.device)

    # Check if model has topology sampling
    if not hasattr(model, 'sample_topology'):
        # Fall back to binarizing weighted samples
        sample_func = lambda: (model.sample(n_samples=1)[0] > 0).to(torch.float64)
    else:
        sample_func = lambda: model.sample_topology(n_samples=1)[0].to(torch.float64)

    # Monte Carlo sampling
    if verbose:
        try:
            from tqdm import trange
            iterator = trange(n_samples, desc="Computing binary projection p-values")
        except ImportError:
            iterator = range(n_samples)
            print(f"Computing binary projection p-values ({n_samples} samples)...")
    else:
        iterator = range(n_samples)

    for _ in iterator:
        # Generate sample topology
        sample_binary = sample_func()

        # Compute projection
        if rows:
            sample_projection = sample_binary @ sample_binary.T
        else:
            sample_projection = sample_binary.T @ sample_binary

        # Count where sample >= observed
        pvalue_counts += (sample_projection >= observed_projection).to(torch.float64)

    # Compute p-values, see pvalues_projection
    pvalues = (pvalue_counts + 1) / (n_samples + 1)

    # Set diagonal to 1
    pvalues.fill_diagonal_(1.0)

    return pvalues
