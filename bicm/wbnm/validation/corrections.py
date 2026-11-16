"""
Multiple testing corrections for statistical validation.
"""

from typing import Tuple, Literal, Optional
import torch


def bonferroni(
    pvalues: torch.Tensor,
    alpha: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bonferroni correction for multiple testing.

    The most conservative correction: rejects H0 if p < alpha/n.

    Parameters
    ----------
    pvalues : torch.Tensor
        Tensor of p-values.
    alpha : float, default=0.05
        Family-wise error rate.

    Returns
    -------
    rejected : torch.Tensor
        Boolean tensor indicating which hypotheses are rejected.
    adjusted : torch.Tensor
        Adjusted p-values (clipped to 1).

    Examples
    --------
    >>> pvalues = torch.tensor([0.01, 0.04, 0.03, 0.001])
    >>> rejected, adjusted = bonferroni(pvalues)
    >>> print(rejected)
    tensor([False, False, False,  True])
    """
    pvalues = torch.as_tensor(pvalues, dtype=torch.float64).flatten()
    n = len(pvalues)

    adjusted = torch.clamp(pvalues * n, max=1.0)
    rejected = pvalues < alpha / n

    return rejected, adjusted


def fdr(
    pvalues: torch.Tensor,
    alpha: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Benjamini-Hochberg procedure for FDR control.

    This is the standard FDR correction method. Controls the False Discovery
    Rate at level alpha under independence or positive dependence.

    Parameters
    ----------
    pvalues : torch.Tensor
        Tensor of p-values.
    alpha : float, default=0.05
        Desired FDR level.

    Returns
    -------
    rejected : torch.Tensor
        Boolean tensor indicating which hypotheses are rejected.
    adjusted : torch.Tensor
        Adjusted p-values (q-values).

    References
    ----------
    - Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"

    Examples
    --------
    >>> pvalues = torch.tensor([0.01, 0.04, 0.03, 0.001])
    >>> rejected, adjusted = fdr(pvalues)

    """
    pvalues = torch.as_tensor(pvalues, dtype=torch.float64).flatten()
    n = len(pvalues)

    # Sort p-values
    sorted_idx = torch.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_idx]

    # Compute BH threshold: p_(i) <= i/n * alpha
    ranks = torch.arange(1, n + 1, dtype=torch.float64, device=pvalues.device)
    thresholds = ranks / n * alpha

    # Find largest k where p_(k) <= k/n * alpha
    below_threshold = sorted_pvalues <= thresholds
    if below_threshold.any():
        max_k = int(torch.nonzero(below_threshold).max().item()) + 1
    else:
        max_k = 0

    # Reject all hypotheses with rank <= max_k
    rejected = torch.zeros(n, dtype=torch.bool, device=pvalues.device)
    rejected[sorted_idx[:max_k]] = True

    # Compute adjusted p-values (q-values)
    # q_i = min_{j>=i} (n/j * p_(j))
    reversed_vals = (n / ranks.flip(0) * sorted_pvalues.flip(0)).flip(0)
    adjusted_sorted = torch.cummin(reversed_vals, dim=0).values
    adjusted_sorted = torch.clamp(adjusted_sorted, max=1.0)

    # Map back to original order
    adjusted = torch.empty(n, dtype=torch.float64, device=pvalues.device)
    adjusted[sorted_idx] = adjusted_sorted

    return rejected, adjusted


# Alias for backwards compatibility
fdr_bh = fdr


def apply_correction(
    pvalues: torch.Tensor,
    method: Literal["bonferroni", "fdr", "fdr_bh", "none"] = "fdr",
    alpha: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a multiple testing correction.

    Parameters
    ----------
    pvalues : torch.Tensor
        Tensor of p-values.
    method : {"bonferroni", "fdr", "none"}, default="fdr"
        Correction method:
        - "bonferroni": Bonferroni correction (FWER control, most conservative)
        - "fdr": Benjamini-Hochberg FDR control (recommended)
        - "none": No correction (just threshold at alpha)
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    rejected : torch.Tensor
        Boolean tensor indicating which hypotheses are rejected.
    adjusted : torch.Tensor
        Adjusted p-values.

    Examples
    --------
    >>> pvalues = torch.tensor([0.01, 0.04, 0.03, 0.001])
    >>> rejected, adjusted = apply_correction(pvalues, method="fdr")
    """
    pvalues = torch.as_tensor(pvalues, dtype=torch.float64)
    original_shape = pvalues.shape
    pvalues_flat = pvalues.flatten()

    if method == "bonferroni":
        rejected, adjusted = bonferroni(pvalues_flat, alpha)
    elif method in ("fdr", "fdr_bh"):
        rejected, adjusted = fdr(pvalues_flat, alpha)
    elif method == "none":
        rejected = pvalues_flat < alpha
        adjusted = pvalues_flat.clone()
    else:
        raise ValueError(f"Unknown method: {method}")

    return rejected.reshape(original_shape), adjusted.reshape(original_shape)
