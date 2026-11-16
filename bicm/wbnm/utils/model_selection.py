"""
Model selection utilities for comparing null models.
"""

from typing import List, Dict, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from bicm.wbnm.models.base import BipartiteModel


def compare_models(
    models: List["BipartiteModel"],
    criterion: str = "aic"
) -> Dict[str, Union[torch.Tensor, List[str], int]]:
    """
    Compare multiple fitted models using information criteria.

    Parameters
    ----------
    models : list of BipartiteModel
        List of solved models to compare. All models must be fitted
        to the same data.
    criterion : {"aic", "aicc", "bic"}, default="aic"
        Information criterion to use for comparison.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "values": tensor of IC values for each model
        - "delta": tensor of differences from minimum IC
        - "weights": tensor of Akaike weights (probability each model is best)
        - "names": list of model class names
        - "best_idx": index of the best model

    Examples
    --------
    >>> from bicm.wbnm.models import BiWCM, BiECM, BiPECM
    >>> from bicm.wbnm.utils import compare_models
    >>>
    >>> # Fit all models to the same data
    >>> models = [
    ...     BiWCM(W).solve(),
    ...     BiECM(W).solve(),
    ...     BiPECM(W).solve(),
    ... ]
    >>>
    >>> # Compare using AIC
    >>> result = compare_models(models, criterion="aic")
    >>> print(f"Best model: {result['names'][result['best_idx']]}")
    >>> print(f"Weights: {result['weights']}")

    Notes
    -----
    Akaike weights represent the probability that each model is the
    best model given the data and the set of candidate models.
    A weight > 0.9 suggests strong evidence for that model.
    """
    if not models:
        raise ValueError("At least one model is required")

    # Check all models are solved
    for model in models:
        if not model.is_solved:
            raise RuntimeError(f"Model {model.__class__.__name__} is not solved")

    # Get IC values
    if criterion == "aic":
        ic_values = torch.tensor([m.aic() for m in models], dtype=torch.float64)
    elif criterion == "aicc":
        ic_values = torch.tensor([m.aicc() for m in models], dtype=torch.float64)
    elif criterion == "bic":
        ic_values = torch.tensor([m.bic() for m in models], dtype=torch.float64)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'aic', 'aicc', or 'bic'.")

    # Compute delta IC (difference from minimum)
    min_ic = ic_values.min()
    delta_ic = ic_values - min_ic

    # Compute Akaike weights
    weights = torch.exp(-delta_ic / 2)
    weights = weights / weights.sum()

    # Get model names
    names = [m.__class__.__name__ for m in models]

    return {
        "values": ic_values,
        "delta": delta_ic,
        "weights": weights,
        "names": names,
        "best_idx": int(torch.argmin(ic_values).item()),
        "criterion": criterion,
    }


def print_model_comparison(
    models: List["BipartiteModel"],
    criterion: str = "aic"
) -> None:
    """
    Print a formatted comparison of models.

    Parameters
    ----------
    models : list of BipartiteModel
        List of solved models to compare.
    criterion : {"aic", "aicc", "bic"}, default="aic"
        Information criterion to use.

    Examples
    --------
    >>> print_model_comparison([model_wcm, model_ecm, model_pecm])
    Model Comparison (AIC)
    ─────────────────────────────────────────
    Model      AIC        ΔAIC     Weight
    ─────────────────────────────────────────
    BiECM      1234.5     0.0      0.85  ***
    BiPECM     1238.2     3.7      0.13
    BiWCM      1245.1     10.6     0.02
    ─────────────────────────────────────────
    """
    result = compare_models(models, criterion)

    # Sort by IC value
    order = torch.argsort(result["values"])

    print(f"\nModel Comparison ({criterion.upper()})")
    print("-" * 50)
    print(f"{'Model':<12} {criterion.upper():<12} {'d'+criterion.upper():<10} {'Weight':<10}")
    print("-" * 50)

    for idx in order.tolist():
        name = result["names"][idx]
        value = result["values"][idx].item()
        delta = result["delta"][idx].item()
        weight = result["weights"][idx].item()

        # Add significance indicator
        if weight > 0.9:
            sig = " ***"
        elif weight > 0.5:
            sig = " **"
        elif weight > 0.25:
            sig = " *"
        else:
            sig = ""

        print(f"{name:<12} {value:<12.1f} {delta:<10.1f} {weight:<10.3f}{sig}")

    print("-" * 50)
    print(f"Best model: {result['names'][result['best_idx']]}")
    print()
