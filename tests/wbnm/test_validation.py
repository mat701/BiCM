"""Tests for validation module."""

import torch
import pytest

from bicm.wbnm.validation import (
    compute_pvalue,
    validate_property,
    bonferroni,
    fdr_bh,
    apply_correction,
)


class TestPValues:
    """Tests for p-value computation."""

    def test_compute_pvalue_greater(self):
        """Test one-sided greater p-value."""
        samples = torch.arange(100, dtype=torch.float64)

        # Value at 95th percentile should have p ~0.05
        pvalue = compute_pvalue(95, samples, alternative="greater")
        assert 0.04 < pvalue < 0.08

    def test_compute_pvalue_less(self):
        """Test one-sided less p-value."""
        samples = torch.arange(100, dtype=torch.float64)

        # Value at 5th percentile should have p ~0.05
        pvalue = compute_pvalue(5, samples, alternative="less")
        assert 0.04 < pvalue < 0.08

    def test_compute_pvalue_two_sided(self):
        """Test two-sided p-value."""
        torch.manual_seed(42)
        samples = torch.randn(1000, dtype=torch.float64)

        # Value at mean should have p ~1
        pvalue = compute_pvalue(0, samples, alternative="two-sided")
        assert pvalue > 0.5

        # Extreme value should have low p
        pvalue = compute_pvalue(3, samples, alternative="two-sided")
        assert pvalue < 0.05

    def test_validate_property_scalar(self):
        """Test validation of scalar property."""
        torch.manual_seed(42)
        samples = torch.randn(1000, dtype=torch.float64)

        pvalue = validate_property(2.5, samples, alternative="greater")
        assert isinstance(pvalue, float)
        assert pvalue < 0.01

    def test_validate_property_array(self):
        """Test validation of array property."""
        torch.manual_seed(42)
        observed = torch.tensor([2.5, 0.0, -1.5], dtype=torch.float64)
        samples = torch.randn(1000, 3, dtype=torch.float64)

        pvalues = validate_property(observed, samples, alternative="greater")

        assert pvalues.shape == (3,)
        assert pvalues[0].item() < 0.02  # 2.5 is extreme
        assert pvalues[1].item() > 0.3   # 0 is not extreme
        assert pvalues[2].item() > 0.9   # -1.5 is low, not high


class TestCorrections:
    """Tests for multiple testing corrections."""

    @pytest.fixture
    def pvalues(self):
        return torch.tensor([0.001, 0.01, 0.04, 0.03, 0.5], dtype=torch.float64)

    def test_bonferroni(self, pvalues):
        """Test Bonferroni correction."""
        rejected, adjusted = bonferroni(pvalues, alpha=0.05)

        # Only the first p-value should survive
        assert rejected[0].item() == True
        assert rejected[1:].sum().item() == 0

        # Adjusted p-values
        torch.testing.assert_close(
            adjusted,
            torch.clamp(pvalues * 5, max=1.0)
        )

    def test_fdr_bh(self, pvalues):
        """Test Benjamini-Hochberg correction."""
        rejected, adjusted = fdr_bh(pvalues, alpha=0.05)

        # More lenient than Bonferroni
        assert rejected.sum().item() >= 1

        # Q-values should be <= 1
        assert (adjusted <= 1.0).all()

    def test_apply_correction(self, pvalues):
        """Test apply_correction wrapper."""
        for method in ["bonferroni", "fdr_bh", "none"]:
            rejected, adjusted = apply_correction(pvalues, method=method)

            assert rejected.shape == pvalues.shape
            assert adjusted.shape == pvalues.shape

    def test_correction_preserves_shape(self):
        """Test that 2D p-values preserve shape."""
        torch.manual_seed(42)
        pvalues = torch.rand(5, 5, dtype=torch.float64)

        rejected, adjusted = apply_correction(pvalues)

        assert rejected.shape == (5, 5)
        assert adjusted.shape == (5, 5)
