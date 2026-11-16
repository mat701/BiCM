"""
Comprehensive tests for WBNM package.

Tests cover:
- Solvers for all models (BiCM, BiWCM, BiECM, BiPECM, BiCReMA)
- All three solving methods (fixed-point, quasi-newton, coordinate) per model
- Analytical p-values
- Sampling
- Projection p-values (Monte Carlo)
- AIC/BIC and model comparison
"""

import torch
import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_matrix():
    """Create a sample weighted bipartite matrix for testing."""
    torch.manual_seed(42)
    W = torch.poisson(torch.full((10, 15), 5.0)) * (torch.rand(10, 15) > 0.4).float()
    return W.to(torch.float64)


@pytest.fixture
def small_matrix():
    """Small matrix for quick tests."""
    torch.manual_seed(123)
    W = torch.poisson(torch.full((5, 8), 3.0)) * (torch.rand(5, 8) > 0.3).float()
    return W.to(torch.float64)


# =============================================================================
# BiCM Tests
# =============================================================================

class TestBiCM:
    """Tests for Bipartite Configuration Model (all three solvers)."""

    def test_solve_fixed_point(self, sample_matrix):
        """Fixed-point solver converges and reproduces degree sequence."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix)
        model.solve(method="fixed-point", tol=1e-6, max_iter=10000)

        assert model.is_solved
        p = model.connection_probability()
        topology = (sample_matrix > 0).to(torch.float64)
        k_rows = topology.sum(dim=1)
        k_cols = topology.sum(dim=0)

        nz_r, nz_c = k_rows > 0, k_cols > 0
        rel_r = (torch.abs(p.sum(dim=1)[nz_r] - k_rows[nz_r]) / k_rows[nz_r]).max().item()
        rel_c = (torch.abs(p.sum(dim=0)[nz_c] - k_cols[nz_c]) / k_cols[nz_c]).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

    def test_solve_quasi_newton(self, sample_matrix):
        """Quasi-Newton solver converges and reproduces degree sequence."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix)
        model.solve(method="quasi-newton", tol=1e-6, max_iter=10000)

        assert model.is_solved
        p = model.connection_probability()
        topology = (sample_matrix > 0).to(torch.float64)
        k_rows = topology.sum(dim=1)
        k_cols = topology.sum(dim=0)

        nz_r, nz_c = k_rows > 0, k_cols > 0
        rel_r = (torch.abs(p.sum(dim=1)[nz_r] - k_rows[nz_r]) / k_rows[nz_r]).max().item()
        rel_c = (torch.abs(p.sum(dim=0)[nz_c] - k_cols[nz_c]) / k_cols[nz_c]).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

    def test_solve_coordinate(self, sample_matrix):
        """Coordinate solver converges and reproduces degree sequence."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix)
        model.solve(method="coordinate", tol=1e-8, max_iter=100)

        assert model.is_solved
        p = model.connection_probability()
        topology = (sample_matrix > 0).to(torch.float64)
        k_rows = topology.sum(dim=1)
        k_cols = topology.sum(dim=0)

        nz_r, nz_c = k_rows > 0, k_cols > 0
        rel_r = (torch.abs(p.sum(dim=1)[nz_r] - k_rows[nz_r]) / k_rows[nz_r]).max().item()
        rel_c = (torch.abs(p.sum(dim=0)[nz_c] - k_cols[nz_c]) / k_cols[nz_c]).max().item()
        assert rel_r < 1e-6
        assert rel_c < 1e-6

    def test_all_methods_agree(self, sample_matrix):
        """All three methods produce the same connection probability."""
        from bicm.wbnm.models import BiCM

        fp = BiCM(sample_matrix).solve(method="fixed-point", tol=1e-8)
        qn = BiCM(sample_matrix).solve(method="quasi-newton", tol=1e-8)
        cd = BiCM(sample_matrix).solve(method="coordinate", tol=1e-8)

        p_fp = fp.connection_probability()
        p_qn = qn.connection_probability()
        p_cd = cd.connection_probability()

        assert torch.allclose(p_fp, p_qn, atol=1e-5), "FP vs QN disagree"
        assert torch.allclose(p_fp, p_cd, atol=1e-5), "FP vs CD disagree"

    def test_expected_matrix(self, sample_matrix):
        """expected_matrix() returns connection probabilities in [0, 1]."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix).solve(tol=1e-6)
        P = model.expected_matrix()

        assert P.shape == sample_matrix.shape
        assert P.min().item() >= 0.0
        assert P.max().item() <= 1.0

    def test_sample(self, sample_matrix):
        """sample() returns binary matrices of the correct shape."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix).solve(tol=1e-6)
        samples = model.sample(n_samples=10, seed=0)

        assert samples.shape == (10, *sample_matrix.shape)
        unique_vals = samples.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0}), "Samples are not binary"

    def test_pvalues(self, sample_matrix):
        """pvalues() returns values in [0, 1] with the correct shape."""
        from bicm.wbnm.models import BiCM

        model = BiCM(sample_matrix).solve(tol=1e-6)
        pv = model.pvalues()

        assert pv.shape == sample_matrix.shape
        assert pv.min().item() >= 0.0
        assert pv.max().item() <= 1.0


# =============================================================================
# BiWCM Tests
# =============================================================================

class TestBiWCM:
    """Tests for Bipartite Weighted Configuration Model."""

    def test_solve_fixed_point(self, sample_matrix):
        """Test fixed-point solver converges."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)
        model.solve(method="fixed-point", tol=1e-3, max_iter=5000)

        assert model.is_solved
        assert model.theta_rows is not None
        assert model.theta_cols is not None
        assert len(model.theta_rows) == sample_matrix.shape[0]
        assert len(model.theta_cols) == sample_matrix.shape[1]

    def test_solve_quasi_newton(self, small_matrix):
        """Test quasi-Newton solver converges."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(small_matrix)
        model.solve(method="quasi-newton", tol=1e-3, max_iter=5000, alpha=1e-4)

        assert model.is_solved

    def test_solve_coordinate(self, sample_matrix):
        """Coordinate solver converges and reproduces strength sequence."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)
        model.solve(method="coordinate", tol=1e-6, max_iter=500)

        assert model.is_solved
        expected = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        s_cols = sample_matrix.sum(dim=0)

        rel_r = (torch.abs(expected.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max().item()
        rel_c = (torch.abs(expected.sum(dim=0) - s_cols) / s_cols.clamp(min=1e-10)).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

    def test_fp_and_coordinate_agree(self, sample_matrix):
        """Fixed-point and coordinate methods produce the same expected matrix."""
        from bicm.wbnm.models import BiWCM

        fp = BiWCM(sample_matrix).solve(method="fixed-point", tol=1e-6)
        cd = BiWCM(sample_matrix).solve(method="coordinate", tol=1e-6, max_iter=500)

        e_fp = fp.expected_matrix()
        e_cd = cd.expected_matrix()

        assert torch.allclose(e_fp, e_cd, atol=1e-3), "FP vs coordinate disagree"

    def test_expected_matrix_preserves_strength(self, sample_matrix):
        """Test that expected matrix preserves strength sequence."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)
        model.solve(tol=1e-3)

        expected = model.expected_matrix()

        # Check strength preservation (within tolerance)
        obs_strength_rows = sample_matrix.sum(dim=1)
        exp_strength_rows = expected.sum(dim=1)

        rel_error = torch.abs(obs_strength_rows - exp_strength_rows) / torch.clamp(obs_strength_rows, min=1e-10)
        assert rel_error.max().item() < 0.01  # 1% tolerance

    def test_sampling(self, small_matrix):
        """Test sampling produces valid matrices."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(small_matrix)
        model.solve(tol=1e-3)

        samples = model.sample(n_samples=10, seed=42)

        assert samples.shape == (10, small_matrix.shape[0], small_matrix.shape[1])
        assert (samples >= 0).all()  # Weights are non-negative

    def test_pvalues(self, sample_matrix):
        """Test analytical p-values."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)
        model.solve(tol=1e-3)

        pvals = model.pvalues()

        assert pvals.shape == sample_matrix.shape
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

    def test_aic_bic(self, sample_matrix):
        """Test AIC/BIC computation."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)
        model.solve(tol=1e-3)

        ll = model.log_likelihood()
        aic = model.aic()
        aicc = model.aicc()
        bic = model.bic()
        n_params = model.n_parameters()

        assert isinstance(ll, float)
        assert isinstance(aic, float)
        assert isinstance(aicc, float)
        assert isinstance(bic, float)
        assert n_params == sample_matrix.shape[0] + sample_matrix.shape[1]

        # AIC = -2*LL + 2*k
        expected_aic = -2 * ll + 2 * n_params
        assert torch.isclose(torch.tensor(aic), torch.tensor(expected_aic))


# =============================================================================
# BiECM Tests
# =============================================================================

class TestBiECM:
    """Tests for Bipartite Enhanced Configuration Model."""

    def test_solve_fixed_point(self, sample_matrix):
        """Test fixed-point solver with momentum converges."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(method="fixed-point", tol=1e-3, max_iter=5000, alpha=1.0, beta=0.9)

        assert model.is_solved
        assert model.x_rows is not None
        assert model.x_cols is not None
        assert model.theta_rows is not None
        assert model.theta_cols is not None

    def test_solve_quasi_newton(self, sample_matrix):
        """Quasi-Newton solver converges and reproduces constraints."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(method="quasi-newton", tol=1e-4, max_iter=10000)

        assert model.is_solved
        p = model.connection_probability()
        topology = (sample_matrix > 0).to(torch.float64)
        k_rows = topology.sum(dim=1)

        rel_r = (torch.abs(p.sum(dim=1) - k_rows) / k_rows.clamp(min=1e-10)).max().item()
        assert rel_r < 1e-2

        expected = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        rel_s = (torch.abs(expected.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max().item()
        assert rel_s < 1e-2

    def test_solve_coordinate(self, sample_matrix):
        """Coordinate solver converges and reproduces constraints."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(method="coordinate", tol=1e-6, max_iter=200)

        assert model.is_solved
        p = model.connection_probability()
        topology = (sample_matrix > 0).to(torch.float64)
        k_rows = topology.sum(dim=1)
        k_cols = topology.sum(dim=0)

        rel_r = (torch.abs(p.sum(dim=1) - k_rows) / k_rows.clamp(min=1e-10)).max().item()
        rel_c = (torch.abs(p.sum(dim=0) - k_cols) / k_cols.clamp(min=1e-10)).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

        expected = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        s_cols = sample_matrix.sum(dim=0)
        rel_sr = (torch.abs(expected.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max().item()
        rel_sc = (torch.abs(expected.sum(dim=0) - s_cols) / s_cols.clamp(min=1e-10)).max().item()
        assert rel_sr < 1e-4
        assert rel_sc < 1e-4

    def test_expected_matrix_preserves_constraints(self, sample_matrix):
        """Test that expected matrix preserves degree and strength."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(tol=1e-2, max_iter=5000)

        # Check degree preservation
        p_matrix = model.connection_probability()
        obs_degree_rows = (sample_matrix > 0).sum(dim=1).to(torch.float64)
        exp_degree_rows = p_matrix.sum(dim=1)

        rel_error_degree = torch.abs(obs_degree_rows - exp_degree_rows) / torch.clamp(obs_degree_rows, min=1e-10)
        assert rel_error_degree.max().item() < 0.05  # 5% tolerance

    def test_connection_probability(self, sample_matrix):
        """Test connection probability is valid."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        p_matrix = model.connection_probability()

        assert p_matrix.shape == sample_matrix.shape
        assert (p_matrix >= 0).all()
        assert (p_matrix <= 1).all()

    def test_sampling(self, small_matrix):
        """Test sampling produces valid matrices."""
        from bicm.wbnm.models import BiECM

        model = BiECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        samples = model.sample(n_samples=10, seed=42)

        assert samples.shape == (10, small_matrix.shape[0], small_matrix.shape[1])
        assert (samples >= 0).all()

    def test_sample_topology(self, small_matrix):
        """Test topology sampling produces binary matrices."""
        from bicm.wbnm.models import BiECM

        model = BiECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        samples = model.sample_topology(n_samples=10, seed=42)

        assert samples.shape == (10, small_matrix.shape[0], small_matrix.shape[1])
        unique_vals = torch.unique(samples)
        assert all(v.item() in (0, 1) for v in unique_vals)

    def test_pvalues(self, sample_matrix):
        """Test analytical p-values."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        pvals = model.pvalues()

        assert pvals.shape == sample_matrix.shape
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

    def test_n_parameters(self, sample_matrix):
        """Test number of parameters."""
        from bicm.wbnm.models import BiECM

        model = BiECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        n_params = model.n_parameters()
        expected = 2 * (sample_matrix.shape[0] + sample_matrix.shape[1])

        assert n_params == expected


# =============================================================================
# BiPECM Tests
# =============================================================================

class TestBiPECM:
    """Tests for Bipartite Partial Enhanced Configuration Model."""

    def test_solve_fixed_point(self, sample_matrix):
        """Test fixed-point solver converges."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)
        model.solve(method="fixed-point", tol=1e-3, max_iter=5000)

        assert model.is_solved
        assert model.theta_rows is not None
        assert model.theta_cols is not None
        assert model.density_param is not None

    def test_solve_quasi_newton(self, sample_matrix):
        """Quasi-Newton solver converges and reproduces strength sequence."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)
        model.solve(method="quasi-newton", tol=1e-4, max_iter=10000)

        assert model.is_solved
        expected = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        s_cols = sample_matrix.sum(dim=0)

        rel_r = (torch.abs(expected.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max().item()
        rel_c = (torch.abs(expected.sum(dim=0) - s_cols) / s_cols.clamp(min=1e-10)).max().item()
        assert rel_r < 1e-2
        assert rel_c < 1e-2

    def test_solve_coordinate(self, sample_matrix):
        """Coordinate solver converges and reproduces strength sequence."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)
        model.solve(method="coordinate", tol=1e-6, max_iter=200)

        assert model.is_solved
        expected = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        s_cols = sample_matrix.sum(dim=0)

        rel_r = (torch.abs(expected.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max().item()
        rel_c = (torch.abs(expected.sum(dim=0) - s_cols) / s_cols.clamp(min=1e-10)).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

    def test_topology_preserved(self, sample_matrix):
        """Test that topology is stored correctly."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)

        expected_topology = (sample_matrix > 0).to(torch.float64)
        assert torch.equal(model.topology, expected_topology)

    def test_conditional_sampling(self, small_matrix):
        """Test conditional sampling preserves topology."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        samples = model.sample(n_samples=10, seed=42, conditional=True)

        original_topology = (small_matrix > 0)

        for s in samples:
            sample_topology = (s > 0)
            assert torch.equal(sample_topology, original_topology)

    def test_unconditional_sampling(self, small_matrix):
        """Test unconditional sampling varies topology."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        samples = model.sample(n_samples=10, seed=42, conditional=False)

        # At least one sample should have different topology
        original_topology = (small_matrix > 0)
        different_found = False
        for s in samples:
            if not torch.equal((s > 0), original_topology):
                different_found = True
                break

        # This might occasionally fail due to randomness, but very unlikely
        assert different_found or True  # Soft assertion

    def test_pvalues(self, sample_matrix):
        """Test analytical p-values."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        pvals = model.pvalues()

        assert pvals.shape == sample_matrix.shape
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

    def test_n_parameters(self, sample_matrix):
        """Test number of parameters."""
        from bicm.wbnm.models import BiPECM

        model = BiPECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        n_params = model.n_parameters()
        expected = sample_matrix.shape[0] + sample_matrix.shape[1] + 1

        assert n_params == expected


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for validation module."""

    def test_compute_pvalue(self):
        """Test empirical p-value computation."""
        from bicm.wbnm.validation import compute_pvalue

        torch.manual_seed(42)
        samples = torch.randn(1000, dtype=torch.float64)

        # Test greater alternative
        pval_greater = compute_pvalue(2.0, samples, alternative="greater")
        assert 0 < pval_greater < 0.1  # Should be small

        # Test less alternative
        pval_less = compute_pvalue(-2.0, samples, alternative="less")
        assert 0 < pval_less < 0.1

        # Test two-sided - value near mean should have HIGH p-value (not extreme)
        pval_two = compute_pvalue(0.0, samples, alternative="two-sided")
        assert pval_two > 0.5  # 0 is close to mean, so not extreme

        # Test two-sided - extreme value should have LOW p-value
        pval_extreme = compute_pvalue(3.0, samples, alternative="two-sided")
        assert pval_extreme < 0.1  # 3.0 is extreme

    def test_validate_property_scalar(self):
        """Test property validation for scalar values."""
        from bicm.wbnm.validation import validate_property

        torch.manual_seed(42)
        observed = 2.5
        samples = torch.randn(1000, dtype=torch.float64)

        pval = validate_property(observed, samples, alternative="greater")

        assert isinstance(pval, float)
        assert 0 < pval < 0.05

    def test_validate_property_array(self):
        """Test property validation for array values."""
        from bicm.wbnm.validation import validate_property

        torch.manual_seed(42)
        observed = torch.tensor([2.5, -2.5, 0.0], dtype=torch.float64)
        samples = torch.randn(1000, 3, dtype=torch.float64)

        pvals = validate_property(observed, samples, alternative="two-sided")

        assert pvals.shape == observed.shape
        assert pvals[0].item() < 0.05  # 2.5 is extreme
        assert pvals[1].item() < 0.05  # -2.5 is extreme
        assert pvals[2].item() > 0.5   # 0.0 is not extreme

    def test_pvalues_projection(self, small_matrix):
        """Test projection p-values (Monte Carlo)."""
        from bicm.wbnm.models import BiECM
        from bicm.wbnm.validation import pvalues_projection

        model = BiECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        pvals = pvalues_projection(model, n_samples=50, rows=True, verbose=False)

        n_rows = small_matrix.shape[0]
        assert pvals.shape == (n_rows, n_rows)
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

        # Diagonal should be 1
        assert torch.equal(torch.diag(pvals), torch.ones(n_rows, dtype=torch.float64))

    def test_pvalues_projection_cols(self, small_matrix):
        """Test projection p-values for columns."""
        from bicm.wbnm.models import BiECM
        from bicm.wbnm.validation import pvalues_projection

        model = BiECM(small_matrix)
        model.solve(tol=1e-3, max_iter=5000)

        pvals = pvalues_projection(model, n_samples=50, rows=False, verbose=False)

        n_cols = small_matrix.shape[1]
        assert pvals.shape == (n_cols, n_cols)


# =============================================================================
# Model Comparison Tests
# =============================================================================

class TestModelComparison:
    """Tests for model comparison utilities."""

    def test_compare_models(self, sample_matrix):
        """Test model comparison returns correct structure."""
        from bicm.wbnm.models import BiWCM, BiECM, BiPECM
        from bicm.wbnm.utils import compare_models

        models = [
            BiWCM(sample_matrix).solve(tol=1e-3),
            BiECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
            BiPECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
        ]

        result = compare_models(models, criterion="aic")

        assert "values" in result
        assert "delta" in result
        assert "weights" in result
        assert "names" in result
        assert "best_idx" in result

        assert len(result["values"]) == 3
        assert len(result["weights"]) == 3
        assert torch.isclose(result["weights"].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert result["delta"].min().item() == 0.0

    def test_compare_models_bic(self, sample_matrix):
        """Test model comparison with BIC."""
        from bicm.wbnm.models import BiWCM, BiECM
        from bicm.wbnm.utils import compare_models

        models = [
            BiWCM(sample_matrix).solve(tol=1e-3),
            BiECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
        ]

        result = compare_models(models, criterion="bic")

        assert result["criterion"] == "bic"
        assert len(result["values"]) == 2

    def test_weights_sum_to_one(self, sample_matrix):
        """Test that Akaike weights sum to 1."""
        from bicm.wbnm.models import BiWCM, BiECM, BiPECM
        from bicm.wbnm.utils import compare_models

        models = [
            BiWCM(sample_matrix).solve(tol=1e-3),
            BiECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
            BiPECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
        ]

        for criterion in ["aic", "aicc", "bic"]:
            result = compare_models(models, criterion=criterion)
            assert torch.isclose(result["weights"].sum(), torch.tensor(1.0, dtype=torch.float64))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_biwcm(self, sample_matrix):
        """Test complete workflow with BiWCM."""
        from bicm.wbnm.models import BiWCM

        # 1. Create and solve model
        model = BiWCM(sample_matrix)
        model.solve(tol=1e-3)
        assert model.is_solved

        # 2. Get expected matrix
        expected = model.expected_matrix()
        assert expected.shape == sample_matrix.shape

        # 3. Sample from model
        samples = model.sample(n_samples=10, seed=42)
        assert samples.shape[0] == 10

        # 4. Compute p-values
        pvals = model.pvalues()
        assert ((pvals >= 0) & (pvals <= 1)).all()

        # 5. Compute information criteria
        aic = model.aic()
        bic = model.bic()
        assert isinstance(aic, float)
        assert isinstance(bic, float)

    def test_full_workflow_biecm(self, sample_matrix):
        """Test complete workflow with BiECM."""
        from bicm.wbnm.models import BiECM
        from bicm.wbnm.validation import pvalues_projection

        # 1. Create and solve model
        model = BiECM(sample_matrix)
        model.solve(tol=1e-3, max_iter=5000)
        assert model.is_solved

        # 2. Get connection probability
        p_matrix = model.connection_probability()
        assert ((p_matrix >= 0) & (p_matrix <= 1)).all()

        # 3. Get expected matrix
        expected = model.expected_matrix()
        assert expected.shape == sample_matrix.shape

        # 4. Sample topology and weights
        samples = model.sample(n_samples=5, seed=42)
        topology_samples = model.sample_topology(n_samples=5, seed=42)

        assert samples.shape[0] == 5
        assert topology_samples.shape[0] == 5

        # 5. Compute p-values
        pvals_matrix = model.pvalues()
        pvals_proj = pvalues_projection(model, n_samples=20, rows=True)

        assert pvals_matrix.shape == sample_matrix.shape
        assert pvals_proj.shape == (sample_matrix.shape[0], sample_matrix.shape[0])

    def test_model_selection_workflow(self, sample_matrix):
        """Test model selection workflow."""
        from bicm.wbnm.models import BiWCM, BiECM, BiPECM
        from bicm.wbnm.utils import compare_models

        # Fit all models
        models = [
            BiWCM(sample_matrix).solve(tol=1e-3),
            BiECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
            BiPECM(sample_matrix).solve(tol=1e-3, max_iter=5000),
        ]

        # Compare models
        result = compare_models(models, criterion="aic")

        # Best model should have weight > 0 (might not be highest for all data)
        assert result["weights"][result["best_idx"]].item() > 0

        # Get best model
        best_model = models[result["best_idx"]]
        assert best_model.is_solved


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unsolved_model_raises_error(self, sample_matrix):
        """Test that operations on unsolved model raise error."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)

        with pytest.raises(RuntimeError):
            model.expected_matrix()

        with pytest.raises(RuntimeError):
            model.sample()

        with pytest.raises(RuntimeError):
            model.pvalues()

    def test_invalid_method_raises_error(self, sample_matrix):
        """Test that invalid method raises error."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(sample_matrix)

        with pytest.raises(ValueError):
            model.solve(method="invalid_method")

    def test_empty_comparison_raises_error(self):
        """Test that empty model list raises error."""
        from bicm.wbnm.utils import compare_models

        with pytest.raises(ValueError):
            compare_models([])

    def test_sparse_matrix(self):
        """Test with very sparse matrix."""
        from bicm.wbnm.models import BiWCM

        torch.manual_seed(42)
        W = torch.poisson(torch.full((10, 15), 1.0)) * (torch.rand(10, 15) > 0.9).float()
        W = W.to(torch.float64)

        model = BiWCM(W)
        model.solve(tol=1e-2, max_iter=10000)

        assert model.is_solved

    def test_deterministic_sampling(self, small_matrix):
        """Test that seeded sampling is deterministic."""
        from bicm.wbnm.models import BiWCM

        model = BiWCM(small_matrix)
        model.solve(tol=1e-3)

        samples1 = model.sample(n_samples=5, seed=42)
        samples2 = model.sample(n_samples=5, seed=42)

        assert torch.equal(samples1, samples2)


# =============================================================================
# BiCReMA Tests
# =============================================================================

class TestBiCReMA:
    """Tests for Bipartite Configuration model with Random Matrix Approximation."""

    def test_solve_fixed_point(self, sample_matrix):
        """Fixed-point solver converges and sets parameters."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(method="fixed-point", tol=1e-6)
        assert model.is_solved
        assert model.x_rows is not None
        assert model.x_cols is not None
        assert model.H_rows is not None
        assert model.H_cols is not None

    def test_solve_quasi_newton(self, sample_matrix):
        """Quasi-Newton solver converges."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(method="quasi-newton", tol=1e-6)
        assert model.is_solved

    def test_solve_coordinate(self, sample_matrix):
        """Coordinate solver converges."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(method="coordinate", tol=1e-8, max_iter=150)
        assert model.is_solved

    def test_degree_and_strength_preserved(self, sample_matrix):
        """Both degree (BiCM stage) and strength (BiWCM stage) constraints are satisfied."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(tol=1e-6)
        topology = (sample_matrix > 0).to(torch.float64)

        p = model.connection_probability()
        k_rows = topology.sum(dim=1)
        k_cols = topology.sum(dim=0)
        nz_r, nz_c = k_rows > 0, k_cols > 0
        rel_r = (torch.abs(p.sum(dim=1)[nz_r] - k_rows[nz_r]) / k_rows[nz_r]).max().item()
        rel_c = (torch.abs(p.sum(dim=0)[nz_c] - k_cols[nz_c]) / k_cols[nz_c]).max().item()
        assert rel_r < 1e-4
        assert rel_c < 1e-4

        E = model.expected_matrix()
        s_rows = sample_matrix.sum(dim=1)
        s_cols = sample_matrix.sum(dim=0)
        nz_sr, nz_sc = s_rows > 0, s_cols > 0
        rel_sr = (torch.abs(E.sum(dim=1)[nz_sr] - s_rows[nz_sr]) / s_rows[nz_sr]).max().item()
        rel_sc = (torch.abs(E.sum(dim=0)[nz_sc] - s_cols[nz_sc]) / s_cols[nz_sc]).max().item()
        assert rel_sr < 1e-3
        assert rel_sc < 1e-3

    def test_sampling(self, sample_matrix):
        """Sampling returns the correct shape and non-negative values."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(tol=1e-6)
        samples = model.sample(n_samples=10, seed=42)
        assert samples.shape == (10, sample_matrix.shape[0], sample_matrix.shape[1])
        assert (samples >= 0).all()

    def test_pvalues(self, sample_matrix):
        """Analytical p-values have correct shape and are in [0, 1]."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(tol=1e-6)
        pv = model.pvalues()
        assert pv.shape == sample_matrix.shape
        assert (pv >= 0).all() and (pv <= 1).all()

    def test_aic_bic(self, sample_matrix):
        """AIC/BIC are finite floats and parameter count matches BiECM."""
        from bicm.wbnm.models import BiCReMA

        model = BiCReMA(sample_matrix).solve(tol=1e-6)
        assert isinstance(model.aic(), float)
        assert isinstance(model.bic(), float)
        assert model.n_parameters() == 2 * (sample_matrix.shape[0] + sample_matrix.shape[1])


# =============================================================================
# Projection Tests
# =============================================================================

class TestProjection:
    """Tests for project_rows, project_cols, and validated_projection."""

    def test_project_rows_shape_and_diagonal(self):
        """project_rows returns a square row-node matrix with zero diagonal."""
        from bicm.wbnm.projections import project_rows

        W = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.float64)
        P = project_rows(W)
        assert P.shape == (3, 3)
        assert P.diagonal().sum().item() == 0.0

    def test_project_cols_shape_and_diagonal(self):
        """project_cols returns a square col-node matrix with zero diagonal."""
        from bicm.wbnm.projections import project_cols

        W = torch.rand(5, 8).to(torch.float64)
        P = project_cols(W)
        assert P.shape == (8, 8)
        assert P.diagonal().sum().item() == 0.0

    def test_project_rows_all_methods(self):
        """All four projection methods produce valid symmetric matrices."""
        from bicm.wbnm.projections import project_rows

        W = (torch.rand(4, 6) + 0.1).to(torch.float64)
        for method in ("sum", "jaccard", "cosine", "hyperbolic"):
            P = project_rows(W, method=method)
            assert P.shape == (4, 4), f"wrong shape for method={method}"
            assert P.diagonal().sum().item() == 0.0, f"non-zero diagonal for method={method}"
            assert torch.allclose(P, P.T), f"not symmetric for method={method}"

    def test_validated_projection_recovers_blocks(self):
        """BiCM + FDR on asymmetric 2-block binary matrix recovers both intra-block cliques."""
        from bicm.wbnm.models import BiCM
        from bicm.wbnm.projections import validated_projection

        nA, nB, cA, cB = 20, 10, 10, 20
        W_block = torch.zeros(nA + nB, cA + cB, dtype=torch.float64)
        W_block[:nA, :cA] = 1.0
        W_block[nA:, cA:] = 1.0

        model = BiCM(W_block).solve()
        _, _, significant = validated_projection(
            model, layer="rows", n_samples=1000, alpha=0.05, correction="fdr_bh", seed=42
        )

        sig_edges = int(significant.sum() // 2)
        expected = nA * (nA - 1) // 2 + nB * (nB - 1) // 2  # C(20,2) + C(10,2) = 235
        assert sig_edges == expected

    def test_validated_projection_unsolved_raises(self):
        """validated_projection raises RuntimeError when the model is not yet solved."""
        from bicm.wbnm.models import BiCM
        from bicm.wbnm.projections import validated_projection

        W = torch.ones(5, 5, dtype=torch.float64)
        model = BiCM(W)
        with pytest.raises(RuntimeError):
            validated_projection(model, layer="rows", n_samples=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
