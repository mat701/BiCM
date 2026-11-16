"""Tests for null models."""

import torch
import pytest

from bicm.wbnm.models import BiECM, BiWCM, BiPECM, BiCReMA
from bicm.wbnm.data import load_example


class TestBiECM:
    """Tests for BiECM model."""

    @pytest.fixture
    def toy_matrix(self):
        """Small test matrix."""
        W, _ = load_example("toy")
        return W

    def test_init(self, toy_matrix):
        """Test model initialization."""
        model = BiECM(toy_matrix)

        assert model.n_rows == 5
        assert model.n_cols == 4
        assert not model.is_solved

    def test_degree_sequences(self, toy_matrix):
        """Test degree sequence computation."""
        model = BiECM(toy_matrix)

        k_rows = model.degree_sequence_rows
        k_cols = model.degree_sequence_cols

        assert len(k_rows) == 5
        assert len(k_cols) == 4
        assert k_rows.sum().item() == k_cols.sum().item()  # Total edges

    def test_strength_sequences(self, toy_matrix):
        """Test strength sequence computation."""
        model = BiECM(toy_matrix)

        s_rows = model.strength_sequence_rows
        s_cols = model.strength_sequence_cols

        assert len(s_rows) == 5
        assert len(s_cols) == 4
        assert torch.allclose(s_rows.sum(), s_cols.sum())

    def test_repr(self, toy_matrix):
        """Test string representation."""
        model = BiECM(toy_matrix)

        repr_str = repr(model)
        assert "BiECM" in repr_str
        assert "n_rows=5" in repr_str
        assert "not solved" in repr_str

    def test_check_solved_raises(self, toy_matrix):
        """Test that operations on unsolved model raise error."""
        model = BiECM(toy_matrix)

        with pytest.raises(RuntimeError, match="must be solved"):
            model.sample()


class TestBiWCM:
    """Tests for BiWCM model."""

    def test_init(self):
        W, _ = load_example("toy")
        model = BiWCM(W)
        assert model.n_rows == 5
        assert model.n_cols == 4


class TestBiPECM:
    """Tests for BiPECM model."""

    def test_init(self):
        W, _ = load_example("toy")
        model = BiPECM(W)
        assert model.n_rows == 5
        assert model.n_cols == 4

    def test_topology_stored(self):
        W, _ = load_example("toy")
        model = BiPECM(W)

        # Check topology is binary version
        assert torch.equal(
            model.topology,
            (W > 0).to(torch.float64)
        )


class TestBiCReMA:
    """Tests for BiCReMA model."""

    @pytest.fixture
    def toy_matrix(self):
        W, _ = load_example("toy")
        return W

    @pytest.fixture
    def solved_model(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        model.solve(tol=1e-8)
        return model

    def test_init(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        assert model.n_rows == 5
        assert model.n_cols == 4
        assert not model.is_solved
        assert model.x_rows is None
        assert model.H_rows is None

    def test_repr(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        assert "BiCReMA" in repr(model)
        assert "not solved" in repr(model)

    def test_check_solved_raises(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        with pytest.raises(RuntimeError, match="must be solved"):
            model.sample()

    def test_solve_sets_params(self, solved_model):
        assert solved_model.is_solved
        assert solved_model.x_rows is not None
        assert solved_model.x_cols is not None
        assert solved_model.H_rows is not None
        assert solved_model.H_cols is not None
        assert len(solved_model.x_rows) == solved_model.n_rows
        assert len(solved_model.x_cols) == solved_model.n_cols
        assert len(solved_model.H_rows) == solved_model.n_rows
        assert len(solved_model.H_cols) == solved_model.n_cols

    def test_params_positive(self, solved_model):
        assert (solved_model.x_rows > 0).all()
        assert (solved_model.x_cols > 0).all()
        assert (solved_model.H_rows > 0).all()
        assert (solved_model.H_cols > 0).all()

    def test_degree_sequence_reproduced(self, toy_matrix, solved_model):
        """BiCM stage must reproduce the degree sequence."""
        tol = 1e-6
        p = solved_model.connection_probability()
        k_rows = solved_model.degree_sequence_rows.to(torch.float64)
        k_cols = solved_model.degree_sequence_cols.to(torch.float64)

        err_r = (torch.abs(p.sum(dim=1) - k_rows) / k_rows.clamp(min=1e-10)).max()
        err_c = (torch.abs(p.sum(dim=0) - k_cols) / k_cols.clamp(min=1e-10)).max()
        assert err_r.item() < tol, f"Row degree error {err_r:.2e} exceeds {tol}"
        assert err_c.item() < tol, f"Col degree error {err_c:.2e} exceeds {tol}"

    def test_strength_sequence_reproduced(self, toy_matrix, solved_model):
        """BiWCM stage must reproduce the strength sequence."""
        tol = 1e-6
        E = solved_model.expected_matrix()
        s_rows = solved_model.strength_sequence_rows.to(torch.float64)
        s_cols = solved_model.strength_sequence_cols.to(torch.float64)

        err_r = (torch.abs(E.sum(dim=1) - s_rows) / s_rows.clamp(min=1e-10)).max()
        err_c = (torch.abs(E.sum(dim=0) - s_cols) / s_cols.clamp(min=1e-10)).max()
        assert err_r.item() < tol, f"Row strength error {err_r:.2e} exceeds {tol}"
        assert err_c.item() < tol, f"Col strength error {err_c:.2e} exceeds {tol}"

    def test_connection_probability_range(self, solved_model):
        p = solved_model.connection_probability()
        assert p.shape == (solved_model.n_rows, solved_model.n_cols)
        assert (p >= 0).all() and (p <= 1).all()

    def test_expected_matrix_positive(self, solved_model):
        E = solved_model.expected_matrix()
        assert E.shape == (solved_model.n_rows, solved_model.n_cols)
        assert (E > 0).all()

    def test_sample_shape(self, solved_model):
        samples = solved_model.sample(n_samples=10, seed=0)
        assert samples.shape == (10, solved_model.n_rows, solved_model.n_cols)

    def test_sample_non_negative(self, solved_model):
        samples = solved_model.sample(n_samples=5, seed=0)
        assert (samples >= 0).all()

    def test_sample_reproducible(self, solved_model):
        s1 = solved_model.sample(n_samples=3, seed=42)
        s2 = solved_model.sample(n_samples=3, seed=42)
        assert torch.allclose(s1, s2)

    def test_pvalues_shape_and_range(self, toy_matrix, solved_model):
        pv = solved_model.pvalues()
        assert pv.shape == toy_matrix.shape
        assert (pv >= 0).all() and (pv <= 1).all()

    def test_pvalues_zero_entries_are_one(self, toy_matrix, solved_model):
        """Zero weights must produce p-value = 1."""
        pv = solved_model.pvalues()
        zero_mask = toy_matrix == 0
        assert torch.allclose(pv[zero_mask], torch.ones(zero_mask.sum(), dtype=torch.float64))

    def test_n_parameters(self, solved_model):
        expected = 2 * (solved_model.n_rows + solved_model.n_cols)
        assert solved_model.n_parameters() == expected

    def test_log_likelihood_finite(self, solved_model):
        ll = solved_model.log_likelihood()
        assert torch.isfinite(torch.tensor(ll))

    def test_aic_greater_than_twice_neg_ll(self, solved_model):
        """AIC = -2*L + 2*k  >  -2*L  always (k > 0)."""
        assert solved_model.aic() > -2 * solved_model.log_likelihood()

    def test_custom_x0(self, toy_matrix):
        """Solve with an explicit initial guess (x0)."""
        n_r, n_c = toy_matrix.shape
        x0 = torch.ones(2 * (n_r + n_c), dtype=torch.float64)
        model = BiCReMA(toy_matrix)
        model.solve(x0=x0, tol=1e-6)
        assert model.is_solved

    def test_beta_schedule_warmup(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        model.solve(tol=1e-6, beta_schedule="warmup", warmup_steps=50)
        assert model.is_solved

    def test_beta_schedule_linear(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        model.solve(tol=1e-6, beta_schedule="linear", warmup_steps=50)
        assert model.is_solved

    def test_invalid_method_raises(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        with pytest.raises(ValueError, match="Unknown method"):
            model.solve(method="gradient-descent")

    # ---- quasi-newton ----

    def test_quasi_newton_converges(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        model.solve(method="quasi-newton", tol=1e-6)
        assert model.is_solved

    def test_quasi_newton_degree_reproduced(self, toy_matrix):
        tol = 1e-5
        model = BiCReMA(toy_matrix)
        model.solve(method="quasi-newton", tol=1e-6)
        p = model.connection_probability()
        k_rows = model.degree_sequence_rows.to(torch.float64)
        k_cols = model.degree_sequence_cols.to(torch.float64)
        assert (torch.abs(p.sum(1) - k_rows) / k_rows.clamp(min=1e-10)).max() < tol
        assert (torch.abs(p.sum(0) - k_cols) / k_cols.clamp(min=1e-10)).max() < tol

    def test_quasi_newton_strength_reproduced(self, toy_matrix):
        tol = 1e-5
        model = BiCReMA(toy_matrix)
        model.solve(method="quasi-newton", tol=1e-6)
        E = model.expected_matrix()
        s_rows = model.strength_sequence_rows.to(torch.float64)
        s_cols = model.strength_sequence_cols.to(torch.float64)
        assert (torch.abs(E.sum(1) - s_rows) / s_rows.clamp(min=1e-10)).max() < tol
        assert (torch.abs(E.sum(0) - s_cols) / s_cols.clamp(min=1e-10)).max() < tol

    # ---- coordinate ----

    def test_coordinate_converges(self, toy_matrix):
        model = BiCReMA(toy_matrix)
        model.solve(method="coordinate", tol=1e-8)
        assert model.is_solved

    def test_coordinate_degree_reproduced(self, toy_matrix):
        tol = 1e-6
        model = BiCReMA(toy_matrix)
        model.solve(method="coordinate", tol=1e-8)
        p = model.connection_probability()
        k_rows = model.degree_sequence_rows.to(torch.float64)
        k_cols = model.degree_sequence_cols.to(torch.float64)
        assert (torch.abs(p.sum(1) - k_rows) / k_rows.clamp(min=1e-10)).max() < tol
        assert (torch.abs(p.sum(0) - k_cols) / k_cols.clamp(min=1e-10)).max() < tol

    def test_coordinate_strength_reproduced(self, toy_matrix):
        tol = 1e-6
        model = BiCReMA(toy_matrix)
        model.solve(method="coordinate", tol=1e-8)
        E = model.expected_matrix()
        s_rows = model.strength_sequence_rows.to(torch.float64)
        s_cols = model.strength_sequence_cols.to(torch.float64)
        assert (torch.abs(E.sum(1) - s_rows) / s_rows.clamp(min=1e-10)).max() < tol
        assert (torch.abs(E.sum(0) - s_cols) / s_cols.clamp(min=1e-10)).max() < tol

    # ---- all methods agree ----

    def test_all_methods_agree(self, toy_matrix):
        """All three methods should converge to the same parameters."""
        tol_solve = 1e-8
        tol_agree = 1e-4
        models = {}
        for method in ("fixed-point", "quasi-newton", "coordinate"):
            m = BiCReMA(toy_matrix)
            m.solve(method=method, tol=tol_solve)
            models[method] = m

        ref_p = models["fixed-point"].connection_probability()
        ref_E = models["fixed-point"].expected_matrix()
        for method in ("quasi-newton", "coordinate"):
            p = models[method].connection_probability()
            E = models[method].expected_matrix()
            assert torch.allclose(p, ref_p, atol=tol_agree), \
                f"{method} p differs from fixed-point"
            assert torch.allclose(E, ref_E, atol=tol_agree), \
                f"{method} E differs from fixed-point"
