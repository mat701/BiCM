"""Tests for projections module."""

import torch
import pytest

from bicm.wbnm.projections import project_rows, project_cols
from bicm.wbnm.data import load_example


class TestProjections:
    """Tests for projection functions."""

    @pytest.fixture
    def toy_matrix(self):
        W, _ = load_example("toy")
        return W

    def test_project_rows_shape(self, toy_matrix):
        """Test output shape of row projection."""
        P = project_rows(toy_matrix)
        assert P.shape == (5, 5)

    def test_project_cols_shape(self, toy_matrix):
        """Test output shape of column projection."""
        P = project_cols(toy_matrix)
        assert P.shape == (4, 4)

    def test_projection_symmetric(self, toy_matrix):
        """Test that projections are symmetric."""
        P_rows = project_rows(toy_matrix)
        P_cols = project_cols(toy_matrix)

        torch.testing.assert_close(P_rows, P_rows.T)
        torch.testing.assert_close(P_cols, P_cols.T)

    def test_projection_no_self_loops(self, toy_matrix):
        """Test that diagonal is zero."""
        P_rows = project_rows(toy_matrix)
        P_cols = project_cols(toy_matrix)

        assert torch.equal(torch.diag(P_rows), torch.zeros(5, dtype=torch.float64))
        assert torch.equal(torch.diag(P_cols), torch.zeros(4, dtype=torch.float64))

    def test_sum_method(self, toy_matrix):
        """Test sum projection method."""
        P = project_rows(toy_matrix, method="sum")

        # Manual calculation for verification
        expected = toy_matrix @ toy_matrix.T
        expected.fill_diagonal_(0)

        torch.testing.assert_close(P, expected)

    def test_jaccard_method(self, toy_matrix):
        """Test Jaccard projection method."""
        P = project_rows(toy_matrix, method="jaccard")

        # Values should be in [0, 1]
        assert P.min().item() >= 0
        assert P.max().item() <= 1

    def test_cosine_method(self, toy_matrix):
        """Test cosine projection method."""
        P = project_rows(toy_matrix, method="cosine")

        # Cosine similarity in [-1, 1], but with positive weights should be [0, 1]
        assert P.min().item() >= -1e-10
        assert P.max().item() <= 1 + 1e-10

    def test_hyperbolic_method(self, toy_matrix):
        """Test hyperbolic projection method."""
        P = project_rows(toy_matrix, method="hyperbolic")

        # tanh values in (-1, 1)
        assert P.min().item() >= -1
        assert P.max().item() <= 1

    def test_invalid_method(self, toy_matrix):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            project_rows(toy_matrix, method="invalid")
