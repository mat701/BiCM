"""
Pytest configuration for the tests of bicm.wbnm (skipped if PyTorch is not installed).

Forces CPU execution for all tests so that tensors created in fixtures
(always CPU) and tensors produced by models (auto-detect device) stay
on the same device. GPU-specific correctness is identical to CPU by
construction (PyTorch guarantees numerical equivalence).
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = ["test_*.py"]
else:
    import torch


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Patch torch.cuda.is_available to return False for every test."""
    if importlib.util.find_spec("torch") is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
