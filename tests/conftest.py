"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_tabpfn_weight_downloads(monkeypatch):
    """Keep tests from fetching TabPFN weights into the repository's models/ folder."""
    monkeypatch.setattr("tfm_finetuning.finetuning.models.tabpfn.model.ensure_checkpoint", lambda _path: None)
