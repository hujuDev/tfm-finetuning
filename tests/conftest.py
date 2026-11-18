"""Shared test setup."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _no_tabpfn_weight_downloads(monkeypatch):
    """Keep tests from fetching TabPFN weights into the repository's models/ folder."""
    try:
        tabpfn_model = importlib.import_module("tfm_finetuning.finetuning.models.tabpfn.model")
    except ImportError:  # the lightweight CI environment has no Hugging Face client
        return
    monkeypatch.setattr(tabpfn_model, "ensure_checkpoint", lambda _path: None)
