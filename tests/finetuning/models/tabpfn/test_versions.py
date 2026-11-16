"""TabPFN version catalog: the upstream model version pin and the pinned inference subsampling."""

from __future__ import annotations

import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import tabpfn  # noqa: F401  (import the real package before tests fake its constants module)

from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.models.registry import get_model_spec
from tfm_finetuning.finetuning.models.tabpfn.model import TabPFNAdapter
from tfm_finetuning.finetuning.models.tabpfn.vendor import TabPFNBindings
from tfm_finetuning.finetuning.models.tabpfn.versions import TABPFN_VERSION_BY_ID


class _FakeModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"


class _FakeWrapper:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _adapter(model_id: str, tmp_path: Path) -> TabPFNAdapter:
    spec = replace(get_model_spec(model_id, repo_root=Path.cwd()), source_path=tmp_path)
    bindings = TabPFNBindings(
        classifier=_FakeWrapper,
        regressor=_FakeWrapper,
        finetuned_classifier=_FakeWrapper,
        finetuned_regressor=_FakeWrapper,
        clone_model_for_evaluation=lambda *args, **kwargs: None,
        save_model=lambda *args, **kwargs: None,
    )
    return TabPFNAdapter(spec, bindings=bindings)


@pytest.fixture
def fake_model_version(monkeypatch):
    module = ModuleType("tabpfn.constants")
    module.ModelVersion = _FakeModelVersion
    monkeypatch.setitem(sys.modules, "tabpfn.constants", module)
    return _FakeModelVersion


def test_v2_models_are_pinned_to_their_own_upstream_version() -> None:
    assert TABPFN_VERSION_BY_ID["tabpfn_v2"].upstream_model_version == "v2"
    assert TABPFN_VERSION_BY_ID["tabpfn_v2_5"].upstream_model_version == "v2.5"


@pytest.mark.parametrize(("model_id", "value"), [("tabpfn_v2", "v2"), ("tabpfn_v2_5", "v2.5")])
def test_v2_finetuners_are_pinned_to_their_own_version(model_id, value, tmp_path, fake_model_version) -> None:
    adapter = _adapter(model_id, tmp_path)

    finetuner = adapter.resolve_finetune_estimator("regression", CommonFineTuneConfig(), seed=0, device="cpu")

    assert finetuner.kwargs["model_version"] is fake_model_version(value)


@pytest.mark.parametrize("model_id", ["tabpfn_v2", "tabpfn_v2_5"])
def test_inference_subsampling_is_pinned_for_every_version(model_id, tmp_path) -> None:
    params = _adapter(model_id, tmp_path).resolved_model_params(CommonFineTuneConfig())

    assert params["n_inference_subsample_samples"] == 50_000
    assert params["n_finetune_ctx_plus_query_samples"] == 10_000
    assert params["use_fixed_preprocessing_seed"] is False
