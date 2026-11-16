"""TabPFN version catalog: v3 / v3.5 checkpoints and the upstream model version pin."""

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
from tfm_finetuning.finetuning.models.tabpfn.versions import (
    TABPFN_VERSION_BY_ID,
    get_tabpfn_version,
)

NEW_MODEL_IDS = ("tabpfn_v3", "tabpfn_v3_5", "tabpfn_v3_5_fast")


class _FakeModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"
    V3 = "v3"
    V3_5 = "v3.5"
    V3_5_FAST = "v3.5-fast"


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


def test_v3_checkpoints_are_per_task_and_v3_5_checkpoints_are_one_multitask_file() -> None:
    v3 = get_tabpfn_version("tabpfn_v3")
    assert v3.classifier_checkpoint != v3.regressor_checkpoint
    for model_id in ("tabpfn_v3_5", "tabpfn_v3_5_fast"):
        version = get_tabpfn_version(model_id)
        assert version.classifier_checkpoint == version.regressor_checkpoint
        assert version.classifier_checkpoint.endswith(".safetensors")
    assert get_tabpfn_version("tabpfn_v3_5").classifier_checkpoint != (
        get_tabpfn_version("tabpfn_v3_5_fast").classifier_checkpoint
    )


def test_upstream_model_versions_match_the_tabpfn_enum_values() -> None:
    assert {model_id: TABPFN_VERSION_BY_ID[model_id].upstream_model_version for model_id in NEW_MODEL_IDS} == {
        "tabpfn_v3": "v3",
        "tabpfn_v3_5": "v3.5",
        "tabpfn_v3_5_fast": "v3.5-fast",
    }


def test_registry_spec_carries_checkpoints() -> None:
    spec = get_model_spec("tabpfn_v3_5", repo_root=Path.cwd())

    assert spec.display_name == "TabPFN-3.5"
    assert spec.classifier_checkpoint == spec.regressor_checkpoint == "models/tabpfn-v3.5-20260909.safetensors"
    assert spec.adapter_cls is TabPFNAdapter


@pytest.mark.parametrize("model_id", NEW_MODEL_IDS)
def test_finetuner_is_pinned_to_the_models_upstream_version(model_id, tmp_path, fake_model_version) -> None:
    adapter = _adapter(model_id, tmp_path)

    classifier = adapter.resolve_finetune_estimator("binary", CommonFineTuneConfig(), seed=0, device="cpu")
    regressor = adapter.resolve_finetune_estimator("regression", CommonFineTuneConfig(), seed=0, device="cpu")

    expected = fake_model_version(TABPFN_VERSION_BY_ID[model_id].upstream_model_version)
    assert classifier.kwargs["model_version"] is expected
    assert regressor.kwargs["model_version"] is expected
    checkpoint = TABPFN_VERSION_BY_ID[model_id]
    assert classifier.kwargs["extra_classifier_kwargs"]["model_path"].endswith(
        Path(checkpoint.classifier_checkpoint).name
    )
    assert regressor.kwargs["extra_regressor_kwargs"]["model_path"].endswith(
        Path(checkpoint.regressor_checkpoint).name
    )


@pytest.mark.parametrize(("model_id", "value"), [("tabpfn_v2", "v2"), ("tabpfn_v2_5", "v2.5")])
def test_v2_finetuners_are_pinned_to_their_own_version(model_id, value, tmp_path, fake_model_version) -> None:
    adapter = _adapter(model_id, tmp_path)

    finetuner = adapter.resolve_finetune_estimator("regression", CommonFineTuneConfig(), seed=0, device="cpu")

    assert finetuner.kwargs["model_version"] is fake_model_version(value)


@pytest.mark.parametrize("model_id", ["tabpfn_v2_5", *NEW_MODEL_IDS])
def test_inference_subsampling_is_pinned_for_every_version(model_id, tmp_path) -> None:
    params = _adapter(model_id, tmp_path).resolved_model_params(CommonFineTuneConfig())

    assert params["n_inference_subsample_samples"] == 50_000
    assert params["n_finetune_ctx_plus_query_samples"] == 10_000
    assert params["use_fixed_preprocessing_seed"] is False


@pytest.mark.parametrize("model_id", NEW_MODEL_IDS)
def test_new_models_use_the_tabpfn_fallbacks(model_id) -> None:
    from tfm_finetuning.finetuning.training import resource_policy

    assert resource_policy._resolve_estimator_fallback_policy({}, model_id=model_id) == "estimator_retry"
