from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from src.finetuning import fit as fit_runner
from src.finetuning.models.base import CommonFineTuneConfig, InMemoryFinetuneLogger
from src.finetuning.models.registry import (
    FoundationModelRegistryError,
    get_foundation_model_spec,
    iter_foundation_model_specs,
)
from src.finetuning.models.tabpfn_versions import (
    TABPFN_VERSIONS,
    VENDOR_ONLY_TABPFN_ARCHITECTURES,
)


def test_foundation_model_dependencies_are_packaged_for_cluster_image() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert "tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena" in dependencies
    assert "tabicl[finetune] @ git+https://github.com/soda-inria/tabicl.git" in dependencies
    assert "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git" in dependencies
    assert not Path(".gitmodules").exists()


def test_section1_registry_reports_pinned_sources() -> None:
    specs = {spec.model_id: spec for spec in iter_foundation_model_specs(repo_root=Path.cwd())}

    assert set(specs) == {"tabpfn_v2", "tabpfn_v2_5", "tabicl_v2", "mitra"}
    assert specs["tabpfn_v2"].version_payload["git_commit"]
    assert specs["tabpfn_v2"].version_payload["source_resolved_path"]
    assert specs["tabpfn_v2"].classifier_checkpoint == "models/tabpfn-v2-classifier.ckpt"
    assert specs["tabpfn_v2"].regressor_checkpoint == "models/tabpfn-v2-regressor.ckpt"
    assert specs["tabpfn_v2_5"].version_payload["git_commit"]
    assert specs["tabpfn_v2_5"].version_payload["source_resolved_path"]
    assert specs["tabicl_v2"].version_payload["git_commit"]
    assert specs["tabicl_v2"].version_payload["source_resolved_path"]
    assert specs["tabicl_v2"].version_payload["package_version"] == "2.1.1"
    assert specs["tabicl_v2"].classifier_checkpoint == "tabicl-classifier-v2-20260212.ckpt"
    assert specs["tabicl_v2"].regressor_checkpoint == "tabicl-regressor-v2-20260212.ckpt"
    assert specs["mitra"].classifier_checkpoint == "autogluon/mitra-classifier"
    assert specs["mitra"].regressor_checkpoint == "autogluon/mitra-regressor"




def test_tabpfn_version_catalog_separates_supported_and_vendor_only_models() -> None:
    assert [version.model_id for version in TABPFN_VERSIONS] == ["tabpfn_v2", "tabpfn_v2_5"]
    assert VENDOR_ONLY_TABPFN_ARCHITECTURES == ("tabpfn_v2_6", "tabpfn_v3")

    for model_id in VENDOR_ONLY_TABPFN_ARCHITECTURES:
        with pytest.raises(FoundationModelRegistryError, match="Unknown"):
            get_foundation_model_spec(model_id, repo_root=Path.cwd())


def test_tabpfn_adapter_maps_common_config_to_official_kwargs() -> None:
    spec = get_foundation_model_spec("tabpfn_v2_5", repo_root=Path.cwd())
    adapter = spec.adapter_cls(spec)
    assert CommonFineTuneConfig().ignore_pretraining_limits is True
    assert adapter._pretraining_limit_kwargs()["ignore_pretraining_limits"] is True

    params = adapter.resolved_model_params(
        CommonFineTuneConfig(learning_rate=3e-5, weight_decay=3e-2, context_size=20_000)
    )

    assert params["learning_rate"] == 3e-5
    assert params["weight_decay"] == 3e-2
    assert params["n_finetune_ctx_plus_query_samples"] == 20_000
    assert params["finetune_ctx_query_split_ratio"] == 0.2
    assert params["n_estimators_finetune"] == 2
    assert params["n_estimators_validation"] == 2
    assert params["n_estimators_final_inference"] == 8
    assert params["use_fixed_preprocessing_seed"] is False


def test_tabpfn_adapter_uses_registry_model_id() -> None:
    spec = get_foundation_model_spec("tabpfn_v2", repo_root=Path.cwd())
    adapter = spec.adapter_cls(spec)

    assert adapter.model_id == "tabpfn_v2"


def test_tabpfn_adapter_resolves_auto_device_for_official_finetuner() -> None:
    spec = get_foundation_model_spec("tabpfn_v2_5", repo_root=Path.cwd())
    adapter = spec.adapter_cls(spec)

    estimator = adapter.resolve_finetune_estimator(
        "binary",
        CommonFineTuneConfig(epochs=1),
        seed=7,
        device=" auto ",
    )

    assert estimator.device in {"cpu", "cuda"}
    assert estimator.device != "auto"


def test_tabicl_adapter_maps_common_config_to_official_kwargs() -> None:
    spec = get_foundation_model_spec("tabicl_v2", repo_root=Path.cwd())
    adapter = spec.adapter_cls(spec)
    params = adapter.resolved_model_params(
        CommonFineTuneConfig(learning_rate=3e-6, weight_decay=1e-3, context_size=2500)
    )

    assert params["learning_rate"] == 3e-6
    assert params["weight_decay"] == 1e-3
    assert params["max_data_size"] == 2500
    assert params["finetune_ctx_query_ratio"] == 0.2
    assert params["n_estimators_finetune"] == 2
    assert params["n_estimators_validation"] == 2
    assert params["n_estimators_inference"] == 8
    assert params["amp"] is True
    assert params["warmup_proportion"] == 0.1


def test_tabicl_epoch_logger_hook_is_captured_by_foundation_model_runner() -> None:
    class DummyTabICLEstimator:
        def _make_experiment_logger(self) -> InMemoryFinetuneLogger:
            raise AssertionError("logger hook should be replaced")

    estimator = DummyTabICLEstimator()
    logger = fit_runner._attach_epoch_logger(estimator)

    assert estimator._make_experiment_logger() is logger
