"""The TabPFN-3 study must reproduce the overview_baseline splits for the new models."""

from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

import pytest
import yaml

from tfm_finetuning.experiments.configuration import expand_runs
from tfm_finetuning.finetuning.models.tabpfn.versions import TABPFN_VERSION_BY_ID

ROOT = Path(__file__).resolve().parents[2]
STUDY = "preprint_tabpfn_v3"
MODEL_KEYS = {"tabpfnv3": "tabpfn_v3", "tabpfnv35": "tabpfn_v3_5", "tabpfnv35fast": "tabpfn_v3_5_fast"}
EXCLUDED_DATASETS = {"airlines", "miniboone"}


def _runs(study: str) -> list[dict]:
    recipe = yaml.safe_load((ROOT / f"configs/experiments/{study}.yaml").read_text())
    return expand_runs(recipe, repo_root=ROOT)


def _split_units(runs: list[dict], model: str) -> set[tuple]:
    return {
        (run["dataset"], run["preprocessing"], run["seed"], run["fold"], run["repeat"], run["sample"])
        for run in runs
        if run["model"] == model
    }


def test_each_new_model_runs_the_same_splits_as_the_baseline_analysis() -> None:
    runs = _runs(STUDY)
    baseline = [run for run in _runs("overview_baseline") if run["dataset"].lower() not in EXCLUDED_DATASETS]
    expected = _split_units(baseline, "tabpfnv25") & {
        unit for unit in _split_units(baseline, "tabpfnv25") if unit[1] == "shared"
    }

    assert {run["model"] for run in runs} == set(MODEL_KEYS)
    assert {run["preprocessing"] for run in runs} == {"shared"}
    assert len({run["dataset"] for run in runs}) == 55
    for model in MODEL_KEYS:
        assert _split_units(runs, model) == expected


def test_run_ids_are_unique_and_balanced_across_models() -> None:
    runs = _runs(STUDY)

    assert len({run["run_id"] for run in runs}) == len(runs)
    assert len(set(Counter(run["model"] for run in runs).values())) == 1


def test_baseline_hyperparameters_are_kept_for_the_new_models() -> None:
    baseline = yaml.safe_load((ROOT / "configs/experiments/overview_baseline.yaml").read_text())["groups"][0][
        "settings"
    ]
    study = yaml.safe_load((ROOT / f"configs/experiments/{STUDY}.yaml").read_text())
    for group in study["groups"]:
        settings = group["settings"]
        for key in (
            "finetune.epochs",
            "finetune.early_stopping",
            "finetune.final_eval_checkpoint",
            "finetune.validation_split_ratio",
            "finetune.n_estimators_finetune",
            "finetune.n_estimators_final_inference",
            "finetune.grad_clip",
            "finetune.warmup_proportion",
            "finetune.train_max_rows",
            "finetune.inference_context_max_rows",
            "finetune.oom_retry_context_sizes",
        ):
            assert settings[key] == baseline[key], key


@pytest.mark.parametrize(("model_key", "model_id"), MODEL_KEYS.items())
def test_model_entries_match_the_version_catalog_and_toml_defaults(model_key: str, model_id: str) -> None:
    version = TABPFN_VERSION_BY_ID[model_id]
    study = yaml.safe_load((ROOT / f"configs/experiments/{STUDY}.yaml").read_text())
    for group in study["groups"]:
        entry = next(model for model in group["models"] if model["name"] == model_key)
        assert entry["settings"]["finetune.model_id"] == model_id
        assert entry["settings"]["finetune.classifier_checkpoint"] == version.classifier_checkpoint
        assert entry["settings"]["finetune.regressor_checkpoint"] == version.regressor_checkpoint
        defaults = tomllib.loads((ROOT / entry["config"]).read_text())["finetune"]
        assert defaults["model_id"] == model_id
        assert defaults["classifier_checkpoint"] == version.classifier_checkpoint
        assert defaults["regressor_checkpoint"] == version.regressor_checkpoint
