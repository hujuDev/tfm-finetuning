"""Mitra-v2's binned regression head: loss switch, mean decode and the patched AutoGluon trainer."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from autogluon.tabular.models.mitra._internal.config.enums import LossName, Task
from autogluon.tabular.models.mitra._internal.core.trainer_finetune import TrainerFinetune

from tfm_finetuning.finetuning.models.mitra.adapter import MitraAdapter
from tfm_finetuning.finetuning.models.mitra.configured import _binned_mitra_class
from tfm_finetuning.finetuning.models.mitra.regression_bins import (
    binned_regression_config,
    install_mean_bin_decode,
    mean_bin_decode,
)
from tfm_finetuning.finetuning.models.mitra.vendor import load_mitra_bindings
from tfm_finetuning.finetuning.models.registry import get_model_spec


def _trainer(n_bins: int) -> SimpleNamespace:
    bins = torch.linspace(-0.5, 1.5, n_bins + 1)
    return SimpleNamespace(bins=bins, bin_width=bins[1] - bins[0])


def test_mean_decode_weights_bin_centers_by_softmax() -> None:
    trainer = _trainer(4)  # centers -0.25, 0.25, 0.75, 1.25
    certain = torch.tensor([[-1e4, 1e4, -1e4, -1e4]])
    split = torch.log(torch.tensor([[0.0, 0.5, 0.5, 0.0]]) + 1e-30)

    assert mean_bin_decode(trainer, certain).item() == pytest.approx(0.25)
    assert mean_bin_decode(trainer, split).item() == pytest.approx(0.5)
    # Non-finite logits must not turn into NaN predictions.
    assert torch.isfinite(mean_bin_decode(trainer, torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))).all()


def test_binned_config_switches_only_regression_with_bins() -> None:
    def config(task: Task) -> SimpleNamespace:
        return SimpleNamespace(task=task, hyperparams={"regression_loss": LossName.MSE, "dim_output": 1})

    regression = binned_regression_config(config(Task.REGRESSION), 1000)
    assert regression.hyperparams == {"regression_loss": LossName.CROSS_ENTROPY, "dim_output": 1000}
    # Mitra-v1 (a single output) and classification keep their configs.
    assert binned_regression_config(config(Task.REGRESSION), 1).hyperparams["regression_loss"] == LossName.MSE
    assert binned_regression_config(config(Task.CLASSIFICATION), 1000).hyperparams["dim_output"] == 1


def test_installed_trainer_decodes_bins_by_mean_and_is_idempotent() -> None:
    install_mean_bin_decode()
    install_mean_bin_decode()

    for name in ("train", "evaluate", "predict"):
        assert "mean_bin_decode" in getattr(TrainerFinetune, name).__code__.co_names


def test_mitra_v2_adapter_uses_binned_regression_only_for_mitra_v2() -> None:
    load_mitra_bindings()
    v1 = MitraAdapter(get_model_spec("mitra"))
    v2 = MitraAdapter(get_model_spec("mitra_v2"))

    assert (v1.model_id, v1.regression_bins) == ("mitra", 1)
    assert (v2.model_id, v2.regression_bins) == ("mitra_v2", 1000)
    zero_shot = v2.resolve_zero_shot_estimator("regression", seed=0, device="cpu", n_estimators_inference=1)
    assert zero_shot.hf_model == "autogluon/mitra-regressor-2"
    assert zero_shot.tfm_regression_bins == 1000
    cfg, _ = zero_shot._create_config(Task.REGRESSION, 1, None)
    assert cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY and cfg.hyperparams["dim_output"] == 1000
    # The dynamically composed estimator class must stay picklable.
    assert type(pickle.loads(pickle.dumps(zero_shot))) is type(zero_shot)
    assert type(zero_shot) is _binned_mitra_class(load_mitra_bindings().regressor)
