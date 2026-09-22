"""Tests for foundation model finetune: runner."""

from __future__ import annotations

from src.finetuning import config as config_runner
from src.finetuning import resource_policy as policy_runner
from src.finetuning.models.base import CommonFineTuneConfig


def test_resolve_common_finetune_config_uses_foundation_model_settings() -> None:
    cfg = config_runner.resolve_common_finetune_config(
        {
            "foundation_model": {
                "model_id": "tabicl_v2",
                "epochs": 5,
                "early_stopping": False,
                "patience": 17,
            },
        }
    )

    assert cfg.epochs == 5
    assert cfg.early_stopping is False
    assert cfg.patience == 17

def test_resource_fallback_policy_can_enable_oom_retry_for_any_model() -> None:
    common_config = CommonFineTuneConfig(context_size=10000)

    assert policy_runner._resolve_oom_retry_context_sizes(
        {"foundation_model": {"resource_fallback_policy": "oom_context_retry"}},
        model_id="mitra",
        common_config=common_config,
    ) == (10000, 5000, 2500, 1000, 500)
