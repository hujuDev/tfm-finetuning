from __future__ import annotations

import pytest

from tfm_finetuning.plots.transfer.balanced_transfer_data import (
    BUDGET_ROWS,
    DATASET_ORDER,
    DATASET_SOURCE,
    DATASET_TASK,
    BalancedBudgetSetting,
    BalancedSourceOnlyResult,
    SourceInitializationAdvantage,
    pair_source_advantages,
)
from tfm_finetuning.plots.transfer.transfer_data import MODEL_ORDER


def test_source_initialization_advantage_uses_direct_paired_metric_gain() -> None:
    common = {
        "dataset": "covertype",
        "task_family": "multiclass",
        "model": "tabpfnv25",
        "seed": 0,
        "target_rows": 64,
        "source": "satimage",
        "metric": "log_loss",
        "icl_value": 1.2,
    }
    settings = tuple(
        BalancedBudgetSetting(
            **common,
            setting=setting,
            value=value,
            gain_over_icl=0.0,
        )
        for setting, value in (
            ("target_100", 1.0),
            ("target_200", 0.8),
            ("source_initialized", 0.5),
        )
    )

    row = pair_source_advantages(settings)[0]

    assert row.target_only_value == pytest.approx(1.0)
    assert row.source_initialized_value == pytest.approx(0.5)
    assert row.advantage == pytest.approx(0.69314718056)


def _complete_advantages() -> tuple[SourceInitializationAdvantage, ...]:
    return tuple(
        SourceInitializationAdvantage(
            dataset=dataset,
            task_family=DATASET_TASK[dataset],
            model=model,
            seed=seed,
            target_rows=target_rows,
            source=DATASET_SOURCE[dataset],
            metric=(
                "roc_auc"
                if DATASET_TASK[dataset] == "binary"
                else "rmse"
                if DATASET_TASK[dataset] == "regression"
                else "log_loss"
            ),
            target_only_value=1.0,
            source_initialized_value=0.9,
            advantage=(model_index - 1) * 0.01 + budget_index * 0.002 + seed * 0.0002,
        )
        for dataset in DATASET_ORDER
        for model_index, model in enumerate(MODEL_ORDER)
        for seed in range(3)
        for budget_index, target_rows in enumerate(BUDGET_ROWS)
    )


def _complete_source_only() -> tuple[BalancedSourceOnlyResult, ...]:
    return tuple(
        BalancedSourceOnlyResult(
            dataset=dataset,
            task_family=DATASET_TASK[dataset],
            model=model,
            seed=seed,
            source=source,
            metric=(
                "roc_auc"
                if DATASET_TASK[dataset] == "binary"
                else "rmse"
                if DATASET_TASK[dataset] == "regression"
                else "log_loss"
            ),
            icl_value=1.0,
            source_value=0.9,
            source_gain=(model_index - 1) * 0.01 + source_index * 0.004 + seed * 0.0002,
            target_only_value=0.85,
            target_only_gain=0.03 + model_index * 0.005 + seed * 0.0002,
        )
        for dataset in DATASET_ORDER
        for model_index, model in enumerate(MODEL_ORDER)
        for seed in range(3)
        for source_index, source in enumerate(
            ("houses",) if DATASET_TASK[dataset] == "regression" else ("satimage", "bank_heloc")
        )
    )
