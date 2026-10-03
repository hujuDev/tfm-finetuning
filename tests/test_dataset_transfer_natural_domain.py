from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.dataset_transfer.natural_domain_context import build_route_matched_evaluation_data
from experiments.dataset_transfer.routes import resolve_transfer_route
from finetuning.datasets.loading import SplitEvaluationData

ROOT = Path(__file__).resolve().parents[1]


def _config(target: str, route: str) -> dict[str, object]:
    return {
        "seed": 0,
        "data": {
            "path": f"data/beyondarena/{target}",
            "split": "fold0_repeat0_sample0",
            "backend": "local_exported_arrays",
        },
        "evaluation": {"backend": "official"},
        "foundation_model": {
            "model_id": "tabpfn_v2_5",
            "classifier_checkpoint": "base.ckpt",
            "test_context_includes_validation_data": True,
            "validation_split_ratio": 0.1,
            "validation_split_seed": 0,
            "use_lr_scheduler": False,
            "early_stopping": False,
        },
        "dataset_transfer": {
            "enabled": True,
            "family": "dataset_transfer_natural_domain",
            "route_id": route,
            "updates_per_dataset": 100,
            "row_budget_per_dataset": 50000,
            "nested_row_seed": 0,
            "context_policy": "target_train_plus_validation",
        },
    }


def _context_eval_config(target: str, checkpoint_route: str) -> dict[str, object]:
    config = _config(target, checkpoint_route)
    transfer = config["dataset_transfer"]
    assert isinstance(transfer, dict)
    transfer.update(
        {
            "family": "dataset_transfer_natural_domain_context_eval",
            "route_id": "route_matched_context_eval",
            "execution_mode": "checkpoint_eval",
            "checkpoint_route": checkpoint_route,
            "checkpoint_origin": target,
            "evaluation_target": target,
            "context_policy": "route_matched_train_plus_validation",
            "dependency_context_policy": "target_train_plus_validation",
            "dependency_updates_per_dataset": 100,
            "dependency_row_budget_per_dataset": 50000,
            "dependency_nested_row_seed": 0,
        }
    )
    return config


def _split(group: str) -> SplitEvaluationData:
    feature = pd.DataFrame({"x": [f"{group}-train", f"{group}-val", f"{group}-test"]})
    return SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=feature.iloc[[0]].reset_index(drop=True),
        train_y=np.array([0]),
        val_context_X=feature.iloc[[0]].reset_index(drop=True),
        val_context_y=np.array([0]),
        val_X=feature.iloc[[1]].reset_index(drop=True),
        val_y=np.array([1]),
        test_context_X=feature.iloc[[0, 1]].reset_index(drop=True),
        test_context_y=np.array([0, 1]),
        test_X=feature.iloc[[2]].reset_index(drop=True),
        test_y=np.array([1]),
        metadata={"test_context_includes_validation_data": True},
        train_ids=np.array(["train"], dtype=object),
        val_ids=np.array(["val"], dtype=object),
        test_context_ids=np.array(["train", "val"], dtype=object),
        test_ids=np.array(["test"], dtype=object),
        test_context_components=np.array(["train", "validation"], dtype=object),
    )


@pytest.mark.parametrize(
    ("target", "route", "datasets", "updates", "boundaries"),
    [
        ("heart_disease_cleveland", "target_control_300_steps", ("heart_disease_cleveland",), 300, (100, 200)),
        (
            "wids_diabetes_hospital_118",
            "source_only_pooled",
            ("wids_diabetes_hospital_19", "wids_diabetes_hospital_188"),
            200,
            (),
        ),
        (
            "covertype_neota",
            "target_inclusive_pooled",
            ("covertype_comanche_peak", "covertype_rawah", "covertype_neota"),
            300,
            (),
        ),
    ],
)
def test_natural_domain_routes_freeze_matched_budgets(
    target: str, route: str, datasets: tuple[str, ...], updates: int, boundaries: tuple[int, ...]
) -> None:
    plan = resolve_transfer_route(_config(target, route), target_dataset=target)
    assert plan is not None
    stage = plan.stages[0]
    assert stage.datasets == datasets
    assert stage.update_budget == updates
    assert stage.view_boundaries == boundaries


def test_route_matched_context_policy_is_checkpoint_evaluation_only() -> None:
    config = _config("heart_disease_cleveland", "source_only_pooled")
    transfer = config["dataset_transfer"]
    assert isinstance(transfer, dict)
    transfer["context_policy"] = "route_matched_train_plus_validation"
    with pytest.raises(ValueError, match="restricted"):
        resolve_transfer_route(config, target_dataset="heart_disease_cleveland")


@pytest.mark.parametrize(
    "checkpoint_route", ["target_control_300_steps", "source_only_pooled", "target_inclusive_pooled"]
)
def test_context_consumer_reconstructs_original_producer_recipe(checkpoint_route: str) -> None:
    target = "heart_disease_cleveland"
    producer = resolve_transfer_route(_config(target, checkpoint_route), target_dataset=target)
    consumer = resolve_transfer_route(_context_eval_config(target, checkpoint_route), target_dataset=target)
    assert producer is not None
    assert consumer is not None
    dependency = consumer.dependency
    assert dependency is not None
    stage = producer.stages[0]
    assert dependency.family == producer.family
    assert dependency.route_id == producer.route_id
    assert dependency.producer_updates_per_dataset == 100
    assert dependency.producer_row_budget_per_dataset == stage.row_budget_per_dataset
    assert dependency.producer_training_recipe == stage.training_recipe
    assert dependency.producer_data_recipe == stage.data_recipe
    assert dependency.producer_context_policy == stage.context_policy


@pytest.mark.parametrize(
    ("checkpoint_route", "expected_groups"),
    [
        ("target_control_300_steps", ("heart_disease_cleveland",)),
        ("source_only_pooled", ("heart_disease_hungary", "heart_disease_va_long_beach")),
        (
            "target_inclusive_pooled",
            ("heart_disease_hungary", "heart_disease_va_long_beach", "heart_disease_cleveland"),
        ),
    ],
)
def test_route_matched_context_uses_fit_groups_and_keeps_c_test(
    checkpoint_route: str, expected_groups: tuple[str, ...]
) -> None:
    target = "heart_disease_cleveland"
    plan = resolve_transfer_route(_context_eval_config(target, checkpoint_route), target_dataset=target)
    assert plan is not None
    assert plan.stages == ()
    assert plan.context_policy == "route_matched_train_plus_validation"
    assert plan.dependency is not None
    assert plan.dependency.family == "dataset_transfer_natural_domain"
    assert plan.dependency.producer_context_policy == "target_train_plus_validation"
    loaded: list[tuple[str, int | None, int | None]] = []

    def loader(
        name: str, *, row_budget: int | None = None, row_budget_policy: str = "cap", row_subset_seed: int | None = None
    ) -> SplitEvaluationData:
        assert row_budget_policy == "cap"
        loaded.append((name, row_budget, row_subset_seed))
        return _split(name)

    data = build_route_matched_evaluation_data(plan, target=target, dataset_loader=loader)
    assert data.test_X["x"].tolist() == [f"{target}-test"]
    assert data.test_context_X["x"].tolist() == [
        value for group in expected_groups for value in (f"{group}-train", f"{group}-val")
    ]
    assert data.metadata["route_matched_context"]["context_groups"] == list(expected_groups)
    assert {budget for _, budget, _ in loaded} == {50000}
    assert {seed for _, _, seed in loaded} == {0}
