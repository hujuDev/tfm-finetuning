from __future__ import annotations

import pytest

from experiments.transfer.routes.balanced_budget import SOURCE_INITIALIZED_ROUTE, TARGET_ONLY_ROUTE
from experiments.transfer.routes.resolve import resolve_transfer_route


def _config(
    *, target: str, route_id: str, task_family: str = "classification", rows: int = 64, row_policy: str = "exact"
) -> dict[str, object]:
    checkpoint_key = "regressor_checkpoint" if task_family == "regression" else "classifier_checkpoint"
    return {
        "seed": 2,
        "data": {
            "path": f"data/tabarena/{target}",
            "split": "fold0_repeat0_sample0",
            "backend": "tabarena_openml_task_wrapper",
        },
        "evaluation": {"backend": "tabarena_aligned"},
        "preprocess": {"steps": ["clamp_train_pool"], "clamp_train_pool": {"max_size": 200000}},
        "foundation_model": {
            "model_id": "tabpfn_v2_5",
            checkpoint_key: "models/base.ckpt",
            "use_lr_scheduler": False,
            "early_stopping": False,
            "validation_split_ratio": 0.1,
            "validation_split_seed": 0,
            "test_context_includes_validation_data": True,
        },
        "dataset_transfer": {
            "enabled": True,
            "family": "dataset_transfer_balanced_budget",
            "task_family": task_family,
            "execution_mode": "train",
            "route_id": route_id,
            "updates_per_dataset": 100,
            "target_train_rows": rows,
            "target_row_budget_policy": row_policy,
            "nested_row_seed": 0,
            "dependency_updates_per_dataset": 100,
            "dependency_row_budget_per_dataset": 50000,
            "dependency_nested_row_seed": 0,
            "context_policy": "target_train_plus_validation",
        },
    }


@pytest.mark.parametrize(
    ("target", "source_label", "producer_family", "producer_route", "origin", "role"),
    [
        (
            "sdss17",
            "satimage",
            "dataset_transfer",
            "sequential_satimage_then_target",
            "covertype",
            "reusable_boundary",
        ),
        (
            "amazon_employee_access",
            "bank_heloc",
            "dataset_transfer",
            "pooled_sources_then_target",
            "credit-g",
            "reusable_boundary",
        ),
        (
            "diabetes130us",
            "bank_heloc",
            "dataset_transfer",
            "pooled_sources_then_target",
            "credit-g",
            "reusable_boundary",
        ),
        (
            "physiochemical_protein",
            "houses",
            "dataset_transfer_source_checkpoint",
            "source_only_checkpoint",
            "houses",
            "terminal",
        ),
        (
            "food_delivery_time",
            "houses",
            "dataset_transfer_source_checkpoint",
            "source_only_checkpoint",
            "houses",
            "terminal",
        ),
    ],
)
def test_balanced_budget_source_routes_bind_declared_boundaries(
    target: str, source_label: str, producer_family: str, producer_route: str, origin: str, role: str
) -> None:
    task_family = "regression" if source_label == "houses" else "classification"
    route = resolve_transfer_route(_config(target=target, route_id=SOURCE_INITIALIZED_ROUTE, task_family=task_family))
    assert route.family == "dataset_transfer_balanced_budget"
    assert route.stages[0].task_family == task_family
    assert route.stages[0].update_budget == 100
    assert route.dependency is not None
    assert (
        route.dependency.family,
        route.dependency.route_id,
        route.dependency.checkpoint_origin,
        route.dependency.checkpoint_role,
    ) == (producer_family, producer_route, origin, role)


def test_balanced_budget_target_only_emits_nested_prefix_views_and_full_cap() -> None:
    route = resolve_transfer_route(
        _config(
            target="food_delivery_time",
            route_id=TARGET_ONLY_ROUTE,
            task_family="regression",
            rows=50000,
            row_policy="cap",
        )
    )
    stage = route.stages[0]
    assert stage.base_checkpoint == "models/base.ckpt"
    assert stage.row_budget_per_dataset == 50000
    assert stage.row_budget_policy == "cap"
    assert stage.update_budget == 200
    assert stage.view_boundaries == (100,)


def test_houses_source_checkpoint_route_is_regression_and_reusable() -> None:
    config = _config(
        target="houses", route_id="source_only_checkpoint", task_family="regression", rows=50000, row_policy="cap"
    )
    transfer = config["dataset_transfer"]
    assert isinstance(transfer, dict)
    transfer.update({"family": "dataset_transfer_source_checkpoint", "row_budget_per_dataset": 50000})
    route = resolve_transfer_route(config)
    assert route.stages[0].datasets == ("houses",)
    assert route.stages[0].task_family == "regression"


def test_balanced_budget_rejects_classifier_checkpoint_for_regression() -> None:
    config = _config(target="physiochemical_protein", route_id=TARGET_ONLY_ROUTE, task_family="regression")
    foundation = config["foundation_model"]
    assert isinstance(foundation, dict)
    foundation["classifier_checkpoint"] = foundation.pop("regressor_checkpoint")
    with pytest.raises(ValueError, match="regressor_checkpoint"):
        resolve_transfer_route(config)
