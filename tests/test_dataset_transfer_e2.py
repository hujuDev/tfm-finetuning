from __future__ import annotations

import pytest

from experiments.transfer.routes.budget import SATIMAGE_INITIALIZED_ROUTE, TARGET_ONLY_ROUTE
from experiments.transfer.routes.resolve import resolve_transfer_route


def _route_config(route_id: str, *, rows: int = 64, seed: int = 2) -> dict[str, object]:
    return {
        "seed": seed,
        "data": {
            "path": "data/tabarena/covertype",
            "split": "fold0_repeat0_sample0",
            "backend": "tabarena_openml_task_wrapper",
        },
        "evaluation": {"backend": "tabarena_aligned"},
        "preprocess": {"steps": ["clamp_train_pool"], "clamp_train_pool": {"max_size": 200000}},
        "foundation_model": {
            "model_id": "tabpfn_v2_5",
            "classifier_checkpoint": "models/base.ckpt",
            "use_lr_scheduler": False,
            "early_stopping": False,
            "validation_split_ratio": 0.1,
            "validation_split_seed": 0,
            "test_context_includes_validation_data": True,
        },
        "dataset_transfer": {
            "enabled": True,
            "family": "dataset_transfer_data_budget",
            "execution_mode": "train",
            "route_id": route_id,
            "updates_per_dataset": 100,
            "target_train_rows": rows,
            "nested_row_seed": 0,
            "dependency_updates_per_dataset": 100,
            "dependency_row_budget_per_dataset": 50000,
            "dependency_nested_row_seed": 0,
            "context_policy": "target_train_plus_validation",
        },
    }


def test_e2_target_only_route_emits_two_exact_horizon_views() -> None:
    route = resolve_transfer_route(_route_config(TARGET_ONLY_ROUTE))
    assert route.family == "dataset_transfer_data_budget"
    assert route.dependency is None
    assert route.stages[0].datasets == ("covertype",)
    assert route.stages[0].row_budget_per_dataset == 64
    assert route.stages[0].row_budget_policy == "exact"
    assert route.stages[0].row_subset_seed == 0
    assert route.stages[0].update_budget == 200
    assert route.stages[0].view_boundaries == (100,)


def test_e2_source_initialized_route_binds_exact_e1_boundary() -> None:
    route = resolve_transfer_route(_route_config(SATIMAGE_INITIALIZED_ROUTE))
    assert route.stages[0].update_budget == 100
    assert route.stages[0].row_subset_seed == 0
    dependency = route.dependency
    assert dependency is not None
    assert dependency.family == "dataset_transfer"
    assert dependency.route_id == "sequential_satimage_then_target"
    assert dependency.checkpoint_role == "reusable_boundary"
    assert dependency.checkpoint_origin == "covertype"
    assert dependency.producer_updates_per_dataset == 100
    assert dependency.producer_row_budget_per_dataset == 50000
    assert dependency.producer_row_subset_seed == 0


def test_e2_rejects_nonmatching_nested_row_seed_policy() -> None:
    config = _route_config(TARGET_ONLY_ROUTE)
    transfer = config["dataset_transfer"]
    assert isinstance(transfer, dict)
    transfer["nested_row_seed"] = "run_seed"
    with pytest.raises(ValueError, match="same fixed/run-seed policy"):
        resolve_transfer_route(config)
