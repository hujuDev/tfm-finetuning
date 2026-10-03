from __future__ import annotations

from dataclasses import asdict

from tfm_finetuning.experiments.transfer.routes.resolve import resolve_transfer_route


def _config(*, family: str, route_id: str, target: str) -> dict[str, object]:
    return {
        "seed": 0,
        "data": {"path": f"data/tabarena/{target}", "split": "fold0_repeat0_sample0"},
        "foundation_model": {
            "model_id": "tabpfn_v2_5",
            "classifier_checkpoint": "models/tabpfn-v2.5-classifier.ckpt",
            "use_lr_scheduler": False,
            "early_stopping": False,
            "test_context_includes_validation_data": True,
        },
        "dataset_transfer": {
            "enabled": True,
            "family": family,
            "execution_mode": "train",
            "route_id": route_id,
            "updates_per_dataset": 100,
            "row_budget_per_dataset": 50_000,
            "nested_row_seed": 0,
            "context_policy": "target_train_plus_validation",
        },
    }


def test_sequential_retention_routes_reuse_exact_e1_source_stages() -> None:
    bank_then_covertype = resolve_transfer_route(
        _config(
            family="dataset_transfer_learned_retention",
            route_id="sequential_bank_then_covertype",
            target="covertype",
        )
    )
    e1_bank_first = resolve_transfer_route(
        _config(
            family="dataset_transfer",
            route_id="sequential_bank_then_heloc_then_target",
            target="credit-g",
        )
    )
    assert asdict(bank_then_covertype.stages[0]) == asdict(e1_bank_first.stages[0])

    covertype_then_bank = resolve_transfer_route(
        _config(
            family="dataset_transfer_learned_retention",
            route_id="sequential_covertype_then_bank",
            target="Bank_Customer_Churn",
        )
    )
    e1_covertype_first = resolve_transfer_route(
        _config(
            family="dataset_transfer",
            route_id="sequential_covertype_then_target",
            target="satimage",
        )
    )
    assert asdict(covertype_then_bank.stages[0]) == asdict(e1_covertype_first.stages[0])


def test_bank_filesystem_slug_resolves_to_canonical_dataset_id() -> None:
    config = _config(
        family="dataset_transfer_learned_retention",
        route_id="sequential_covertype_then_bank",
        target="bank_customer_churn",
    )

    route = resolve_transfer_route(config)

    assert route.target_dataset == "Bank_Customer_Churn"


def test_retention_routes_have_the_declared_two_dataset_budget_and_selectors() -> None:
    sequential = resolve_transfer_route(
        _config(
            family="dataset_transfer_learned_retention",
            route_id="sequential_bank_then_covertype",
            target="covertype",
        )
    )
    assert [stage.datasets for stage in sequential.stages] == [
        ("Bank_Customer_Churn",),
        ("covertype",),
    ]
    assert [stage.update_budget for stage in sequential.stages] == [100, 100]

    pooled = resolve_transfer_route(
        _config(
            family="dataset_transfer_learned_retention",
            route_id="pooled_bank_covertype",
            target="covertype",
        )
    )
    assert pooled.stages[0].datasets == ("Bank_Customer_Churn", "covertype")
    assert pooled.stages[0].update_budget == 200
    assert pooled.stages[0].selector.kind == "equal_mean_validation_improvement"


def test_retention_eval_can_target_both_datasets() -> None:
    config = _config(
        family="dataset_transfer_source_only_eval",
        route_id="learned_retention_probe",
        target="covertype",
    )
    transfer = config["dataset_transfer"]
    assert isinstance(transfer, dict)
    transfer.update(
        {
            "execution_mode": "checkpoint_eval",
            "evaluation_targets": ["Bank_Customer_Churn", "covertype"],
            "checkpoint_family": "dataset_transfer_learned_retention",
            "checkpoint_route": "sequential_bank_then_covertype",
            "checkpoint_role": "terminal",
            "checkpoint_origin": "covertype",
            "dependency_updates_per_dataset": 100,
            "dependency_row_budget_per_dataset": 50_000,
            "dependency_nested_row_seed": 0,
        }
    )
    route = resolve_transfer_route(config)

    assert route.execution_mode == "checkpoint_eval"
    assert route.evaluation_targets == ("Bank_Customer_Churn", "covertype")
    assert route.dependency is not None
    assert route.dependency.family == "dataset_transfer_learned_retention"
    assert route.dependency.checkpoint_role == "terminal"
