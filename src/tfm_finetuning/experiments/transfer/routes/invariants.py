"""Cross-field scientific invariants for maintained transfer routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tfm_finetuning.experiments.transfer.routes.config import row_seed_policy
from tfm_finetuning.experiments.transfer.spec import TransferStageSpec


def validate_checkpoint_boundaries(stages: tuple[TransferStageSpec, ...]) -> None:
    """Require a horizon-independent training path for shorter-budget controls."""
    for stage in stages:
        if stage.view_boundaries and (stage.training_recipe.use_lr_scheduler or stage.training_recipe.early_stopping):
            raise ValueError(
                "Checkpoint-boundary controls require use_lr_scheduler=false and "
                "early_stopping=false so a shorter run has the same optimization path."
            )


def validate_budget_row_seed_policy(
    transfer: Mapping[str, Any],
    *,
    resolved_seed: int,
    run_seed: int,
) -> None:
    """Bind every E2 row prefix to the fixed seed-zero E1 endpoint."""
    required = ("nested_row_seed", "dependency_nested_row_seed")
    missing = [f"dataset_transfer.{key}" for key in required if key not in transfer]
    if missing:
        raise ValueError(
            "E2 nested-budget routes require explicit curve and E1-endpoint "
            "row seed policies; missing: " + ", ".join(missing)
        )
    curve_policy = row_seed_policy(
        transfer["nested_row_seed"],
        field="dataset_transfer.nested_row_seed",
    )
    endpoint_policy = row_seed_policy(
        transfer["dependency_nested_row_seed"],
        field="dataset_transfer.dependency_nested_row_seed",
    )
    if curve_policy != endpoint_policy:
        raise ValueError(
            "E2 nested_row_seed must use the same fixed/run-seed policy as "
            "dependency_nested_row_seed so the reused E1 endpoint is a true superset."
        )
    policy_name, fixed_seed = curve_policy
    expected = run_seed if policy_name == "run_seed" else fixed_seed
    if resolved_seed != expected:
        raise RuntimeError(
            f"Resolved E2 row subset seed {resolved_seed} disagrees with declared policy {curve_policy!r}."
        )
