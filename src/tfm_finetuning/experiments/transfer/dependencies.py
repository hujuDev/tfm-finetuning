"""Find completed transfer producers from fresh run records, without artifact caches.

Consumers never launch training themselves. Run the documented producer study
first, or explicitly set ``dataset_transfer.checkpoint_path``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tfm_finetuning.experiments.transfer.routes.resolve import resolve_transfer_route
from tfm_finetuning.experiments.transfer.runner.executor import require_checkpoint
from tfm_finetuning.experiments.transfer.spec import TransferRouteSpec


def producer_config(config: Mapping[str, Any], route: TransferRouteSpec) -> dict[str, Any]:
    """Reconstruct the producer's training and data recipe from a consumer route."""
    dependency = route.dependency
    if dependency is None:
        raise ValueError("This route has no checkpoint dependency.")
    result = copy.deepcopy(dict(config))
    transfer = result.setdefault("dataset_transfer", {})
    transfer.update(
        enabled=True,
        family=dependency.family,
        execution_mode="train",
        route_id=dependency.route_id,
        evaluation_target=dependency.checkpoint_origin,
        updates_per_dataset=dependency.producer_updates_per_dataset,
        row_budget_per_dataset=dependency.producer_row_budget_per_dataset,
        context_policy=dependency.producer_context_policy,
    )
    transfer.pop("checkpoint_path", None)
    if dependency.producer_row_subset_seed is None:
        transfer.pop("nested_row_seed", None)
    else:
        transfer["nested_row_seed"] = dependency.producer_row_subset_seed
    foundation = result.setdefault("foundation_model", {})
    foundation.update(asdict(dependency.producer_training_recipe))
    foundation["test_context_includes_validation_data"] = (
        dependency.producer_context_policy == "target_train_plus_validation"
    )
    recipe = dependency.producer_data_recipe
    result.setdefault("data", {})["backend"] = recipe.data_backend
    result.setdefault("evaluation", {})["backend"] = recipe.evaluation_backend
    foundation.update(
        validation_split_ratio=recipe.validation_split_ratio,
        validation_split_seed=recipe.validation_split_seed,
    )
    for key in (
        "train_max_rows",
        "fit_train_max_rows",
        "max_train_rows",
        "inference_context_max_rows",
        "eval_context_max_rows",
        "eval_query_max_rows",
        "inference_query_max_rows",
    ):
        foundation.pop(key, None)
    for key in ("train_max_rows", "inference_context_max_rows", "eval_query_max_rows"):
        value = getattr(recipe, key)
        if value is not None:
            foundation[key] = value
    result.pop("preprocess", None)
    if recipe.preprocess_steps:
        result["preprocess"] = {"steps": list(recipe.preprocess_steps)}
        if "clamp_train_pool" in recipe.preprocess_steps:
            result["preprocess"]["clamp_train_pool"] = {
                key: value
                for key, value in {
                    "max_size": recipe.clamp_train_pool_max_size,
                    "seed": recipe.clamp_train_pool_seed,
                }.items()
                if value is not None
            }
    return result


def _stage_recipe(route: TransferRouteSpec, role: str) -> list[dict[str, Any]]:
    stages = []
    for stage in route.stages:
        record = asdict(stage)
        # The parent model is checked through model_id; local download paths may differ.
        record.pop("base_checkpoint", None)
        stages.append(record)
        if stage.role == role:
            return stages
    raise ValueError(f"Producer route {route.route_id!r} has no {role!r} checkpoint.")


def resolve_checkpoint_dependency(
    config: Mapping[str, Any], route: TransferRouteSpec, *, results_root: Path, repo_root: Path
) -> Path | None:
    """Return one matching completed producer's checkpoint, requiring its weights.

    Model, preprocessing, seed, OpenML split, nested row budgets, and every
    optimization/data recipe field must agree. A report alone is insufficient.
    """
    dependency = route.dependency
    if dependency is None:
        return None
    explicit = config.get("dataset_transfer", {}).get("checkpoint_path")
    if explicit:
        path = Path(explicit).expanduser()
        return require_checkpoint(path if path.is_absolute() else repo_root / path)
    expected = resolve_transfer_route(producer_config(config, route), target_dataset=dependency.checkpoint_origin)
    expected_stages = _stage_recipe(expected, dependency.checkpoint_role)
    matches: set[Path] = set()
    for run_path in Path(results_root).rglob("run.json"):
        metadata = json.loads(run_path.read_text())
        if metadata.get("status") != "complete":
            continue
        report_path = run_path.with_name("finetune_report.json")
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        transfer = report.get("dataset_transfer", {})
        if (transfer.get("family"), transfer.get("route")) != (dependency.family, dependency.route_id):
            continue
        candidate_config = report.get("config", metadata.get("config"))
        if not isinstance(candidate_config, Mapping):
            raise ValueError(f"Completed producer lacks its resolved configuration: {run_path}")
        candidate = resolve_transfer_route(candidate_config, target_dataset=dependency.checkpoint_origin)
        if _stage_recipe(candidate, dependency.checkpoint_role) != expected_stages:
            continue
        stage_name = next(stage.name for stage in candidate.stages if stage.role == dependency.checkpoint_role)
        raw_path = report.get("stage_checkpoints", {}).get(stage_name)
        if not raw_path:
            raise ValueError(f"Producer report has no checkpoint for stage {stage_name!r}: {report_path}")
        checkpoint = Path(raw_path)
        matches.add(require_checkpoint(checkpoint if checkpoint.is_absolute() else report_path.parent / checkpoint))
    if not matches:
        raise FileNotFoundError(
            f"Run producer {dependency.family}/{dependency.route_id} for {dependency.checkpoint_origin} "
            "with the same model, preprocessing, seed, split and budgets before this consumer, "
            "or set dataset_transfer.checkpoint_path explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple completed producer checkpoints match. Set dataset_transfer.checkpoint_path explicitly."
        )
    return matches.pop()
