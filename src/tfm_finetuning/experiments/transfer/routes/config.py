"""Decode and validate Section 3 route configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tfm_finetuning.experiments.transfer.spec import (
    CheckpointDependency,
    SplitIdentity,
    TransferDataRecipe,
    TransferTrainingRecipe,
)
from tfm_finetuning.finetuning.config import (
    resolve_common_finetune_config,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import normalize_preprocessing_protocol

_TARGET_PATH_ALIASES = {
    "bank_customer_churn": "Bank_Customer_Churn",
    "credit_g": "credit-g",
}


@dataclass(frozen=True)
class DecodedTransferRouteConfig:
    """Validated inputs consumed by the maintained route-family builders."""

    transfer: Mapping[str, Any]
    target: str
    family: str
    route_id: str
    execution_mode: str
    training_recipe: TransferTrainingRecipe
    data_recipe: TransferDataRecipe
    context_policy: str
    updates_per_dataset: int | None
    row_budget_per_dataset: int | None
    target_train_rows: int | None
    stage_common: Mapping[str, Any] | None


def decode_transfer_route_config(
    config: Mapping[str, Any],
    *,
    target_dataset: str | None,
) -> DecodedTransferRouteConfig:
    """Parse one concrete config once before executable route construction."""
    transfer = mapping(config.get("dataset_transfer"))
    if transfer.get("enabled") is not True and not transfer.get("route_id"):
        raise ValueError("dataset_transfer.enabled=true or dataset_transfer.route_id is required.")
    target = clean_dataset(target_dataset or transfer.get("evaluation_target") or target_from_config(config))
    family = str(transfer.get("family") or "dataset_transfer").strip()
    route_id = str(transfer.get("route_id") or "").strip()
    if not route_id:
        raise ValueError("dataset_transfer.route_id is required.")
    execution_mode = str(transfer.get("execution_mode") or "train").strip().lower()
    training_recipe = TransferTrainingRecipe.from_common_config(resolve_common_finetune_config(config))
    data_recipe = transfer_data_recipe(config)
    context_policy = required_context_policy(config, transfer=transfer)
    frozen_transfer = MappingProxyType(dict(transfer))
    if execution_mode == "checkpoint_eval":
        return DecodedTransferRouteConfig(
            transfer=frozen_transfer,
            target=target,
            family=family,
            route_id=route_id,
            execution_mode=execution_mode,
            training_recipe=training_recipe,
            data_recipe=data_recipe,
            context_policy=context_policy,
            updates_per_dataset=None,
            row_budget_per_dataset=None,
            target_train_rows=None,
            stage_common=None,
        )

    updates = positive_int(
        transfer.get("updates_per_dataset"),
        field="dataset_transfer.updates_per_dataset",
    )
    row_budget = optional_positive_int(transfer.get("row_budget_per_dataset"))
    target_rows = optional_positive_int(transfer.get("target_train_rows"))
    stage_common = MappingProxyType(
        {
            "split": split_identity(config),
            "preprocessing": preprocessing(config),
            "model_id": model_id(config),
            "base_checkpoint": base_checkpoint(config, task_family=task_family(transfer)),
            "task_family": task_family(transfer),
            "training_recipe": training_recipe,
            "data_recipe": data_recipe,
            "context_policy": context_policy,
            "seed": seed(config),
            "row_subset_seed": row_subset_seed(config, transfer=transfer),
            "strict_resource_semantics": bool(transfer.get("strict_resource_semantics", True)),
        }
    )
    return DecodedTransferRouteConfig(
        transfer=frozen_transfer,
        target=target,
        family=family,
        route_id=route_id,
        execution_mode=execution_mode,
        training_recipe=training_recipe,
        data_recipe=data_recipe,
        context_policy=context_policy,
        updates_per_dataset=updates,
        row_budget_per_dataset=row_budget,
        target_train_rows=target_rows,
        stage_common=stage_common,
    )


def split_identity(config: Mapping[str, Any]) -> SplitIdentity:
    """Decode the fixed OpenML split identity."""
    data = mapping(config.get("data"))
    split_name = str(data.get("split") or "fold0_repeat0_sample0")
    match = re.fullmatch(
        r"fold(\d+)_repeat(\d+)_sample(\d+)",
        split_name,
    )
    if match is None:
        raise ValueError("Section 3 requires data.split=fold<k>_repeat<r>_sample<s>.")
    return SplitIdentity(*(int(value) for value in match.groups()))


def target_from_config(config: Mapping[str, Any]) -> str:
    """Resolve the target dataset from the materialized data path."""
    raw = str(mapping(config.get("data")).get("path") or "").removeprefix(":")
    path_name = Path(raw).name
    return _TARGET_PATH_ALIASES.get(path_name, path_name)


def model_id(config: Mapping[str, Any]) -> str:
    """Resolve the maintained foundation-model identifier."""
    value = str(mapping(config.get("foundation_model")).get("model_id") or "").strip()
    if not value:
        raise ValueError("foundation_model.model_id is required for transfer routes.")
    return value


def base_checkpoint(config: Mapping[str, Any], *, task_family: str) -> str:
    """Resolve the configured base-checkpoint identifier."""
    foundation = mapping(config.get("foundation_model"))
    key = "regressor_checkpoint" if task_family == "regression" else "classifier_checkpoint"
    value = foundation.get(key)
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"foundation_model.{key} is required for {task_family} transfer routes.")
    return text


def task_family(transfer: Mapping[str, Any]) -> str:
    """Resolve the homogeneous task family for every stage in one route."""
    value = str(transfer.get("task_family") or "classification").strip().lower()
    if value not in {"classification", "regression"}:
        raise ValueError("dataset_transfer.task_family must be classification or regression.")
    return value


def preprocessing(config: Mapping[str, Any]) -> str:
    """Resolve the shared or native preprocessing protocol."""
    value = str(mapping(config.get("foundation_model")).get("preprocessing_protocol") or "").strip()
    return normalize_preprocessing_protocol(value) or "shared"


def seed(config: Mapping[str, Any]) -> int:
    """Resolve the concrete run seed."""
    return int(config.get("seed") or 0)


def row_subset_seed(
    config: Mapping[str, Any],
    *,
    transfer: Mapping[str, Any],
) -> int:
    """Resolve the nested row-prefix seed."""
    policy, fixed_seed = row_seed_policy(
        transfer.get("nested_row_seed"),
        field="dataset_transfer.nested_row_seed",
        implicit_run_seed=True,
    )
    return seed(config) if policy == "run_seed" else int(fixed_seed)


def row_seed_policy(
    raw: Any,
    *,
    field: str,
    implicit_run_seed: bool = False,
) -> tuple[str, int | None]:
    """Decode a fixed or run-seed row-selection policy."""
    text = "" if raw is None else str(raw).strip().lower()
    if text == "run_seed" or (implicit_run_seed and text in {"", "none", "null"}):
        return ("run_seed", None)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be run_seed or a non-negative integer.") from exc
    if isinstance(raw, bool) or value < 0 or value != raw:
        raise ValueError(f"{field} must be run_seed or a non-negative integer.")
    return ("fixed", value)


def mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and normalize absent sections."""
    return value if isinstance(value, Mapping) else {}


def clean_dataset(value: Any) -> str:
    """Validate a non-empty dataset identifier."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("Dataset names must be non-empty.")
    return text


def positive_int(value: Any, *, field: str) -> int:
    """Validate a positive integer config field."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return parsed


def optional_positive_int(value: Any) -> int | None:
    """Decode an optional positive row budget."""
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return positive_int(value, field="row budget")


def required_context_policy(
    config: Mapping[str, Any],
    *,
    transfer: Mapping[str, Any],
) -> str:
    """Enforce a maintained ICL policy with validation in each fit pool."""
    policy = str(transfer.get("context_policy") or "").strip()
    supported = {
        "target_train_plus_validation",
        "route_matched_train_plus_validation",
    }
    if policy not in supported:
        raise ValueError(
            f"Section 3 requires dataset_transfer.context_policy to be one of: {', '.join(sorted(supported))}."
        )
    if (
        policy == "route_matched_train_plus_validation"
        and str(transfer.get("family") or "") != "dataset_transfer_natural_domain_context_eval"
    ):
        raise ValueError(
            "route_matched_train_plus_validation is restricted to the "
            "same-schema natural-domain checkpoint-evaluation family."
        )
    foundation = mapping(config.get("foundation_model"))
    raw_flag = foundation.get("test_context_includes_validation_data")
    if raw_flag is None:
        raw_flag = foundation.get("test_context_includes_explicit_validation_data")
    if not strict_bool(raw_flag):
        raise ValueError(
            "Section 3 requires "
            "foundation_model.test_context_includes_validation_data=true "
            "so each declared train-plus-validation context policy matches "
            "runtime evaluation."
        )
    return policy


def strict_bool(value: Any) -> bool:
    """Decode the strict boolean forms used by config materialization."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def dependency_training_recipe(
    current_recipe: TransferTrainingRecipe,
    *,
    transfer: Mapping[str, Any],
) -> TransferTrainingRecipe:
    """Resolve optional exact producer-recipe overrides."""
    raw = transfer.get("dependency_training_recipe")
    if raw is None:
        return current_recipe
    if not isinstance(raw, Mapping):
        raise ValueError("dataset_transfer.dependency_training_recipe must be a mapping.")
    return current_recipe.with_overrides(raw)


def transfer_data_recipe(
    config: Mapping[str, Any],
) -> TransferDataRecipe:
    """Resolve exact path-independent producer data preparation."""
    foundation = mapping(config.get("foundation_model"))
    evaluation = mapping(config.get("evaluation"))
    data = mapping(config.get("data"))
    preprocess = mapping(config.get("preprocess"))
    raw_steps = preprocess.get("steps")
    if raw_steps is None:
        steps: tuple[str, ...] = ()
    elif isinstance(raw_steps, list | tuple):
        steps = tuple(str(step).strip() for step in raw_steps)
    else:
        raise ValueError("preprocess.steps must be a list for Section 3 transfer routes.")
    unsupported = sorted(set(steps).difference({"clamp_train_pool"}))
    if unsupported:
        raise ValueError(
            "Section 3 transfer routes only support clamp_train_pool data "
            f"preprocessing; unsupported: {', '.join(unsupported)}."
        )
    clamp = mapping(preprocess.get("clamp_train_pool"))
    clamp_max = optional_recipe_int(
        clamp.get("max_size", clamp.get("size")),
        field="preprocess.clamp_train_pool.max_size",
    )
    clamp_seed = optional_recipe_int(
        clamp.get("seed"),
        field="preprocess.clamp_train_pool.seed",
        allow_zero=True,
    )
    if clamp_max is not None and "clamp_train_pool" not in steps:
        raise ValueError("A configured clamp_train_pool max_size must be declared in preprocess.steps.")

    raw_fraction = foundation.get(
        "validation_split_ratio",
        evaluation.get("val_fraction", 0.2),
    )
    try:
        validation_fraction = float(raw_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("Section 3 validation split ratio must be numeric.") from exc
    raw_validation_seed = foundation.get(
        "validation_split_seed",
        evaluation.get(
            "val_seed",
            data.get("seed", config.get("seed", 0)),
        ),
    )
    validation_seed = nonnegative_int(
        raw_validation_seed,
        field="validation split seed",
    )
    return TransferDataRecipe(
        data_backend=str(data.get("backend") or "").strip(),
        evaluation_backend=str(evaluation.get("backend") or "").strip(),
        validation_split_ratio=validation_fraction,
        validation_split_seed=validation_seed,
        preprocess_steps=steps,
        clamp_train_pool_max_size=clamp_max,
        clamp_train_pool_seed=clamp_seed,
        train_max_rows=first_optional_recipe_int(
            foundation,
            ("train_max_rows", "fit_train_max_rows", "max_train_rows"),
        ),
        inference_context_max_rows=first_optional_recipe_int(
            foundation,
            ("inference_context_max_rows", "eval_context_max_rows"),
        ),
        eval_query_max_rows=first_optional_recipe_int(
            foundation,
            ("eval_query_max_rows", "inference_query_max_rows"),
        ),
    )


def dependency_data_recipe(
    current_recipe: TransferDataRecipe,
    *,
    transfer: Mapping[str, Any],
) -> TransferDataRecipe:
    """Resolve optional exact producer data-recipe overrides."""
    raw = transfer.get("dependency_data_recipe")
    if raw is None:
        return current_recipe
    if not isinstance(raw, Mapping):
        raise ValueError("dataset_transfer.dependency_data_recipe must be a mapping.")
    return current_recipe.with_overrides(raw)


def first_optional_recipe_int(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
) -> int | None:
    """Resolve the first present optional integer recipe field."""
    for key in keys:
        if config.get(key) is not None:
            return optional_recipe_int(
                config[key],
                field=f"foundation_model.{key}",
            )
    return None


def optional_recipe_int(
    value: Any,
    *,
    field: str,
    allow_zero: bool = False,
) -> int | None:
    """Decode an optional non-negative recipe field."""
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    parsed = nonnegative_int(value, field=field)
    if parsed == 0 and not allow_zero:
        raise ValueError(f"{field} must be positive when provided.")
    return parsed


def nonnegative_int(value: Any, *, field: str) -> int:
    """Validate a non-negative integer field."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer.") from exc
    if isinstance(value, bool) or parsed < 0 or parsed != value:
        raise ValueError(f"{field} must be a non-negative integer.")
    return parsed


def checkpoint_dependency(
    transfer: Mapping[str, Any],
    *,
    family: str,
    route_id: str,
    checkpoint_role: str,
    checkpoint_origin: str,
    producer_training_recipe: TransferTrainingRecipe,
    producer_data_recipe: TransferDataRecipe,
    producer_context_policy: str,
) -> CheckpointDependency:
    """Describe the producer recipe required by a continuation or frozen evaluation."""
    updates = positive_int(
        transfer.get("dependency_updates_per_dataset"),
        field="dataset_transfer.dependency_updates_per_dataset",
    )
    row_budget = optional_positive_int(transfer.get("dependency_row_budget_per_dataset"))
    seed_policy, fixed_seed = row_seed_policy(
        transfer.get("dependency_nested_row_seed"),
        field="dataset_transfer.dependency_nested_row_seed",
        implicit_run_seed=True,
    )
    row_subset_seed = None if seed_policy == "run_seed" else fixed_seed
    return CheckpointDependency(
        family=family,
        route_id=route_id,
        checkpoint_role=checkpoint_role,  # type: ignore[arg-type]
        checkpoint_origin=checkpoint_origin,
        producer_updates_per_dataset=updates,
        producer_row_budget_per_dataset=row_budget,
        producer_row_subset_seed=row_subset_seed,
        producer_training_recipe=producer_training_recipe,
        producer_data_recipe=producer_data_recipe,
        producer_context_policy=producer_context_policy,  # type: ignore[arg-type]
    )
