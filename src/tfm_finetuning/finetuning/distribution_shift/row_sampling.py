"""Train-row and context-row distribution interventions."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from tfm_finetuning.finetuning.distribution_shift._common import is_noop_token, jsonable_label, parse_fraction
from tfm_finetuning.finetuning.distribution_shift.fit_pool import (
    combine_fit_pool,
    nested_selection_indices,
    rebuild_fit_roles,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def apply_subsample_train(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Select one nested labeled-budget prefix and rebuild all fit-side roles."""
    requested = cfg.get("size")
    pool = combine_fit_pool(data)
    target = _target_count(requested, len(pool.X))
    if target is None or target >= len(pool.X):
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "requested": requested,
            "fit_pool_size_before": len(pool.X),
            "skipped": True,
            "reason": "subsample_noop_or_full",
        }
    selected = nested_selection_indices(pool, target=target, task_type=str(data.task_type), seed=seed)
    transformed, pool_report = rebuild_fit_roles(
        data,
        pool,
        selected,
        task_type=str(data.task_type),
        seed=seed,
    )
    return transformed, {
        "stage": "pre_finite",
        "seed": seed,
        "requested": requested,
        "strategy": "nested_stratified" if data.task_type in {"binary", "multiclass"} else "nested_seeded",
        "transformation_setting": "fit_side_transformation",
        "fit_side_role": "subsampled_total_labeled_pool",
        "test_side_role": "original",
        **pool_report,
    }


def apply_rebalance_train(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Select a binary fit pool at the requested class prior.

    With ``budget_policy=common_feasible``, every configured prior uses the
    largest row budget feasible without replacement for all target rates. The
    ``original`` condition samples that same budget at the observed prior and
    is therefore the family-specific causal reference.

    With ``budget_policy=retain_original_minority``, the ``original`` condition
    retains the full fit pool and transformed conditions keep every row from
    the original minority class while downsampling only the original majority.

    Example:
        ``target_rates=[0.05, 0.2, 0.5]`` gives every condition the same number
        of fit-side rows while only their class composition changes.
    """
    focal_label_policy = str(cfg.get("focal_label_policy") or "encoded_class_1").strip().lower().replace("-", "_")
    if focal_label_policy not in {"encoded_class_1", "original_fit_majority"}:
        raise ValueError(
            "rebalance_train.focal_label_policy must be 'encoded_class_1' or "
            f"'original_fit_majority', got {cfg.get('focal_label_policy')!r}"
        )
    majority_oriented = focal_label_policy == "original_fit_majority"
    requested_key = "target_focal_rate" if majority_oriented else "target_pos_rate"
    requested = cfg.get(requested_key)
    budget_policy = str(cfg.get("budget_policy") or "").strip().lower().replace("-", "_")
    original_reference = str(requested).strip().lower() == "original"
    intervention_reference_policies = {"common_feasible", "retain_original_minority"}
    if (is_noop_token(requested) or parse_fraction(requested) == 0.0) and not (
        original_reference and budget_policy in intervention_reference_policies
    ):
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            f"requested_{requested_key}": requested,
            "skipped": True,
            "reason": "class_prior_noop",
        }
    if data.task_type != "binary":
        raise ValueError(f"rebalance_train is a binary-classification intervention, got task_type={data.task_type!r}")
    pool = combine_fit_pool(data)
    selected, label_report = _binary_prior_indices(
        pool.query_y,
        requested=requested,
        pos_label_cfg=cfg.get("pos_label"),
        focal_label_policy=focal_label_policy,
        majority_tie_break=cfg.get("majority_tie_break"),
        budget_policy=budget_policy,
        target_rates_cfg=cfg.get("target_rates"),
        rng=rng,
    )
    transformed, pool_report = rebuild_fit_roles(
        data,
        pool,
        selected,
        task_type=str(data.task_type),
        seed=seed,
    )
    report = {
        "stage": "pre_finite",
        "seed": seed,
        f"requested_{requested_key}": requested,
        "transformation_setting": "fit_side_transformation",
        "fit_side_role": (
            "fixed_budget_rebalanced_total_labeled_pool"
            if budget_policy == "common_feasible"
            else "original_minority_retained_majority_downsampled_total_labeled_pool"
        ),
        "test_side_role": "original",
        "family_reference": original_reference,
        **pool_report,
        **label_report,
    }
    return transformed, report


def _target_count(requested: Any, total: int) -> int | None:
    if is_noop_token(requested):
        return None
    value = parse_fraction(requested)
    if value is None:
        raise ValueError(f"subsample size must be numeric or fraction, got {requested!r}")
    if value in {0.0, 1.0}:
        return None
    if 0.0 < value < 1.0:
        # A rebuilt fit pool needs at least one training and one validation row.
        return min(total, max(2, math.floor(total * value)))
    if float(value).is_integer() and value > 1:
        return min(total, int(value))
    raise ValueError(f"subsample size must be an integer or fraction in (0, 1], got {requested!r}")


def _binary_prior_indices(
    labels: np.ndarray,
    *,
    requested: Any,
    pos_label_cfg: Any,
    budget_policy: str,
    target_rates_cfg: Any,
    rng: np.random.Generator,
    focal_label_policy: str = "encoded_class_1",
    majority_tie_break: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic within-class selections for one class-prior policy."""
    supported_budget_policies = {"common_feasible", "retain_original_minority"}
    if budget_policy not in supported_budget_policies:
        raise ValueError(
            "rebalance_train.budget_policy must be 'common_feasible' or "
            f"'retain_original_minority', got {budget_policy!r}"
        )
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError(f"rebalance_train expected two labels, got {unique.size}")
    policy = str(focal_label_policy or "encoded_class_1").strip().lower().replace("-", "_")
    if policy == "original_fit_majority":
        if pos_label_cfg is not None:
            raise ValueError("pos_label must be omitted when focal_label_policy='original_fit_majority'")
        focal_label, orientation_report = _resolve_original_fit_majority_label(
            labels,
            tie_break=majority_tie_break,
        )
    elif policy == "encoded_class_1":
        focal_label = unique[-1] if pos_label_cfg is None else pos_label_cfg
        if focal_label not in set(unique.tolist()):
            raise ValueError(f"pos_label={pos_label_cfg!r} is not present in labels {unique.tolist()}")
        orientation_report = {
            "focal_label_policy": "encoded_class_1",
            "focal_label_source": "configured_pos_label" if pos_label_cfg is not None else "largest_encoded_label",
            "majority_tie": False,
            "majority_tie_break": None,
            "test_labels_used_for_orientation": False,
        }
    else:
        raise ValueError(
            "rebalance_train.focal_label_policy must be 'encoded_class_1' or "
            f"'original_fit_majority', got {focal_label_policy!r}"
        )
    other_label = unique[0] if unique[0] != focal_label else unique[1]
    focal_idx = np.flatnonzero(labels == focal_label)
    other_idx = np.flatnonzero(labels == other_label)
    if focal_idx.size == 0 or other_idx.size == 0:
        raise ValueError("rebalance_train requires both classes in the fit pool")
    current_rate = float(focal_idx.size / len(labels))
    target_rate = current_rate if str(requested).strip().lower() == "original" else parse_fraction(requested)
    if target_rate is None or not 0.0 < target_rate < 1.0:
        raise ValueError(f"rebalance_train target class rate must be in (0, 1), got {requested!r}")
    if budget_policy == "common_feasible":
        target_rates = _class_prior_target_rates(target_rates_cfg)
        budget = _common_feasible_budget(
            pos_count=len(focal_idx),
            neg_count=len(other_idx),
            target_rates=target_rates,
        )
        target_focal, target_other = _counts_for_prior(budget, target_rate)
        if target_focal > len(focal_idx) or target_other > len(other_idx):
            raise ValueError(
                "common class-prior budget cannot realize requested rate without replacement: "
                f"budget={budget}, target_rate={target_rate}, "
                f"available=({len(focal_idx)}, {len(other_idx)})"
            )
        selected = np.sort(
            np.concatenate(
                [rng.permutation(focal_idx)[:target_focal], rng.permutation(other_idx)[:target_other]]
            ).astype(int)
        )
        policy_report = {
            "common_fit_pool_budget": budget,
            "configured_target_rates": target_rates,
            "original_fixed_budget_control": str(requested).strip().lower() == "original",
            "original_full_pool_control": False,
            "retained_original_minority_rows": False,
            "selection_strategy": "deterministic_within_class_prefixes_without_replacement",
        }
    else:
        if policy != "original_fit_majority":
            raise ValueError(
                "rebalance_train.budget_policy='retain_original_minority' requires "
                "focal_label_policy='original_fit_majority'"
            )
        if target_rate > current_rate and not math.isclose(target_rate, current_rate):
            raise ValueError(
                "retain_original_minority cannot increase the original-majority rate "
                f"without removing minority rows: target_rate={target_rate}, original_rate={current_rate}"
            )
        original_reference = str(requested).strip().lower() == "original"
        if original_reference:
            target_focal = len(focal_idx)
            selected = np.arange(len(labels), dtype=int)
        else:
            target_focal = _focal_count_with_fixed_other(
                other_count=len(other_idx),
                target_rate=float(target_rate),
            )
            if target_focal > len(focal_idx):
                raise ValueError(
                    "retain_original_minority cannot realize requested rate without adding majority rows: "
                    f"target_rate={target_rate}, available=({len(focal_idx)}, {len(other_idx)})"
                )
            selected = np.sort(np.concatenate([rng.permutation(focal_idx)[:target_focal], other_idx]).astype(int))
        policy_report = {
            "fixed_original_minority_count": int(other_idx.size),
            "variable_fit_pool_budget": int(len(selected)),
            "original_fixed_budget_control": False,
            "original_full_pool_control": original_reference,
            "retained_original_minority_rows": True,
            "selection_strategy": (
                "full_original_fit_pool"
                if original_reference
                else "deterministic_original_majority_prefix_with_all_original_minority_rows"
            ),
        }
    selected_labels = labels[selected]
    report = {
        "budget_policy": budget_policy,
        "count_rounding_rule": (
            "nearest_integer_half_up_toward_more_focal_rows"
            if budget_policy == "retain_original_minority"
            else "nearest_integer_ties_to_even"
        ),
        "resolved_target_focal_rate": float(target_rate),
        "focal_label": jsonable_label(focal_label),
        "other_label": jsonable_label(other_label),
        "focal_rate_before": current_rate,
        "original_majority_rate": current_rate if policy == "original_fit_majority" else None,
        "focal_rate_after": float(np.mean(selected_labels == focal_label)),
        "untouched_fit_class_counts": [
            {"label": jsonable_label(focal_label), "count": int(focal_idx.size), "role": "focal"},
            {"label": jsonable_label(other_label), "count": int(other_idx.size), "role": "other"},
        ],
        "class_counts_before": {"focal": int(focal_idx.size), "other": int(other_idx.size)},
        "class_counts_after": {
            "focal": int(np.sum(selected_labels == focal_label)),
            "other": int(np.sum(selected_labels == other_label)),
        },
        **policy_report,
        **orientation_report,
    }
    if policy == "encoded_class_1":
        report.update(
            {
                "resolved_target_pos_rate": report["resolved_target_focal_rate"],
                "pos_label": report["focal_label"],
                "pos_rate_before": report["focal_rate_before"],
                "pos_rate_after": report["focal_rate_after"],
                "class_counts_before": {"pos": int(focal_idx.size), "neg": int(other_idx.size)},
                "class_counts_after": {
                    "pos": int(np.sum(selected_labels == focal_label)),
                    "neg": int(np.sum(selected_labels == other_label)),
                },
            }
        )
    return selected, report


def _class_prior_target_rates(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("rebalance_train.target_rates must list every transformed prior")
    rates: list[float] = []
    for raw in value:
        rate = parse_fraction(raw)
        if rate is None or not 0.0 < rate < 1.0:
            raise ValueError(f"rebalance_train.target_rates contains invalid rate {raw!r}")
        rates.append(float(rate))
    return sorted(set(rates))


def _common_feasible_budget(*, pos_count: int, neg_count: int, target_rates: list[float]) -> int:
    budget = min(
        pos_count + neg_count,
        *(math.floor(pos_count / rate) for rate in target_rates),
        *(math.floor(neg_count / (1.0 - rate)) for rate in target_rates),
    )
    while budget >= 2:
        if all(
            (counts := _counts_for_prior(budget, rate))[0] <= pos_count
            and counts[1] <= neg_count
            for rate in target_rates
        ):
            return budget
        budget -= 1
    raise ValueError("rebalance_train cannot form a two-class common budget for the configured rates")


def _counts_for_prior(budget: int, rate: float) -> tuple[int, int]:
    positive = max(1, min(budget - 1, round(budget * rate)))
    return positive, budget - positive


def apply_test_context_majority_downsampling(
    data: Any,
    cfg: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Balance only the test ICL context by downsampling its original majority.

    Fine-tuning train data, validation data, and test queries remain unchanged.
    The majority label is resolved from the original test ICL context without
    inspecting test-query labels.
    """
    if data.task_type != "binary":
        raise ValueError(
            "test-context majority downsampling is a binary-classification intervention, "
            f"got task_type={data.task_type!r}"
        )
    rng = np.random.default_rng(int(seed))
    selected, label_report = _binary_prior_indices(
        np.asarray(data.test_context_y),
        requested=cfg.get("target_focal_rate", 0.5),
        pos_label_cfg=None,
        focal_label_policy="original_fit_majority",
        majority_tie_break=cfg.get("majority_tie_break"),
        budget_policy="retain_original_minority",
        target_rates_cfg=None,
        rng=rng,
    )
    updates: dict[str, Any] = {
        "test_context_X": data.test_context_X.iloc[selected].reset_index(drop=True),
        "test_context_y": np.asarray(data.test_context_y)[selected],
    }
    for field_name in ("test_context_ids", "test_context_components"):
        values = getattr(data, field_name, None)
        if values is not None:
            updates[field_name] = np.asarray(values)[selected]

    report = {
        "stage": "test_context_comparison",
        "seed": int(seed),
        "transformation_setting": "test_icl_context_only",
        "fine_tuning_train_data": "full_original",
        "test_query_data": "original",
        "test_context_rows_before": int(len(data.test_context_X)),
        "test_context_rows_after": int(len(selected)),
        **label_report,
        "focal_label_source": "full_original_test_icl_context",
        "test_labels_used_for_orientation": False,
    }
    metadata = dict(data.metadata)
    metadata["test_context_comparison_policy"] = "balanced_original_majority_downsampled_for_test_icl"
    metadata["test_context_majority_downsampling"] = report
    updates["metadata"] = metadata
    return replace(data, **updates), report



def _resolve_original_fit_majority_label(
    labels: np.ndarray,
    *,
    tie_break: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve one focal label from untouched fit labels without test data."""
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    if unique.size != 2:
        raise ValueError(f"rebalance_train expected two labels, got {unique.size}")
    rule = str(tie_break or "smallest_label_by_type_then_value_v1").strip().lower().replace("-", "_")
    if rule != "smallest_label_by_type_then_value_v1":
        raise ValueError(
            f"rebalance_train.majority_tie_break must be 'smallest_label_by_type_then_value_v1', got {tie_break!r}"
        )
    maximum = int(np.max(counts))
    candidates = [label for label, count in zip(unique.tolist(), counts.tolist(), strict=True) if count == maximum]
    tied = len(candidates) > 1
    focal_label = min(candidates, key=_stable_label_sort_key)
    return focal_label, {
        "focal_label_policy": "original_fit_majority",
        "focal_label_source": "untouched_combined_original_training_validation_fit_pool",
        "majority_tie": tied,
        "majority_tie_break": rule,
        "focal_label_held_fixed_across_conditions": True,
        "test_labels_used_for_orientation": False,
    }



def _stable_label_sort_key(label: Any) -> tuple[str, float | str, str]:
    value = jsonable_label(label)
    if isinstance(value, bool):
        return ("bool", float(value), repr(value))
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("rebalance_train labels must be finite")
        return ("number", numeric, repr(value))
    return ("string", str(value), repr(value))



def _focal_count_with_fixed_other(*, other_count: int, target_rate: float) -> int:
    """Resolve focal rows while retaining every row of the other class."""
    decimal_rate = Decimal(str(target_rate))
    focal = int(
        (Decimal(other_count) * decimal_rate / (Decimal(1) - decimal_rate)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )
    return max(1, focal)
