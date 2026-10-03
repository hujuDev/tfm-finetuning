"""Config resolution helpers for foundation-model fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tfm_finetuning.finetuning.datasets.loading import _coerce_bool_config
from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.models.registry import FoundationModelSpec
from tfm_finetuning.finetuning.run import shared as _pipeline_shared
from tfm_finetuning.utils.config import get_nested


def resolve_foundation_model_id(config: Mapping[str, Any]) -> str:
    """Resolve the registry model id from a generated finetune config.

    Example:
        >>> resolve_foundation_model_id({"foundation_model": {"model_id": "tabicl_v2"}})
        'tabicl_v2'
    """
    for dotted_key in ("foundation_model.model_id", "model.foundation_model_id", "model.id"):
        value = get_nested(config, dotted_key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    raise SystemExit("foundation_model.model_id is required for the unified foundation-model runner.")


def resolve_configured_foundation_model_spec(
    spec: FoundationModelSpec,
    config: Mapping[str, Any],
) -> FoundationModelSpec:
    """Apply checkpoint identifiers from the generated config to a registry spec.

    Example:
        >>> isinstance(resolve_configured_foundation_model_spec, object)
        True
    """
    classifier_checkpoint = get_nested(config, "foundation_model.classifier_checkpoint")
    regressor_checkpoint = get_nested(config, "foundation_model.regressor_checkpoint")
    classifier_text = str(classifier_checkpoint).strip() if classifier_checkpoint is not None else ""
    regressor_text = str(regressor_checkpoint).strip() if regressor_checkpoint is not None else ""
    if not classifier_text and not regressor_text:
        return spec

    version_payload = dict(spec.version_payload)
    if classifier_text:
        version_payload["checkpoint_identifier_classifier"] = classifier_text
    if regressor_text:
        version_payload["checkpoint_identifier_regressor"] = regressor_text
    return replace(
        spec,
        classifier_checkpoint=classifier_text or spec.classifier_checkpoint,
        regressor_checkpoint=regressor_text or spec.regressor_checkpoint,
        version_payload=version_payload,
    )


def resolve_common_finetune_config(config: Mapping[str, Any]) -> CommonFineTuneConfig:
    """Resolve common foundation-model fine-tuning settings.

    Example:
        >>> cfg = resolve_common_finetune_config({"foundation_model": {"epochs": 2}})
        >>> cfg.epochs
        2
    """
    payload = dict(config.get("foundation_model") if isinstance(config.get("foundation_model"), Mapping) else {})
    if "n_epochs" in config and "epochs" not in payload:
        payload["epochs"] = config["n_epochs"]
    return CommonFineTuneConfig.from_mapping(payload)


def resolve_test_eval_interval_epochs(config: Mapping[str, Any]) -> int:
    """Resolve the requested periodic test-evaluation cadence.

    Example:
        >>> resolve_test_eval_interval_epochs({"foundation_model": {"test_eval_interval_epochs": 1}})
        1
    """
    value = get_nested(config, "foundation_model.test_eval_interval_epochs")
    if value is None:
        value = get_nested(config, "evaluation.test_eval_interval_epochs")
    if value is None:
        return 0
    try:
        interval = int(value)
    except Exception as exc:
        raise SystemExit(f"foundation_model.test_eval_interval_epochs must be an integer, got {value!r}.") from exc
    if interval < 0:
        raise SystemExit("foundation_model.test_eval_interval_epochs must be >= 0.")
    return interval




def _resolve_checkpoint_mode(config: Mapping[str, Any]) -> str:
    value = get_nested(config, "foundation_model.final_eval_checkpoint")
    text = str(value or "best").strip().lower()
    if text in {"best", "final", "latest", "last", "final_epoch"}:
        return "final" if text in {"final", "latest", "last", "final_epoch"} else "best"
    raise SystemExit("foundation_model.final_eval_checkpoint must be one of: best, final.")


def _resolve_baseline_only(config: Mapping[str, Any]) -> bool:
    for dotted_key in ("foundation_model.baseline_only",):
        value = get_nested(config, dotted_key)
        if value is not None:
            return _coerce_bool_config(value, dotted_key=dotted_key)
    return False


def _resolve_keep_intermediate_checkpoints(config: Mapping[str, Any]) -> bool:
    """Return whether TabICL should keep every upstream checkpoint file.

    Example:
        >>> _resolve_keep_intermediate_checkpoints({"foundation_model": {"keep_intermediate_checkpoints": True}})
        True
    """
    value = get_nested(config, "foundation_model.keep_intermediate_checkpoints")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise SystemExit("foundation_model.keep_intermediate_checkpoints must be a boolean.")


def _resolve_save_best_val_checkpoint(config: Mapping[str, Any]) -> bool:
    """Return whether the best-validation checkpoint should be retained as its own artifact.

    Example:
        >>> _resolve_save_best_val_checkpoint({"foundation_model": {"save_best_val_checkpoint": True}})
        True
    """
    value = get_nested(config, "foundation_model.save_best_val_checkpoint")
    if value is None:
        return False
    return _coerce_bool_config(value, dotted_key="foundation_model.save_best_val_checkpoint")


def _prepare_config(config: dict[str, Any]) -> tuple[int, int, int, str | None]:
    return _pipeline_shared.apply_split_seed(config)
