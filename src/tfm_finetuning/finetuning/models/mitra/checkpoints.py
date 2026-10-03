"""Mitra inference-checkpoint serialization and in-memory state handling."""

from __future__ import annotations

import contextlib
import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from tfm_finetuning.finetuning.models.mitra.common import MITRA_INTERFACE_LOCAL_CHUNKED

MITRA_PORTABLE_CHECKPOINT_FORMAT = "thesis_mitra_portable_weights"
MITRA_PORTABLE_CHECKPOINT_VERSION = 1

_SOURCE_BOUND_ESTIMATOR_ATTRS = (
    "X",
    "y",
    "X_val",
    "y_val",
    "classes_",
    "_tfm_mitra_feature_preprocessor",
    "_tfm_in_memory_controller",
    "_tfm_initial_checkpoint",
    "_tfm_suppress_adapter_checkpoints",
    "experiment_logger",
)


def write_checkpoint_package(estimator: Any, path: Path) -> None:
    """Write a reset-safe Mitra package when local-chunked state is available.

    The portable package deliberately omits dataset-bound labels, feature
    preprocessors, optimizer/scheduler state, and fitted ICL context. The original upstream training interface stores its estimator directly.
    That representation cannot be used for cross-dataset warm starts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _portable_checkpoint_payload(estimator)
    with path.open("wb") as handle:
        pickle.dump(payload if payload is not None else estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_checkpoint_with_trainer_states(estimator: Any, path: Path, states: list[dict[str, Any]]) -> None:
    original_states = [clone_torch_state_dict(trainer.model.state_dict()) for trainer in estimator.trainers]
    try:
        apply_trainer_states(estimator, states)
        write_checkpoint_package(estimator, path)
    finally:
        apply_trainer_states(estimator, original_states)


def write_checkpoint_with_finetuned_state(
    estimator: Any,
    path: Path,
    state: dict[str, Any],
    *,
    epoch: int,
    checkpoint_mode: str,
) -> None:
    trainer_states = state.get("trainer_states")
    if not isinstance(trainer_states, list) or not trainer_states:
        raise RuntimeError("Mitra in-memory checkpoint export requires captured trainer states.")
    original_state = clone_estimator_state(estimator)
    try:
        apply_estimator_state(estimator, state)
        estimator._tfm_checkpoint_epoch = int(epoch)
        estimator._tfm_checkpoint_mode = str(checkpoint_mode)
        write_checkpoint_package(estimator, path)
    finally:
        apply_estimator_state(estimator, original_state)


def read_checkpoint_package(path: Path, *, device: str | None = None) -> Any:
    """Load a checkpoint from the original or adapted Mitra training interface."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not is_portable_checkpoint_payload(payload):
        return payload
    estimator = payload["estimator_template"]
    if device is not None:
        estimator.device = str(device)
    states = payload["trainer_states"]
    metadata = dict(payload["metadata"])
    from tfm_finetuning.finetuning.models.mitra.configured import restore_portable_mitra_ensemble

    restore_portable_mitra_ensemble(
        estimator,
        states,
        task=metadata["task"],
        dim_output=int(metadata["dim_output"]),
        n_classes=int(metadata["n_classes"]),
    )
    estimator._tfm_portable_checkpoint_metadata = metadata
    return estimator


def read_checkpoint_for_finetuning(
    path: Path,
    *,
    task_type: str,
    expected_ensemble_size: int,
) -> list[dict[str, Any]]:
    """Return reset-safe learned weights for a fresh Mitra training stage.

    Only local-chunked portable packages are accepted. The caller must request
    the exact saved ensemble size so no member is silently dropped or repeated.
    """
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not is_portable_checkpoint_payload(payload):
        raise RuntimeError(
            f"Mitra fine-tuning warm starts require a reset-safe portable checkpoint; non-portable package received: {path}."
        )
    metadata = payload["metadata"]
    if metadata.get("finetune_interface") != MITRA_INTERFACE_LOCAL_CHUNKED:
        raise RuntimeError("Mitra warm starts are supported only for the local_chunked interface.")
    checkpoint_family = str(metadata.get("task_family"))
    requested_family = _task_family_from_value(task_type)
    if checkpoint_family != requested_family:
        raise RuntimeError(
            "Mitra checkpoint task family is incompatible with the requested stage "
            f"({checkpoint_family!r} != {requested_family!r})."
        )
    states = list(payload.get("trainer_states") or [])
    saved_size = int(metadata.get("ensemble_size", len(states)))
    expected_size = int(expected_ensemble_size)
    if saved_size != len(states) or saved_size != expected_size:
        raise RuntimeError(
            "Mitra warm starts require an exact ensemble match "
            f"(checkpoint={saved_size}, states={len(states)}, requested={expected_size})."
        )
    return [clone_torch_state_dict(state) for state in states]


def is_portable_checkpoint_payload(payload: Any) -> bool:
    """Return whether ``payload`` follows the reset-safe Mitra schema."""
    return bool(
        isinstance(payload, dict)
        and payload.get("format") == MITRA_PORTABLE_CHECKPOINT_FORMAT
        and payload.get("schema_version") == MITRA_PORTABLE_CHECKPOINT_VERSION
    )


def clone_torch_state_dict(state_dict: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in dict(state_dict).items()
    }


def clone_current_trainer_states(estimator: Any) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for trainer in getattr(estimator, "trainers", []):
        model = getattr(trainer, "model", None)
        if model is not None and hasattr(model, "state_dict"):
            states.append(clone_torch_state_dict(model.state_dict()))
    return states


def clone_estimator_state(estimator: Any) -> dict[str, Any]:
    """Capture the Mitra inference state; optimizer and RNG state are intentionally excluded."""
    state: dict[str, Any] = {
        "trainer_states": clone_current_trainer_states(estimator),
        "n_estimators": getattr(estimator, "n_estimators", None),
        "n_estimators_inference": getattr(estimator, "n_estimators_inference", None),
    }
    if hasattr(estimator, "classes_"):
        classes = estimator.classes_
        state["classes_"] = None if classes is None else np.asarray(classes).copy()
    preprocessor = getattr(estimator, "_tfm_mitra_feature_preprocessor", None)
    if preprocessor is not None:
        state["_tfm_mitra_feature_preprocessor"] = copy.deepcopy(preprocessor)
    return state


def estimator_state_with_trainer_states(estimator: Any, trainer_states: list[dict[str, Any]]) -> dict[str, Any]:
    state = clone_estimator_state(estimator)
    state["trainer_states"] = [clone_torch_state_dict(trainer_state) for trainer_state in trainer_states]
    return state


def apply_estimator_state(estimator: Any, state: dict[str, Any]) -> None:
    trainer_states = state.get("trainer_states")
    if isinstance(trainer_states, list):
        apply_trainer_states(estimator, trainer_states)
    if "classes_" in state:
        classes = state["classes_"]
        estimator.classes_ = None if classes is None else np.asarray(classes).copy()
    if "_tfm_mitra_feature_preprocessor" in state:
        estimator._tfm_mitra_feature_preprocessor = copy.deepcopy(state["_tfm_mitra_feature_preprocessor"])
    for attr in ("n_estimators", "n_estimators_inference"):
        if attr in state and state[attr] is not None:
            setattr(estimator, attr, int(state[attr]))


def apply_trainer_states(estimator: Any, states: list[dict[str, Any]], *, exact: bool = False) -> None:
    trainers = list(getattr(estimator, "trainers", []))
    if exact and len(trainers) != len(states):
        raise RuntimeError(
            "Mitra checkpoint ensemble does not match the estimator template "
            f"(trainers={len(trainers)}, states={len(states)})."
        )
    for trainer, state in zip(trainers, states, strict=exact):
        trainer.model.load_state_dict(state)


def captured_trainer_states(estimator: Any, attr: str) -> list[dict[str, Any]]:
    states = [
        getattr(getattr(trainer, "checkpoint", None), attr, None) for trainer in getattr(estimator, "trainers", [])
    ]
    return [clone_torch_state_dict(state) for state in states if isinstance(state, dict)]


def epoch_checkpoint_sort_key(path: Path) -> tuple[int, str]:
    return epoch_from_checkpoint_name(path), path.name


def epoch_from_checkpoint_name(path: Path) -> int:
    stem = path.stem
    if stem.startswith("epoch"):
        with contextlib.suppress(ValueError):
            return int(stem.removeprefix("epoch"))
    return -1


def _portable_checkpoint_payload(estimator: Any) -> dict[str, Any] | None:
    interface = getattr(estimator, "tfm_mitra_finetune_interface", None)
    trainers = list(getattr(estimator, "trainers", []) or [])
    states = clone_current_trainer_states(estimator)
    if interface != MITRA_INTERFACE_LOCAL_CHUNKED:
        return None
    if not trainers or not states:
        if getattr(estimator, "task", None) is not None:
            raise RuntimeError("A fitted local_chunked Mitra estimator has no portable trainer weights.")
        return None
    if len(trainers) != len(states):
        raise RuntimeError(
            "Cannot write a portable Mitra checkpoint with incomplete trainer state "
            f"(trainers={len(trainers)}, states={len(states)})."
        )
    template = _inference_template_without_source_state(estimator)
    task_family = _task_family_from_value(getattr(estimator, "task", estimator.__class__.__name__))
    trainer = trainers[0]
    cfg = getattr(trainer, "cfg", None)
    hyperparams = getattr(cfg, "hyperparams", {})
    dim_output = hyperparams.get("dim_output")
    n_classes = getattr(trainer, "n_classes", None)
    task = getattr(estimator, "task", None)
    if cfg is None or dim_output is None or n_classes is None or task is None:
        raise RuntimeError("A fitted local_chunked Mitra estimator lacks portable architecture metadata.")
    return {
        "format": MITRA_PORTABLE_CHECKPOINT_FORMAT,
        "schema_version": MITRA_PORTABLE_CHECKPOINT_VERSION,
        "metadata": {
            "task_family": task_family,
            "task": task,
            "dim_output": int(dim_output),
            "n_classes": int(n_classes),
            "ensemble_size": len(states),
            "finetune_interface": MITRA_INTERFACE_LOCAL_CHUNKED,
            "portable_state": "model_weights_only",
            "optimizer_state_included": False,
            "source_preprocessing_included": False,
            "source_classes_included": False,
        },
        "estimator_template": template,
        "trainer_states": states,
    }


def _inference_template_without_source_state(estimator: Any) -> Any:
    template = copy.copy(estimator)
    for attr in _SOURCE_BOUND_ESTIMATOR_ATTRS:
        _remove_or_clear_attr(template, attr)
    template._tfm_initial_trainer_states = None
    template.trainers = []
    return template


def _remove_or_clear_attr(target: Any, attr: str) -> None:
    if not hasattr(target, attr):
        return
    with contextlib.suppress(Exception):
        delattr(target, attr)
        return
    with contextlib.suppress(Exception):
        setattr(target, attr, None)


def _task_family_from_value(value: Any) -> str:
    normalized = str(getattr(value, "value", value)).strip().lower()
    if normalized in {"binary", "multiclass", "classification", "classifier"} or "classifier" in normalized:
        return "classification"
    if normalized in {"regression", "regressor"} or "regressor" in normalized:
        return "regression"
    raise RuntimeError(f"Cannot resolve Mitra checkpoint task family from {value!r}.")
