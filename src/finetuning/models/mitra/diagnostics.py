"""Mitra epoch instrumentation and local-trainer diagnostics."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from finetuning.models.mitra.checkpoints import clone_torch_state_dict


@contextmanager
def capture_epoch_logging(estimator: Any) -> Iterator[None]:
    """Bridge Mitra's validation callbacks to the shared in-memory epoch logger."""
    logger_obj = getattr(estimator, "experiment_logger", None)
    if logger_obj is None:
        yield
        return
    module = importlib.import_module("autogluon.tabular.models.mitra.sklearn_interface")
    trainer_cls = module.TrainerFinetune
    callbacks = importlib.import_module("autogluon.tabular.models.mitra._internal.core.callbacks")
    checkpoint_cls = callbacks.Checkpoint
    original = trainer_cls.log_metrics
    original_checkpoint_call = checkpoint_cls.__call__
    original_set_to_best = checkpoint_cls.set_to_best

    def wrapped(trainer: Any, epoch: int, metrics_train: Any, metrics_valid: Any) -> Any:
        logger_obj.log_epoch(epoch_payload(epoch, metrics_train, metrics_valid), step=int(epoch))
        checkpoint = getattr(trainer, "checkpoint", None)
        if checkpoint is not None:
            checkpoint._foundation_current_epoch = int(epoch)
        return original(trainer, epoch, metrics_train, metrics_valid)

    def wrapped_checkpoint_call(checkpoint: Any, net: Any, loss: float) -> Any:
        previous = getattr(checkpoint, "curr_best_loss", None)
        result = original_checkpoint_call(checkpoint, net, loss)
        with contextlib.suppress(Exception):
            if previous is None or float(loss) < float(previous):
                checkpoint._foundation_best_epoch = getattr(checkpoint, "_foundation_current_epoch", None)
        return result

    def wrapped_set_to_best(checkpoint: Any, net: Any) -> Any:
        checkpoint._foundation_final_epoch = getattr(checkpoint, "_foundation_current_epoch", None)
        checkpoint._foundation_final_state = clone_torch_state_dict(net.state_dict())
        checkpoint._foundation_best_state = clone_torch_state_dict(getattr(checkpoint, "best_model", {}))
        return original_set_to_best(checkpoint, net)

    trainer_cls.log_metrics = wrapped
    checkpoint_cls.__call__ = wrapped_checkpoint_call
    checkpoint_cls.set_to_best = wrapped_set_to_best
    try:
        yield
    finally:
        trainer_cls.log_metrics = original
        checkpoint_cls.__call__ = original_checkpoint_call
        checkpoint_cls.set_to_best = original_set_to_best


def epoch_payload(epoch: int, metrics_train: Any, metrics_valid: Any) -> dict[str, float]:
    train_mse = _metric_value(metrics_train, "mse")
    train_log_loss = _metric_value(metrics_train, "log_loss")
    payload: dict[str, float] = {"train/epoch": float(int(epoch) - 1)}
    if train_log_loss is not None:
        payload["train/mean_loss"] = train_log_loss
        payload["train/log_loss"] = train_log_loss
    if train_mse is not None:
        payload["train/mean_loss"] = train_mse
        payload["train/mse"] = train_mse
    for metric in ("log_loss", "accuracy", "auc", "f1", "rmse", "mse", "mae", "r2"):
        train_value = _metric_value(metrics_train, metric)
        val_value = _metric_value(metrics_valid, metric)
        target_metric = "roc_auc" if metric == "auc" else metric
        if train_value is not None:
            payload[f"train/{target_metric}"] = train_value
        if val_value is not None:
            payload[f"val/{target_metric}"] = val_value
    return payload


def _metric_value(metrics: Any, metric_name: str) -> float | None:
    values = getattr(metrics, "metrics", None)
    if not isinstance(values, dict):
        return None
    target = metric_name.lower()
    for key, value in values.items():
        key_name = str(getattr(key, "value", getattr(key, "name", key))).lower()
        if key_name == target:
            return float(value)
    return None


def final_logged_epoch(estimator: Any) -> int:
    records = getattr(getattr(estimator, "experiment_logger", None), "epoch_records", None)
    if records:
        return max(int(float(record.get("train/epoch", 0))) + 1 for record in records)
    return max(1, int(getattr(estimator, "fine_tune_steps", 1) or 1))


def best_logged_epoch(estimator: Any) -> int | None:
    epochs = [
        getattr(getattr(trainer, "checkpoint", None), "_foundation_best_epoch", None)
        for trainer in getattr(estimator, "trainers", [])
    ]
    numeric = [int(epoch) for epoch in epochs if epoch is not None]
    return max(set(numeric), key=numeric.count) if numeric else None


def local_trainer_diagnostics(estimator: Any) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for trainer in getattr(estimator, "trainers", []):
        as_payload = getattr(getattr(trainer, "local_diagnostics", None), "as_payload", None)
        if callable(as_payload):
            payloads.append(as_payload())
    if not payloads:
        return {}
    aggregate: dict[str, Any] = dict(payloads[0])
    aggregate["n_trainers"] = len(payloads)
    for key in (
        "epochs_completed",
        "optimizer_steps_completed",
        "skipped_chunks",
        "query_size_clamped_count",
        "rare_class_fallback_count",
        "singleton_chunk_count",
        "grad_clip_applied_steps",
    ):
        values: list[int] = []
        for item in payloads:
            value = item.get(key)
            if value is not None:
                values.append(int(value))
        if values:
            aggregate[key] = sum(values)
    grad_norms: list[float] = []
    for item in payloads:
        value = item.get("grad_norm_max")
        if value is not None:
            grad_norms.append(float(value))
    if grad_norms:
        aggregate["grad_norm_max"] = max(grad_norms)
    aggregate["trainer_diagnostics"] = payloads
    return aggregate
