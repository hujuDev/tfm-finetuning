"""Evaluate live fine-tuned states without writing checkpoint files."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig, InMemoryStateAdapter
from tfm_finetuning.utils.gpu_memory import CudaMemoryTracker

EvaluateEstimatorFn = Callable[..., dict[str, Any]]
LogEvalTimingFn = Callable[[Mapping[str, Any]], None]


class InMemoryEvaluator:
    """Build inference estimators around live weights and evaluate data roles."""

    def __init__(
        self,
        *,
        adapter: InMemoryStateAdapter,
        data: Any,
        common_config: CommonFineTuneConfig,
        device: str | None,
        evaluate_estimator_on_split: EvaluateEstimatorFn,
        log_eval_timing: LogEvalTimingFn,
    ) -> None:
        self.adapter = adapter
        self.data = data
        self.common_config = common_config
        self.device = device
        self.evaluate_estimator_on_split = evaluate_estimator_on_split
        self.log_eval_timing = log_eval_timing
        self.timing_rows: list[dict[str, Any]] | None = None

    def evaluate_state(
        self,
        estimator: Any,
        *,
        state: Mapping[str, Any],
        epoch: int,
        validation_device: Any,
        phase: str,
        eval_source: str,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate a captured state and restore the live estimator afterward."""
        original_state = self.adapter.clone_finetuned_state(estimator)
        try:
            self.adapter.apply_finetuned_state(estimator, state)
            return self.evaluate_current(
                estimator,
                epoch=epoch,
                validation_device=validation_device,
                phase=phase,
                eval_source=eval_source,
            )
        finally:
            self.adapter.apply_finetuned_state(estimator, original_state)

    def evaluate_current(
        self,
        estimator: Any,
        *,
        epoch: int,
        validation_device: Any,
        phase: str,
        eval_source: str,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate validation and test roles from the current live state."""
        return self.evaluate_splits(
            estimator,
            epoch=epoch,
            validation_device=validation_device,
            phase=phase,
            eval_source=eval_source,
            splits=("val", "test"),
        )

    def evaluate_splits(
        self,
        estimator: Any,
        *,
        epoch: int,
        validation_device: Any,
        phase: str,
        eval_source: str,
        splits: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        """Evaluate selected data roles while preserving model train/eval mode."""
        metrics: dict[str, dict[str, Any]] = {}
        split_specs = {
            "val": (
                self.common_config.n_estimators_validation,
                self.data.val_context_X,
                self.data.val_context_y,
                self.data.val_X,
                self.data.val_y,
            ),
            "test": (
                self.common_config.n_estimators_final_inference,
                self.data.test_context_X,
                self.data.test_context_y,
                self.data.test_X,
                self.data.test_y,
            ),
        }
        was_training = _model_was_training(estimator)
        _set_eval_mode(estimator)
        try:
            for split in splits:
                n_estimators, context_X, context_y, query_X, query_y = split_specs[split]
                split_started = time.monotonic()
                timing = {
                    "phase": phase,
                    "split": split,
                    "epoch": int(epoch),
                    "checkpoint_path": None,
                    "eval_source": eval_source,
                    "n_estimators": int(n_estimators),
                    "context_rows": len(context_X),
                    "query_rows": len(query_X),
                }
                memory_device = validation_device if validation_device is not None else self.device
                with CudaMemoryTracker(memory_device) as memory_tracker:
                    estimator_started = time.monotonic()
                    inner = self.adapter.build_in_memory_inference_estimator(
                        estimator,
                        self.data.task_type,
                        memory_device,
                        n_estimators,
                    )
                    timing["estimator_init_seconds"] = time.monotonic() - estimator_started
                    metrics[split] = self.evaluate_estimator_on_split(
                        adapter=self.adapter,
                        estimator=inner,
                        context_X=context_X,
                        context_y=context_y,
                        query_X=query_X,
                        query_y=query_y,
                        task_type=self.data.task_type,
                        timing=timing,
                        target_transform_state=getattr(self.data, "target_transform_state", None),
                    )
                timing.update(memory_tracker.as_report_fields())
                timing["seconds"] = time.monotonic() - split_started
                timing["score"] = metrics[split].get("score")
                if self.timing_rows is not None:
                    self.timing_rows.append(timing)
                self.log_eval_timing(timing)
        finally:
            _restore_model_mode(estimator, was_training=was_training)
        return metrics


def _set_eval_mode(estimator: Any) -> None:
    model = getattr(estimator, "model_", None)
    if hasattr(model, "eval"):
        model.eval()


def _model_was_training(estimator: Any) -> bool:
    model = getattr(estimator, "model_", None)
    training = getattr(model, "training", None)
    if isinstance(training, bool):
        return training
    return getattr(model, "mode", None) == "train"


def _restore_model_mode(estimator: Any, *, was_training: bool) -> None:
    setter = getattr(estimator, "_set_training_mode", None)
    if was_training and callable(setter):
        setter(True)
        return
    model = getattr(estimator, "model_", None)
    if hasattr(model, "train"):
        model.train(was_training)
