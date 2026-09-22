from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from finetuning.in_memory_evaluation import (
    EvaluateEstimatorFn,
    InMemoryEvaluator,
    LogEvalTimingFn,
)
from finetuning.models.base import (
    CapturedFinetunedStatesAdapter,
    CommonFineTuneConfig,
    FoundationModelAdapter,
    InMemoryStateAdapter,
)

if TYPE_CHECKING:
    from pathlib import Path

class InMemoryFitController:
    """Keep TabICL fine-tune eval/checkpoint state in memory instead of epoch files.

    Example:
        >>> callable(InMemoryFitController.maybe_create)
        True
    """

    def __init__(
        self,
        *,
        adapter: InMemoryStateAdapter,
        data: Any,
        common_config: CommonFineTuneConfig,
        device: str | None,
        checkpoint_mode: str,
        test_eval_interval: int,
        evaluate_estimator_on_split: EvaluateEstimatorFn,
        log_eval_timing: LogEvalTimingFn,
    ) -> None:
        self.adapter = adapter
        self.data = data
        self.common_config = common_config
        self.device = device
        self.checkpoint_mode = checkpoint_mode
        self.test_eval_interval = int(test_eval_interval)
        self._evaluator = InMemoryEvaluator(
            adapter=adapter,
            data=data,
            common_config=common_config,
            device=device,
            evaluate_estimator_on_split=evaluate_estimator_on_split,
            log_eval_timing=log_eval_timing,
        )
        self.original_save_interval: Any = None
        self.validation_rows: list[dict[str, Any]] = []
        self.validation_eval_seconds = 0.0
        self.periodic_rows: list[dict[str, Any]] = []
        self.periodic_eval_seconds = 0.0
        self.best_state: Mapping[str, Any] | None = None
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, dict[str, Any]] | None = None
        self.best_metrics_source: str | None = None
        self.best_eval_seconds = 0.0
        self.best_eval_count = 0
        self.final_state: Mapping[str, Any] | None = None
        self.final_metric: float | None = None
        self.final_epoch: int | None = None
        self.final_metrics: dict[str, dict[str, Any]] | None = None
        self.final_eval_seconds = 0.0
        self.final_eval_count = 0
        self.export_source: str | None = None
        self.exported_checkpoint_path: Path | None = None

    @classmethod
    def maybe_create(
        cls,
        *,
        adapter: FoundationModelAdapter,
        data: Any,
        common_config: CommonFineTuneConfig,
        device: str | None,
        checkpoint_mode: str,
        test_eval_interval: int,
        keep_intermediate_checkpoints: bool,
        evaluate_estimator_on_split: EvaluateEstimatorFn,
        log_eval_timing: LogEvalTimingFn,
    ) -> InMemoryFitController | None:
        """Create a controller when the adapter can snapshot and restore fine-tuned state."""
        if keep_intermediate_checkpoints:
            return None
        if not isinstance(adapter, InMemoryStateAdapter):
            return None
        return cls(
            adapter=adapter,
            data=data,
            common_config=common_config,
            device=device,
            checkpoint_mode=checkpoint_mode,
            test_eval_interval=test_eval_interval,
            evaluate_estimator_on_split=evaluate_estimator_on_split,
            log_eval_timing=log_eval_timing,
        )

    def prepare_estimator(self, estimator: Any) -> None:
        """Disable disk checkpoint emission on an estimator before fitting starts."""
        self.original_save_interval = getattr(estimator, "save_interval", None)
        if self.original_save_interval is not None:
            estimator.save_interval = 0
        estimator._foundation_suppress_adapter_checkpoints = True
        estimator._foundation_in_memory_controller = self
        logger.info(
            (
                "Foundation-model in-memory fit controller enabled: "
                "checkpoint_mode={} test_eval_interval={} original_save_interval={} "
                "active_save_interval={}"
            ),
            self.checkpoint_mode,
            self.test_eval_interval,
            self.original_save_interval,
            getattr(estimator, "save_interval", None),
        )

    def bind_timing_rows(self, timing_rows: list[dict[str, Any]]) -> None:
        """Attach the caller-owned timing row list used in the final report."""
        self._evaluator.timing_rows = timing_rows

    @property
    def suppress_disk_checkpoints(self) -> bool:
        """Return whether this controller expects adapter disk checkpoints to be suppressed."""
        return True

    def after_validation(
        self,
        estimator: Any,
        *,
        epoch: int | None,
        primary: Any,
        validation_device: Any,
    ) -> None:
        """Capture best/final states and optional periodic metrics after validation."""
        primary_value = _coerce_float(primary)
        periodic_metrics: dict[str, dict[str, Any]] | None = None
        validation_metrics: dict[str, Any] | None = None
        if epoch is not None:
            self._capture_final(estimator, epoch=epoch, primary=primary_value)
        if epoch is not None and self.test_eval_interval > 0 and epoch % self.test_eval_interval == 0:
            periodic_metrics = self._run_periodic_eval(estimator, epoch=epoch, validation_device=validation_device)
            validation_metrics = periodic_metrics.get("val")
        elif epoch is not None:
            validation_metrics = self._run_validation_eval(
                estimator,
                epoch=epoch,
                validation_device=validation_device,
            )
        selection_score = _periodic_validation_score({"val": validation_metrics})
        if selection_score is None:
            selection_score = primary_value
        improved = False
        if epoch is not None:
            improved = self._maybe_capture_best(estimator, epoch=epoch, primary=selection_score)
        if epoch is not None and improved and periodic_metrics is not None:
            self.best_metrics = periodic_metrics
            self.best_metrics_source = "periodic_in_memory"
        elif epoch is not None and improved and validation_metrics is not None:
            # Checkpoint search may inspect validation only. The selected best
            # state is evaluated on test once, after fitting has finished.
            self.best_metrics = {"val": validation_metrics}
            self.best_metrics_source = "in_memory_validation"

    def after_fit(self, estimator: Any, *, validation_device: Any) -> None:
        """Consume adapter-captured best/final states when validation callbacks did not run."""
        captured = self._capture_adapter_finetuned_states(estimator)
        final = captured.get("final")
        if isinstance(final, Mapping):
            state = final.get("state")
            epoch = _coerce_int(final.get("epoch"))
            if state is not None and self.final_state is None:
                self.final_state = state
                self.final_epoch = epoch
                self.final_metric = _coerce_float(final.get("metric"))

        best = captured.get("best")
        if isinstance(best, Mapping):
            state = best.get("state")
            epoch = _coerce_int(best.get("epoch"))
            if state is not None and self.best_state is None:
                self.best_state = state
                self.best_epoch = epoch
                self.best_metric = _coerce_float(best.get("metric"))
            if state is not None and self.best_metrics is None:
                epoch = epoch or self.final_epoch or int(self.common_config.epochs)
                self.best_epoch = epoch
                if not self.periodic_rows:
                    self._run_state_best_eval(
                        estimator,
                        state=state,
                        epoch=epoch,
                        validation_device=validation_device,
                    )
        if (
            self.best_state is not None
            and not self.periodic_rows
            and (not isinstance(self.best_metrics, Mapping) or "test" not in self.best_metrics)
        ):
            epoch = self.best_epoch or self.final_epoch or int(self.common_config.epochs)
            self._run_state_best_eval(
                estimator,
                state=self.best_state,
                epoch=epoch,
                validation_device=validation_device,
            )

    def _capture_adapter_finetuned_states(self, estimator: Any) -> Mapping[str, Any]:
        if not isinstance(self.adapter, CapturedFinetunedStatesAdapter):
            return {}
        payload = self.adapter.capture_finetuned_states(estimator)
        return payload if isinstance(payload, Mapping) else {}

    def _maybe_capture_best(self, estimator: Any, *, epoch: int | None, primary: float | None) -> bool:
        if primary is None or np.isnan(primary):
            return False
        improved = self.best_metric is None
        metric_improved = getattr(estimator, "_metric_improved", None)
        if self.best_metric is not None:
            improved = (
                bool(metric_improved(primary, self.best_metric))
                if callable(metric_improved)
                else primary > self.best_metric
            )
        if not improved:
            return False
        started_at = time.monotonic()
        self.best_state = self.adapter.clone_finetuned_state(estimator)
        self.best_metric = primary
        self.best_epoch = epoch
        logger.debug(
            "Foundation-model in-memory best state captured: epoch={} primary={} seconds={:.3f}",
            epoch,
            primary,
            time.monotonic() - started_at,
        )
        return True

    def _capture_final(self, estimator: Any, *, epoch: int, primary: float | None) -> None:
        started_at = time.monotonic()
        self.final_state = self.adapter.clone_finetuned_state(estimator)
        self.final_metric = primary
        self.final_epoch = int(epoch)
        logger.debug(
            "Foundation-model in-memory final state captured: epoch={} primary={} seconds={:.3f}",
            epoch,
            primary,
            time.monotonic() - started_at,
        )

    def _run_periodic_eval(self, estimator: Any, *, epoch: int, validation_device: Any) -> dict[str, dict[str, Any]]:
        started_at = time.monotonic()
        metrics = self._evaluator.evaluate_splits(
            estimator,
            epoch=epoch,
            validation_device=validation_device,
            phase="periodic_in_memory",
            eval_source="in_memory",
            splits=("val", "test"),
        )
        elapsed = time.monotonic() - started_at
        self.periodic_eval_seconds += elapsed
        self.periodic_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint_path": None,
                "eval_source": "in_memory",
                "runtime_seconds": elapsed,
                "metrics": metrics,
            }
        )
        logger.debug(
            "Foundation-model periodic in-memory eval finished: epoch={} seconds={:.3f}",
            epoch,
            elapsed,
        )
        return metrics

    def _run_validation_eval(self, estimator: Any, *, epoch: int, validation_device: Any) -> dict[str, Any]:
        started_at = time.monotonic()
        metrics = self._evaluator.evaluate_splits(
            estimator,
            epoch=epoch,
            validation_device=validation_device,
            phase="validation_in_memory",
            eval_source="in_memory_validation",
            splits=("val",),
        )
        elapsed = time.monotonic() - started_at
        val_metrics = metrics["val"]
        self.validation_eval_seconds += elapsed
        self.validation_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint_path": None,
                "eval_source": "in_memory_validation",
                "runtime_seconds": elapsed,
                "metrics": {"val": val_metrics},
            }
        )
        logger.debug(
            "Foundation-model validation in-memory eval finished: epoch={} seconds={:.3f}",
            epoch,
            elapsed,
        )
        return val_metrics

    def _run_state_best_eval(self, estimator: Any, *, state: Any, epoch: int, validation_device: Any) -> None:
        started_at = time.monotonic()
        metrics = self._evaluator.evaluate_state(
            estimator,
            state=state,
            epoch=epoch,
            validation_device=validation_device,
            phase="selected_checkpoint",
            eval_source="in_memory_best",
        )
        elapsed = time.monotonic() - started_at
        self.best_eval_seconds += elapsed
        self.best_eval_count += 1
        self.best_metrics = metrics
        self.best_metrics_source = "in_memory_best"
        logger.debug(
            "Foundation-model best-validation captured-state eval finished: epoch={} seconds={:.3f}",
            epoch,
            elapsed,
        )

    def evaluate_final(self, estimator: Any, *, epoch: int, validation_device: Any) -> dict[str, dict[str, Any]]:
        """Evaluate the live final estimator without exporting a checkpoint file."""
        started_at = time.monotonic()
        if self.final_state is not None:
            metrics = self._evaluator.evaluate_state(
                estimator,
                state=self.final_state,
                epoch=epoch,
                validation_device=validation_device,
                phase="final_in_memory",
                eval_source="in_memory_final",
            )
        else:
            metrics = self._evaluator.evaluate_current(
                estimator,
                epoch=epoch,
                validation_device=validation_device,
                phase="final_in_memory",
                eval_source="in_memory_final",
            )
        elapsed = time.monotonic() - started_at
        self.final_eval_seconds += elapsed
        self.final_eval_count += 1
        self.final_metrics = metrics
        logger.debug(
            "Foundation-model final in-memory eval finished: epoch={} seconds={:.3f}",
            epoch,
            elapsed,
        )
        return metrics

    def export_checkpoint(self, estimator: Any, output_dir: Path, *, checkpoint_mode: str) -> Path:
        """Export the selected in-memory state as a project-owned checkpoint package."""
        if checkpoint_mode == "final":
            state = self.final_state
            epoch = self.final_epoch
            metric = self.final_metric
            source = "in_memory_final_state"
        else:
            state = self.best_state
            epoch = self.best_epoch
            metric = self.best_metric
            source = "in_memory_best_state"
        if state is None:
            raise RuntimeError(
                f"Cannot export {checkpoint_mode!r} checkpoint because no in-memory state was captured."
            )
        started_at = time.monotonic()
        checkpoint_path = self.adapter.export_finetuned_state(
            estimator,
            state,
            output_dir,
            epoch=epoch,
            metric=metric,
            checkpoint_mode=checkpoint_mode,
        )
        self.export_source = source
        self.exported_checkpoint_path = checkpoint_path
        logger.info(
            (
                "Foundation-model in-memory checkpoint exported: mode={} source={} "
                "epoch={} seconds={:.3f} checkpoint={}"
            ),
            checkpoint_mode,
            source,
            epoch,
            time.monotonic() - started_at,
            checkpoint_path,
        )
        return checkpoint_path

    def export_selected_checkpoint(self, estimator: Any, output_dir: Path) -> Path:
        """Export the configured selected checkpoint mode."""
        return self.export_checkpoint(estimator, output_dir, checkpoint_mode=self.checkpoint_mode)

    def summary(self) -> dict[str, Any]:
        """Return report metadata for the in-memory checkpoint/evaluation policy."""
        return {
            "enabled": True,
            "intermediate_checkpoints": "suppressed",
            "periodic_eval_source": "in_memory" if self.test_eval_interval > 0 else "disabled",
            "original_save_interval": self.original_save_interval,
            "active_save_interval": 0 if self.original_save_interval is not None else None,
            "validation_eval_seconds": self.validation_eval_seconds,
            "validation_eval_rows": len(self.validation_rows),
            "periodic_eval_seconds": self.periodic_eval_seconds,
            "periodic_eval_rows": len(self.periodic_rows),
            "best_eval_seconds": self.best_eval_seconds,
            "best_eval_count": self.best_eval_count,
            "best_metrics_source": self.best_metrics_source,
            "best_epoch": self.best_epoch,
            "final_epoch": self.final_epoch,
            "final_eval_seconds": self.final_eval_seconds,
            "final_eval_count": self.final_eval_count,
            "export_source": self.export_source,
            "exported_checkpoint_path": str(self.exported_checkpoint_path) if self.exported_checkpoint_path else None,
        }


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    with contextlib.suppress(Exception):
        return float(value)
    return None


def _periodic_validation_score(metrics: Mapping[str, Any] | None) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    val_metrics = metrics.get("val")
    if not isinstance(val_metrics, Mapping):
        return None
    return _coerce_float(val_metrics.get("score"))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    with contextlib.suppress(Exception):
        return int(value)
    return None
