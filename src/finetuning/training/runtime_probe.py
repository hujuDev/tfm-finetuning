"""Runtime instrumentation for in-memory foundation-model fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from finetuning.training.in_memory import InMemoryFitController


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _checkpoint_size_mib(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def _validation_device_from_args(args: tuple[Any, ...]) -> Any:
    if not args:
        return None
    first = args[0]
    if isinstance(first, Mapping):
        return first.get("device")
    return first


class _FitRuntimeProbe:
    def __init__(self, controller: InMemoryFitController | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._validation_index = 0
        self._validation_context: dict[str, Any] | None = None
        self._controller = controller

    def install(self, estimator: Any) -> _FitRuntimeProbe:
        self._wrap_validation(estimator)
        self._wrap_checkpoint_save(estimator)
        self._wrap_inner_builder(estimator)
        return self

    def _record_event(self, *, event: str, started_at: float, **fields: Any) -> None:
        seconds = time.monotonic() - started_at
        payload = {"event": event, "seconds": seconds, **fields}
        self.events.append(payload)

    def _wrap_validation(self, estimator: Any) -> None:
        target_name = "_validate_current_model"
        original = getattr(estimator, target_name, None)
        if not callable(original):
            target_name = "_evaluate_model"
            original = getattr(estimator, target_name, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            validation_index = self._validation_index
            self._validation_index += 1
            epoch = None if validation_index == 0 else validation_index
            stage = "baseline_validation" if epoch is None else "epoch_validation"
            train_rows = _safe_len(args[1]) if len(args) > 1 else None
            val_rows = _safe_len(args[3]) if len(args) > 3 else None
            self._validation_context = {"stage": stage, "epoch": epoch}
            started_at = time.monotonic()
            logger.debug(
                (
                    "Foundation-model fit validation start: stage={} epoch={} "
                    "train_rows={} val_rows={}"
                ),
                stage,
                epoch,
                train_rows,
                val_rows,
            )
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                self._record_event(
                    event="fit_validation",
                    started_at=started_at,
                    stage=stage,
                    epoch=epoch,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    status="failed",
                    error=repr(exc),
                )
                logger.warning(
                    "Foundation-model fit validation failed: stage={} epoch={} seconds={:.3f} error={}",
                    stage,
                    epoch,
                    time.monotonic() - started_at,
                    repr(exc),
                )
                if self._controller is not None:
                    self._validation_context = None
                raise
            finally:
                if self._controller is None:
                    self._validation_context = None

            primary = getattr(result, "primary", None)
            if self._controller is not None:
                try:
                    if epoch is not None:
                        self._validation_context = {"stage": "periodic_in_memory", "epoch": epoch}
                    self._controller.after_validation(
                        estimator,
                        epoch=epoch,
                        primary=primary,
                        validation_device=_validation_device_from_args(args),
                    )
                finally:
                    self._validation_context = None
            self._record_event(
                event="fit_validation",
                started_at=started_at,
                stage=stage,
                epoch=epoch,
                train_rows=train_rows,
                val_rows=val_rows,
                status="finished",
                primary=primary,
            )
            logger.debug(
                (
                    "Foundation-model fit validation finished: stage={} epoch={} "
                    "seconds={:.3f} primary={}"
                ),
                stage,
                epoch,
                time.monotonic() - started_at,
                primary,
            )
            return result

        setattr(estimator, target_name, wrapped)

    def _wrap_checkpoint_save(self, estimator: Any) -> None:
        original = getattr(estimator, "_save_checkpoint", None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            epoch = kwargs.get("epoch")
            output_dir = kwargs.get("output_dir")
            output_path = Path(output_dir) if output_dir is not None else None
            is_best = bool(kwargs.get("is_best"))
            save_interval = bool(kwargs.get("save_interval"))
            started_at = time.monotonic()
            if self._controller is not None and self._controller.suppress_disk_checkpoints:
                self._record_event(
                    event="fit_checkpoint_save",
                    started_at=started_at,
                    epoch=epoch,
                    is_best=is_best,
                    save_interval=save_interval,
                    status="skipped",
                    reason="in_memory_checkpoint_policy",
                )
                logger.debug(
                    (
                        "Foundation-model fit checkpoint save skipped: epoch={} "
                        "is_best={} save_interval={} reason=in_memory_checkpoint_policy"
                    ),
                    epoch,
                    is_best,
                    save_interval,
                )
                return None
            logger.debug(
                (
                    "Foundation-model fit checkpoint save start: epoch={} "
                    "is_best={} save_interval={} output_dir={}"
                ),
                epoch,
                is_best,
                save_interval,
                output_dir,
            )
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                self._record_event(
                    event="fit_checkpoint_save",
                    started_at=started_at,
                    epoch=epoch,
                    is_best=is_best,
                    save_interval=save_interval,
                    status="failed",
                    error=repr(exc),
                )
                logger.warning(
                    "Foundation-model fit checkpoint save failed: epoch={} seconds={:.3f} error={}",
                    epoch,
                    time.monotonic() - started_at,
                    repr(exc),
                )
                raise

            best_path = output_path / "best.ckpt" if output_path is not None and is_best else None
            epoch_path = output_path / f"epoch{epoch}.ckpt" if output_path is not None and save_interval else None
            best_mib = _checkpoint_size_mib(best_path)
            epoch_mib = _checkpoint_size_mib(epoch_path)
            self._record_event(
                event="fit_checkpoint_save",
                started_at=started_at,
                epoch=epoch,
                is_best=is_best,
                save_interval=save_interval,
                status="finished",
                best_checkpoint_mib=best_mib,
                epoch_checkpoint_mib=epoch_mib,
            )
            logger.debug(
                (
                    "Foundation-model fit checkpoint save finished: epoch={} seconds={:.3f} "
                    "best_mib={} epoch_mib={}"
                ),
                epoch,
                time.monotonic() - started_at,
                best_mib,
                epoch_mib,
            )
            return result

        estimator._save_checkpoint = wrapped

    def _wrap_inner_builder(self, estimator: Any) -> None:
        original = getattr(estimator, "_build_inner_estimator", None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started_at = time.monotonic()
            inner = original(*args, **kwargs)
            n_estimators = args[1] if len(args) > 1 else kwargs.get("n_estimators")
            context = dict(self._validation_context or {})
            self._record_event(
                event="fit_inner_estimator_build",
                started_at=started_at,
                stage=context.get("stage"),
                epoch=context.get("epoch"),
                n_estimators=n_estimators,
                status="finished",
            )
            logger.debug(
                (
                    "Foundation-model fit inner estimator built: stage={} epoch={} "
                    "n_estimators={} seconds={:.3f}"
                ),
                context.get("stage"),
                context.get("epoch"),
                n_estimators,
                time.monotonic() - started_at,
            )
            self._wrap_inner_method(inner, "fit", context, n_estimators)
            self._wrap_inner_method(inner, "predict_proba", context, n_estimators)
            self._wrap_inner_method(inner, "predict", context, n_estimators)
            return inner

        estimator._build_inner_estimator = wrapped

    def _wrap_inner_method(
        self,
        inner: Any,
        method_name: str,
        context: Mapping[str, Any],
        n_estimators: Any,
    ) -> None:
        original = getattr(inner, method_name, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            input_rows = _safe_len(args[0]) if args else None
            started_at = time.monotonic()
            logger.debug(
                (
                    "Foundation-model fit inner method start: method={} stage={} "
                    "epoch={} n_estimators={} input_rows={}"
                ),
                method_name,
                context.get("stage"),
                context.get("epoch"),
                n_estimators,
                input_rows,
            )
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                self._record_event(
                    event=f"fit_inner_{method_name}",
                    started_at=started_at,
                    stage=context.get("stage"),
                    epoch=context.get("epoch"),
                    n_estimators=n_estimators,
                    input_rows=input_rows,
                    status="failed",
                    error=repr(exc),
                )
                logger.warning(
                    (
                        "Foundation-model fit inner method failed: method={} stage={} "
                        "epoch={} seconds={:.3f} error={}"
                    ),
                    method_name,
                    context.get("stage"),
                    context.get("epoch"),
                    time.monotonic() - started_at,
                    repr(exc),
                )
                raise
            self._record_event(
                event=f"fit_inner_{method_name}",
                started_at=started_at,
                stage=context.get("stage"),
                epoch=context.get("epoch"),
                n_estimators=n_estimators,
                input_rows=input_rows,
                status="finished",
            )
            logger.debug(
                (
                    "Foundation-model fit inner method finished: method={} stage={} "
                    "epoch={} seconds={:.3f}"
                ),
                method_name,
                context.get("stage"),
                context.get("epoch"),
                time.monotonic() - started_at,
            )
            return result

        setattr(inner, method_name, wrapped)
