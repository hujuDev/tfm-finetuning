"""Checkpoint selection and periodic checkpoint evaluation helpers."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from finetuning.datasets.loading import SplitEvaluationData
from finetuning.evaluation.core import _evaluate_checkpoint
from finetuning.models.base import (
    BaselineCheckpointAdapter,
    CommonFineTuneConfig,
    FoundationModelAdapter,
    TaskKind,
)
from finetuning.training.in_memory import InMemoryFitController


def _epoch_from_checkpoint(path: Path) -> int | None:
    patterns = (
        r"epoch(\d+)\.ckpt",
        r"checkpoint_.+_(\d+)\.pth",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, path.name)
        if match is not None:
            return int(match.group(1))
    return None


def _epoch_checkpoint_candidates(output_dir: Path) -> list[tuple[int, Path]]:
    candidates = [
        (epoch, path)
        for pattern in ("epoch*.ckpt", "checkpoint_*_*.pth")
        for path in output_dir.glob(pattern)
        if (epoch := _epoch_from_checkpoint(path)) is not None
    ]
    return sorted(candidates, key=lambda item: (item[0], item[1].name))


def select_best_checkpoint_path(output_dir: Path) -> Path:
    """Select the best-validation checkpoint path produced by supported adapters.

    Example:
        >>> select_best_checkpoint_path(Path('/tmp/missing')).name
        'best.ckpt'
    """
    for name in ("best.ckpt", "checkpoint_best.ckpt"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    best = sorted(output_dir.glob("checkpoint_*_best.pth"))
    if best:
        return best[-1]
    return output_dir / "best.ckpt"


def select_checkpoint_path(output_dir: Path, *, checkpoint_mode: str) -> Path:
    """Select the checkpoint used for final evaluation and `model.ckpt`.

    Example:
        >>> select_checkpoint_path(Path('/tmp/missing'), checkpoint_mode='best').name
        'best.ckpt'
    """
    if checkpoint_mode == "final":
        epoch_checkpoints = _epoch_checkpoint_candidates(output_dir)
        if epoch_checkpoints:
            return max(epoch_checkpoints, key=lambda item: item[0])[1]
    return select_best_checkpoint_path(output_dir)


def _materialize_model_checkpoint(*, source: Path, output_dir: Path) -> Path:
    target = output_dir / "model.ckpt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _local_baseline_checkpoint_path(
    adapter: FoundationModelAdapter,
    task_type: TaskKind,
) -> Path | None:
    """Return a local base-model checkpoint when the adapter exposes one.

    Example:
        >>> _local_baseline_checkpoint_path(object(), "regression") is None
        True
    """
    if not isinstance(adapter, BaselineCheckpointAdapter):
        return None
    path = adapter.baseline_checkpoint_path(task_type)
    return path if path is not None and path.exists() else None


def _fit_controller_checkpoint_policy(
    fit_controller: InMemoryFitController,
    *,
    checkpoint_mode: str,
    save_best_val_checkpoint: bool,
) -> dict[str, Any]:
    payload = fit_controller.summary()
    payload["configured_checkpoint_mode"] = checkpoint_mode
    payload["save_best_val_checkpoint"] = bool(save_best_val_checkpoint)
    payload["configured_export_source"] = f"in_memory_{checkpoint_mode}_state"
    payload["best_export_source"] = "in_memory_best_state"
    payload["final_export_source"] = "in_memory_final_state"
    payload["export_source"] = fit_controller.export_source
    return payload


def _periodic_epoch_metrics(
    *,
    adapter: FoundationModelAdapter,
    output_dir: Path,
    data: SplitEvaluationData,
    seed: int,
    device: str | None,
    common_config: CommonFineTuneConfig,
    interval: int,
    timing_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if interval <= 0:
        logger.debug("Foundation-model periodic checkpoint eval disabled: interval={}", interval)
        return rows
    epoch_checkpoints = _epoch_checkpoint_candidates(output_dir)
    selected = [(epoch, path) for epoch, path in epoch_checkpoints if epoch % interval == 0]
    logger.info(
        (
            "Foundation-model periodic checkpoint eval start: interval={} "
            "selected_checkpoints={} available_checkpoints={} output_dir={}"
        ),
        interval,
        len(selected),
        len(epoch_checkpoints),
        output_dir,
    )
    for epoch, checkpoint_path in selected:
        checkpoint_started = time.monotonic()
        metrics = _evaluate_checkpoint(
            adapter=adapter,
            checkpoint_path=checkpoint_path,
            data=data,
            seed=int(seed),
            device=device,
            common_config=common_config,
            timing_rows=timing_rows,
            phase="periodic_checkpoint",
            epoch=epoch,
        )
        checkpoint_seconds = time.monotonic() - checkpoint_started
        logger.debug(
            "Foundation-model periodic checkpoint eval finished: epoch={} seconds={:.3f} checkpoint={}",
            epoch,
            checkpoint_seconds,
            checkpoint_path,
        )
        rows.append(
            {
                "epoch": epoch,
                "checkpoint_path": str(checkpoint_path),
                "eval_source": "checkpoint",
                "runtime_seconds": checkpoint_seconds,
                "metrics": metrics,
            }
        )
    return rows


def _fallback_periodic_rows_from_final_checkpoint(
    *,
    final_metrics: Mapping[str, Any],
    final_epoch: int | None,
    final_checkpoint_path: Path | None,
    interval: int,
    timing_rows: list[dict[str, Any]],
    eval_source: str = "final_checkpoint",
    timing_phase: str = "final_checkpoint",
) -> list[dict[str, Any]]:
    if interval <= 0 or final_epoch is None or final_epoch <= 0:
        return []
    if final_epoch % interval != 0 or not isinstance(final_metrics, Mapping):
        return []
    split_metrics = {
        split: dict(metrics)
        for split, metrics in final_metrics.items()
        if split in {"val", "test"} and isinstance(metrics, Mapping)
    }
    if set(split_metrics) != {"val", "test"}:
        return []

    aliases: list[dict[str, Any]] = []
    for row in list(timing_rows):
        if row.get("phase") != timing_phase or row.get("split") not in split_metrics:
            continue
        alias = dict(row)
        alias["phase"] = f"periodic_{timing_phase}"
        alias["epoch"] = int(final_epoch)
        alias["eval_source"] = eval_source
        alias["alias_of"] = timing_phase
        aliases.append(alias)
    timing_rows.extend(aliases)
    runtime_seconds = sum(float(row.get("seconds") or 0.0) for row in aliases) if aliases else None
    return [
        {
            "epoch": int(final_epoch),
            "checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
            "eval_source": eval_source,
            "runtime_seconds": runtime_seconds,
            "metrics": split_metrics,
            "fallback_source": eval_source,
        }
    ]
