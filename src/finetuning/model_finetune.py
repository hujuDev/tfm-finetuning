"""Fine-tune registry-backed foundation models through the main pipeline.

Example:
    uv run python src/finetuning/model_finetune.py path/to/finetune.toml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from finetuning.config import (
    _resolve_baseline_only,
)
from finetuning.models.registry import get_foundation_model_spec
from finetuning.run.baseline import complete_baseline_only_run
from finetuning.run.data import _prepare_foundation_run_data
from finetuning.run.final import complete_finetuned_run
from finetuning.run.setup import initialize_foundation_run
from finetuning.run.training import run_finetuned_training_phases
from finetuning.run.zero_shot import evaluate_zero_shot_baselines
from finetuning.training.fit import _record_runtime_phase
from finetuning.training.resource_policy import (
    _apply_estimator_fallback_policy,
    _configure_adapter_common_config,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")


def _resolve_evaluation_splits(config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve optional validation-only evaluation used by calibration runs."""
    foundation = config.get("foundation_model")
    raw = foundation.get("evaluation_splits") if isinstance(foundation, dict) else None
    if raw is None:
        return ("val", "test")
    if not isinstance(raw, list) or not raw:
        raise ValueError("foundation_model.evaluation_splits must be a non-empty list")
    splits = tuple(str(value).strip().lower() for value in raw)
    if any(value not in {"val", "test"} for value in splits):
        raise ValueError("foundation_model.evaluation_splits may contain only val and test")
    return splits


def main(config: dict[str, Any], output_path: Path, *, force: bool = False) -> dict[str, Any] | Path | None:
    """Run one model and write its results directly to output_path.

    Example:
        >>> callable(main)
        True
    """
    transfer = config.get("dataset_transfer", {})
    if transfer.get("enabled") or transfer.get("route_id"):
        from experiments.dataset_transfer.runner import run_dataset_transfer

        return run_dataset_transfer(
            config,
            output_path,
            force=force,
        )

    setup = initialize_foundation_run(
        config,
        output_path,
        force=force,
        model_spec_resolver=get_foundation_model_spec,
    )
    config = setup.config
    paths = setup.paths
    spec = setup.spec
    adapter = setup.adapter
    common_config = setup.common_config
    configured_common_config = common_config
    device = setup.device
    resolved_seed = setup.resolved_seed
    preprocessing_protocol = setup.preprocessing_protocol

    start = time.monotonic()
    runtime_phases: list[dict[str, Any]] = []
    eval_timing_rows: list[dict[str, Any]] = []
    data, feature_preprocessing, comparison_data = _prepare_foundation_run_data(
        config=config,
        repo_root=paths.repo_root,
        adapter=adapter,
        runtime_phases=runtime_phases,
        resolved_seed=int(resolved_seed),
        preprocessing_protocol=preprocessing_protocol,
        model_id=spec.model_id,
    )
    phase_started = time.monotonic()
    common_config, estimator_fallback = _apply_estimator_fallback_policy(
        adapter=adapter,
        data=data,
        spec=spec,
        config=config,
        common_config=configured_common_config,
    )
    _configure_adapter_common_config(adapter, common_config)
    _record_runtime_phase(
        runtime_phases,
        phase="estimator_fallback",
        started_at=phase_started,
        policy=estimator_fallback.get("estimator_fallback_policy"),
        triggered=estimator_fallback.get("estimator_fallback_triggered"),
        required_min_n_estimators=estimator_fallback.get("required_min_n_estimators"),
        feature_count=estimator_fallback.get("feature_count_for_estimator_requirement"),
    )
    baseline_metrics, comparison_baseline_metrics = evaluate_zero_shot_baselines(
        adapter=adapter,
        data=data,
        comparison_data=comparison_data,
        resolved_seed=int(setup.resolved_seed),
        device=device,
        common_config=common_config,
        runtime_phases=runtime_phases,
        eval_timing_rows=eval_timing_rows,
        splits=_resolve_evaluation_splits(config),
    )
    if _resolve_baseline_only(config):
        return complete_baseline_only_run(
            setup,
            data=data,
            comparison_data=comparison_data,
            feature_preprocessing=feature_preprocessing,
            common_config=common_config,
            configured_common_config=configured_common_config,
            estimator_fallback=estimator_fallback,
            baseline_metrics=baseline_metrics,
            comparison_baseline_metrics=comparison_baseline_metrics,
            runtime_phases=runtime_phases,
            eval_timing_rows=eval_timing_rows,
            started_at=start,
        )

    training = run_finetuned_training_phases(
        setup,
        data=data,
        comparison_data=comparison_data,
        common_config=common_config,
        baseline_metrics=baseline_metrics,
        comparison_baseline_metrics=comparison_baseline_metrics,
        runtime_phases=runtime_phases,
        eval_timing_rows=eval_timing_rows,
    )
    return complete_finetuned_run(
        setup,
        data=data,
        comparison_data=comparison_data,
        feature_preprocessing=feature_preprocessing,
        training=training,
        configured_common_config=configured_common_config,
        estimator_fallback=estimator_fallback,
        baseline_metrics=baseline_metrics,
        comparison_baseline_metrics=comparison_baseline_metrics,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
        started_at=start,
    )


def _run_main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run one registry-backed foundation-model fine-tune config.")
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"The config {config_path} does not exist")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    main(config, args.output, force=bool(args.force))


if __name__ == "__main__":
    _configure_logging()
    _run_main_cli()
