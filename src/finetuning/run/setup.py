"""Resolve one model, its scientific settings, and its explicit run directory."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetuning.config import (
    _prepare_config,
    _resolve_checkpoint_mode,
    _resolve_keep_intermediate_checkpoints,
    _resolve_save_best_val_checkpoint,
    resolve_common_finetune_config,
    resolve_configured_foundation_model_spec,
    resolve_foundation_model_id,
    resolve_test_eval_interval_epochs,
)
from finetuning.datasets.preprocessing import (
    _configure_adapter_preprocessing_protocol,
    _resolve_preprocessing_protocol,
)
from finetuning.models.base import CommonFineTuneConfig, FoundationModelAdapter
from finetuning.models.registry import FoundationModelSpec, get_foundation_model_spec
from finetuning.reporting.report import _init_wandb_run
from finetuning.run.runtime import prepare_output
from finetuning.training.fit import _resolve_device
from utils.paths import RepoPaths, get_paths


@dataclass
class FoundationRunSetup:
    """Shared training state; ``initialize_foundation_run`` constructs it."""

    config: dict[str, Any]
    base_seed: int
    resolved_seed: int
    seed_delta: int
    split_name: str | None
    paths: RepoPaths
    spec: FoundationModelSpec
    adapter: FoundationModelAdapter
    common_config: CommonFineTuneConfig
    test_eval_interval: int
    checkpoint_mode: str
    keep_intermediate_checkpoints: bool
    save_best_val_checkpoint: bool
    device: str | None
    preprocessing_protocol: str
    output: Path
    wandb_run: Any


def initialize_foundation_run(
    config: dict[str, Any],
    output_path: Path,
    *,
    force: bool,
    model_spec_resolver: Callable[..., FoundationModelSpec] = get_foundation_model_spec,
) -> FoundationRunSetup:
    """Create fresh state; ``output_path`` is used directly and never redirected to a cache."""
    config = copy.deepcopy(config)
    base_seed, resolved_seed, seed_delta, split_name = _prepare_config(config)
    paths = get_paths(config)
    spec = resolve_configured_foundation_model_spec(
        model_spec_resolver(resolve_foundation_model_id(config), repo_root=paths.repo_root),
        config,
    )
    spec.ensure_source_available()
    adapter = spec.adapter_cls(spec)
    preprocessing_protocol = _resolve_preprocessing_protocol(config)
    _configure_adapter_preprocessing_protocol(adapter, preprocessing_protocol)
    output = prepare_output(config, Path(output_path), force=force)
    return FoundationRunSetup(
        config=config,
        base_seed=base_seed,
        resolved_seed=resolved_seed,
        seed_delta=seed_delta,
        split_name=split_name,
        paths=paths,
        spec=spec,
        adapter=adapter,
        common_config=resolve_common_finetune_config(config),
        test_eval_interval=resolve_test_eval_interval_epochs(config),
        checkpoint_mode=_resolve_checkpoint_mode(config),
        keep_intermediate_checkpoints=_resolve_keep_intermediate_checkpoints(config),
        save_best_val_checkpoint=_resolve_save_best_val_checkpoint(config),
        device=_resolve_device(config),
        preprocessing_protocol=preprocessing_protocol,
        output=output,
        wandb_run=_init_wandb_run(config, output=output),
    )
