"""Exact one-update TabICL engine for Section 3 dataset-transfer stages.

This module is intentionally model-specific.  It composes the pinned TabICL
fine-tuner's model loading, task loss, AMP, scheduler, and inference helpers,
but owns the update loop so a pooled stage can keep one optimizer while
switching between independently preprocessed table schemas.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
)
from finetuning.evaluation.core import (
    _evaluate_checkpoint,
    _evaluate_in_memory_estimator_on_data,
    _evaluate_zero_shot,
)
from finetuning.models.tabicl.finetune_compat import patch_tabicl_finetune_meta_batches
from finetuning.models.tabicl.preprocessing import (
    prepare_native_finetune_inputs,
    prepare_shared_finetune_inputs,
)

if TYPE_CHECKING:
    from experiments.transfer.spec import TransferStageSpec
    from finetuning.models.base import CommonFineTuneConfig
    from finetuning.models.tabicl.model import TabICLAdapter

_MAX_EPISODE_ATTEMPTS = 32
_MAX_AMP_STEP_ATTEMPTS = 32


@dataclass(frozen=True)
class TabICLUpdateResult:
    """Audit values emitted by one completed optimizer update."""

    loss: float
    examples: int


@dataclass(frozen=True)
class _PreparedDataset:
    data: Any
    train_X: np.ndarray
    train_y: np.ndarray


class TabICLTransferEngine:
    """One TabICL model/optimizer/scheduler session for one transfer stage.

    A new instance is constructed at every stage boundary.  ``train_update``
    never recreates optimizer or scheduler state, so interleaved datasets share
    the exact same optimization trajectory while retaining native schemas.
    """

    def __init__(
        self,
        *,
        adapter: TabICLAdapter,
        stage: TransferStageSpec,
        common_config: CommonFineTuneConfig,
        device: str | None,
        datasets: Mapping[str, Any],
        initial_checkpoint: Path | None,
    ) -> None:
        if int(os.environ.get("RANK", "-1")) != -1:
            raise RuntimeError("Section 3 TabICL transfer stages currently require one process and one GPU.")
        if stage.task_family not in {"classification", "regression"}:
            raise RuntimeError(f"Unsupported Section 3 TabICL task family: {stage.task_family!r}.")
        self.classification = stage.task_family == "classification"
        training_datasets = set(stage.datasets)
        provided_datasets = set(datasets)
        if not training_datasets.issubset(provided_datasets) or (
            provided_datasets != training_datasets and stage.role != "terminal"
        ):
            raise ValueError(
                "TabICL transfer engine must receive every stage dataset; "
                "only terminal stages may add evaluation-only datasets."
            )

        self.adapter = adapter
        self.stage = stage
        self.common_config = common_config
        self.requested_device = device
        self.preprocessing_diagnostics: dict[str, dict[str, Any]] = {}
        self.meta_batch_diagnostics: dict[str, int] = {}
        self.datasets = {name: self._prepare_dataset(name, data) for name, data in datasets.items()}
        task_types = {prepared.data.task_type for prepared in self.datasets.values()}
        allowed = {"binary", "multiclass"} if self.classification else {"regression"}
        if not task_types.issubset(allowed):
            raise RuntimeError(
                f"TabICL transfer stage requires {stage.task_family} datasets, got {sorted(task_types)}."
            )

        # Binary and multiclass share classifier weights. Dataset-local label
        # encoders keep their vocabularies apart; regression preserves floats.
        representative_task = next(iter(self.datasets.values())).data.task_type
        self.estimator = adapter.resolve_finetune_estimator(
            representative_task,
            common_config,
            int(stage.seed),
            device,
            initial_checkpoint=initial_checkpoint,
        )
        self.device = self.estimator._resolve_device()
        self._seed_torch(int(stage.seed))
        self.estimator._load_pretrained(self.device)
        self.estimator.model_.to(self.device)
        self.estimator._apply_freezing(self.estimator.model_)
        self.estimator._ddp_model_ = None
        self.estimator.is_fitted_ = False

        torch = self._torch()
        parameters = [parameter for parameter in self.estimator.model_.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("TabICL transfer stage has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(common_config.learning_rate),
            weight_decay=float(common_config.weight_decay),
        )
        self.scheduler = self._build_scheduler()
        _, self.scaler, self.amp_context = self.estimator._make_amp(self.device)
        self.eval_timing_rows: list[dict[str, Any]] = []
        self.gpu_time_seconds = 0.0

    def train_update(self, dataset_name: str, *, rng: np.random.Generator) -> TabICLUpdateResult:
        """Sample one dataset-native episode and take exactly one optimizer step."""
        prepared = self.datasets[dataset_name]
        row_count = min(len(prepared.train_y), int(self.common_config.context_size))
        if row_count < 2:
            raise RuntimeError(f"Dataset {dataset_name!r} has fewer than two fit rows.")
        batch = self._sample_supported_episode(dataset_name, prepared=prepared, row_count=row_count, rng=rng)

        from tabicl._finetune import data as tabicl_data

        torch = self._torch()
        batch = tabicl_data.move_meta_batch(batch, self.device)
        self.estimator._set_training_mode(True)
        self._synchronize_device()
        started = time.monotonic()
        for _attempt in range(_MAX_AMP_STEP_ATTEMPTS):
            self.optimizer.zero_grad(set_to_none=True)
            with self.amp_context():
                loss = self.estimator._compute_batch_loss(batch, self.estimator.model_)
            if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
                raise RuntimeError("TabICL produced a non-finite transfer loss before the optimizer step.")
            self.scaler.scale(loss).backward()
            if self.common_config.grad_clip is not None and float(self.common_config.grad_clip) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.estimator.model_.parameters(), float(self.common_config.grad_clip))
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if float(self.scaler.get_scale()) >= scale_before:
                break
            self.meta_batch_diagnostics["amp_overflow_retry_count"] = (
                self.meta_batch_diagnostics.get("amp_overflow_retry_count", 0) + 1
            )
        else:
            raise RuntimeError(
                f"TabICL AMP skipped {_MAX_AMP_STEP_ATTEMPTS} attempts for one optimizer update "
                f"(final scale={float(self.scaler.get_scale()):g})."
            )
        if self.scheduler is not None:
            self.scheduler.step()
        self._synchronize_device()
        self.gpu_time_seconds += time.monotonic() - started
        return TabICLUpdateResult(
            loss=float(loss.detach().item()),
            examples=int(batch.X.shape[1]),
        )

    def clone_state(self) -> Mapping[str, Any]:
        """Return an immutable CPU snapshot of the live model weights."""
        return self.adapter.clone_finetuned_state(self.estimator)

    def evaluate_current(self, dataset_name: str, *, splits: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Evaluate the live weights with the named dataset's ICL context."""
        return _evaluate_in_memory_estimator_on_data(
            adapter=self.adapter,
            estimator=self.estimator,
            data=self.datasets[dataset_name].data,
            device=self.requested_device,
            common_config=self.common_config,
            timing_rows=self.eval_timing_rows,
            phase="transfer_stage_validation",
            eval_source="tabicl_transfer_in_memory",
            epoch=None,
            splits=splits,
        )

    def evaluate_base(self, dataset_name: str, *, splits: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """Evaluate the unchanged TabICL base model on matched ICL rows."""
        return _evaluate_zero_shot(
            adapter=self.adapter,
            data=self.datasets[dataset_name].data,
            seed=int(self.stage.seed),
            device=self.requested_device,
            common_config=self.common_config,
            timing_rows=self.eval_timing_rows,
            phase="transfer_stage_base_icl",
            eval_source="base_pure_icl",
            splits=splits,
        )

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        dataset_name: str,
        *,
        splits: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        """Evaluate serialized selected weights on matched ICL rows."""
        return _evaluate_checkpoint(
            adapter=self.adapter,
            checkpoint_path=checkpoint_path,
            data=self.datasets[dataset_name].data,
            seed=int(self.stage.seed),
            device=self.requested_device,
            common_config=self.common_config,
            timing_rows=self.eval_timing_rows,
            phase="transfer_stage_selected_checkpoint",
            eval_source="tabicl_transfer_checkpoint",
            splits=splits,
        )

    def export_state(
        self,
        state: Mapping[str, Any],
        output_dir: Path,
        *,
        update: int,
        metric: float,
    ) -> Path:
        """Write a weights-only TabICL-native checkpoint for one selection."""
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.adapter.export_finetuned_state(
            self.estimator,
            state,
            output_dir,
            epoch=int(update),
            metric=float(metric),
            checkpoint_mode="best",
        )

    def validate_reload(self, checkpoint_path: Path) -> bool:
        """Reload, fit a class-complete ICL context, and predict one val row."""
        prepared = next(iter(self.datasets.values()))
        data = prepared.data
        context_indices = _reload_context_indices(
            np.asarray(data.train_y),
            max_rows=min(int(self.common_config.context_size), 64),
            classification=self.classification,
        )
        if len(data.val_X) < 1:
            raise RuntimeError("TabICL checkpoint reload validation requires one validation query row.")
        estimator = self.adapter.load_checkpoint_for_inference(
            Path(checkpoint_path),
            data.task_type,
            self.requested_device,
            1,
            seed=int(self.stage.seed),
        )
        fitted = self.adapter.fit_inference_context(
            estimator,
            data.train_X.iloc[context_indices],
            np.asarray(data.train_y)[context_indices],
        )
        if fitted is not None:
            estimator = fitted
        prediction = np.asarray(
            self.adapter.predict_for_metrics(
                estimator,
                data.val_X.iloc[:1],
                data.task_type,
            )
        )
        return bool(prediction.size and np.isfinite(prediction).all())

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe model-specific training diagnostics."""
        return {
            "tabicl_meta_batch_split": dict(self.meta_batch_diagnostics),
            "tabicl_dataset_preprocessing": dict(self.preprocessing_diagnostics),
            "evaluation_timing": list(self.eval_timing_rows),
        }

    def _prepare_dataset(self, dataset_name: str, data: Any) -> _PreparedDataset:
        if self.adapter._preprocessing_protocol == PREPROCESSING_NATIVE:
            train_X, _val_X, diagnostics = prepare_native_finetune_inputs(data.train_X, data.val_X)
        else:
            train_X, _val_X, diagnostics = prepare_shared_finetune_inputs(
                data.train_X,
                data.val_X,
                protocol=self.adapter._preprocessing_protocol,
            )
        raw_labels = np.asarray(data.train_y)
        labels = (
            LabelEncoder().fit_transform(raw_labels).astype(np.int64, copy=False)
            if self.classification
            else raw_labels.astype(np.float32, copy=False)
        )
        self.preprocessing_diagnostics[dataset_name] = diagnostics
        return _PreparedDataset(data=data, train_X=train_X, train_y=labels)

    def _sample_supported_episode(
        self,
        dataset_name: str,
        *,
        prepared: _PreparedDataset,
        row_count: int,
        rng: np.random.Generator,
    ) -> Any:
        from tabicl._finetune import data as tabicl_data

        for _attempt in range(_MAX_EPISODE_ATTEMPTS):
            indices = rng.choice(len(prepared.train_y), size=row_count, replace=False)
            episode_seed = int(rng.integers(0, np.iinfo(np.int32).max, endpoint=False))
            with patch_tabicl_finetune_meta_batches() as diagnostics:
                batch = tabicl_data._build_meta_batch(
                    prepared.train_X[indices],
                    prepared.train_y[indices],
                    classification=self.classification,
                    n_estimators=int(self.common_config.n_estimators_finetune),
                    query_size=max(1, int(row_count * float(self.common_config.query_ratio))),
                    epoch_seed=episode_seed,
                    chunk_idx=0,
                    norm_methods=self.estimator.norm_methods,
                    feat_shuffle_method=self.estimator.feat_shuffle_method,
                    class_shuffle_method=getattr(self.estimator, "class_shuffle_method", "shift"),
                    outlier_threshold=float(self.estimator.outlier_threshold),
                    preprocessing_seed=int(self.stage.seed),
                )
            for key, value in diagnostics.as_dict().items():
                self.meta_batch_diagnostics[key] = self.meta_batch_diagnostics.get(key, 0) + int(value)
            if not self.estimator._skip_batch(batch, False, self.device):
                return batch
            self.meta_batch_diagnostics["rejected_episode_count"] = (
                self.meta_batch_diagnostics.get("rejected_episode_count", 0) + 1
            )
        raise RuntimeError(
            f"TabICL rejected {_MAX_EPISODE_ATTEMPTS} deterministic episodes for {dataset_name!r}; "
            "no allocated optimizer update was consumed."
        )

    def _build_scheduler(self) -> Any | None:
        if not self.common_config.use_lr_scheduler:
            return None
        from tabicl.train._optim import get_scheduler

        scheduler_config = SimpleNamespace(
            max_steps=int(self.stage.update_budget),
            warmup_proportion=float(self.common_config.warmup_proportion),
            warmup_steps=0,
            scheduler="cosine_warmup",
        )
        return get_scheduler(scheduler_config, self.optimizer)

    def _seed_torch(self, seed: int) -> None:
        torch = self._torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _synchronize_device(self) -> None:
        torch = self._torch()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch


def _reload_context_indices(
    labels: np.ndarray,
    *,
    max_rows: int,
    classification: bool,
) -> np.ndarray:
    """Return deterministic reload rows, class-complete for classifiers."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise RuntimeError("TabICL checkpoint reload validation requires one-dimensional labels.")
    if not classification:
        return np.arange(min(int(max_rows), len(labels)), dtype=np.int64)
    required = [int(np.flatnonzero(labels == label)[0]) for label in np.unique(labels)]
    limit = max(int(max_rows), len(required))
    required_set = set(required)
    fill = [index for index in range(len(labels)) if index not in required_set]
    return np.asarray([*required, *fill[: max(0, min(limit, len(labels)) - len(required))]], dtype=np.int64)


__all__ = ["TabICLTransferEngine", "TabICLUpdateResult"]
