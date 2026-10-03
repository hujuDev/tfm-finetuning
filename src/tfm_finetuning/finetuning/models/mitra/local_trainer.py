"""Local chunked Mitra training built on AutoGluon's trainer internals.

This module owns support/query chunk execution, optimizer-step scheduling, and
gradient clipping. The official sklearn strategy never enters this loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from autogluon.tabular.models.mitra._internal.config.enums import LossName, ModelName, Task
from autogluon.tabular.models.mitra._internal.core.prediction_metrics import PredictionMetricsTracker
from autogluon.tabular.models.mitra._internal.core.trainer_finetune import TrainerFinetune
from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from tfm_finetuning.finetuning.models.mitra.chunking import (
    LocalMitraFineTuneDiagnostics,
    SplitResult,
    count_mitra_epoch_chunks,
    dropped_mitra_tail_rows,
    iter_mitra_epoch_chunks,
    split_mitra_support_query_indices,
)
from tfm_finetuning.finetuning.models.mitra.common import (
    MITRA_SCHEDULER_COSINE_WARMUP,
    MITRA_SCHEDULER_LINEAR_WARMUP_ONLY,
    MITRA_SCHEDULER_OFFICIAL_PLATEAU,
)


@dataclass(frozen=True)
class _EpochTrainResult:
    batch_count: int
    global_step: int
    metrics_train: Any


@dataclass
class _LocalFitState:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_train_transformed: np.ndarray
    y_train_transformed: np.ndarray
    diagnostics: LocalMitraFineTuneDiagnostics
    scheduler: LambdaLR | None
    start_time: float
    global_step: int = 0
    stopped: bool = False
    finished: bool = False


class LocalMitraTrainer(TrainerFinetune):
    """Mitra trainer with TabPFN/TabICL-like local context/query chunking.

    Example:
        >>> issubclass(LocalMitraTrainer, TrainerFinetune)
        True
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.local_diagnostics = LocalMitraFineTuneDiagnostics()
        min_delta = self.cfg.hyperparams.get("early_stopping_delta")
        if min_delta is not None:
            self.early_stopping.delta = float(min_delta)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fine-tune with multiple context/query meta-batches per epoch."""
        state = self.begin_local_fit(x_train, y_train, x_val, y_val)
        max_epochs = int(self.cfg.hyperparams["max_epochs"])
        for epoch in range(1, max_epochs + 1):
            if self.run_local_epoch(state, epoch=epoch):
                break
        self.finish_local_fit(state)

    def begin_local_fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _LocalFitState:
        """Prepare local chunked fine-tuning and return mutable fit state."""
        self.preprocessor.fit(x_train, y_train)
        x_train_transformed = self.preprocessor.transform_X(x_train)
        y_train_transformed = self.preprocessor.transform_y(y_train)

        diagnostics = self._initialize_diagnostics(n_samples=len(y_train_transformed))
        scheduler = self._make_step_scheduler(diagnostics)

        self.checkpoint.reset(self.model)
        self._set_checkpoint_epoch(0)
        metrics_valid = self.evaluate(x_train, y_train, x_val, y_val)
        if self.verbose:
            self.log_start_metrics(metrics_valid)
        self.checkpoint(self.model, metrics_valid.loss)

        return _LocalFitState(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_train_transformed=x_train_transformed,
            y_train_transformed=y_train_transformed,
            diagnostics=diagnostics,
            scheduler=scheduler,
            start_time=time.time(),
        )

    def run_local_epoch(self, state: _LocalFitState, *, epoch: int) -> bool:
        """Train one local-chunked epoch and return whether this trainer stopped."""
        if state.stopped:
            return True
        epoch_result = self._run_training_epoch(
            x_train_transformed=state.x_train_transformed,
            y_train_transformed=state.y_train_transformed,
            diagnostics=state.diagnostics,
            scheduler=state.scheduler,
            epoch=epoch,
            global_step=state.global_step,
        )
        state.global_step = epoch_result.global_step
        if epoch_result.batch_count == 0:
            state.diagnostics.skipped_chunks += state.diagnostics.chunks_per_epoch
            state.stopped = True
            logger.warning("Local Mitra trainer produced no batches for epoch={}; stopping early.", epoch)
            return True

        metrics_valid = self.evaluate(state.x_train, state.y_train, state.x_val, state.y_val)
        if self.verbose:
            self.log_metrics(epoch, epoch_result.metrics_train, metrics_valid)
        self._set_checkpoint_epoch(epoch)
        self.checkpoint(self.model, metrics_valid.loss)
        state.diagnostics.epochs_completed = epoch
        state.diagnostics.optimizer_steps_completed = state.global_step

        if self._should_stop_after_epoch(
            metrics_valid=metrics_valid,
            diagnostics=state.diagnostics,
            start_time=state.start_time,
        ):
            state.stopped = True
        return state.stopped

    def finish_local_fit(self, state: _LocalFitState) -> None:
        """Finalize a local fit by restoring the best checkpointed model."""
        if state.finished:
            return
        self.local_diagnostics = state.diagnostics
        self._set_checkpoint_epoch(state.diagnostics.epochs_completed)
        self.checkpoint.set_to_best(self.model)
        state.finished = True

    def _set_checkpoint_epoch(self, epoch: int) -> None:
        checkpoint = getattr(self, "checkpoint", None)
        if checkpoint is not None:
            checkpoint._foundation_current_epoch = int(epoch)

    def _run_training_epoch(
        self,
        *,
        x_train_transformed: np.ndarray,
        y_train_transformed: np.ndarray,
        diagnostics: LocalMitraFineTuneDiagnostics,
        scheduler: LambdaLR | None,
        epoch: int,
        global_step: int,
    ) -> _EpochTrainResult:
        epoch_batches = 0
        tracker = PredictionMetricsTracker(task=self.cfg.task, preprocessor=self.preprocessor)
        self.model.train()
        epoch_seed = int(self.cfg.seed) + epoch - 1
        chunks = iter_mitra_epoch_chunks(
            len(y_train_transformed),
            context_size=diagnostics.context_size,
            epoch_seed=epoch_seed,
            min_chunk_size=diagnostics.min_chunk_size,
        )
        for chunk_index, indices in enumerate(chunks):
            dataset = self._build_training_dataset(
                x_train_transformed=x_train_transformed,
                y_train_transformed=y_train_transformed,
                diagnostics=diagnostics,
                indices=indices,
                seed=epoch_seed + chunk_index * 7919,
            )
            for batch in self.make_loader(dataset, training=True):
                loss = self._train_batch(batch, tracker=tracker, diagnostics=diagnostics)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                epoch_batches += 1
                self._log_step(loss=loss, step=global_step, epoch=epoch, scheduler=scheduler)
        metrics_train = tracker.get_metrics() if epoch_batches > 0 else None
        return _EpochTrainResult(batch_count=epoch_batches, global_step=global_step, metrics_train=metrics_train)

    def _build_training_dataset(
        self,
        *,
        x_train_transformed: np.ndarray,
        y_train_transformed: np.ndarray,
        diagnostics: LocalMitraFineTuneDiagnostics,
        indices: np.ndarray,
        seed: int,
    ) -> DatasetFinetune:
        split = self._split_support_query(
            y_train_transformed[indices],
            query_ratio=diagnostics.query_ratio,
            seed=seed,
        )
        diagnostics.query_size_clamped_count += int(split.query_size_clamped)
        diagnostics.rare_class_fallback_count += int(split.rare_class_fallback)
        diagnostics.singleton_chunk_count += int(split.singleton_chunk)
        return DatasetFinetune(
            self.cfg,
            x_support=x_train_transformed[indices][split.support_idx],
            y_support=y_train_transformed[indices][split.support_idx],
            x_query=x_train_transformed[indices][split.query_idx],
            y_query=y_train_transformed[indices][split.query_idx],
            max_samples_support=diagnostics.max_samples_support,
            max_samples_query=diagnostics.max_samples_query,
            rng=self.rng,
        )

    def _should_stop_after_epoch(
        self,
        *,
        metrics_valid: Any,
        diagnostics: LocalMitraFineTuneDiagnostics,
        start_time: float,
    ) -> bool:
        self.early_stopping(metrics_valid.metrics[self.metric])
        if self.early_stopping.we_should_stop():
            diagnostics.early_stopped = True
            if self.verbose:
                logger.info("Early stopping")
            return True
        if self._budget_reached(start_time):
            diagnostics.budget_reached = True
            logger.info("Time limit reached")
            return True
        if self._uses_official_plateau_scheduler(diagnostics):
            self.scheduler_reduce_on_plateau.step(metrics_valid.loss)
        return False

    def _initialize_diagnostics(self, *, n_samples: int) -> LocalMitraFineTuneDiagnostics:
        context_size = max(2, int(self.cfg.hyperparams.get("foundation_context_size") or 0))
        if context_size <= 2:
            context_size = int(self.cfg.hyperparams["max_samples_support"]) + int(
                self.cfg.hyperparams["max_samples_query"]
            )
        min_chunk_size = max(2, int(self.cfg.hyperparams.get("foundation_mitra_chunk_min_rows", 50)))
        chunks_per_epoch = count_mitra_epoch_chunks(n_samples, context_size, min_chunk_size)
        max_epochs = int(self.cfg.hyperparams["max_epochs"])
        diagnostics = LocalMitraFineTuneDiagnostics(
            context_size=context_size,
            query_ratio=float(self.cfg.hyperparams.get("foundation_query_ratio", 0.2)),
            min_chunk_size=min_chunk_size,
            max_samples_support=int(self.cfg.hyperparams["max_samples_support"]),
            max_samples_query=int(self.cfg.hyperparams["max_samples_query"]),
            use_lr_scheduler=bool(self.cfg.hyperparams.get("lr_scheduler", False)),
            scheduler_policy=str(
                self.cfg.hyperparams.get("foundation_scheduler_policy", MITRA_SCHEDULER_COSINE_WARMUP)
            ),
            warmup_proportion=float(self.cfg.hyperparams.get("foundation_warmup_proportion", 0.1)),
            chunks_per_epoch=chunks_per_epoch,
            epochs_configured=max_epochs,
            total_optimizer_steps_planned=max(1, chunks_per_epoch * max_epochs),
            dropped_tail_rows=dropped_mitra_tail_rows(n_samples, context_size, min_chunk_size),
            grad_clip=_optional_positive_float(self.cfg.hyperparams.get("grad_clip")),
        )
        diagnostics.warmup_steps = (
            max(1, round(diagnostics.total_optimizer_steps_planned * diagnostics.warmup_proportion))
            if diagnostics.use_lr_scheduler
            else 0
        )
        return diagnostics

    def _make_step_scheduler(self, diagnostics: LocalMitraFineTuneDiagnostics) -> LambdaLR | None:
        if not diagnostics.use_lr_scheduler:
            return None
        policy = diagnostics.scheduler_policy
        step_policy = MITRA_SCHEDULER_LINEAR_WARMUP_ONLY if policy == MITRA_SCHEDULER_OFFICIAL_PLATEAU else policy
        if step_policy not in {MITRA_SCHEDULER_COSINE_WARMUP, MITRA_SCHEDULER_LINEAR_WARMUP_ONLY}:
            raise ValueError(f"Unsupported Mitra scheduler policy: {policy!r}")

        total_steps = max(1, int(diagnostics.total_optimizer_steps_planned))
        warmup_steps = min(max(0, int(diagnostics.warmup_steps)), max(0, total_steps - 1))

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if step_policy == MITRA_SCHEDULER_LINEAR_WARMUP_ONLY:
                return 1.0
            decay_steps = max(1, total_steps - warmup_steps)
            progress = float(current_step - warmup_steps) / float(decay_steps)
            progress = max(0.0, min(1.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _split_support_query(self, y_chunk: np.ndarray, *, query_ratio: float, seed: int) -> SplitResult:
        return split_mitra_support_query_indices(
            y_chunk,
            query_ratio=query_ratio,
            seed=seed,
            classification=self.cfg.task == Task.CLASSIFICATION,
        )

    def _train_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        tracker: PredictionMetricsTracker,
        diagnostics: LocalMitraFineTuneDiagnostics,
    ) -> torch.Tensor:
        with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams["precision"])):
            x_support = batch["x_support"].to(self.device, non_blocking=True)
            y_support = batch["y_support"].to(self.device, non_blocking=True)
            x_query = batch["x_query"].to(self.device, non_blocking=True)
            y_query = batch["y_query"].to(self.device, non_blocking=True)
            padding_features = batch["padding_features"].to(self.device, non_blocking=True)
            padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
            padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)

            if self.cfg.task == Task.REGRESSION and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY:
                y_support = torch.bucketize(y_support, self.bins) - 1
                y_support = torch.clamp(y_support, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)
                y_query_bin_ids = torch.bucketize(y_query, self.bins) - 1
                y_query_bin_ids = torch.clamp(
                    y_query_bin_ids,
                    0,
                    self.cfg.hyperparams["dim_output"] - 1,
                ).to(torch.int64)

            if self.cfg.model_name == ModelName.TABPFN:
                y_hat = self.model(x_support, y_support, x_query, task=self.cfg.task).squeeze(-1)
            elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
                y_hat = self.model(
                    x_support,
                    y_support,
                    x_query,
                    padding_features,
                    padding_obs_support,
                    padding_obs_query,
                )
            else:
                raise NotImplementedError(f"Model {self.cfg.model_name} not implemented")

            if self.cfg.task == Task.REGRESSION and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY:
                loss = self.loss(y_hat, y_query_bin_ids)
            else:
                loss = self.loss(y_hat, y_query)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if diagnostics.grad_clip is not None and diagnostics.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), diagnostics.grad_clip)
            diagnostics.grad_clip_applied_steps += 1
            grad_norm_value = float(grad_norm.detach().cpu().item() if hasattr(grad_norm, "detach") else grad_norm)
            diagnostics.grad_norm_max = (
                grad_norm_value
                if diagnostics.grad_norm_max is None
                else max(diagnostics.grad_norm_max, grad_norm_value)
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.cfg.task == Task.REGRESSION and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY:
            y_hat = torch.argmax(y_hat, dim=-1)
            y_hat = self.bins[y_hat] + self.bin_width / 2
        y_hat = y_hat.float()
        tracker.update(y_hat, y_query, train=self.cfg.task == Task.REGRESSION)
        return loss

    def _log_step(
        self,
        *,
        loss: torch.Tensor,
        step: int,
        epoch: int,
        scheduler: LambdaLR | None,
    ) -> None:
        logger_obj = getattr(self, "experiment_logger", None)
        if logger_obj is None:
            return
        log_step = getattr(logger_obj, "log_step", None)
        if not callable(log_step):
            return
        lr = scheduler.get_last_lr()[0] if scheduler is not None else float(self.cfg.hyperparams["lr"])
        log_step(
            {"train/loss": float(loss.detach().item()), "train/lr": float(lr), "train/epoch": float(epoch - 1)},
            step=int(step),
        )

    def _budget_reached(self, start_time: float) -> bool:
        budget = self.cfg.hyperparams.get("budget")
        return bool(budget is not None and budget > 0 and time.time() - start_time > budget)

    @staticmethod
    def _uses_official_plateau_scheduler(diagnostics: LocalMitraFineTuneDiagnostics) -> bool:
        return diagnostics.use_lr_scheduler and diagnostics.scheduler_policy == MITRA_SCHEDULER_OFFICIAL_PLATEAU


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None
