"""Translate shared fine-tuning settings into Mitra's concrete behavior."""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any

from finetuning.models.base import CommonFineTuneConfig, TaskKind
from finetuning.models.mitra.common import (
    MITRA_FINETUNE_INTERFACES,
    MITRA_INTERFACE_LOCAL_CHUNKED,
    MITRA_SCHEDULER_OFFICIAL_PLATEAU,
    MITRA_SCHEDULER_POLICIES,
)


@dataclass(frozen=True)
class MitraFineTuneOptions:
    """Resolved Mitra settings, including the local/official training strategy."""

    epochs: int
    learning_rate: float
    weight_decay: float
    context_size: int
    query_ratio: float
    n_estimators: int
    n_estimators_finetune: int
    n_estimators_validation: int
    n_estimators_inference: int
    early_stopping: bool
    patience: int
    min_delta: float
    grad_clip: float
    use_lr_scheduler: bool
    warmup_proportion: float
    interface: str
    scheduler_policy: str
    chunk_min_rows: int

    @classmethod
    def from_common(cls, config: CommonFineTuneConfig) -> MitraFineTuneOptions:
        """Resolve and validate Mitra-specific options from shared settings."""
        interface = resolve_mitra_finetune_interface(config)
        return cls(
            epochs=int(config.epochs),
            learning_rate=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            context_size=int(config.context_size),
            query_ratio=float(config.query_ratio),
            n_estimators=resolve_finetune_ensemble_size(config),
            n_estimators_finetune=int(config.n_estimators_finetune),
            n_estimators_validation=int(config.n_estimators_validation),
            n_estimators_inference=int(config.n_estimators_final_inference),
            early_stopping=bool(config.early_stopping),
            patience=resolve_mitra_patience(config),
            min_delta=float(config.min_delta),
            grad_clip=float(config.grad_clip),
            use_lr_scheduler=bool(config.use_lr_scheduler),
            warmup_proportion=float(config.warmup_proportion),
            interface=interface,
            scheduler_policy=resolve_mitra_scheduler_policy(config),
            chunk_min_rows=int(config.mitra_chunk_min_rows),
        )

    @property
    def query_size(self) -> int:
        """Return query rows in each local context/query batch."""
        context_size = max(2, self.context_size)
        query = max(1, round(context_size * self.query_ratio))
        return min(query, context_size - 1)

    @property
    def support_size(self) -> int:
        """Return support rows in each local context/query batch."""
        return max(1, max(2, self.context_size) - self.query_size)

    @property
    def warmup_steps(self) -> int:
        """Return the configured warmup count used at the selected interface."""
        return max(1, round(self.epochs * self.warmup_proportion))

    @property
    def local_chunked(self) -> bool:
        """Return whether thesis-owned local chunking is selected."""
        return self.interface == MITRA_INTERFACE_LOCAL_CHUNKED

    def estimator_kwargs(
        self,
        *,
        task_type: TaskKind,
        checkpoint: str,
        seed: int,
        device: str | None,
    ) -> dict[str, Any]:
        """Return exact kwargs passed to AutoGluon's Mitra estimator."""
        return {
            "n_estimators": self.n_estimators,
            "device": resolve_mitra_device(device),
            "fine_tune": True,
            "fine_tune_steps": self.epochs,
            "metric": mitra_metric_for_task(task_type),
            "hf_model": checkpoint,
            "patience": self.patience,
            "lr": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "seed": int(seed),
            "verbose": True,
            "weight_decay": self.weight_decay,
            "max_samples_support": self.support_size,
            "max_samples_query": self.query_size,
            "query_ratio": self.query_ratio,
            "grad_clip": self.grad_clip,
            "use_lr_scheduler": self.use_lr_scheduler,
            "warmup_proportion": self.warmup_proportion,
            "min_delta": self.min_delta,
            "mitra_finetune_interface": self.interface,
            "mitra_scheduler_policy": self.scheduler_policy,
            "mitra_chunk_min_rows": self.chunk_min_rows,
        }

    def report_params(self) -> dict[str, Any]:
        """Return stable resolved settings for reports and run metadata."""
        return {
            "epochs": self.epochs,
            "fine_tune_steps": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "context_size": self.context_size,
            "max_samples_support": self.support_size,
            "max_samples_query": self.query_size,
            "query_ratio": self.query_ratio,
            "n_estimators": self.n_estimators,
            "n_estimators_finetune": self.n_estimators_finetune,
            "n_estimators_validation": self.n_estimators_validation,
            "n_estimators_final_inference": self.n_estimators_inference,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip if self.local_chunked else None,
            "use_lr_scheduler": self.use_lr_scheduler if self.local_chunked else False,
            "warmup_proportion": self.warmup_proportion,
            "warmup_steps": self.warmup_steps,
            "warmup_steps_scope": "fit_time_optimizer_step" if self.local_chunked else "official_epoch",
            "mitra_finetune_interface": self.interface,
            "mitra_scheduler_policy": self.scheduler_policy if self.local_chunked else "official_sklearn_default",
            "mitra_chunk_min_rows": self.chunk_min_rows,
        }


def resolve_mitra_device(device: str | None) -> str:
    requested = str(device).strip() if device is not None else ""
    if requested and requested.lower() != "auto":
        return requested
    with contextlib.suppress(Exception):
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def resolve_mitra_finetune_interface(config: CommonFineTuneConfig) -> str:
    value = str(config.mitra_finetune_interface).strip().lower()
    if value not in MITRA_FINETUNE_INTERFACES:
        allowed = ", ".join(sorted(MITRA_FINETUNE_INTERFACES))
        raise ValueError(f"Unsupported Mitra fine-tune interface {value!r}; expected one of: {allowed}.")
    return value


def resolve_mitra_scheduler_policy(config: CommonFineTuneConfig) -> str:
    value = str(config.mitra_scheduler_policy).strip().lower()
    if value == "official_sklearn_default":
        return MITRA_SCHEDULER_OFFICIAL_PLATEAU
    if value not in MITRA_SCHEDULER_POLICIES:
        allowed = ", ".join(sorted(MITRA_SCHEDULER_POLICIES))
        raise ValueError(f"Unsupported Mitra scheduler policy {value!r}; expected one of: {allowed}.")
    return value


def mitra_metric_for_task(task_type: TaskKind) -> str:
    return "mse" if task_type == "regression" else "log_loss"


def resolve_mitra_patience(config: CommonFineTuneConfig) -> int:
    return int(config.patience) if config.early_stopping else int(config.epochs) + 1


def resolve_finetune_ensemble_size(config: CommonFineTuneConfig) -> int:
    values = {
        int(config.n_estimators_finetune),
        int(config.n_estimators_validation),
        int(config.n_estimators_final_inference),
    }
    if len(values) != 1:
        raise ValueError(
            "Mitra fine-tuning requires matching n_estimators_finetune, "
            "n_estimators_validation, and n_estimators_final_inference so checkpoints "
            "and epoch curves use the same full ensemble."
        )
    return values.pop()
