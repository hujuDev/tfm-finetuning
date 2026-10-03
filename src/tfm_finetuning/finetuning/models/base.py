from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from tfm_finetuning.experiments.transfer.runner.executor import StageProduction
    from tfm_finetuning.experiments.transfer.spec import TransferRouteSpec, TransferStageSpec

TaskKind = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class CommonFineTuneConfig:
    """Shared fine-tuning settings before model-specific translation.

    Example:
        >>> cfg = CommonFineTuneConfig(learning_rate=3e-5, context_size=5000)
        >>> cfg.n_estimators_final_inference
        8
    """

    epochs: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    context_size: int = 10_000
    query_ratio: float = 0.2
    n_estimators_finetune: int = 2
    n_estimators_validation: int = 2
    n_estimators_final_inference: int = 8
    grad_clip: float = 1.0
    use_lr_scheduler: bool = True
    warmup_proportion: float = 0.1
    early_stopping: bool = True
    patience: int = 101
    min_delta: float = 0.0
    ignore_pretraining_limits: bool = True
    mitra_finetune_interface: str = "local_chunked"
    mitra_scheduler_policy: str = "cosine_warmup"
    mitra_chunk_min_rows: int = 50

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> CommonFineTuneConfig:
        """Build config from a partial mapping while preserving runner defaults.

        Example:
            >>> CommonFineTuneConfig.from_mapping({"epochs": 1}).epochs
            1
        """
        if payload is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in payload.items() if key in allowed}
        return cls(**kwargs)

    @classmethod
    def default_finetune(cls, *, epochs: int = 100) -> CommonFineTuneConfig:
        """Return the maintained default fine-tuning recipe.

        Example:
            >>> CommonFineTuneConfig.default_finetune().context_size
            10000
        """
        return cls(epochs=epochs, learning_rate=1e-5, weight_decay=0.01, context_size=10_000)


def task_kind_from_value(value: Any) -> TaskKind:
    """Normalize local task labels to the shared adapter vocabulary.

    Example:
        >>> task_kind_from_value("binclass")
        'binary'
    """
    text = str(getattr(value, "value", value)).strip().lower()
    if text in {"binary", "binclass"}:
        return "binary"
    if text in {"multiclass", "multi"}:
        return "multiclass"
    if text == "regression":
        return "regression"
    raise ValueError(f"Unsupported task type for foundation-model fine-tuning: {value!r}")


def validation_metric_for_task(task_type: TaskKind, *, model_id: str) -> str:
    """Return the official validation metric used for checkpoint selection.

    Example:
        >>> validation_metric_for_task("multiclass", model_id="tabicl_v2")
        'log_loss'
    """
    _ = model_id
    if task_type == "binary":
        return "roc_auc"
    if task_type == "multiclass":
        return "log_loss"
    return "mse"


@runtime_checkable
class FoundationModelAdapter(Protocol):
    """Core behavior every maintained model Adapter implements.

    Model-specific hooks live in the named capability protocols below.

    Example:
        >>> hasattr(FoundationModelAdapter, "__call__")
        False
    """

    model_id: str

    def resolve_zero_shot_estimator(
        self,
        task_type: TaskKind,
        seed: int,
        device: str | None,
        n_estimators_inference: int,
    ) -> Any:
        """Return an unfitted official estimator for zero-shot inference."""

    def resolve_finetune_estimator(
        self,
        task_type: TaskKind,
        config: CommonFineTuneConfig,
        seed: int,
        device: str | None,
    ) -> Any:
        """Return an unfitted official fine-tuning estimator."""

    def fit_finetuned(
        self,
        estimator: Any,
        train_X: pd.DataFrame,
        train_y: np.ndarray,
        val_X: pd.DataFrame,
        val_y: np.ndarray,
        output_dir: Path,
    ) -> Any:
        """Fit a fine-tuning estimator with explicit validation data."""

    def export_checkpoint(self, output_dir: Path, checkpoint_mode: str) -> Path:
        """Return the checkpoint path selected for downstream inference."""

    def load_checkpoint_for_inference(
        self,
        checkpoint_path: Path,
        task_type: TaskKind,
        device: str | None,
        n_estimators_inference: int,
        seed: int,
    ) -> Any:
        """Load a saved fine-tuned checkpoint with the run's inference seed."""

    def fit_inference_context(self, estimator: Any, context_X: pd.DataFrame, context_y: np.ndarray) -> Any:
        """Fit the in-context inference estimator on context rows."""

    def predict_for_metrics(self, estimator: Any, query_X: pd.DataFrame, task_type: TaskKind) -> Any:
        """Return predictions or probabilities appropriate for metric computation."""

    def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, Any]:
        """Return adapter-specific constructor parameters for logging and tables."""


@runtime_checkable
class PreprocessingProtocolAdapter(Protocol):
    """Adapter that distinguishes model-native from shared preprocessing."""

    def set_preprocessing_protocol(self, protocol: str) -> None:
        """Record the selected preprocessing protocol."""


@runtime_checkable
class FeatureSchemaAdapter(Protocol):
    """Adapter that consumes categorical or typed feature-schema metadata."""

    def set_feature_schema(self, schema: Mapping[str, Any] | None) -> None:
        """Configure the schema of the features passed to the model."""


@runtime_checkable
class CommonConfigAdapter(Protocol):
    """Adapter that needs common settings outside estimator construction."""

    def set_common_config(self, config: CommonFineTuneConfig) -> None:
        """Configure run-level behavior from the common fine-tuning settings."""


@runtime_checkable
class EstimatorMinimumAdapter(Protocol):
    """Adapter with a feature-count-dependent estimator minimum."""

    def minimum_required_estimators(self, feature_count: int) -> int:
        """Return the minimum ensemble size for the active feature count."""


@runtime_checkable
class FineTuneDiagnosticsAdapter(Protocol):
    """Adapter that exposes diagnostics captured by its last fit call."""

    def consume_finetune_diagnostics(self) -> Mapping[str, Any]:
        """Return and clear model-specific fit diagnostics."""


@runtime_checkable
class BaselineCheckpointAdapter(Protocol):
    """Adapter whose zero-shot checkpoint is a local file."""

    def baseline_checkpoint_path(self, task_type: TaskKind) -> Path | None:
        """Return the local base checkpoint, if one exists."""


@runtime_checkable
class InMemoryStateAdapter(Protocol):
    """Adapter supporting live state evaluation and checkpoint export."""

    def clone_finetuned_state(self, estimator: Any) -> Mapping[str, Any]:
        """Capture a CPU inference snapshot of the current fine-tuned state."""

    def apply_finetuned_state(self, estimator: Any, state: Mapping[str, Any]) -> None:
        """Restore a previously captured fine-tuned state."""

    def build_in_memory_inference_estimator(
        self,
        estimator: Any,
        task_type: TaskKind,
        device: Any,
        n_estimators_inference: int,
    ) -> Any:
        """Build an inference estimator around live fine-tuned weights."""

    def export_finetuned_state(
        self,
        estimator: Any,
        state: Mapping[str, Any],
        output_dir: Path,
        *,
        epoch: int | None,
        metric: float | None,
        checkpoint_mode: str,
    ) -> Path:
        """Persist an in-memory inference state in the model's checkpoint format."""


@runtime_checkable
class CapturedFinetunedStatesAdapter(Protocol):
    """Adapter whose vendor loop captures distinct best/final states."""

    def capture_finetuned_states(self, estimator: Any) -> Mapping[str, Any]:
        """Return vendor-captured named states after fitting."""


@runtime_checkable
class WarmStartFineTuneAdapter(Protocol):
    """Adapter that can initialize a fresh fine-tuning stage from learned weights."""

    def resolve_finetune_estimator(
        self,
        task_type: TaskKind,
        config: CommonFineTuneConfig,
        seed: int,
        device: str | None,
        initial_checkpoint: Path | None = None,
    ) -> Any:
        """Build a fresh estimator and load only portable learned weights."""


@dataclass(frozen=True)
class DatasetTransferTrainerRequest:
    """Typed construction context for one executable Dataset Transfer Plan."""

    route: TransferRouteSpec
    config: Mapping[str, Any]
    common_config: CommonFineTuneConfig
    device: str | None
    dataset_loader: Any


@dataclass(frozen=True)
class DatasetTransferTrainerContract:
    """Immutable scientific guarantees made by an exact stage trainer."""

    exact_optimizer_updates: bool
    fresh_optimizer_per_stage: bool
    applies_nested_train_row_budget: bool
    honors_row_budget_policy: bool
    honors_row_subset_seed: bool
    checkpoint_reload_validation: bool
    weights_only_stage_initialization: bool
    declared_validation_selector: bool
    strict_resource_semantics: bool
    checkpoint_view_boundaries: bool
    dataset_native_episodes: bool
    persistent_optimizer_within_stage: bool
    exact_balanced_dataset_updates: bool
    raw_row_concatenation: bool
    source_only_target_evaluation: bool
    logical_update_unit: Literal[
        "optimizer_step",
        "dataset_episode_all_ensemble_members",
    ] = "optimizer_step"
    physical_member_steps_reported: bool = False
    validation_cadence: Literal[
        "every_update",
        "complete_balanced_round",
    ] = "complete_balanced_round"

    @classmethod
    def exact_balanced(
        cls,
        *,
        logical_update_unit: Literal[
            "optimizer_step",
            "dataset_episode_all_ensemble_members",
        ] = "optimizer_step",
        physical_member_steps_reported: bool = False,
    ) -> DatasetTransferTrainerContract:
        """Declare the maintained exact, dataset-native stage semantics."""
        return cls(
            exact_optimizer_updates=True,
            fresh_optimizer_per_stage=True,
            applies_nested_train_row_budget=True,
            honors_row_budget_policy=True,
            honors_row_subset_seed=True,
            checkpoint_reload_validation=True,
            weights_only_stage_initialization=True,
            declared_validation_selector=True,
            strict_resource_semantics=True,
            checkpoint_view_boundaries=True,
            dataset_native_episodes=True,
            persistent_optimizer_within_stage=True,
            exact_balanced_dataset_updates=True,
            raw_row_concatenation=False,
            source_only_target_evaluation=True,
            logical_update_unit=logical_update_unit,
            physical_member_steps_reported=physical_member_steps_reported,
        )

    def missing_for(self, route: TransferRouteSpec) -> tuple[str, ...]:
        """Return fail-closed contract violations for one executable route."""
        required_true = (
            "exact_optimizer_updates",
            "fresh_optimizer_per_stage",
            "applies_nested_train_row_budget",
            "honors_row_budget_policy",
            "honors_row_subset_seed",
            "checkpoint_reload_validation",
            "weights_only_stage_initialization",
            "declared_validation_selector",
            "strict_resource_semantics",
        )
        missing = [
            name
            for name in required_true
            if getattr(self, name) is not True
        ]
        if any(stage.view_boundaries for stage in route.stages):
            if self.checkpoint_view_boundaries is not True:
                missing.append("checkpoint_view_boundaries")
        if any(stage.training_mode == "pooled_episodic" for stage in route.stages):
            pooled_required = (
                "dataset_native_episodes",
                "persistent_optimizer_within_stage",
                "exact_balanced_dataset_updates",
            )
            missing.extend(
                name
                for name in pooled_required
                if getattr(self, name) is not True
            )
            if self.raw_row_concatenation is not False:
                missing.append("raw_row_concatenation")
        terminal = route.terminal_stage
        if (
            terminal is not None
            and route.target_dataset not in terminal.datasets
            and self.source_only_target_evaluation is not True
        ):
            missing.append("source_only_target_evaluation")
        return tuple(missing)


@runtime_checkable
class DatasetTransferStageTrainer(Protocol):
    """Exact stage trainer returned by a model Adapter."""

    common_config: CommonFineTuneConfig
    contract: DatasetTransferTrainerContract

    def train_stage(
        self,
        *,
        stage: TransferStageSpec,
        initial_checkpoint: Path | None,
        output_dir: Path,
    ) -> StageProduction:
        """Train exactly one immutable stage and return durable weights."""


@runtime_checkable
class DatasetTransferStageTrainerAdapter(Protocol):
    """Adapter exposing an exact update-counted Section 3 stage trainer."""

    def create_dataset_transfer_stage_trainer(
        self,
        request: DatasetTransferTrainerRequest,
    ) -> DatasetTransferStageTrainer:
        """Build the model-specific persistent-optimizer stage trainer."""


class InMemoryFinetuneLogger:
    """Capture official per-epoch fine-tuning logs without requiring W&B.

    Example:
        >>> logger = InMemoryFinetuneLogger()
        >>> logger.log_epoch({"train/epoch": 0, "val/roc_auc": 0.5}, step=1)
        >>> logger.epoch_records[0]["val/roc_auc"]
        0.5
    """

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.step_records: list[dict[str, Any]] = []
        self.epoch_records: list[dict[str, Any]] = []

    def setup(self, config: dict[str, Any]) -> None:
        """Store logger setup payload."""
        self.config = dict(config)

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Store one training-step metric payload."""
        self.step_records.append({"step": int(step), **dict(metrics)})

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """Store one epoch metric payload."""
        self.epoch_records.append({"step": int(step), **dict(metrics)})

    def finish(self) -> None:
        """Match the official logger protocol."""
