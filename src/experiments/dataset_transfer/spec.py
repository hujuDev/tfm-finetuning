"""Scientific transfer routes, update budgets and validation selection rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from finetuning.preprocessing_protocol import normalize_preprocessing_protocol

StageRole = Literal["intermediate", "reusable_boundary", "terminal"]
StageTrainingMode = Literal["single_dataset", "pooled_episodic"]
SelectorKind = Literal["target_validation", "equal_mean_validation_improvement"]
ExecutionMode = Literal["train", "checkpoint_eval"]
TransferContextPolicy = Literal["target_train_plus_validation", "route_matched_train_plus_validation"]


@dataclass(frozen=True)
class SplitIdentity:
    """Fixed OpenML fold, repeat and sample; test rows never enter checkpoint selection."""

    fold: int = 0
    repeat: int = 0
    sample: int = 0


@dataclass(frozen=True)
class ValidationSelector:
    """Checkpoint-selection rule for one stage."""

    kind: SelectorKind
    datasets: tuple[str, ...]
    metric_orientation: Literal["higher_is_better"] = "higher_is_better"

    def __post_init__(self) -> None:
        """Validate selector shape."""
        if not self.datasets:
            raise ValueError("A transfer-stage selector needs at least one dataset.")
        if self.kind == "target_validation" and len(self.datasets) != 1:
            raise ValueError("target_validation must name exactly one target dataset.")


@dataclass(frozen=True)
class TransferTrainingRecipe:
    """Optimization recipe shared by training stages and their checkpoint consumers."""

    epochs: int = 100
    learning_rate: float = 1e-05
    weight_decay: float = 0.01
    context_size: int = 10000
    query_ratio: float = 0.2
    n_estimators_finetune: int = 2
    n_estimators_validation: int = 2
    n_estimators_final_inference: int = 8
    grad_clip: float | None = 1.0
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
    def from_common_config(cls, config: Any) -> TransferTrainingRecipe:
        """Snapshot every shared fine-tuning field from a resolved config."""
        values = {name: getattr(config, name) for name in cls.__dataclass_fields__}
        return cls(**values)

    def with_overrides(self, overrides: Mapping[str, Any]) -> TransferTrainingRecipe:
        """Return a validated recipe with explicit producer-field overrides."""
        unknown = sorted(set(overrides).difference(self.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown transfer training recipe fields: {', '.join(unknown)}")
        return type(self)(**{**asdict(self), **dict(overrides)})


@dataclass(frozen=True)
class TransferDataRecipe:
    """Split, preprocessing and row-limit settings needed to reproduce a producer."""

    data_backend: str = ""
    evaluation_backend: str = ""
    validation_split_ratio: float = 0.2
    validation_split_seed: int = 0
    preprocess_steps: tuple[str, ...] = ()
    clamp_train_pool_max_size: int | None = None
    clamp_train_pool_seed: int | None = None
    train_max_rows: int | None = None
    inference_context_max_rows: int | None = None
    eval_query_max_rows: int | None = None

    def __post_init__(self) -> None:
        """Reject data recipes that cannot reproduce a deterministic split."""
        if not 0.0 < float(self.validation_split_ratio) < 1.0:
            raise ValueError("validation_split_ratio must be a fraction in (0, 1).")
        for name in ("validation_split_seed", "clamp_train_pool_seed"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in (
            "clamp_train_pool_max_size",
            "train_max_rows",
            "inference_context_max_rows",
            "eval_query_max_rows",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be positive when provided.")

    def with_overrides(self, overrides: Mapping[str, Any]) -> TransferDataRecipe:
        """Return a validated producer recipe with explicit overrides."""
        unknown = sorted(set(overrides).difference(self.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown transfer data recipe fields: {', '.join(unknown)}")
        values = {**asdict(self), **dict(overrides)}
        steps = values.get("preprocess_steps", ())
        if isinstance(steps, list):
            values["preprocess_steps"] = tuple((str(step) for step in steps))
        return type(self)(**values)


@dataclass(frozen=True)
class TransferStageSpec:
    """A stage with its own optimizer, dataset budgets and validation selector."""

    name: str
    role: StageRole
    training_mode: StageTrainingMode
    datasets: tuple[str, ...]
    row_budget_per_dataset: int | None
    update_budget: int
    sampling_weights: tuple[float, ...]
    selector: ValidationSelector
    training_recipe: TransferTrainingRecipe = field(default_factory=TransferTrainingRecipe)
    data_recipe: TransferDataRecipe = field(default_factory=TransferDataRecipe)
    context_policy: TransferContextPolicy = "target_train_plus_validation"
    row_budget_policy: Literal["cap", "exact"] = "cap"
    split: SplitIdentity = field(default_factory=SplitIdentity)
    preprocessing: str = "shared"
    model_id: str = ""
    base_checkpoint: str = ""
    task_family: Literal["classification", "regression"] = "classification"
    seed: int = 0
    row_subset_seed: int = 0
    parent_stage: str | None = None
    optimizer_reset: Literal["fresh"] = "fresh"
    strict_resource_semantics: bool = True
    view_boundaries: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate stage invariants and normalize imported protocol names."""
        object.__setattr__(self, "preprocessing", normalize_preprocessing_protocol(self.preprocessing))
        if not self.name.strip():
            raise ValueError("Transfer stage names must be non-empty.")
        if not self.datasets:
            raise ValueError(f"Transfer stage {self.name!r} has no datasets.")
        if self.update_budget <= 0:
            raise ValueError(f"Transfer stage {self.name!r} update_budget must be positive.")
        if self.row_budget_policy == "exact" and self.row_budget_per_dataset is None:
            raise ValueError("An exact row budget policy requires row_budget_per_dataset.")
        if (
            isinstance(self.row_subset_seed, bool)
            or not isinstance(self.row_subset_seed, int)
            or self.row_subset_seed < 0
        ):
            raise ValueError("row_subset_seed must be a non-negative integer.")
        if len(self.sampling_weights) != len(self.datasets):
            raise ValueError("sampling_weights must align one-to-one with datasets.")
        if any((weight <= 0 for weight in self.sampling_weights)):
            raise ValueError("sampling_weights must all be positive.")
        if self.training_mode == "single_dataset" and len(self.datasets) != 1:
            raise ValueError("single_dataset stages must name exactly one dataset.")
        if self.optimizer_reset != "fresh":
            raise ValueError("Section 3 transfer stages require optimizer_reset='fresh'.")
        if any((boundary <= 0 or boundary >= self.update_budget for boundary in self.view_boundaries)):
            raise ValueError("view boundaries must lie strictly inside the stage update budget.")


@dataclass(frozen=True)
class CheckpointDependency:
    """A producer-stage query used by checkpoint-only consumer routes."""

    family: str
    route_id: str
    checkpoint_role: Literal["terminal", "reusable_boundary"]
    checkpoint_origin: str
    producer_updates_per_dataset: int
    producer_row_budget_per_dataset: int | None
    producer_training_recipe: TransferTrainingRecipe
    producer_data_recipe: TransferDataRecipe
    producer_context_policy: TransferContextPolicy
    producer_row_subset_seed: int | None = None

    def __post_init__(self) -> None:
        """Reject dependency queries that cannot reconstruct an exact producer."""
        if (
            isinstance(self.producer_updates_per_dataset, bool)
            or not isinstance(self.producer_updates_per_dataset, int)
            or self.producer_updates_per_dataset <= 0
        ):
            raise ValueError("Checkpoint dependencies require a positive producer update budget.")
        if self.producer_row_budget_per_dataset is not None and (
            isinstance(self.producer_row_budget_per_dataset, bool)
            or not isinstance(self.producer_row_budget_per_dataset, int)
            or self.producer_row_budget_per_dataset <= 0
        ):
            raise ValueError("Checkpoint dependency row budgets must be positive when provided.")
        if self.producer_row_subset_seed is not None and (
            isinstance(self.producer_row_subset_seed, bool)
            or not isinstance(self.producer_row_subset_seed, int)
            or self.producer_row_subset_seed < 0
        ):
            raise ValueError("Checkpoint dependency row-subset seeds must be non-negative.")


@dataclass(frozen=True)
class TransferRouteSpec:
    """Ordered training stages, or a frozen-checkpoint evaluation on declared targets."""

    route_id: str
    family: str
    target_dataset: str
    execution_mode: ExecutionMode
    stages: tuple[TransferStageSpec, ...] = ()
    evaluation_targets: tuple[str, ...] = ()
    dependency: CheckpointDependency | None = None
    context_policy: TransferContextPolicy = "target_train_plus_validation"

    def __post_init__(self) -> None:
        """Validate route mode and dependency order."""
        if not self.route_id.strip() or not self.family.strip() or (not self.target_dataset.strip()):
            raise ValueError("route_id, family, and target_dataset must be non-empty.")
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError(f"Route {self.route_id!r} contains duplicate stage names.")
        previous: str | None = None
        for stage in self.stages:
            if stage.parent_stage != previous:
                raise ValueError(
                    f"Route {self.route_id!r} must be a dependency-ordered linear DAG: stage {stage.name!r} expected parent {previous!r}, got {stage.parent_stage!r}."
                )
            previous = stage.name
        if self.execution_mode == "train" and (not self.stages):
            raise ValueError("Training routes require at least one stage.")
        if self.execution_mode == "checkpoint_eval":
            if self.stages:
                raise ValueError("Checkpoint-only routes cannot contain training stages.")
            if self.dependency is None or not self.evaluation_targets:
                raise ValueError("Checkpoint-only routes require a dependency and evaluation targets.")

    @property
    def terminal_stage(self) -> TransferStageSpec | None:
        """Return the final training stage, if this is a producer route."""
        return self.stages[-1] if self.stages else None
