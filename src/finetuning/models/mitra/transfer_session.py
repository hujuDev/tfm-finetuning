"""Persistent Mitra ensemble updates for Section 3 dataset transfer.

One logical dataset episode updates every ensemble member exactly once.  The
logical update is the cross-model comparison unit; physical member optimizer
steps are tracked separately so the distinction remains auditable.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from autogluon.tabular.models.mitra._internal.config.enums import Task
from autogluon.tabular.models.mitra._internal.core.prediction_metrics import (
    PredictionMetricsTracker,
)
from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune

from finetuning.models.mitra.chunking import (
    LocalMitraFineTuneDiagnostics,
    split_mitra_support_query_indices,
)
from finetuning.models.mitra.configured import _fresh_local_trainers
from finetuning.models.mitra.preprocessing import (
    _new_trainer_preprocessor,
    fit_dense_label_encoder,
)
from finetuning.models.preprocessing import FiniteFeaturePreprocessor

_MITRA_CLASS_OUTPUTS = 10


@dataclass(frozen=True)
class MitraDatasetEpisode:
    """One raw-row support/query episode owned by one dataset schema."""

    support_indices: np.ndarray
    query_indices: np.ndarray

    @property
    def example_count(self) -> int:
        """Return support plus query rows consumed by one ensemble member."""
        return len(self.support_indices) + len(self.query_indices)


@dataclass(frozen=True)
class PreparedMitraDataset:
    """Per-member dataset preprocessing cached for one persistent stage."""

    name: str
    data: Any
    classes: np.ndarray
    member_features: tuple[np.ndarray, ...]
    member_labels: tuple[np.ndarray, ...]
    member_preprocessors: tuple[Any, ...]

    @property
    def row_count(self) -> int:
        """Return the number of fit rows available to episode sampling."""
        return len(self.member_labels[0])


class MitraDatasetEpisodeSource:
    """Sample deterministic schema-native Mitra support/query episodes."""

    def __init__(self, dataset: PreparedMitraDataset, *, context_size: int, query_ratio: float) -> None:
        self.dataset = dataset
        self.context_size = max(2, int(context_size))
        self.query_ratio = float(query_ratio)
        self.classification = dataset.data.task_type in {"binary", "multiclass"}
        if dataset.row_count < 2:
            raise ValueError(f"Mitra transfer dataset {dataset.name!r} needs at least two training rows.")

    def sample_episode(self, *, rng: np.random.Generator) -> MitraDatasetEpisode:
        """Sample rows and split them without consulting another dataset."""
        size = min(self.context_size, self.dataset.row_count)
        selected = np.asarray(rng.choice(self.dataset.row_count, size=size, replace=False), dtype=np.int64)
        split_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        split = split_mitra_support_query_indices(
            self.dataset.member_labels[0][selected],
            query_ratio=self.query_ratio,
            seed=split_seed,
            classification=self.classification,
        )
        return MitraDatasetEpisode(
            support_indices=selected[split.support_idx],
            query_indices=selected[split.query_idx],
        )


MemberStep = Callable[
    [Any, PreparedMitraDataset, MitraDatasetEpisode, Any, LocalMitraFineTuneDiagnostics, int, int],
    float,
]


class MitraPersistentStageSession:
    """Keep every Mitra model, optimizer, and scheduler alive for one stage."""

    def __init__(
        self,
        *,
        trainers: tuple[Any, ...],
        datasets: Mapping[str, PreparedMitraDataset],
        schedulers: tuple[Any, ...],
        diagnostics: tuple[LocalMitraFineTuneDiagnostics, ...],
        seed: int,
        member_step: MemberStep | None = None,
    ) -> None:
        if not trainers or len(trainers) != len(schedulers) or len(trainers) != len(diagnostics):
            raise ValueError("Mitra transfer session requires aligned non-empty ensemble state.")
        self.trainers = trainers
        self.datasets = dict(datasets)
        self.schedulers = schedulers
        self.diagnostics = diagnostics
        self.seed = int(seed)
        self.member_step = member_step or _train_member_episode
        self.optimizer_ids = tuple(id(trainer.optimizer) for trainer in trainers)
        self.member_optimizer_steps = [0 for _ in trainers]
        self.logical_examples = 0
        self.member_examples = 0
        self.losses: list[float] = []

    def train_episode(
        self,
        episode: MitraDatasetEpisode,
        *,
        dataset_name: str,
        global_update_index: int,
        dataset_update_index: int,
    ) -> dict[str, Any]:
        """Apply one logical update, comprising one step per ensemble member."""
        try:
            dataset = self.datasets[dataset_name]
        except KeyError as exc:
            raise KeyError(f"Unknown Mitra transfer dataset: {dataset_name!r}.") from exc
        member_losses: list[float] = []
        for member_index, trainer in enumerate(self.trainers):
            if id(trainer.optimizer) != self.optimizer_ids[member_index]:
                raise RuntimeError("Mitra optimizer was replaced inside a persistent transfer stage.")
            step_seed = _derived_seed(
                self.seed,
                dataset_name,
                dataset_update_index,
                member_index,
            )
            loss = self.member_step(
                trainer,
                dataset,
                episode,
                self.schedulers[member_index],
                self.diagnostics[member_index],
                step_seed,
                global_update_index,
            )
            member_losses.append(float(loss))
            self.member_optimizer_steps[member_index] += 1
        self.logical_examples += episode.example_count
        self.member_examples += episode.example_count * len(self.trainers)
        mean_loss = float(np.mean(member_losses))
        self.losses.append(mean_loss)
        return {
            "loss": mean_loss,
            "member_losses": member_losses,
            "logical_examples": episode.example_count,
            "member_optimizer_steps": len(self.trainers),
        }


def initialize_stage_estimator(
    adapter: Any,
    *,
    task_type: str,
    common_config: Any,
    seed: int,
    device: str | None,
    initial_checkpoint: Any,
) -> Any:
    """Build a fresh local-chunked ensemble and fresh optimizers from weights."""
    if task_type not in {"binary", "multiclass", "regression"}:
        raise ValueError(f"Unsupported Section 3 Mitra task type: {task_type!r}.")
    seed_training_randomness(seed)
    estimator = adapter.resolve_finetune_estimator(
        task_type,
        common_config,
        int(seed),
        device,
        initial_checkpoint=initial_checkpoint,
    )
    if getattr(estimator, "foundation_mitra_finetune_interface", None) != "local_chunked":
        raise ValueError("Section 3 Mitra training requires mitra_finetune_interface='local_chunked'.")
    task = Task.REGRESSION if task_type == "regression" else Task.CLASSIFICATION
    dim_output = 1 if task == Task.REGRESSION else _MITRA_CLASS_OUTPUTS
    n_classes = 0 if task == Task.REGRESSION else _MITRA_CLASS_OUTPUTS
    estimator.task = task
    cfg, model_cls = estimator._create_config(task, dim_output, None)
    cfg.task = task
    initial_states = getattr(estimator, "_foundation_initial_trainer_states", None)
    estimator.trainers = _fresh_local_trainers(
        estimator,
        cfg=cfg,
        model_cls=model_cls,
        task=task,
        dim_output=dim_output,
        n_classes=n_classes,
        states=initial_states,
    )
    estimator._foundation_initial_trainer_states = None
    if len(estimator.trainers) != int(common_config.n_estimators_finetune):
        raise RuntimeError("Mitra transfer ensemble size differs from n_estimators_finetune.")
    for trainer in estimator.trainers:
        hyperparams = getattr(getattr(trainer, "cfg", None), "hyperparams", {})
        if int(hyperparams.get("dim_output", -1)) != dim_output:
            raise RuntimeError("Mitra transfer trainer lacks portable output architecture metadata.")
        if int(getattr(trainer, "n_classes", -1)) != n_classes:
            raise RuntimeError("Mitra transfer trainer lacks portable class architecture metadata.")
    return estimator


def prepare_stage_datasets(
    trainers: tuple[Any, ...],
    raw_datasets: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, PreparedMitraDataset]:
    """Fit isolated per-dataset/per-member preprocessing without raw concatenation."""
    prepared: dict[str, PreparedMitraDataset] = {}
    for dataset_index, (name, data) in enumerate(raw_datasets.items()):
        if data.task_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Mitra transfer dataset {name!r} has unsupported task {data.task_type!r}.")
        outer = FiniteFeaturePreprocessor.fit(data.train_X)
        numeric = outer.transform(data.train_X).to_numpy(dtype=np.float32, copy=True)
        classification = data.task_type != "regression"
        if classification:
            classes, labels = fit_dense_label_encoder(data.train_y)
        else:
            classes = np.asarray([], dtype=np.float32)
            labels = np.asarray(data.train_y, dtype=np.float32)
        if classification and len(classes) > _MITRA_CLASS_OUTPUTS:
            raise ValueError(
                f"Mitra transfer dataset {name!r} has {len(classes)} classes; maximum is {_MITRA_CLASS_OUTPUTS}."
            )
        member_features: list[np.ndarray] = []
        member_labels: list[np.ndarray] = []
        member_preprocessors: list[Any] = []
        for member_index, trainer in enumerate(trainers):
            preprocessing_seed = _derived_seed(seed, name, dataset_index, member_index)
            with isolated_random_seed(preprocessing_seed):
                preprocessor = _new_trainer_preprocessor(trainer)
                preprocessor.fit(numeric.copy(), labels.copy())
                member_features.append(preprocessor.transform_X(numeric.copy()))
                member_labels.append(preprocessor.transform_y(labels.copy()))
            member_preprocessors.append(preprocessor)
        prepared[name] = PreparedMitraDataset(
            name=name,
            data=data,
            classes=classes,
            member_features=tuple(member_features),
            member_labels=tuple(member_labels),
            member_preprocessors=tuple(member_preprocessors),
        )
    return prepared


def exact_stage_schedulers(
    trainers: tuple[Any, ...],
    *,
    update_budget: int,
) -> tuple[tuple[Any, ...], tuple[LocalMitraFineTuneDiagnostics, ...]]:
    """Build one step scheduler per optimizer for the exact logical budget."""
    schedulers: list[Any] = []
    diagnostics: list[LocalMitraFineTuneDiagnostics] = []
    for trainer in trainers:
        params = trainer.cfg.hyperparams
        item = LocalMitraFineTuneDiagnostics(
            context_size=int(params["max_samples_support"]) + int(params["max_samples_query"]),
            query_ratio=float(params.get("foundation_query_ratio", 0.2)),
            max_samples_support=int(params["max_samples_support"]),
            max_samples_query=int(params["max_samples_query"]),
            use_lr_scheduler=bool(params.get("lr_scheduler", False)),
            scheduler_policy=str(params.get("foundation_scheduler_policy", "cosine_warmup")),
            warmup_proportion=float(params.get("foundation_warmup_proportion", 0.1)),
            total_optimizer_steps_planned=int(update_budget),
            epochs_configured=int(update_budget),
            grad_clip=float(params.get("grad_clip") or 0.0) or None,
        )
        item.warmup_steps = (
            max(1, round(item.total_optimizer_steps_planned * item.warmup_proportion)) if item.use_lr_scheduler else 0
        )
        diagnostics.append(item)
        schedulers.append(trainer._make_step_scheduler(item))
    return tuple(schedulers), tuple(diagnostics)


def seed_training_randomness(seed: int) -> None:
    """Seed Python, NumPy, and torch for one deterministic stage."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@contextmanager
def isolated_random_seed(seed: int) -> Iterator[None]:
    """Run preprocessing/evaluation without perturbing later training RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = None
    torch_state = None
    cuda_states = None
    try:
        import torch

        torch_module = torch
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    seed_training_randomness(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)


def _train_member_episode(
    trainer: Any,
    dataset: PreparedMitraDataset,
    episode: MitraDatasetEpisode,
    scheduler: Any,
    diagnostics: LocalMitraFineTuneDiagnostics,
    step_seed: int,
    global_update_index: int,
) -> float:
    member_index = _member_index(trainer, dataset)
    trainer.preprocessor = dataset.member_preprocessors[member_index]
    training_data = DatasetFinetune(
        trainer.cfg,
        x_support=dataset.member_features[member_index][episode.support_indices],
        y_support=dataset.member_labels[member_index][episode.support_indices],
        x_query=dataset.member_features[member_index][episode.query_indices],
        y_query=dataset.member_labels[member_index][episode.query_indices],
        max_samples_support=int(trainer.cfg.hyperparams["max_samples_support"]),
        max_samples_query=int(trainer.cfg.hyperparams["max_samples_query"]),
        rng=np.random.RandomState(int(step_seed)),
    )
    tracker = PredictionMetricsTracker(task=trainer.cfg.task, preprocessor=trainer.preprocessor)
    trainer.model.train()
    batches = list(trainer.make_loader(training_data, training=True))
    if len(batches) != 1:
        raise RuntimeError(f"A Section 3 Mitra episode must map to exactly one optimizer batch; got {len(batches)}.")
    scale_before = _scaler_scale(getattr(trainer, "scaler", None))
    optimizer_step_before = _optimizer_step_signature(trainer.optimizer)
    loss = trainer._train_batch(batches[0], tracker=tracker, diagnostics=diagnostics)
    loss_value = float(loss.detach().cpu().item())
    if not np.isfinite(loss_value):
        raise RuntimeError("Mitra transfer member produced a non-finite loss.")
    scale_after = _scaler_scale(getattr(trainer, "scaler", None))
    optimizer_step_after = _optimizer_step_signature(trainer.optimizer)
    if scale_before is not None and scale_after is not None and scale_after < scale_before:
        raise RuntimeError("Mitra AMP scaler skipped a member optimizer step.")
    if (
        optimizer_step_before is not None
        and optimizer_step_after is not None
        and optimizer_step_after <= optimizer_step_before
    ):
        raise RuntimeError("Mitra member optimizer did not advance for a logical update.")
    if scheduler is not None:
        scheduler.step()
    diagnostics.optimizer_steps_completed += 1
    trainer._log_step(
        loss=loss,
        step=diagnostics.optimizer_steps_completed,
        epoch=int(global_update_index) + 1,
        scheduler=scheduler,
    )
    return loss_value


def _scaler_scale(scaler: Any) -> float | None:
    getter = getattr(scaler, "get_scale", None)
    if not callable(getter):
        return None
    value = float(getter())
    return value if np.isfinite(value) else None


def _optimizer_step_signature(optimizer: Any) -> float | None:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        return None
    steps: list[float] = []
    for payload in state.values():
        if not isinstance(payload, Mapping) or "step" not in payload:
            continue
        value = payload["step"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        steps.append(float(value))
    return float(sum(steps)) if steps else 0.0


def _member_index(trainer: Any, dataset: PreparedMitraDataset) -> int:
    for index, preprocessor in enumerate(dataset.member_preprocessors):
        if preprocessor is getattr(trainer, "preprocessor", None):
            return index
    # At call entry the session assigns by trainer order, while the first call
    # has no cached preprocessor identity yet. The constructor stores the index.
    index = getattr(trainer, "_foundation_transfer_member_index", None)
    if isinstance(index, int) and 0 <= index < len(dataset.member_preprocessors):
        return index
    raise RuntimeError("Mitra transfer trainer lacks an ensemble-member index.")


def assign_transfer_member_indices(trainers: tuple[Any, ...]) -> None:
    """Bind stable member indices used by per-dataset preprocessing caches."""
    for index, trainer in enumerate(trainers):
        trainer._foundation_transfer_member_index = index


def _derived_seed(seed: int, dataset_name: str, *indices: int) -> int:
    digest = hashlib.sha256(str(dataset_name).encode("utf-8")).digest()
    token = int.from_bytes(digest[:4], "big", signed=False)
    sequence = np.random.SeedSequence([int(seed), token, *(int(index) for index in indices)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


__all__ = [
    "MitraDatasetEpisode",
    "MitraDatasetEpisodeSource",
    "MitraPersistentStageSession",
    "PreparedMitraDataset",
    "assign_transfer_member_indices",
    "exact_stage_schedulers",
    "initialize_stage_estimator",
    "isolated_random_seed",
    "prepare_stage_datasets",
]
