"""Pinned TabPFN vendor runtime for one-step dataset-transfer episodes.

This is the only thesis module that reaches into the official fine-tuner's
private one-step surface.  Keeping those calls here makes version drift explicit
and leaves stage scheduling, validation, and reporting model-independent of the
vendor mechanics.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig

_REQUIRED_WRAPPER_MEMBERS = (
    "_build_estimator_config",
    "_create_estimator",
    "_estimator_kwargs",
    "_forward_with_loss",
    "_get_valid_finetuning_query_size",
    "_setup_batch",
    "_setup_estimator",
    "_should_skip_batch",
)
_MAX_EPISODE_ATTEMPTS = 32
_MAX_AMP_STEP_ATTEMPTS = 8


@dataclass(frozen=True)
class TabPFNTransferEpisode:
    """One already-preprocessed, schema-native TabPFN optimizer input."""

    batch: Any
    example_count: int


class TabPFNTransferRuntime:
    """Pinned vendor-private one-step runtime with one optimizer per stage."""

    def __init__(
        self,
        *,
        adapter: Any,
        common_config: CommonFineTuneConfig,
        task_type: str,
        seed: int,
        device: str | None,
        initial_checkpoint: Path | None,
        total_updates: int,
    ) -> None:
        if os.environ.get("LOCAL_RANK") is not None:
            raise RuntimeError("TabPFN transfer episodic training currently supports one process/GPU only.")
        if task_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Unsupported TabPFN transfer task type: {task_type!r}.")
        self.task_type = task_type
        adapter._vendor()  # Install the repository's official-import shims first.
        self.bindings = _load_runtime_bindings()
        self.adapter = adapter
        self.common_config = common_config
        self.wrapper = adapter.resolve_finetune_estimator(
            task_type,
            common_config,
            seed,
            device,
            initial_checkpoint=initial_checkpoint,
        )
        _require_members(self.wrapper, _REQUIRED_WRAPPER_MEMBERS, owner="official TabPFN fine-tuner")
        self.device = str(self.wrapper.device)
        self._initialize_model()
        model = self.wrapper.finetuned_estimator_.model_
        self.optimizer = self.bindings.get_and_init_optimizer(
            model_parameters=model.parameters(),
            learning_rate=float(common_config.learning_rate),
            weight_decay=float(common_config.weight_decay),
            checkpoint_path=None,
            device=self.device,
        )
        self.scheduler = None
        if common_config.use_lr_scheduler:
            schedule = self.bindings.get_cosine_schedule_with_warmup(
                total_steps=int(total_updates),
                warmup_steps=int(total_updates * float(common_config.warmup_proportion)),
                warmup_only=False,
            )
            self.scheduler = self.bindings.LambdaLR(self.optimizer, lr_lambda=schedule)
        self.use_amp = self.device.startswith("cuda") and self.bindings.torch.cuda.is_available()
        self.scaler = self.bindings.GradScaler() if self.use_amp else None
        self.amp_overflow_retry_count = 0
        self.unstratified_episode_count = 0
        self.performance_options = self.bindings.PerformanceOptions(
            force_recompute_layer=True,
            use_chunkwise_inference=False,
        )

    def _initialize_model(self) -> None:
        kwargs = copy.deepcopy(self.wrapper._estimator_kwargs)
        model_path = kwargs.pop("model_path", None)
        inference_config = copy.deepcopy(kwargs.get("inference_config", {}))
        base = {
            **kwargs,
            "ignore_pretraining_limits": True,
            "device": self.wrapper.device,
            "random_state": self.wrapper.random_state,
            "inference_config": inference_config,
        }
        config = self.wrapper._build_estimator_config(base, self.wrapper.n_estimators_finetune)
        if model_path is not None:
            config["model_path"] = model_path
        inner = self.wrapper._create_estimator(config)
        self.wrapper.finetuned_estimator_ = inner
        self.wrapper._setup_estimator()
        inner._initialize_model_variables()
        inner.model_.to(self.device)
        self.wrapper._ddp_module_ = None
        # In-memory validation consumes live weights through the fitted wrapper.
        # Dataset-bound X/y/schema fields are intentionally not installed here:
        # each validation call supplies its own independently preprocessed data.
        self.wrapper.is_fitted_ = True

    def sample_episode(self, data: Any, *, rng: np.random.Generator) -> TabPFNTransferEpisode:
        """Sample one valid support/query episode from one dataset schema."""
        for _ in range(_MAX_EPISODE_ATTEMPTS):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            episode = self._sample_episode_once(data, seed=seed)
            if not self.wrapper._should_skip_batch(episode.batch):
                return episode
        raise RuntimeError(f"TabPFN could not sample a class-complete episode in {_MAX_EPISODE_ATTEMPTS} attempts.")

    def _sample_episode_once(self, data: Any, *, seed: int) -> TabPFNTransferEpisode:
        X = data.train_X.to_numpy() if hasattr(data.train_X, "to_numpy") else np.asarray(data.train_X)
        y = np.asarray(data.train_y)
        episode_rows = min(int(self.common_config.context_size), len(y))
        query_rows = self.wrapper._get_valid_finetuning_query_size(
            query_size=int(episode_rows * float(self.common_config.query_ratio)),
            y_train=y,
        )
        if query_rows <= 0 or query_rows >= episode_rows:
            raise ValueError(f"Invalid TabPFN support/query split: rows={episode_rows}, query_rows={query_rows}.")
        task_type = getattr(self, "task_type", "binary")
        force_no_stratify = task_type == "regression" or _requires_unstratified_episode_split(
            y, episode_rows=episode_rows, query_rows=query_rows
        )
        if force_no_stratify:
            self.unstratified_episode_count += 1
        splitter = partial(train_test_split, test_size=query_rows, random_state=seed)
        collection = self.bindings.get_preprocessed_dataset_chunks(
            calling_instance=self.wrapper.finetuned_estimator_,
            X_raw=X,
            y_raw=y,
            split_fn=splitter,
            max_data_size=episode_rows,
            model_type="regressor" if task_type == "regression" else "classifier",
            equal_split_size=False,
            data_shuffle_seed=seed,
            preprocessing_random_state=seed,
            force_no_stratify=force_no_stratify,
        )
        if len(collection) == 0:
            raise RuntimeError("TabPFN preprocessing produced no dataset-native episode.")
        item = collection[seed % len(collection)]
        context_rows = int(item.y_context[0].shape[0])
        query_count = int(item.y_query.shape[0])
        return TabPFNTransferEpisode(
            batch=self.bindings.meta_dataset_collator([item]),
            example_count=context_rows + query_count,
        )

    def train_episode(self, episode: TabPFNTransferEpisode) -> float:
        """Apply exactly one optimizer and scheduler update."""
        torch = self.bindings.torch
        batch = episode.batch
        self.wrapper._setup_batch(batch)
        inner = self.wrapper.finetuned_estimator_
        inner.fit_from_preprocessed(
            batch.X_context,
            batch.y_context,
            batch.cat_indices,
            batch.configs,
            performance_options=self.performance_options,
        )
        use_scaler = self.use_amp and self.scaler is not None
        if use_scaler:
            for _attempt in range(_MAX_AMP_STEP_ATTEMPTS):
                self.optimizer.zero_grad()
                with self.bindings.autocast(enabled=True), self.bindings.sdpa_kernel_context():
                    loss = self.wrapper._forward_with_loss(batch)
                if not torch.isfinite(loss.detach()).item():
                    raise FloatingPointError("TabPFN transfer optimizer produced a non-finite loss.")
                scale_before = float(self.scaler.get_scale())
                with self.bindings.sdpa_kernel_context():
                    self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.common_config.grad_clip is not None:
                    self.bindings.clip_grad_norm_(inner.model_.parameters(), float(self.common_config.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if not _amp_step_was_skipped(
                    scale_before=scale_before,
                    scale_after=float(self.scaler.get_scale()),
                ):
                    break
                self.amp_overflow_retry_count += 1
            else:
                raise FloatingPointError(
                    f"TabPFN AMP skipped {_MAX_AMP_STEP_ATTEMPTS} attempts for one optimizer update."
                )
        else:
            self.optimizer.zero_grad()
            with self.bindings.autocast(enabled=False), self.bindings.sdpa_kernel_context():
                loss = self.wrapper._forward_with_loss(batch)
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError("TabPFN transfer optimizer produced a non-finite loss.")
            with self.bindings.sdpa_kernel_context():
                loss.backward()
            if self.common_config.grad_clip is not None:
                self.bindings.clip_grad_norm_(inner.model_.parameters(), float(self.common_config.grad_clip))
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.detach().item())

    def clone_state(self) -> Mapping[str, Any]:
        """Capture the current learned weights without optimizer state."""
        return self.adapter.clone_finetuned_state(self.wrapper)

    def export_state(
        self,
        state: Mapping[str, Any],
        output_dir: Path,
        *,
        update: int,
        score: float,
        checkpoint_mode: str,
    ) -> Path:
        """Persist one validation-selected weights-only state."""
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.adapter.export_finetuned_state(
            self.wrapper,
            state,
            output_dir,
            epoch=update,
            metric=score,
            checkpoint_mode=checkpoint_mode,
        )

    def synchronize(self) -> None:
        """Synchronize the active CUDA device for accurate timing."""
        if self.device.startswith("cuda") and self.bindings.torch.cuda.is_available():
            self.bindings.torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Release gradient references held by the stage optimizer."""
        self.optimizer.zero_grad(set_to_none=True)


@dataclass(frozen=True)
class _RuntimeBindings:
    torch: Any
    PerformanceOptions: Any
    GradScaler: Any
    autocast: Any
    sdpa_kernel_context: Any
    get_preprocessed_dataset_chunks: Any
    meta_dataset_collator: Any
    get_and_init_optimizer: Any
    get_cosine_schedule_with_warmup: Any
    clip_grad_norm_: Any
    LambdaLR: Any


def _load_runtime_bindings() -> _RuntimeBindings:
    torch = importlib.import_module("torch")
    architectures = importlib.import_module("tabpfn.architectures.interface")
    compat = importlib.import_module("tabpfn.finetuning._torch_compat")
    data_util = importlib.import_module("tabpfn.finetuning.data_util")
    train_util = importlib.import_module("tabpfn.finetuning.train_util")
    nn_utils = importlib.import_module("torch.nn.utils")
    lr_scheduler = importlib.import_module("torch.optim.lr_scheduler")
    return _RuntimeBindings(
        torch=torch,
        PerformanceOptions=architectures.PerformanceOptions,
        GradScaler=compat.GradScaler,
        autocast=compat.autocast,
        sdpa_kernel_context=compat.sdpa_kernel_context,
        get_preprocessed_dataset_chunks=data_util.get_preprocessed_dataset_chunks,
        meta_dataset_collator=data_util.meta_dataset_collator,
        get_and_init_optimizer=train_util.get_and_init_optimizer,
        get_cosine_schedule_with_warmup=train_util.get_cosine_schedule_with_warmup,
        clip_grad_norm_=nn_utils.clip_grad_norm_,
        LambdaLR=lr_scheduler.LambdaLR,
    )


def _require_members(value: Any, names: tuple[str, ...], *, owner: str) -> None:
    missing = [name for name in names if not hasattr(value, name)]
    if missing:
        raise RuntimeError(f"Incompatible {owner}; missing pinned transfer primitives: {', '.join(missing)}.")


def _amp_step_was_skipped(*, scale_before: float, scale_after: float) -> bool:
    """Return whether GradScaler detected overflow and skipped the step."""
    return float(scale_after) < float(scale_before)


def _requires_unstratified_episode_split(
    labels: np.ndarray,
    *,
    episode_rows: int,
    query_rows: int,
) -> bool:
    """Return whether sklearn cannot stratify the requested episode split."""
    _, class_counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
    class_count = len(class_counts)
    context_rows = int(episode_rows) - int(query_rows)
    return bool(class_counts.min() < 2 or int(query_rows) < class_count or context_rows < class_count)


def dataset_episode_rng(seed: int, dataset_name: str, dataset_update: int) -> np.random.Generator:
    """Return the stable dataset-local episode stream for one logical update."""
    token = int.from_bytes(
        hashlib.sha256(dataset_name.encode("utf-8")).digest()[:4],
        byteorder="big",
        signed=False,
    )
    return np.random.default_rng(np.random.SeedSequence([int(seed), token, int(dataset_update)]))


__all__ = ["TabPFNTransferEpisode", "TabPFNTransferRuntime", "dataset_episode_rng"]
