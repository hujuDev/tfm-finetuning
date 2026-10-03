"""Choose official Mitra training or the thesis-owned local chunked strategy.

Official training delegates to AutoGluon's sklearn implementation. Local
training keeps the vendor model and optimizer but interleaves ensemble members
epoch-by-epoch when the shared in-memory controller is active.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from tfm_finetuning.finetuning.models.mitra.common import (
    MITRA_INTERFACE_LOCAL_CHUNKED,
    MITRA_INTERFACE_OFFICIAL_SKLEARN,
)

_PUBLIC_MITRA_MODULE = "tfm_finetuning.finetuning.models.mitra.model"
_CONFIGURED_MITRA_CLASSES: dict[type[Any], type[Any]] = {}


class _MitraEstimatorHost(Protocol):
    """Vendor methods required by the dynamically composed Mitra class."""

    def _create_config(self, *args: Any, **kwargs: Any) -> Any: ...

    def _train_ensemble(self, *args: Any, **kwargs: Any) -> Any: ...


class _ConfiguredMitraMixin:
    """Configuration and local-training behavior mixed into a vendor estimator."""

    device: str
    hf_model: str | None
    n_estimators: int
    state_dict: str | Path
    train_time: float
    trainers: list[Any]
    verbose: bool

    def __init__(
        self,
        *args: Any,
        weight_decay: float,
        max_samples_support: int,
        max_samples_query: int,
        query_ratio: float,
        grad_clip: float,
        use_lr_scheduler: bool,
        warmup_proportion: float,
        min_delta: float,
        mitra_finetune_interface: str,
        mitra_scheduler_policy: str,
        mitra_chunk_min_rows: int,
        **kwargs: Any,
    ) -> None:
        self.foundation_weight_decay = float(weight_decay)
        self.foundation_max_samples_support = int(max_samples_support)
        self.foundation_max_samples_query = int(max_samples_query)
        self.foundation_query_ratio = float(query_ratio)
        self.foundation_grad_clip = float(grad_clip)
        self.foundation_use_lr_scheduler = bool(use_lr_scheduler)
        self.foundation_warmup_proportion = float(warmup_proportion)
        self.foundation_min_delta = float(min_delta)
        self.foundation_mitra_finetune_interface = str(mitra_finetune_interface)
        self.foundation_mitra_scheduler_policy = str(mitra_scheduler_policy)
        self.foundation_mitra_chunk_min_rows = int(mitra_chunk_min_rows)
        super().__init__(*args, **kwargs)

    def _create_config(self, *args: Any, **kwargs: Any) -> Any:
        host = cast("_MitraEstimatorHost", super())
        cfg, model_cls = host._create_config(*args, **kwargs)
        cfg.hyperparams["weight_decay"] = float(self.foundation_weight_decay)
        cfg.hyperparams["max_samples_support"] = int(self.foundation_max_samples_support)
        cfg.hyperparams["max_samples_query"] = int(self.foundation_max_samples_query)
        cfg.hyperparams["foundation_mitra_finetune_interface"] = str(self.foundation_mitra_finetune_interface)
        if self.foundation_mitra_finetune_interface == MITRA_INTERFACE_LOCAL_CHUNKED:
            cfg.hyperparams["grad_clip"] = float(self.foundation_grad_clip)
            cfg.hyperparams["lr_scheduler"] = bool(self.foundation_use_lr_scheduler)
            cfg.hyperparams["foundation_warmup_proportion"] = float(self.foundation_warmup_proportion)
            cfg.hyperparams["early_stopping_delta"] = float(self.foundation_min_delta)
            cfg.hyperparams["foundation_context_size"] = int(self.foundation_max_samples_support) + int(
                self.foundation_max_samples_query
            )
            cfg.hyperparams["foundation_query_ratio"] = float(self.foundation_query_ratio)
            cfg.hyperparams["foundation_scheduler_policy"] = str(self.foundation_mitra_scheduler_policy)
            cfg.hyperparams["foundation_mitra_chunk_min_rows"] = int(self.foundation_mitra_chunk_min_rows)
        return cfg, model_cls

    def _train_local_chunked_trainers(
        self,
        *,
        trainers: list[Any],
        controller: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        max_epochs: int,
    ) -> None:
        if controller is None:
            for trainer in trainers:
                trainer.train(X_train, y_train, X_valid, y_valid)
            return

        states = [trainer.begin_local_fit(X_train, y_train, X_valid, y_valid) for trainer in trainers]
        for epoch in range(1, int(max_epochs) + 1):
            for trainer, state in zip(trainers, states, strict=True):
                if not state.stopped:
                    trainer.run_local_epoch(state, epoch=epoch)
                if state.stopped and not state.finished:
                    trainer.finish_local_fit(state)
            controller.after_validation(
                self,
                epoch=epoch,
                primary=None,
                validation_device=self.device,
            )
            if all(state.stopped for state in states):
                break
        for trainer, state in zip(trainers, states, strict=True):
            trainer.finish_local_fit(state)

    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        task: Any,
        dim_output: int,
        n_classes: int = 0,
        time_limit: float | None = None,
    ) -> Any:
        if self.foundation_mitra_finetune_interface == MITRA_INTERFACE_OFFICIAL_SKLEARN:
            host = cast("_MitraEstimatorHost", super())
            return host._train_ensemble(
                X_train,
                y_train,
                X_valid,
                y_valid,
                task,
                dim_output,
                n_classes,
                time_limit,
            )
        if self.foundation_mitra_finetune_interface != MITRA_INTERFACE_LOCAL_CHUNKED:
            raise ValueError(f"Unsupported Mitra fine-tune interface: {self.foundation_mitra_finetune_interface!r}")

        cfg, model_cls = self._create_config(task, dim_output, time_limit)
        torch = importlib.import_module("torch")

        self.trainers.clear()
        self.train_time = 0.0
        controller = getattr(self, "_foundation_in_memory_controller", None)
        initial_states = getattr(self, "_foundation_initial_trainer_states", None)
        if initial_states is not None and len(initial_states) != int(self.n_estimators):
            raise RuntimeError(
                "Mitra warm starts require an exact ensemble match "
                f"(checkpoint={len(initial_states)}, configured={self.n_estimators})."
            )
        try:
            self.trainers.extend(
                _fresh_local_trainers(
                    self,
                    cfg=cfg,
                    model_cls=model_cls,
                    task=task,
                    dim_output=dim_output,
                    n_classes=n_classes,
                    states=initial_states,
                )
            )

            started_at = time.time()
            self._train_local_chunked_trainers(
                trainers=self.trainers,
                controller=controller,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                max_epochs=int(cfg.hyperparams["max_epochs"]),
            )
            self.train_time += time.time() - started_at
            self._foundation_initial_trainer_states = None
        except torch.cuda.OutOfMemoryError:
            self.trainers.clear()
            raise
        return self


def restore_portable_mitra_ensemble(
    estimator: Any,
    states: list[dict[str, Any]],
    *,
    task: Any,
    dim_output: int,
    n_classes: int,
) -> None:
    """Rebuild inference trainers from CPU weights without serialized training state."""
    if getattr(estimator, "foundation_mitra_finetune_interface", None) != MITRA_INTERFACE_LOCAL_CHUNKED:
        raise RuntimeError("Portable Mitra checkpoints require the local_chunked interface.")
    if len(states) != int(estimator.n_estimators):
        raise RuntimeError(
            "Mitra portable checkpoint ensemble does not match its estimator template "
            f"(states={len(states)}, configured={estimator.n_estimators})."
        )
    cfg, model_cls = estimator._create_config(task, int(dim_output), None)
    estimator.trainers = _fresh_local_trainers(
        estimator,
        cfg=cfg,
        model_cls=model_cls,
        task=task,
        dim_output=int(dim_output),
        n_classes=int(n_classes),
        states=states,
    )


def _fresh_local_trainers(
    estimator: Any,
    *,
    cfg: Any,
    model_cls: type[Any],
    task: Any,
    dim_output: int,
    n_classes: int,
    states: list[dict[str, Any]] | None,
) -> list[Any]:
    local_trainer_module = importlib.import_module("tfm_finetuning.finetuning.models.mitra.local_trainer")
    local_trainer_cls = local_trainer_module.LocalMitraTrainer
    sklearn_module = importlib.import_module("autogluon.tabular.models.mitra.sklearn_interface")
    rng = np.random.RandomState(cfg.seed)
    trainers: list[Any] = []
    for estimator_index in range(int(estimator.n_estimators)):
        model = _fresh_mitra_model(
            estimator,
            cfg=cfg,
            model_cls=model_cls,
            task=task,
            dim_output=dim_output,
            use_hf=bool(getattr(sklearn_module, "USE_HF", True)),
        )
        if states is not None:
            try:
                model.load_state_dict(states[estimator_index])
            except Exception as exc:
                raise RuntimeError(
                    "Mitra warm-start weights are incompatible with the freshly "
                    f"constructed ensemble member {estimator_index}."
                ) from exc
        trainer = local_trainer_cls(
            cfg,
            model,
            n_classes=n_classes,
            device=estimator.device,
            rng=rng,
            verbose=estimator.verbose,
        )
        trainer.experiment_logger = getattr(estimator, "experiment_logger", None)
        trainers.append(trainer)
    return trainers


def _fresh_mitra_model(
    estimator: Any,
    *,
    cfg: Any,
    model_cls: type[Any],
    task: Any,
    dim_output: int,
    use_hf: bool,
) -> Any:
    if use_hf:
        if estimator.hf_model is None:
            raise AssertionError("hf_model must not be None.")
        snapshot = Path(estimator.hf_model).expanduser()
        if snapshot.is_dir():
            return _load_local_mitra_snapshot(model_cls, snapshot=snapshot, device=estimator.device)
        return model_cls.from_pretrained(estimator.hf_model, device=estimator.device)
    return model_cls(
        dim=cfg.hyperparams["dim"],
        dim_output=dim_output,
        n_layers=cfg.hyperparams["n_layers"],
        n_heads=cfg.hyperparams["n_heads"],
        task=str(getattr(task, "value", task)).upper(),
        use_pretrained_weights=True,
        path_to_weights=Path(estimator.state_dict),
        device=estimator.device,
    )


def _load_local_mitra_snapshot(model_cls: type[Any], *, snapshot: Path, device: str) -> Any:
    """Load the exact project-materialized Mitra config and safetensor bytes."""
    config_path = Path(snapshot) / "config.json"
    weights_path = Path(snapshot) / "model.safetensors"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid materialized Mitra config: {config_path}") from exc
    required = {"dim", "dim_output", "n_layers", "n_heads", "task"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise RuntimeError(f"Materialized Mitra config lacks required architecture fields: {config_path}")
    try:
        from safetensors.torch import load_file

        state = load_file(str(weights_path), device=str(device))
    except Exception as exc:
        raise RuntimeError(f"Could not load materialized Mitra weights: {weights_path}") from exc
    model = model_cls(
        dim=config["dim"],
        dim_output=config["dim_output"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        task=str(config["task"]).upper(),
        use_pretrained_weights=False,
        path_to_weights="",
        device=device,
    )
    model.load_state_dict(state)
    return model


@contextmanager
def local_mitra_snapshot_loading(snapshot: str | Path | None) -> Iterator[None]:
    """Make the vendor zero-shot fit load a resolved local snapshot exactly.

    AutoGluon's public estimator always calls ``Tab2D.from_pretrained`` with a
    Hugging Face repository ID. Transfer training replaces that call only while fitting
    a local content-addressed base snapshot; ordinary Mitra runs are unchanged.
    """
    if snapshot is None or not Path(snapshot).expanduser().is_dir():
        yield
        return
    module = importlib.import_module("autogluon.tabular.models.mitra._internal.models.tab2d")
    model_cls = module.Tab2D
    original_descriptor = model_cls.__dict__["from_pretrained"]
    original = model_cls.from_pretrained
    resolved_snapshot = Path(snapshot).expanduser().resolve()

    def _from_pretrained(cls: type[Any], path_or_repo_id: str, device: str = "cuda") -> Any:
        candidate = Path(path_or_repo_id).expanduser()
        if candidate.is_dir() and candidate.resolve() == resolved_snapshot:
            return _load_local_mitra_snapshot(cls, snapshot=resolved_snapshot, device=device)
        return original(path_or_repo_id, device=device)

    model_cls.from_pretrained = classmethod(_from_pretrained)
    try:
        yield
    finally:
        model_cls.from_pretrained = original_descriptor


def _configured_mitra_class(base_cls: type[Any]) -> type[Any]:
    existing = _CONFIGURED_MITRA_CLASSES.get(base_cls)
    if existing is not None:
        return existing

    class ConfiguredMitra(_ConfiguredMitraMixin, base_cls):
        pass

    class_name = f"Configured{base_cls.__name__}"
    ConfiguredMitra.__name__ = class_name
    ConfiguredMitra.__qualname__ = class_name
    ConfiguredMitra.__module__ = _PUBLIC_MITRA_MODULE
    globals()[class_name] = ConfiguredMitra
    public_module = sys.modules.get(_PUBLIC_MITRA_MODULE) or importlib.import_module(_PUBLIC_MITRA_MODULE)
    setattr(public_module, class_name, ConfiguredMitra)
    _CONFIGURED_MITRA_CLASSES[base_cls] = ConfiguredMitra
    return ConfiguredMitra
