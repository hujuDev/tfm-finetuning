"""TabPFN v2/v2.5 adapter: preprocessing boundary, vendor calls, state, and prediction."""

from __future__ import annotations

import contextlib
import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
    normalize_preprocessing_protocol,
)
from finetuning.evaluation.official_epoch_metrics import install_full_official_epoch_eval_metrics
from finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferStageTrainer,
    DatasetTransferTrainerRequest,
    TaskKind,
    validation_metric_for_task,
)
from finetuning.models.registry import FoundationModelSpec
from finetuning.models.tabpfn.preprocessing import NativeCategoricalEncoder
from finetuning.models.tabpfn.vendor import TabPFNBindings, load_tabpfn_bindings
from utils.paths import get_repo_root

_TABPFN_MAX_FEATURES_PER_ESTIMATOR = 500


class TabPFNAdapter:
    """Complete thesis-side fine-tuning slice for TabPFN v2 and v2.5.

    Example:
        >>> from finetuning.models.registry import get_foundation_model_spec
        >>> adapter = TabPFNAdapter(get_foundation_model_spec("tabpfn_v2_5"))
        >>> adapter.resolved_model_params(CommonFineTuneConfig())["use_fixed_preprocessing_seed"]
        False
    """

    model_id = "tabpfn_v2_5"

    def __init__(self, spec: FoundationModelSpec, *, bindings: TabPFNBindings | None = None) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self._bindings = bindings
        self._last_finetuned_estimator: Any | None = None
        self._categorical_features_indices: tuple[int, ...] = ()
        self._preprocessing_protocol = PREPROCESSING_NATIVE
        self._native_category_encoders: dict[int, NativeCategoricalEncoder] = {}
        self._ignore_pretraining_limits = True

    def set_preprocessing_protocol(self, protocol: str) -> None:
        """Configure whether TabPFN sees native or shared-preprocessed inputs."""
        self._preprocessing_protocol = normalize_preprocessing_protocol(protocol)

    def set_common_config(self, config: CommonFineTuneConfig) -> None:
        """Configure run-level TabPFN options used outside the fine-tuning wrapper."""
        self._ignore_pretraining_limits = bool(config.ignore_pretraining_limits)

    def set_feature_schema(self, schema: Mapping[str, Any] | None) -> None:
        """Configure TabPFN categorical columns from the active preprocessing protocol."""
        if not isinstance(schema, Mapping):
            self._categorical_features_indices = ()
            return
        raw_indices = schema.get("categorical_feature_indices") or ()
        self._categorical_features_indices = tuple(int(index) for index in raw_indices)

    def create_dataset_transfer_stage_trainer(
        self,
        request: DatasetTransferTrainerRequest,
    ) -> DatasetTransferStageTrainer:
        """Build the exact update-based Section 3 trainer for TabPFN v2.5."""
        from finetuning.models.tabpfn.transfer import (
            TabPFNDatasetTransferStageTrainer,
        )

        return TabPFNDatasetTransferStageTrainer(
            adapter=self,
            route=request.route,
            config=request.config,
            common_config=request.common_config,
            device=request.device,
            dataset_loader=request.dataset_loader,
        )

    def _checkpoint_for_task(self, task_type: TaskKind) -> str:
        checkpoint = self.spec.regressor_checkpoint if task_type == "regression" else self.spec.classifier_checkpoint
        if checkpoint is None:
            raise ValueError(f"{self.model_id} has no checkpoint for task_type={task_type}.")
        path = Path(checkpoint)
        if not path.is_absolute():
            path = get_repo_root(Path(__file__)) / path
        return str(path)

    def baseline_checkpoint_path(self, task_type: TaskKind) -> Path:
        """Return the local base checkpoint used for zero-shot TabPFN inference."""
        return Path(self._checkpoint_for_task(task_type))

    def _vendor(self) -> TabPFNBindings:
        if self._bindings is None:
            self._bindings = load_tabpfn_bindings()
        return self._bindings

    def _extra_tabpfn_kwargs(self) -> dict[str, Any]:
        if self._preprocessing_protocol != PREPROCESSING_NATIVE or not self._categorical_features_indices:
            return {}
        return {"categorical_features_indices": list(self._categorical_features_indices)}

    def _pretraining_limit_kwargs(self) -> dict[str, Any]:
        return {"ignore_pretraining_limits": bool(self._ignore_pretraining_limits)}

    def resolve_zero_shot_estimator(
        self,
        task_type: TaskKind,
        seed: int,
        device: str | None,
        n_estimators_inference: int,
    ) -> Any:
        """Return a TabPFN estimator backed by the configured base checkpoint."""
        cls = self._vendor().estimator_class(regression=task_type == "regression")
        return cls(
            model_path=self._checkpoint_for_task(task_type),
            n_estimators=int(n_estimators_inference),
            device=device or "auto",
            random_state=int(seed),
            **self._pretraining_limit_kwargs(),
            **self._extra_tabpfn_kwargs(),
        )

    def resolve_finetune_estimator(
        self,
        task_type: TaskKind,
        config: CommonFineTuneConfig,
        seed: int,
        device: str | None,
        initial_checkpoint: Path | None = None,
    ) -> Any:
        """Return a fresh TabPFN wrapper, optionally initialized from learned weights."""
        cls = self._vendor().finetuner_class(regression=task_type == "regression")
        params = self.resolved_model_params(config)
        params["device"] = _resolve_finetune_device(device)
        params["random_state"] = int(seed)
        params["eval_metric"] = validation_metric_for_task(task_type, model_id=self.model_id)
        model_path = initial_checkpoint or Path(self._checkpoint_for_task(task_type))
        extra_kwargs = {"model_path": str(model_path), **self._extra_tabpfn_kwargs()}
        extra_kwargs.update(self._pretraining_limit_kwargs())
        if task_type == "regression":
            params["extra_regressor_kwargs"] = extra_kwargs
        else:
            params["extra_classifier_kwargs"] = extra_kwargs
        estimator = cls(**params)
        install_full_official_epoch_eval_metrics(estimator)
        return estimator

    def fit_finetuned(
        self,
        estimator: Any,
        train_X: pd.DataFrame,
        train_y: np.ndarray,
        val_X: pd.DataFrame,
        val_y: np.ndarray,
        output_dir: Path,
    ) -> Any:
        """Fit the official wrapper with explicit inner validation data."""
        encoder = self._fit_native_category_encoder(train_X)
        self._native_category_encoders[id(estimator)] = encoder
        result = estimator.fit(
            encoder.transform(train_X),
            train_y,
            X_val=encoder.transform(val_X),
            y_val=val_y,
            output_dir=output_dir,
        )
        self._last_finetuned_estimator = result
        return result

    def export_checkpoint(self, output_dir: Path, checkpoint_mode: str) -> Path:
        """Return the selected official checkpoint written by TabPFN."""
        if checkpoint_mode == "final":
            checkpoints = sorted(
                (path for path in output_dir.glob("checkpoint_*_*.pth") if not path.name.endswith("_best.pth")),
                key=_tabpfn_checkpoint_sort_key,
            )
            if checkpoints:
                return checkpoints[-1]
            live = self._export_live_finetuned_estimator(output_dir, checkpoint_mode)
            if live is not None:
                return live
        best = sorted(output_dir.glob("checkpoint_*_best.pth"))
        if best:
            return best[-1]
        raise FileNotFoundError(f"No TabPFN best-validation checkpoint found in {output_dir}.")

    def _export_live_finetuned_estimator(self, output_dir: Path, checkpoint_mode: str) -> Path | None:
        model_ckpt = output_dir / "model.ckpt"
        finetuned_estimator = getattr(self._last_finetuned_estimator, "finetuned_estimator_", None)
        if finetuned_estimator is None:
            return model_ckpt if model_ckpt.exists() else None
        self._vendor().save_model(
            model=finetuned_estimator,
            save_path=model_ckpt,
            additional_fields={"checkpoint_export_mode": f"{checkpoint_mode}_in_memory_fallback"},
        )
        return model_ckpt

    def load_checkpoint_for_inference(
        self,
        checkpoint_path: Path,
        task_type: TaskKind,
        device: str | None,
        n_estimators_inference: int,
        seed: int,
    ) -> Any:
        """Load a fine-tuned checkpoint into a TabPFN inference estimator."""
        cls = self._vendor().estimator_class(regression=task_type == "regression")
        return cls(
            model_path=str(checkpoint_path),
            n_estimators=int(n_estimators_inference),
            device=device or "auto",
            random_state=int(seed),
            **self._pretraining_limit_kwargs(),
            **self._extra_tabpfn_kwargs(),
        )

    def fit_inference_context(self, estimator: Any, context_X: pd.DataFrame, context_y: np.ndarray) -> Any:
        """Fit the TabPFN in-context estimator."""
        encoder = self._fit_native_category_encoder(context_X)
        self._native_category_encoders[id(estimator)] = encoder
        return estimator.fit(encoder.transform(context_X), context_y)

    def predict_for_metrics(self, estimator: Any, query_X: pd.DataFrame, task_type: TaskKind) -> Any:
        """Return probabilities for classifiers and labels for regression."""
        query_X = self._transform_with_native_category_encoder(estimator, query_X)
        if task_type == "regression":
            return estimator.predict(query_X)
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(query_X)
        return estimator.predict(query_X)

    def clone_finetuned_state(self, estimator: Any) -> dict[str, Any]:
        """Return a CPU snapshot of the live fine-tuned TabPFN model weights."""
        inner = _inner_finetuned_estimator(estimator)
        models = getattr(inner, "models_", None)
        if not models:
            raise RuntimeError("TabPFN fine-tuned model is not available for in-memory checkpointing.")
        state: dict[str, Any] = {
            "model_state_dicts": [_clone_state_dict_to_cpu(model.state_dict()) for model in models],
        }
        bardist = getattr(inner, "znorm_space_bardist_", None)
        if bardist is not None and hasattr(bardist, "state_dict"):
            state["znorm_space_bardist_state_dict"] = _clone_state_dict_to_cpu(bardist.state_dict())
        return state

    def apply_finetuned_state(self, estimator: Any, state: Mapping[str, Any]) -> None:
        """Restore a TabPFN in-memory state onto the live fine-tuned estimator."""
        inner = _inner_finetuned_estimator(estimator)
        models = list(getattr(inner, "models_", None) or [])
        state_dicts = list(state.get("model_state_dicts") or [])
        if not models or len(models) != len(state_dicts):
            raise RuntimeError(
                "TabPFN in-memory state does not match the live estimator "
                f"(models={len(models)} state_dicts={len(state_dicts)})."
            )
        for model, state_dict in zip(models, state_dicts, strict=True):
            model.load_state_dict(state_dict)
        bardist_state = state.get("znorm_space_bardist_state_dict")
        bardist = getattr(inner, "znorm_space_bardist_", None)
        if bardist_state is not None and bardist is not None and hasattr(bardist, "load_state_dict"):
            bardist.load_state_dict(bardist_state)

    def build_in_memory_inference_estimator(
        self,
        estimator: Any,
        task_type: TaskKind,
        device: Any,
        n_estimators_inference: int,
    ) -> Any:
        """Build a TabPFN inference estimator from the live fine-tuned model."""
        vendor = self._vendor()
        model_class = vendor.estimator_class(regression=task_type == "regression")
        eval_init_args = {
            "n_estimators": int(n_estimators_inference),
            "device": device or "auto",
            **self._pretraining_limit_kwargs(),
            **self._extra_tabpfn_kwargs(),
        }
        return vendor.clone_model_for_evaluation(
            _inner_finetuned_estimator(estimator),
            eval_init_args,
            model_class,
        )

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
        """Write a TabPFN checkpoint from an in-memory state without keeping epoch files."""
        inner = _inner_finetuned_estimator(estimator)
        original_state = self.clone_finetuned_state(estimator)
        checkpoint_epoch = 0 if epoch is None else int(epoch)
        checkpoint_path = output_dir / ("best.pth" if checkpoint_mode == "best" else f"epoch{checkpoint_epoch}.pth")
        try:
            self.apply_finetuned_state(estimator, state)
            additional_fields: dict[str, Any] = {
                "epoch": checkpoint_epoch,
                "checkpoint_export_mode": f"{checkpoint_mode}_in_memory_state",
            }
            if metric is not None:
                additional_fields["best_metric"] = float(metric)
            self._vendor().save_model(
                model=inner,
                save_path=checkpoint_path,
                additional_fields=additional_fields,
            )
        finally:
            self.apply_finetuned_state(estimator, original_state)
        return checkpoint_path

    def minimum_required_estimators(self, feature_count: int) -> int:
        """Return TabPFN's estimator minimum for feature-block coverage."""
        return max(1, math.ceil(max(1, int(feature_count)) / _TABPFN_MAX_FEATURES_PER_ESTIMATOR))

    def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, Any]:
        """Map shared settings to official TabPFN parameter names."""
        return {
            "epochs": int(common_config.epochs),
            "learning_rate": float(common_config.learning_rate),
            "weight_decay": float(common_config.weight_decay),
            "n_finetune_ctx_plus_query_samples": int(common_config.context_size),
            "finetune_ctx_query_split_ratio": float(common_config.query_ratio),
            "n_estimators_finetune": int(common_config.n_estimators_finetune),
            "n_estimators_validation": int(common_config.n_estimators_validation),
            "n_estimators_final_inference": int(common_config.n_estimators_final_inference),
            "early_stopping": bool(common_config.early_stopping),
            "early_stopping_patience": int(common_config.patience),
            "min_delta": float(common_config.min_delta),
            "grad_clip_value": float(common_config.grad_clip),
            "use_lr_scheduler": bool(common_config.use_lr_scheduler),
            "lr_warmup_only": False,
            "use_activation_checkpointing": True,
            "use_fixed_preprocessing_seed": False,
        }

    def _fit_native_category_encoder(self, frame: pd.DataFrame) -> NativeCategoricalEncoder:
        if self._preprocessing_protocol != PREPROCESSING_NATIVE:
            return NativeCategoricalEncoder.empty()
        return NativeCategoricalEncoder.fit(
            frame,
            categorical_feature_indices=self._categorical_features_indices,
        )

    def _transform_with_native_category_encoder(self, estimator: Any, frame: pd.DataFrame) -> pd.DataFrame:
        if self._preprocessing_protocol != PREPROCESSING_NATIVE:
            return frame
        encoder = self._native_category_encoders.get(id(estimator))
        if encoder is None:
            return frame
        return encoder.transform(frame)


def _inner_finetuned_estimator(estimator: Any) -> Any:
    inner = getattr(estimator, "finetuned_estimator_", estimator)
    if inner is None:
        raise RuntimeError("TabPFN fine-tuned estimator is not available.")
    return inner


def _clone_state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        item = value
        if hasattr(item, "detach"):
            item = item.detach()
        if hasattr(item, "cpu"):
            item = item.cpu()
        if hasattr(item, "clone"):
            item = item.clone()
        cloned[key] = item
    return cloned


def _resolve_finetune_device(device: str | None) -> str:
    """Return an explicit Torch device for official TabPFN fine-tuning.

    Example:
        >>> _resolve_finetune_device("cpu")
        'cpu'
        >>> _resolve_finetune_device(" auto ") in {"cpu", "cuda"}
        True
    """
    requested = str(device).strip() if device is not None else ""
    if requested and requested.lower() != "auto":
        return requested
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tabpfn_checkpoint_sort_key(path: Path) -> tuple[int, str]:
    with contextlib.suppress(ValueError):
        return int(path.stem.rsplit("_", 1)[-1]), path.name
    return -1, path.name
