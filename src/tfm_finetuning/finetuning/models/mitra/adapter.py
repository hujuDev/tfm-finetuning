"""Mitra adapter: common setup around official and local-chunked training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
    normalize_preprocessing_protocol,
)
from tfm_finetuning.finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferStageTrainer,
    DatasetTransferTrainerRequest,
    TaskKind,
)
from tfm_finetuning.finetuning.models.mitra.checkpoints import (
    apply_estimator_state,
    captured_trainer_states,
    clone_estimator_state,
    epoch_checkpoint_sort_key,
    estimator_state_with_trainer_states,
    read_checkpoint_for_finetuning,
    read_checkpoint_package,
    write_checkpoint_package,
    write_checkpoint_with_finetuned_state,
    write_checkpoint_with_trainer_states,
)
from tfm_finetuning.finetuning.models.mitra.common import MITRA_INTERFACE_LOCAL_CHUNKED
from tfm_finetuning.finetuning.models.mitra.config import (
    MitraFineTuneOptions,
    mitra_metric_for_task,
    resolve_mitra_device,
)
from tfm_finetuning.finetuning.models.mitra.configured import (
    _configured_mitra_class,
    local_mitra_snapshot_loading,
)
from tfm_finetuning.finetuning.models.mitra.diagnostics import (
    best_logged_epoch,
    capture_epoch_logging,
    final_logged_epoch,
    local_trainer_diagnostics,
)
from tfm_finetuning.finetuning.models.mitra.preprocessing import (
    adapter_input_preprocessing_metadata,
    context_features,
    encode_with_classes,
    fit_dense_label_encoder,
    frame_values,
    has_trained_ensemble,
    is_classifier,
    predict_features,
    refit_trainer_preprocessors,
)
from tfm_finetuning.finetuning.models.mitra.vendor import MitraBindings, load_mitra_bindings
from tfm_finetuning.finetuning.models.preprocessing import (
    FiniteFeaturePreprocessor,
)
from tfm_finetuning.finetuning.models.registry import ModelSpec


class MitraAdapter:
    """Complete thesis-side Mitra slice for official and local-chunked training.

    Example:
        >>> from tfm_finetuning.finetuning.models.registry import get_model_spec
        >>> adapter = MitraAdapter(get_model_spec("mitra"))
        >>> adapter.resolved_model_params(CommonFineTuneConfig(epochs=3))["fine_tune_steps"]
        3
    """

    model_id = "mitra"

    def __init__(self, spec: ModelSpec, *, bindings: MitraBindings | None = None) -> None:
        self.spec = spec
        self._bindings = bindings
        self._last_finetuned_estimator: Any | None = None
        self._last_finetune_diagnostics: dict[str, Any] = {}
        self._preprocessing_protocol = PREPROCESSING_NATIVE

    def set_preprocessing_protocol(self, protocol: str) -> None:
        """Record whether Mitra sees native or shared-preprocessed features."""
        self._preprocessing_protocol = normalize_preprocessing_protocol(protocol)

    def _vendor(self) -> MitraBindings:
        if self._bindings is None:
            self._bindings = load_mitra_bindings()
        return self._bindings

    def _checkpoint_for_task(self, task_type: TaskKind) -> str:
        checkpoint = self.spec.regressor_checkpoint if task_type == "regression" else self.spec.classifier_checkpoint
        if checkpoint is None:
            raise ValueError(f"{self.model_id} has no checkpoint for task_type={task_type}.")
        return checkpoint

    def resolve_zero_shot_estimator(
        self,
        task_type: TaskKind,
        seed: int,
        device: str | None,
        n_estimators_inference: int,
    ) -> Any:
        """Return a Mitra estimator backed by the configured Hugging Face checkpoint."""
        cls = self._vendor().estimator_class(regression=task_type == "regression")
        return cls(
            n_estimators=int(n_estimators_inference),
            device=resolve_mitra_device(device),
            fine_tune=False,
            fine_tune_steps=0,
            metric=mitra_metric_for_task(task_type),
            hf_model=self._checkpoint_for_task(task_type),
            seed=int(seed),
            verbose=False,
        )

    def resolve_finetune_estimator(
        self,
        task_type: TaskKind,
        config: CommonFineTuneConfig,
        seed: int,
        device: str | None,
        initial_checkpoint: Path | None = None,
    ) -> Any:
        """Return a fresh Mitra estimator, optionally warm-started by model weights."""
        base_cls = self._vendor().estimator_class(regression=task_type == "regression")
        cls = _configured_mitra_class(base_cls)
        options = MitraFineTuneOptions.from_common(config)
        if initial_checkpoint is not None and options.interface != MITRA_INTERFACE_LOCAL_CHUNKED:
            raise ValueError("Mitra checkpoint warm starts require mitra_finetune_interface='local_chunked'.")
        estimator = cls(
            **options.estimator_kwargs(
                task_type=task_type,
                checkpoint=self._checkpoint_for_task(task_type),
                seed=seed,
                device=device,
            )
        )
        estimator.n_estimators_finetune = int(config.n_estimators_finetune)
        estimator.n_estimators_validation = int(config.n_estimators_validation)
        estimator.n_estimators_inference = int(config.n_estimators_final_inference)
        estimator._tfm_common_config = config
        if initial_checkpoint is not None:
            estimator._tfm_initial_trainer_states = read_checkpoint_for_finetuning(
                initial_checkpoint,
                task_type=task_type,
                expected_ensemble_size=options.n_estimators,
            )
            estimator._tfm_initial_checkpoint = str(initial_checkpoint)
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
        """Fit Mitra with explicit validation data and write project checkpoints."""
        preprocessor = FiniteFeaturePreprocessor.fit(train_X)
        estimator._tfm_mitra_feature_preprocessor = preprocessor
        train_X_fit = preprocessor.transform(train_X)
        val_X_fit = preprocessor.transform(val_X)
        self._last_finetune_diagnostics = {
            "mitra_adapter_input_preprocessing": adapter_input_preprocessing_metadata(
                preprocessor=preprocessor,
                protocol=self._preprocessing_protocol,
                train_X=train_X,
                train_X_fit=train_X_fit,
                applied_to="mitra_fit_inputs",
            )
        }
        with capture_epoch_logging(estimator):
            if is_classifier(estimator):
                classes, train_dense = fit_dense_label_encoder(train_y)
                val_dense = encode_with_classes(val_y, classes)
                estimator.classes_ = classes
                result = estimator.fit(train_X_fit, train_dense, X_val=val_X_fit, y_val=val_dense)
            else:
                estimator.classes_ = None
                result = estimator.fit(
                    train_X_fit,
                    np.asarray(train_y, dtype=np.float32),
                    X_val=val_X_fit,
                    y_val=np.asarray(val_y),
                )
        self._last_finetuned_estimator = result
        local_diagnostics = local_trainer_diagnostics(result)
        if local_diagnostics:
            self._last_finetune_diagnostics["mitra_local_trainer"] = local_diagnostics
        final_epoch = final_logged_epoch(result)
        best_epoch = best_logged_epoch(result)
        result._tfm_final_epoch = final_epoch
        result._tfm_best_epoch = best_epoch
        suppress_checkpoints = bool(
            getattr(result, "_tfm_suppress_adapter_checkpoints", False)
            or getattr(estimator, "_tfm_suppress_adapter_checkpoints", False)
        )
        if not suppress_checkpoints:
            write_checkpoint_package(result, output_dir / "best.ckpt")
            final_states = captured_trainer_states(result, "_tfm_final_state")
            if final_states:
                write_checkpoint_with_trainer_states(
                    result,
                    output_dir / f"epoch{final_epoch}.ckpt",
                    final_states,
                )
            else:
                write_checkpoint_package(result, output_dir / f"epoch{final_epoch}.ckpt")
        return result

    def export_checkpoint(self, output_dir: Path, checkpoint_mode: str) -> Path:
        """Return the selected project-owned Mitra checkpoint package."""
        if checkpoint_mode == "final":
            checkpoints = sorted(output_dir.glob("epoch*.ckpt"), key=epoch_checkpoint_sort_key)
            if checkpoints:
                return checkpoints[-1]
        best = output_dir / "best.ckpt"
        if best.exists():
            return best
        checkpoints = sorted(output_dir.glob("epoch*.ckpt"), key=epoch_checkpoint_sort_key)
        if checkpoints:
            return checkpoints[-1]
        if self._last_finetuned_estimator is not None:
            write_checkpoint_package(self._last_finetuned_estimator, best)
            return best
        raise FileNotFoundError(f"No Mitra checkpoint found in {output_dir}.")

    def consume_finetune_diagnostics(self) -> dict[str, Any]:
        """Return and clear diagnostics captured during the last Mitra fine-tune."""
        diagnostics = dict(self._last_finetune_diagnostics)
        self._last_finetune_diagnostics = {}
        return diagnostics

    def load_checkpoint_for_inference(
        self,
        checkpoint_path: Path,
        task_type: TaskKind,
        device: str | None,
        n_estimators_inference: int,
        seed: int,
    ) -> Any:
        """Load a project-owned Mitra checkpoint package."""
        # Portable packages retain the configured estimator constructor but not
        # the source-bound trainers. Register its stable public pickle name
        # before unpickling in a fresh evaluation process.
        base_cls = self._vendor().estimator_class(regression=task_type == "regression")
        _configured_mitra_class(base_cls)
        estimator = read_checkpoint_package(
            checkpoint_path,
            device=resolve_mitra_device(str(device)) if device is not None else None,
        )
        if not getattr(estimator, "trainers", None):
            raise RuntimeError(f"Mitra checkpoint has no trained ensemble members: {checkpoint_path}.")
        portable_metadata = getattr(estimator, "_tfm_portable_checkpoint_metadata", None)
        if isinstance(portable_metadata, dict):
            task_family = "regression" if task_type == "regression" else "classification"
            checkpoint_family = portable_metadata.get("task_family")
            if checkpoint_family != task_family:
                raise RuntimeError(
                    "Mitra checkpoint task family is incompatible with inference "
                    f"({checkpoint_family!r} != {task_family!r})."
                )
            ensemble_size = int(portable_metadata.get("ensemble_size", len(estimator.trainers)))
            if ensemble_size != len(estimator.trainers) or ensemble_size != int(n_estimators_inference):
                raise RuntimeError(
                    "Mitra portable inference requires an exact ensemble match "
                    f"(checkpoint={ensemble_size}, trainers={len(estimator.trainers)}, "
                    f"requested={n_estimators_inference})."
                )
        estimator.n_estimators = int(n_estimators_inference)
        estimator.n_estimators_inference = int(n_estimators_inference)
        estimator.seed = int(seed)
        shared_rng = np.random.RandomState(int(seed))
        for trainer in estimator.trainers:
            cfg = getattr(trainer, "cfg", None)
            if cfg is not None:
                cfg.seed = int(seed)
            if hasattr(trainer, "rng"):
                trainer.rng = shared_rng
        if device is not None:
            estimator.device = resolve_mitra_device(str(device))
        return estimator

    def fit_inference_context(self, estimator: Any, context_X: pd.DataFrame, context_y: np.ndarray) -> Any:
        """Bind the ICL context rows to a zero-shot or fine-tuned Mitra estimator."""
        context_X_fit = context_features(estimator, context_X)
        if is_classifier(estimator):
            classes, dense = fit_dense_label_encoder(context_y)
            estimator.classes_ = classes
            if not has_trained_ensemble(estimator):
                with local_mitra_snapshot_loading(getattr(estimator, "hf_model", None)):
                    return estimator.fit(context_X_fit, dense, X_val=context_X_fit, y_val=dense)
            refit_trainer_preprocessors(estimator, context_X_fit, dense)
            estimator.X = frame_values(context_X_fit)
            estimator.y = dense
            return estimator
        if not has_trained_ensemble(estimator):
            y = np.asarray(context_y, dtype=np.float32)
            with local_mitra_snapshot_loading(getattr(estimator, "hf_model", None)):
                return estimator.fit(context_X_fit, y, X_val=context_X_fit, y_val=y)
        refit_trainer_preprocessors(estimator, context_X_fit, np.asarray(context_y, dtype=np.float32))
        estimator.X = frame_values(context_X_fit)
        estimator.y = np.asarray(context_y, dtype=np.float32)
        return estimator

    def predict_for_metrics(self, estimator: Any, query_X: pd.DataFrame, task_type: TaskKind) -> Any:
        """Return Mitra probabilities or regression predictions for metric computation."""
        query_X = predict_features(estimator, query_X)
        if task_type == "regression":
            return estimator.predict(query_X)
        proba = np.asarray(estimator.predict_proba(query_X), dtype=np.float64)
        classes = getattr(estimator, "classes_", None)
        if classes is None or proba.shape[1] == len(classes):
            return proba
        padded = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
        padded[:, : min(proba.shape[1], padded.shape[1])] = proba[:, : padded.shape[1]]
        return padded

    def clone_finetuned_state(self, estimator: Any) -> dict[str, Any]:
        """Return a CPU snapshot of the current Mitra ensemble weights."""
        state = clone_estimator_state(estimator)
        if not state["trainer_states"]:
            raise RuntimeError("Mitra fine-tuned trainers are not available for in-memory checkpointing.")
        return state

    def apply_finetuned_state(self, estimator: Any, state: dict[str, Any]) -> None:
        """Apply a captured Mitra ensemble state to the live estimator."""
        apply_estimator_state(estimator, state)

    def capture_finetuned_states(self, estimator: Any) -> dict[str, dict[str, Any]]:
        """Return best/final states captured from Mitra trainer checkpoint callbacks."""
        payload: dict[str, dict[str, Any]] = {}
        best_states = captured_trainer_states(estimator, "_tfm_best_state")
        if best_states:
            payload["best"] = {
                "state": estimator_state_with_trainer_states(estimator, best_states),
                "epoch": best_logged_epoch(estimator),
                "metric": None,
            }
        final_states = captured_trainer_states(estimator, "_tfm_final_state")
        if final_states:
            payload["final"] = {
                "state": estimator_state_with_trainer_states(estimator, final_states),
                "epoch": final_logged_epoch(estimator),
                "metric": None,
            }
        return payload

    def build_in_memory_inference_estimator(
        self,
        estimator: Any,
        task_type: TaskKind,
        device: Any,
        n_estimators_inference: int,
    ) -> Any:
        """Build a Mitra inference estimator from the live fine-tuned ensemble."""
        _ = task_type
        if not has_trained_ensemble(estimator):
            raise RuntimeError("Mitra fine-tuned trainers are not available for in-memory evaluation.")
        inner = copy.copy(estimator)
        inner.trainers = list(getattr(estimator, "trainers", []))
        inner.n_estimators = int(n_estimators_inference)
        inner.n_estimators_inference = int(n_estimators_inference)
        if device is not None:
            inner.device = resolve_mitra_device(str(device))
        return inner

    def export_finetuned_state(
        self,
        estimator: Any,
        state: dict[str, Any],
        output_dir: Path,
        *,
        epoch: int | None,
        metric: float | None,
        checkpoint_mode: str,
    ) -> Path:
        """Write one project-owned Mitra checkpoint from an in-memory state."""
        _ = metric
        checkpoint_epoch = 0 if epoch is None else int(epoch)
        checkpoint_path = output_dir / ("best.ckpt" if checkpoint_mode == "best" else f"epoch{checkpoint_epoch}.ckpt")
        write_checkpoint_with_finetuned_state(
            estimator,
            checkpoint_path,
            state,
            epoch=checkpoint_epoch,
            checkpoint_mode=checkpoint_mode,
        )
        return checkpoint_path

    def create_dataset_transfer_stage_trainer(
        self,
        request: DatasetTransferTrainerRequest,
    ) -> DatasetTransferStageTrainer:
        """Build the exact persistent-optimizer transfer Mitra trainer."""
        from tfm_finetuning.finetuning.models.mitra.transfer import (
            MitraDatasetTransferStageTrainer,
        )

        return MitraDatasetTransferStageTrainer(
            self,
            route=request.route,
            config=request.config,
            common_config=request.common_config,
            device=request.device,
            dataset_loader=request.dataset_loader,
        )

    def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, Any]:
        """Map common settings to Mitra constructor parameters."""
        return MitraFineTuneOptions.from_common(common_config).report_params()
