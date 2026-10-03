"""TabICLv2 adapter: preprocessing boundary, vendor calls, state, and prediction."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

from finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
    normalize_preprocessing_protocol,
)
from finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferStageTrainer,
    DatasetTransferTrainerRequest,
    TaskKind,
    validation_metric_for_task,
)
from finetuning.models.registry import FoundationModelSpec
from finetuning.models.tabicl.finetune_compat import patch_tabicl_finetune_meta_batches
from finetuning.models.tabicl.preprocessing import (
    prepare_native_finetune_inputs,
    prepare_shared_finetune_inputs,
)
from finetuning.models.tabicl.vendor import TabICLBindings, load_tabicl_bindings


class TabICLAdapter:
    """Complete thesis-side fine-tuning slice for TabICLv2.

    Example:
        >>> from finetuning.models.registry import get_foundation_model_spec
        >>> adapter = TabICLAdapter(get_foundation_model_spec("tabicl_v2"))
        >>> adapter.resolved_model_params(CommonFineTuneConfig())["max_data_size"]
        10000
    """

    model_id = "tabicl_v2"

    def __init__(self, spec: FoundationModelSpec, *, bindings: TabICLBindings | None = None) -> None:
        self.spec = spec
        self._bindings = bindings
        self._preprocessing_protocol = PREPROCESSING_NATIVE
        self._last_finetune_diagnostics: dict[str, Any] = {}

    def set_preprocessing_protocol(self, protocol: str) -> None:
        """Configure whether fit inputs are native or already shared-preprocessed."""
        self._preprocessing_protocol = normalize_preprocessing_protocol(protocol)

    def _vendor(self) -> TabICLBindings:
        if self._bindings is None:
            self._bindings = load_tabicl_bindings()
        return self._bindings

    def _checkpoint_for_task(self, task_type: TaskKind) -> str:
        checkpoint = self.spec.regressor_checkpoint if task_type == "regression" else self.spec.classifier_checkpoint
        if checkpoint is None:
            raise ValueError(f"{self.model_id} has no checkpoint for task_type={task_type}.")
        return checkpoint

    def _checkpoint_init_kwargs(self, task_type: TaskKind) -> dict[str, Any]:
        """Prefer exact absolute checkpoint bytes over TabICL's download fallback."""
        checkpoint = self._checkpoint_for_task(task_type)
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_absolute():
            return {"checkpoint_version": checkpoint}
        if not candidate.is_file():
            raise FileNotFoundError(f"Configured TabICL checkpoint does not exist: {candidate}")
        return {
            "model_path": str(candidate.resolve()),
            "allow_auto_download": False,
        }

    def resolve_zero_shot_estimator(
        self,
        task_type: TaskKind,
        seed: int,
        device: str | None,
        n_estimators_inference: int,
    ) -> Any:
        """Return a TabICL estimator backed by the configured checkpoint id."""
        cls = self._vendor().estimator_class(regression=task_type == "regression")
        return cls(
            n_estimators=int(n_estimators_inference),
            device=device,
            random_state=int(seed),
            **self._checkpoint_init_kwargs(task_type),
        )

    def resolve_finetune_estimator(
        self,
        task_type: TaskKind,
        config: CommonFineTuneConfig,
        seed: int,
        device: str | None,
        initial_checkpoint: Path | None = None,
    ) -> Any:
        """Return a fresh TabICL wrapper, optionally initialized from learned weights."""
        cls = self._vendor().finetuner_class(regression=task_type == "regression")
        params = self.resolved_model_params(config)
        params["device"] = device
        params["random_state"] = int(seed)
        params["eval_metric"] = validation_metric_for_task(task_type, model_id=self.model_id)
        params.update(self._checkpoint_init_kwargs(task_type))
        if initial_checkpoint is not None:
            params["model_path"] = str(initial_checkpoint)
            params["allow_auto_download"] = False
        return cls(**params)

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
        if self._preprocessing_protocol == PREPROCESSING_NATIVE:
            train_X_fit, val_X_fit, preprocessing_diagnostics = prepare_native_finetune_inputs(train_X, val_X)
        else:
            train_X_fit, val_X_fit, preprocessing_diagnostics = prepare_shared_finetune_inputs(
                train_X,
                val_X,
                protocol=self._preprocessing_protocol,
            )
        with patch_tabicl_finetune_meta_batches() as diagnostics:
            result = estimator.fit(train_X_fit, train_y, X_val=val_X_fit, y_val=val_y, output_dir=output_dir)
        payload = diagnostics.as_dict()
        self._last_finetune_diagnostics = {
            "tabicl_meta_batch_split": payload,
            "tabicl_native_finetune_input_preprocessing": preprocessing_diagnostics,
        }
        return result

    def consume_finetune_diagnostics(self) -> dict[str, Any]:
        """Return and clear diagnostics captured during the last TabICL fine-tune."""
        diagnostics = dict(self._last_finetune_diagnostics)
        self._last_finetune_diagnostics = {}
        return diagnostics

    def export_checkpoint(self, output_dir: Path, checkpoint_mode: str) -> Path:
        """Return the checkpoint written by TabICL fine-tuning."""
        _ = checkpoint_mode
        for name in ("best.ckpt", "checkpoint_best.ckpt", "model.ckpt"):
            candidate = output_dir / name
            if candidate.exists():
                return candidate
        checkpoints = sorted(output_dir.glob("*.ckpt"))
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(f"No TabICL checkpoint found in {output_dir}.")

    def load_checkpoint_for_inference(
        self,
        checkpoint_path: Path,
        task_type: TaskKind,
        device: str | None,
        n_estimators_inference: int,
        seed: int,
    ) -> Any:
        """Load a fine-tuned TabICL checkpoint into an inference estimator."""
        cls = self._vendor().estimator_class(regression=task_type == "regression")
        return cls(
            model_path=str(checkpoint_path),
            n_estimators=int(n_estimators_inference),
            device=device,
            random_state=int(seed),
        )

    def fit_inference_context(self, estimator: Any, context_X: pd.DataFrame, context_y: np.ndarray) -> Any:
        """Fit the TabICL in-context estimator."""
        return estimator.fit(context_X, context_y)

    def predict_for_metrics(self, estimator: Any, query_X: pd.DataFrame, task_type: TaskKind) -> Any:
        """Return probabilities for classifiers and labels for regression."""
        if task_type == "regression":
            return estimator.predict(query_X)
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(query_X)
        return estimator.predict(query_X)

    def clone_finetuned_state(self, estimator: Any) -> dict[str, Any]:
        """Return a CPU snapshot of the current fine-tuned TabICL weights."""
        model = getattr(estimator, "model_", None)
        if model is None or not hasattr(model, "state_dict"):
            raise RuntimeError("TabICL fine-tuned model is not available for in-memory checkpointing.")
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def apply_finetuned_state(self, estimator: Any, state: Mapping[str, Any]) -> None:
        """Restore a captured TabICL state onto the live fine-tuned estimator."""
        model = getattr(estimator, "model_", None)
        if model is None or not hasattr(model, "load_state_dict"):
            raise RuntimeError("TabICL fine-tuned model is not available for in-memory state restore.")
        model.load_state_dict(dict(state))

    def build_in_memory_inference_estimator(
        self,
        estimator: Any,
        task_type: TaskKind,
        device: Any,
        n_estimators_inference: int,
    ) -> Any:
        """Build a TabICL inference estimator from the live fine-tuned model."""
        _ = task_type
        model = getattr(estimator, "model_", None)
        builder = getattr(estimator, "_build_inner_estimator", None)
        if model is None or not callable(builder):
            raise RuntimeError("TabICL fine-tuned model is not available for in-memory evaluation.")
        return builder(model, int(n_estimators_inference), device)

    def export_finetuned_state(
        self,
        estimator: Any,
        state_dict: Mapping[str, Any],
        output_dir: Path,
        *,
        epoch: int | None,
        metric: float | None,
        checkpoint_mode: str,
    ) -> Path:
        """Write one TabICL-compatible checkpoint from an in-memory state."""
        torch = importlib.import_module("torch")

        _ = checkpoint_mode
        model_config = getattr(estimator, "model_config_", None)
        if model_config is None:
            raise RuntimeError("TabICL model_config_ is required to export an in-memory checkpoint.")
        checkpoint_epoch = 0 if epoch is None else int(epoch)
        payload: dict[str, Any] = {
            "config": model_config,
            "state_dict": dict(state_dict),
            "curr_step": checkpoint_epoch,
            "epoch": checkpoint_epoch,
        }
        if metric is not None:
            payload["best_metric"] = float(metric)
        checkpoint_path = output_dir / ("best.ckpt" if checkpoint_mode == "best" else f"epoch{checkpoint_epoch}.ckpt")
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, Any]:
        """Map shared settings to official TabICL parameter names."""
        return {
            "epochs": int(common_config.epochs),
            "learning_rate": float(common_config.learning_rate),
            "weight_decay": float(common_config.weight_decay),
            "max_data_size": int(common_config.context_size),
            "finetune_ctx_query_ratio": float(common_config.query_ratio),
            "n_estimators_finetune": int(common_config.n_estimators_finetune),
            "n_estimators_validation": int(common_config.n_estimators_validation),
            "n_estimators_inference": int(common_config.n_estimators_final_inference),
            "early_stopping": bool(common_config.early_stopping),
            "patience": int(common_config.patience),
            "min_delta": float(common_config.min_delta),
            "grad_clip": float(common_config.grad_clip),
            "amp": True,
            "use_lr_scheduler": bool(common_config.use_lr_scheduler),
            "warmup_proportion": float(common_config.warmup_proportion),
        }

    def create_dataset_transfer_stage_trainer(
        self,
        request: DatasetTransferTrainerRequest,
    ) -> DatasetTransferStageTrainer:
        """Return the exact persistent-optimizer Section 3 stage trainer."""
        from finetuning.models.tabicl.transfer import (
            TabICLDatasetTransferStageTrainer,
        )

        return TabICLDatasetTransferStageTrainer(
            adapter=self,
            route=request.route,
            config=request.config,
            common_config=request.common_config,
            device=request.device,
            dataset_loader=request.dataset_loader,
        )
