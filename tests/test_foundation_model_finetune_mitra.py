"""Tests for foundation model finetune: mitra."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.finetuning import config as config_runner
from src.finetuning.models import mitra_configured as mitra_module
from src.finetuning.models.base import CommonFineTuneConfig
from src.finetuning.models.mitra import MitraAdapter
from src.finetuning.models.mitra_local_trainer import (
    LocalMitraTrainer,
    count_mitra_epoch_chunks,
    dropped_mitra_tail_rows,
    iter_mitra_epoch_chunks,
    split_mitra_support_query_indices,
)
from src.finetuning.models.preprocessing import (
    FiniteFeaturePreprocessor,
)
from src.finetuning.models.registry import FoundationModelSpec

from tests.support.split_suites.foundation_model_finetune_support_1 import (
    FakeStatefulMitraClassifier,
    FakeTorchTensor,
    _fake_mitra_bindings,
)


def test_resolve_common_finetune_config_accepts_mitra_local_trainer_options() -> None:
    cfg = config_runner.resolve_common_finetune_config(
        {
            "foundation_model": {
                "model_id": "mitra",
                "mitra_finetune_interface": "official_sklearn",
                "mitra_scheduler_policy": "official_plateau",
                "mitra_chunk_min_rows": 12,
            }
        }
    )

    assert cfg.mitra_finetune_interface == "official_sklearn"
    assert cfg.mitra_scheduler_policy == "official_plateau"
    assert cfg.mitra_chunk_min_rows == 12

def test_mitra_local_chunk_helpers_match_context_size_semantics() -> None:
    chunks = iter_mitra_epoch_chunks(25, context_size=10, epoch_seed=0, min_chunk_size=2)

    assert count_mitra_epoch_chunks(25, context_size=10, min_chunk_size=2) == 3
    assert [len(chunk) for chunk in chunks] == [10, 10, 5]
    assert dropped_mitra_tail_rows(23, context_size=10, min_chunk_size=5) == 3

def test_mitra_local_split_keeps_query_classes_in_support_when_possible() -> None:
    y = np.array([0, 0, 1, 1, 2])

    split = split_mitra_support_query_indices(y, query_ratio=0.8, seed=0, classification=True)

    assert split.query_size_clamped is True
    assert split.singleton_chunk is True
    assert set(y[split.query_idx]).issubset(set(y[split.support_idx]))

def test_mitra_local_split_falls_back_for_all_singleton_class_chunks() -> None:
    y = np.array([0, 1, 2])

    split = split_mitra_support_query_indices(y, query_ratio=0.5, seed=0, classification=True)

    assert split.rare_class_fallback is True
    assert split.singleton_chunk is True
    assert len(split.query_idx) == 2

def test_mitra_local_trainer_marks_checkpoint_epoch_for_state_capture() -> None:
    checkpoint = SimpleNamespace()
    trainer = SimpleNamespace(checkpoint=checkpoint)

    LocalMitraTrainer._set_checkpoint_epoch(trainer, 2)

    assert checkpoint._foundation_current_epoch == 2


def test_mitra_adapter_dense_labels_and_project_checkpoint_packages(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec, bindings=_fake_mitra_bindings())
    estimator = adapter.resolve_finetune_estimator(
        "binary",
        CommonFineTuneConfig(
            epochs=2,
            n_estimators_finetune=2,
            n_estimators_validation=2,
            n_estimators_final_inference=2,
        ),
        seed=3,
        device="cpu",
    )
    fitted = adapter.fit_finetuned(
        estimator,
        pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        np.array(["no", "yes", "yes"]),
        pd.DataFrame({"x": [3.0, 4.0]}),
        np.array(["yes", "no"]),
        tmp_path,
    )
    loaded = adapter.load_checkpoint_for_inference(
        tmp_path / "best.ckpt",
        "binary",
        "cpu",
        2,
        seed=3,
    )
    adapter.fit_inference_context(loaded, pd.DataFrame({"x": [5.0, 6.0]}), np.array(["no", "yes"]))
    proba = adapter.predict_for_metrics(loaded, pd.DataFrame({"x": [7.0, 8.0]}), "binary")
    diagnostics = adapter.consume_finetune_diagnostics()

    assert fitted.y.tolist() == [0, 1, 1]
    assert fitted.y_val.tolist() == [1, 0]
    assert fitted.classes_.tolist() == ["no", "yes"]
    assert (tmp_path / "best.ckpt").exists()
    assert (tmp_path / "epoch2.ckpt").exists()
    assert loaded.classes_.tolist() == ["no", "yes"]
    assert proba.shape == (2, 2)
    assert estimator.foundation_weight_decay == 0.01
    assert estimator.foundation_mitra_finetune_interface == "local_chunked"
    assert estimator.foundation_mitra_scheduler_policy == "cosine_warmup"
    assert estimator.foundation_grad_clip == 1.0
    assert estimator.foundation_use_lr_scheduler is True
    assert diagnostics["mitra_adapter_input_preprocessing"]["method"] == "mitra_adapter_finite_numeric_guard"
    assert diagnostics["mitra_adapter_input_preprocessing"]["input_already_shared_preprocessed"] is False

def test_mitra_adapter_resolved_params_expose_local_and_legacy_interfaces(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec, bindings=_fake_mitra_bindings())

    local_cfg = CommonFineTuneConfig(
        epochs=4,
        context_size=100,
        query_ratio=0.25,
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_final_inference=2,
    )
    official_cfg = CommonFineTuneConfig(
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_final_inference=2,
        mitra_finetune_interface="official_sklearn",
    )
    estimator = adapter.resolve_finetune_estimator("binary", official_cfg, seed=0, device="cpu")

    local_params = adapter.resolved_model_params(local_cfg)
    official_params = adapter.resolved_model_params(official_cfg)

    assert local_params["mitra_finetune_interface"] == "local_chunked"
    assert local_params["mitra_scheduler_policy"] == "cosine_warmup"
    assert local_params["warmup_steps_scope"] == "fit_time_optimizer_step"
    assert local_params["max_samples_support"] == 75
    assert local_params["max_samples_query"] == 25
    assert official_params["mitra_finetune_interface"] == "official_sklearn"
    assert official_params["warmup_steps_scope"] == "official_epoch"
    assert estimator.foundation_mitra_finetune_interface == "official_sklearn"

def test_mitra_local_chunked_oom_propagates_without_mutating_caps(monkeypatch) -> None:
    class FakeCudaOom(RuntimeError):
        pass

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _checkpoint, *, device):
            return cls()

    class FakeTrainer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.experiment_logger = None

        def train(self, *_args) -> None:
            raise FakeCudaOom("CUDA out of memory")

    class FakeMitraBase:
        def __init__(self, **kwargs) -> None:
            self.n_estimators = int(kwargs["n_estimators"])
            self.hf_model = kwargs["hf_model"]
            self.device = kwargs["device"]
            self.verbose = bool(kwargs["verbose"])
            self.trainers: list[object] = []
            self.train_time = 0.0

        def _create_config(self, *_args):
            return (
                SimpleNamespace(
                    seed=0,
                    hyperparams={
                        "max_epochs": 2,
                        "max_samples_support": 8_000,
                        "max_samples_query": 2_000,
                    },
                ),
                FakeModel,
            )

    class ConfiguredMitra(mitra_module._ConfiguredMitraMixin, FakeMitraBase):
        pass

    modules = {
        "finetuning.models.mitra_local_trainer": SimpleNamespace(LocalMitraTrainer=FakeTrainer),
        "autogluon.tabular.models.mitra.sklearn_interface": SimpleNamespace(USE_HF=True),
        "torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeCudaOom)),
    }
    monkeypatch.setattr(mitra_module.importlib, "import_module", modules.__getitem__)
    estimator = ConfiguredMitra(
        n_estimators=1,
        hf_model="autogluon/mitra-classifier",
        device="cuda",
        verbose=False,
        weight_decay=0.01,
        max_samples_support=8_000,
        max_samples_query=2_000,
        query_ratio=0.2,
        grad_clip=1.0,
        use_lr_scheduler=True,
        warmup_proportion=0.1,
        min_delta=0.0,
        mitra_finetune_interface="local_chunked",
        mitra_scheduler_policy="cosine_warmup",
        mitra_chunk_min_rows=50,
    )

    with pytest.raises(FakeCudaOom, match="CUDA out of memory"):
        estimator._train_ensemble(
            np.zeros((10, 2)),
            np.zeros(10),
            np.zeros((4, 2)),
            np.zeros(4),
            task="binary",
            dim_output=2,
        )

    assert estimator.trainers == []
    assert estimator.foundation_max_samples_support == 8_000
    assert estimator.foundation_max_samples_query == 2_000

def test_mitra_adapter_in_memory_state_export_round_trips_epoch_state(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec)
    estimator = FakeStatefulMitraClassifier(value=1)
    estimator._foundation_mitra_feature_preprocessor = FiniteFeaturePreprocessor.fit(pd.DataFrame({"x": [0.0, 1.0]}))

    state = adapter.clone_finetuned_state(estimator)
    estimator.trainers[0].model.load_state_dict({"weight": FakeTorchTensor(2)})
    inner = adapter.build_in_memory_inference_estimator(
        estimator,
        task_type="binary",
        device="cpu",
        n_estimators_inference=3,
    )
    exported = adapter.export_finetuned_state(
        estimator,
        state,
        tmp_path,
        epoch=1,
        metric=0.9,
        checkpoint_mode="best",
    )
    loaded = adapter.load_checkpoint_for_inference(
        exported,
        "binary",
        "cpu",
        n_estimators_inference=5,
        seed=0,
    )

    assert inner is not estimator
    assert inner.trainers[0] is estimator.trainers[0]
    assert inner.n_estimators == 3
    assert estimator.trainers[0].model.value == 2
    assert loaded.trainers[0].model.value == 1
    assert loaded.classes_.tolist() == ["no", "yes"]
    assert loaded.n_estimators == 5
    assert loaded._foundation_checkpoint_epoch == 1
    assert loaded._foundation_checkpoint_mode == "best"

def test_mitra_adapter_captures_best_and_final_states_from_trainer_checkpoints(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec)
    estimator = FakeStatefulMitraClassifier(value=3)
    estimator.trainers[0].checkpoint = SimpleNamespace(
        _foundation_best_state={"weight": FakeTorchTensor(1)},
        _foundation_final_state={"weight": FakeTorchTensor(2)},
        _foundation_best_epoch=1,
    )
    estimator._foundation_mitra_feature_preprocessor = FiniteFeaturePreprocessor.fit(pd.DataFrame({"x": [0.0, 1.0]}))

    captured = adapter.capture_finetuned_states(estimator)
    adapter.apply_finetuned_state(estimator, captured["best"]["state"])
    best_value = estimator.trainers[0].model.value
    adapter.apply_finetuned_state(estimator, captured["final"]["state"])
    final_value = estimator.trainers[0].model.value

    assert captured["best"]["epoch"] == 1
    assert captured["final"]["epoch"] == 1
    assert best_value == 1
    assert final_value == 2

def test_mitra_adapter_suppresses_legacy_checkpoints_under_in_memory_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec, bindings=_fake_mitra_bindings())
    estimator = adapter.resolve_finetune_estimator(
        "binary",
        CommonFineTuneConfig(
            epochs=2,
            n_estimators_finetune=2,
            n_estimators_validation=2,
            n_estimators_final_inference=2,
        ),
        seed=3,
        device="cpu",
    )
    estimator._foundation_suppress_adapter_checkpoints = True

    adapter.fit_finetuned(
        estimator,
        pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        np.array(["no", "yes", "yes"]),
        pd.DataFrame({"x": [3.0, 4.0]}),
        np.array(["yes", "no"]),
        tmp_path,
    )

    assert not (tmp_path / "best.ckpt").exists()
    assert not list(tmp_path.glob("epoch*.ckpt"))

def test_mitra_zero_shot_context_fit_initializes_empty_ensemble(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="finetuning.models.mitra.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="autogluon/mitra-classifier",
        regressor_checkpoint="autogluon/mitra-regressor",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = MitraAdapter(spec, bindings=_fake_mitra_bindings())
    estimator = adapter.resolve_zero_shot_estimator("binary", seed=0, device="cpu", n_estimators_inference=1)

    adapter.fit_inference_context(
        estimator,
        pd.DataFrame({"x": [0.0, 1.0], "weekday": ["Mon", "Thu"]}),
        np.array(["no", "yes"]),
    )
    proba = adapter.predict_for_metrics(
        estimator,
        pd.DataFrame({"x": [2.0, 3.0], "weekday": ["Sun", None]}),
        "binary",
    )

    assert estimator.fit_call_count == 1
    assert estimator.trainers == ["trainer"]
    assert estimator.classes_.tolist() == ["no", "yes"]
    assert estimator.y.tolist() == [0, 1]
    assert estimator.X.dtype.kind == "f"
    assert estimator.predict_X.dtype.kind == "f"
    assert np.isfinite(estimator.X).all()
    assert np.isfinite(estimator.predict_X).all()
    assert proba.shape == (2, 2)
