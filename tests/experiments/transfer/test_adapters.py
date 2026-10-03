from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.models.mitra import configured as mitra_configured
from tfm_finetuning.finetuning.models.mitra import preprocessing as mitra_preprocessing
from tfm_finetuning.finetuning.models.mitra.adapter import MitraAdapter
from tfm_finetuning.finetuning.models.mitra.checkpoints import (
    MITRA_PORTABLE_CHECKPOINT_FORMAT,
    read_checkpoint_for_finetuning,
    read_checkpoint_package,
    write_checkpoint_package,
)
from tfm_finetuning.finetuning.models.mitra.configured import _ConfiguredMitraMixin
from tfm_finetuning.finetuning.models.registry import ModelSpec
from tfm_finetuning.finetuning.models.tabicl.model import TabICLAdapter
from tfm_finetuning.finetuning.models.tabicl.vendor import TabICLBindings
from tfm_finetuning.finetuning.models.tabpfn.model import TabPFNAdapter
from tfm_finetuning.finetuning.models.tabpfn.vendor import TabPFNBindings


class _Tensor:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def clone(self) -> _Tensor:
        return _Tensor(self.value)


class _Model:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    @classmethod
    def from_pretrained(cls, _checkpoint: str, *, device: str) -> _Model:
        _ = device
        return cls()

    def state_dict(self) -> dict[str, _Tensor]:
        return {"weight": _Tensor(self.value)}

    def load_state_dict(self, state: dict[str, _Tensor]) -> None:
        self.value = state["weight"].value


class _PortableTrainer:
    def __init__(self, value: int, *, seed: int = 0) -> None:
        self.model = _Model(value)
        self.cfg = SimpleNamespace(seed=seed, hyperparams={"dim_output": 10})
        self.n_classes = 10
        self.preprocessor = {"source_width": 2}
        self.optimizer = {"source_step": 99}


class _PortableEstimator:
    def __init__(self, values: tuple[int, ...] = (3, 5)) -> None:
        self.task = "classification"
        self.tfm_mitra_finetune_interface = "local_chunked"
        self.n_estimators = len(values)
        self.n_estimators_inference = len(values)
        self.device = "cuda"
        self.hf_model = "autogluon/mitra-classifier"
        self.verbose = False
        self.trainers = [_PortableTrainer(value, seed=index) for index, value in enumerate(values)]
        self.classes_ = np.array(["source_no", "source_yes"])
        self._tfm_mitra_feature_preprocessor = {"source_width": 2}
        self.X = np.zeros((2, 2), dtype=np.float32)
        self.y = np.array([0, 1])

    def _create_config(self, *_args):
        return SimpleNamespace(seed=0, hyperparams={"dim_output": 10}), _Model


def _spec(model_id: str, tmp_path: Path) -> ModelSpec:
    if model_id == "tabpfn_v2_5":
        adapter = "tfm_finetuning.finetuning.models.tabpfn.model.TabPFNAdapter"
        classifier = "models/tabpfn-classifier.ckpt"
        regressor = "models/tabpfn-regressor.ckpt"
    elif model_id == "tabicl_v2":
        adapter = "tfm_finetuning.finetuning.models.tabicl.model.TabICLAdapter"
        classifier = "tabicl-classifier.ckpt"
        regressor = "tabicl-regressor.ckpt"
    else:
        adapter = "tfm_finetuning.finetuning.models.mitra.model.MitraAdapter"
        classifier = "autogluon/mitra-classifier"
        regressor = "autogluon/mitra-regressor"
    return ModelSpec(
        model_id=model_id,
        display_name=model_id,
        adapter_cls_path=adapter,
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint=classifier,
        regressor_checkpoint=regressor,
        source_path=tmp_path,
        version_payload={"source_available": True},
    )


def test_tabpfn_and_tabicl_warm_starts_use_model_path(tmp_path: Path) -> None:
    class CapturingEstimator:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    tabpfn_bindings = TabPFNBindings(
        classifier=CapturingEstimator,
        regressor=CapturingEstimator,
        finetuned_classifier=CapturingEstimator,
        finetuned_regressor=CapturingEstimator,
        clone_model_for_evaluation=lambda *_args, **_kwargs: None,
        save_model=lambda *_args, **_kwargs: None,
    )
    tabicl_bindings = TabICLBindings(
        classifier=CapturingEstimator,
        regressor=CapturingEstimator,
        finetuned_classifier=CapturingEstimator,
        finetuned_regressor=CapturingEstimator,
    )
    checkpoint = tmp_path / "parent.ckpt"

    tabpfn = TabPFNAdapter(_spec("tabpfn_v2_5", tmp_path), bindings=tabpfn_bindings)
    tabicl = TabICLAdapter(_spec("tabicl_v2", tmp_path), bindings=tabicl_bindings)
    tabpfn_estimator = tabpfn.resolve_finetune_estimator(
        "binary", CommonFineTuneConfig(), 7, "cpu", initial_checkpoint=checkpoint
    )
    tabicl_estimator = tabicl.resolve_finetune_estimator(
        "binary", CommonFineTuneConfig(), 7, "cpu", initial_checkpoint=checkpoint
    )

    assert tabpfn_estimator.kwargs["extra_classifier_kwargs"]["model_path"] == str(checkpoint)
    assert tabicl_estimator.kwargs["model_path"] == str(checkpoint)
    assert tabicl_estimator.kwargs["checkpoint_version"] == "tabicl-classifier.ckpt"


def test_mitra_portable_checkpoint_contains_only_reset_safe_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mitra.ckpt"
    write_checkpoint_package(_PortableEstimator(), checkpoint)

    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)

    assert payload["format"] == MITRA_PORTABLE_CHECKPOINT_FORMAT
    assert payload["metadata"]["portable_state"] == "model_weights_only"
    assert payload["metadata"]["optimizer_state_included"] is False
    assert payload["metadata"]["source_preprocessing_included"] is False
    assert payload["metadata"]["source_classes_included"] is False
    assert payload["estimator_template"].trainers == []
    assert not hasattr(payload["estimator_template"], "classes_")
    assert not hasattr(payload["estimator_template"], "_tfm_mitra_feature_preprocessor")
    states = read_checkpoint_for_finetuning(checkpoint, task_type="multiclass", expected_ensemble_size=2)
    assert [state["weight"].value for state in states] == [3, 5]


def test_mitra_portable_warm_start_rejects_task_and_ensemble_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mitra.ckpt"
    write_checkpoint_package(_PortableEstimator(), checkpoint)

    with pytest.raises(RuntimeError, match="task family"):
        read_checkpoint_for_finetuning(checkpoint, task_type="regression", expected_ensemble_size=2)
    with pytest.raises(RuntimeError, match="exact ensemble match"):
        read_checkpoint_for_finetuning(checkpoint, task_type="binary", expected_ensemble_size=1)


def test_mitra_portable_inference_rebuilds_fresh_trainers_from_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FreshTrainer:
        def __init__(self, _cfg, model, **_kwargs) -> None:
            self.model = model
            self.optimizer = object()
            self.preprocessor = None
            self.experiment_logger = None

    modules = {
        "tfm_finetuning.finetuning.models.mitra.local_trainer": SimpleNamespace(LocalMitraTrainer=FreshTrainer),
        "autogluon.tabular.models.mitra.sklearn_interface": SimpleNamespace(USE_HF=True),
    }
    monkeypatch.setattr(mitra_configured.importlib, "import_module", modules.__getitem__)
    checkpoint = tmp_path / "mitra.ckpt"
    write_checkpoint_package(_PortableEstimator(), checkpoint)

    loaded = read_checkpoint_package(checkpoint, device="cpu")

    assert loaded.device == "cpu"
    assert [trainer.model.value for trainer in loaded.trainers] == [3, 5]
    assert all(trainer.preprocessor is None for trainer in loaded.trainers)
    assert not hasattr(loaded, "classes_")
    assert not hasattr(loaded, "_tfm_mitra_feature_preprocessor")


def test_mitra_loads_warm_start_weights_before_fresh_trainer_construction(monkeypatch) -> None:
    class FreshTrainer:
        def __init__(self, _cfg, model, **_kwargs) -> None:
            self.model = model
            self.model_value_at_construction = model.value
            self.experiment_logger = None

        def train(self, *_args) -> None:
            return None

    class BaseEstimator:
        def __init__(self, **kwargs: Any) -> None:
            self.n_estimators = int(kwargs["n_estimators"])
            self.hf_model = kwargs["hf_model"]
            self.device = kwargs["device"]
            self.verbose = kwargs["verbose"]
            self.trainers: list[Any] = []
            self.train_time = 0.0

        def _create_config(self, *_args):
            return SimpleNamespace(seed=0, hyperparams={"max_epochs": 1}), _Model

    class ConfiguredEstimator(_ConfiguredMitraMixin, BaseEstimator):
        pass

    class FakeCudaOom(RuntimeError):
        pass

    modules = {
        "tfm_finetuning.finetuning.models.mitra.local_trainer": SimpleNamespace(LocalMitraTrainer=FreshTrainer),
        "autogluon.tabular.models.mitra.sklearn_interface": SimpleNamespace(USE_HF=True),
        "torch": SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeCudaOom)),
    }
    monkeypatch.setattr(mitra_configured.importlib, "import_module", modules.__getitem__)
    estimator = ConfiguredEstimator(
        n_estimators=2,
        hf_model="autogluon/mitra-classifier",
        device="cpu",
        verbose=False,
        weight_decay=0.01,
        max_samples_support=8,
        max_samples_query=2,
        query_ratio=0.2,
        grad_clip=1.0,
        use_lr_scheduler=True,
        warmup_proportion=0.1,
        min_delta=0.0,
        mitra_finetune_interface="local_chunked",
        mitra_scheduler_policy="cosine_warmup",
        mitra_chunk_min_rows=2,
    )
    estimator._tfm_initial_trainer_states = [
        {"weight": _Tensor(11)},
        {"weight": _Tensor(13)},
    ]

    estimator._train_ensemble(
        np.zeros((4, 3)),
        np.zeros(4),
        np.zeros((2, 3)),
        np.zeros(2),
        task="classification",
        dim_output=10,
        n_classes=10,
    )

    assert [trainer.model_value_at_construction for trainer in estimator.trainers] == [11, 13]
    assert estimator._tfm_initial_trainer_states is None


def test_mitra_inference_context_refits_target_classes_and_both_preprocessors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RecordingPreprocessor:
        def fit(self, X, y) -> None:
            self.fit_shape = X.shape
            self.fit_y = np.asarray(y).copy()

    class Trainer:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(seed=17)
            self.preprocessor = {"source_width": 2}

    class TargetClassifier:
        def __init__(self) -> None:
            self.trainers = [Trainer(), Trainer()]
            self.classes_ = np.array(["source_no", "source_yes"])
            self._tfm_mitra_feature_preprocessor = {"source_width": 2}

    created: list[RecordingPreprocessor] = []

    def fresh(_trainer) -> RecordingPreprocessor:
        item = RecordingPreprocessor()
        created.append(item)
        return item

    monkeypatch.setattr(mitra_preprocessing, "_new_trainer_preprocessor", fresh)
    estimator = TargetClassifier()
    adapter = MitraAdapter(_spec("mitra", tmp_path))
    context = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0],
            "b": [1.0, 2.0, 3.0],
            "new_width": [4.0, 5.0, 6.0],
        }
    )

    adapter.fit_inference_context(estimator, context, np.array(["red", "green", "blue"]))

    assert estimator.classes_.tolist() == ["blue", "green", "red"]
    assert estimator.X.shape == (3, 3)
    assert estimator._tfm_mitra_feature_preprocessor.numeric_columns == ("a", "b", "new_width")
    assert created[0].fit_shape == (3, 3)
    assert sorted(created[0].fit_y.tolist()) == [0, 1, 2]
    assert estimator.trainers[0].preprocessor is created[0]
    assert isinstance(estimator.trainers[0].rng, np.random.RandomState)
    assert estimator.trainers[0].rng is estimator.trainers[1].rng
