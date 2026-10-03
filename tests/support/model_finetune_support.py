"""Shared helpers extracted from `the model_finetune tests`."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from tfm_finetuning.finetuning import model_finetune as runner
from tfm_finetuning.finetuning.models.mitra.vendor import MitraBindings
from tfm_finetuning.finetuning.models.registry import FoundationModelSpec

run_data_runner = importlib.import_module(runner._prepare_foundation_run_data.__module__)

preprocessing_runner = importlib.import_module(
    run_data_runner._prepare_alternate_test_context_data.__module__
)

class FakeMitraClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.trainers = ["trainer"] if kwargs.get("fine_tune", True) else []
        self.fine_tune_steps = int(kwargs.get("fine_tune_steps", 1) or 1)
        self.fit_call_count = 0

    def fit(self, X, y, X_val=None, y_val=None):
        self.fit_call_count += 1
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.trainers = ["trainer"]
        return self

    def predict_proba(self, X):
        self.predict_X = np.asarray(X)
        return np.tile(np.array([[0.25, 0.75]]), (len(X), 1))

class FakeMitraRegressor:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.trainers = ["trainer"] if kwargs.get("fine_tune", True) else []
        self.fine_tune_steps = int(kwargs.get("fine_tune_steps", 1) or 1)
        self.fit_call_count = 0

    def fit(self, X, y, X_val=None, y_val=None):
        self.fit_call_count += 1
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.trainers = ["trainer"]
        return self

    def predict(self, X):
        self.predict_X = np.asarray(X)
        return np.zeros(len(X), dtype=np.float32)

def _fake_mitra_bindings() -> MitraBindings:
    return MitraBindings(classifier=FakeMitraClassifier, regressor=FakeMitraRegressor)

class FakeTorchTensor:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTorchTensor(self.value)

class FakeStatefulMitraModel:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def state_dict(self) -> dict[str, FakeTorchTensor]:
        return {"weight": FakeTorchTensor(self.value)}

    def load_state_dict(self, state: dict[str, FakeTorchTensor]) -> None:
        self.value = int(state["weight"].value)

class FakeStatefulMitraTrainer:
    def __init__(self, value: int) -> None:
        self.model = FakeStatefulMitraModel(value)

class FakeStatefulMitraClassifier:
    def __init__(self, value: int) -> None:
        self.trainers = [FakeStatefulMitraTrainer(value)]
        self.classes_ = np.array(["no", "yes"])
        self.n_estimators = 1
        self.n_estimators_inference = 1
        self.device = "cpu"

def _tabpfn_test_spec(tmp_path: Path) -> FoundationModelSpec:
    return FoundationModelSpec(
        model_id="tabpfn_v2_5",
        display_name="TabPFNv2.5",
        adapter_cls_path="tfm_finetuning.finetuning.models.tabpfn.model.TabPFNAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="models/tabpfn-v2.5-classifier.ckpt",
        regressor_checkpoint="models/tabpfn-v2.5-regressor.ckpt",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
