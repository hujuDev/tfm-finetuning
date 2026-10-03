from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.transfer.spec import (
    SplitIdentity,
    TransferRouteSpec,
    TransferStageSpec,
    ValidationSelector,
)
from finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferTrainerRequest,
)
from finetuning.models.tabicl.model import TabICLAdapter
from finetuning.models.tabicl.transfer import TabICLDatasetTransferStageTrainer
from finetuning.models.tabicl.transfer_engine import (
    TabICLTransferEngine,
    _PreparedDataset,
    _reload_context_indices,
)


@dataclass(frozen=True)
class _Update:
    examples: int = 10


@dataclass(frozen=True)
class _Data:
    metric_name: str = "roc_auc"
    task_type: str = "binary"


def test_tabicl_regression_reload_context_is_a_deterministic_prefix() -> None:
    labels = np.linspace(-1.0, 1.0, 20)

    assert _reload_context_indices(labels, max_rows=7, classification=False).tolist() == list(range(7))


def test_tabicl_regression_episode_uses_vendor_class_shuffle_default() -> None:
    engine = object.__new__(TabICLTransferEngine)
    engine.classification = False
    engine.estimator = SimpleNamespace(
        norm_methods=None,
        feat_shuffle_method="latin",
        outlier_threshold=4.0,
        _skip_batch=lambda batch, using_ddp, device: False,
    )
    engine.stage = SimpleNamespace(seed=7)
    engine.common_config = CommonFineTuneConfig(n_estimators_finetune=2, query_ratio=0.2)
    engine.device = torch.device("cpu")
    engine.meta_batch_diagnostics = {}
    prepared = _PreparedDataset(
        data=None,
        train_X=np.arange(128, dtype=np.float32).reshape(64, 2),
        train_y=np.linspace(-1.0, 1.0, 64, dtype=np.float32),
    )

    batch = engine._sample_supported_episode(
        "regression",
        prepared=prepared,
        row_count=32,
        rng=np.random.default_rng(11),
    )

    assert batch.y_query.dtype == torch.float32
    assert batch.X.shape[0] == 2


class _Loader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, str, int | None]] = []
        self.runtime_phases: list[dict[str, Any]] = []

    def __call__(
        self,
        name: str,
        *,
        row_budget: int | None,
        row_budget_policy: str,
        row_subset_seed: int | None,
    ) -> _Data:
        self.calls.append((name, row_budget, row_budget_policy, row_subset_seed))
        self.runtime_phases.append({"phase": "load", "dataset": name})
        return _Data()


class _FakeEngine:
    def __init__(self, *, score_table: dict[str, dict[int, float]], **kwargs: Any) -> None:
        self.stage = kwargs["stage"]
        self.score_table = score_table
        self.initial_checkpoint = kwargs["initial_checkpoint"]
        self.update = 0
        self.dataset_order: list[str] = []
        self.gpu_time_seconds = 0.125

    def train_update(self, dataset_name: str, *, rng: Any) -> _Update:
        _ = rng.random()
        self.update += 1
        self.dataset_order.append(dataset_name)
        return _Update()

    def clone_state(self) -> dict[str, int]:
        return {"update": self.update}

    def evaluate_current(self, dataset_name: str, *, splits: tuple[str, ...]) -> dict[str, Any]:
        assert splits == ("val",)
        return {"val": {"score": self.score_table[dataset_name][self.update]}}

    def evaluate_base(self, dataset_name: str, *, splits: tuple[str, ...]) -> dict[str, Any]:
        _ = dataset_name
        return {split: {"score": 0.5 if split == "val" else 0.51} for split in splits}

    def export_state(
        self,
        state: dict[str, int],
        output_dir: Path,
        *,
        update: int,
        metric: float,
    ) -> Path:
        _ = metric
        assert update == state["update"]
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "best.ckpt"
        path.write_text(str(update), encoding="utf-8")
        return path

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        dataset_name: str,
        *,
        splits: tuple[str, ...],
    ) -> dict[str, Any]:
        update = int(checkpoint_path.read_text(encoding="utf-8"))
        val_score = self.score_table[dataset_name][update]
        return {
            split: {"score": val_score if split == "val" else val_score - 0.01}
            for split in splits
        }

    def validate_reload(self, checkpoint_path: Path) -> bool:
        return checkpoint_path.is_file() and checkpoint_path.read_text(encoding="utf-8").isdigit()

    def diagnostics(self) -> dict[str, Any]:
        return {"fake": True}


def _stage(
    *,
    datasets: tuple[str, ...],
    update_budget: int,
    role: str,
    selector: ValidationSelector,
    views: tuple[int, ...] = (),
) -> TransferStageSpec:
    return TransferStageSpec(
        name="stage",
        role=role,
        training_mode="single_dataset" if len(datasets) == 1 else "pooled_episodic",
        datasets=datasets,
        row_budget_per_dataset=64,
        update_budget=update_budget,
        sampling_weights=tuple(1.0 for _ in datasets),
        selector=selector,
        split=SplitIdentity(),
        preprocessing="shared",
        model_id="tabicl_v2",
        base_checkpoint="tabicl.ckpt",
        seed=7,
        row_subset_seed=11,
        view_boundaries=views,
    )


def _route(stage: TransferStageSpec, *, target: str) -> TransferRouteSpec:
    return TransferRouteSpec(
        route_id="test",
        family="dataset_transfer",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
    )


def _trainer(
    stage: TransferStageSpec,
    *,
    target: str,
    scores: dict[str, dict[int, float]],
    common: CommonFineTuneConfig | None = None,
) -> tuple[TabICLDatasetTransferStageTrainer, _Loader, list[_FakeEngine]]:
    loader = _Loader()
    engines: list[_FakeEngine] = []

    def engine_factory(**kwargs: Any) -> _FakeEngine:
        engine = _FakeEngine(score_table=scores, **kwargs)
        engines.append(engine)
        return engine

    trainer = TabICLDatasetTransferStageTrainer(
        adapter=SimpleNamespace(),
        route=_route(stage, target=target),
        config={},
        common_config=common or CommonFineTuneConfig(early_stopping=False),
        device="cpu",
        dataset_loader=loader,
        engine_factory=engine_factory,
    )
    return trainer, loader, engines


def test_tabicl_adapter_exposes_exact_transfer_factory() -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=2,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
    )
    adapter = TabICLAdapter(SimpleNamespace())
    trainer = adapter.create_dataset_transfer_stage_trainer(
        DatasetTransferTrainerRequest(
            route=_route(stage, target="target"),
            config={},
            common_config=CommonFineTuneConfig(),
            device="cpu",
            dataset_loader=_Loader(),
        )
    )

    assert trainer.contract.persistent_optimizer_within_stage is True
    assert trainer.contract.raw_row_concatenation is False


def test_pooled_stage_balances_updates_and_selects_equal_source_mean(tmp_path: Path) -> None:
    stage = _stage(
        datasets=("source-a", "source-b"),
        update_budget=6,
        role="reusable_boundary",
        selector=ValidationSelector(
            "equal_mean_validation_improvement",
            ("source-a", "source-b"),
        ),
    )
    scores = {
        "source-a": {2: 0.60, 4: 0.55, 6: 0.80},
        "source-b": {2: 0.50, 4: 0.70, 6: 0.40},
    }
    trainer, loader, engines = _trainer(stage, target="target", scores=scores)
    initial = tmp_path / "initial.ckpt"
    initial.write_bytes(b"weights")

    production = trainer.train_stage(stage=stage, initial_checkpoint=initial, output_dir=tmp_path / "out")

    assert len(engines) == 1
    assert engines[0].initial_checkpoint == initial
    assert engines[0].dataset_order.count("source-a") == 3
    assert engines[0].dataset_order.count("source-b") == 3
    assert production.report["dataset_optimizer_updates"] == {"source-a": 3, "source-b": 3}
    assert production.report["checkpoint_selection"]["selected_update"] == 4
    assert production.checkpoint_path.read_text(encoding="utf-8") == "4"
    assert loader.calls == [
        ("source-a", 64, "cap", 11),
        ("source-b", 64, "cap", 11),
    ]


def test_source_only_terminal_uses_target_for_evaluation_without_target_updates(
    tmp_path: Path,
) -> None:
    stage = _stage(
        datasets=("source-a", "source-b"),
        update_budget=4,
        role="terminal",
        selector=ValidationSelector(
            "equal_mean_validation_improvement",
            ("source-a", "source-b"),
        ),
    )
    scores = {
        "source-a": {2: 0.60, 4: 0.70},
        "source-b": {2: 0.55, 4: 0.65},
        "target": {2: 0.52, 4: 0.53},
    }
    trainer, loader, engines = _trainer(stage, target="target", scores=scores)

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert loader.calls == [
        ("source-a", 64, "cap", 11),
        ("source-b", 64, "cap", 11),
        ("target", 64, "cap", 11),
    ]
    assert set(engines[0].dataset_order) == {"source-a", "source-b"}
    assert production.report["dataset_optimizer_updates"] == {
        "source-a": 2,
        "source-b": 2,
    }
    assert production.report["metrics"]["test"]["score"] == 0.52


def test_stage_trainer_rejects_non_strict_resource_semantics(tmp_path: Path) -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=1,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
    )
    stage = replace(stage, strict_resource_semantics=False)
    trainer, _loader, engines = _trainer(stage, target="target", scores={"target": {1: 0.6}})

    with pytest.raises(ValueError, match="strict resource semantics"):
        trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)
    assert engines == []


def test_terminal_stage_emits_validation_selected_prefix_view_and_metrics(tmp_path: Path) -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=6,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
        views=(2,),
    )
    scores = {"target": {1: 0.55, 2: 0.70, 3: 0.60, 4: 0.80, 5: 0.75, 6: 0.77}}
    trainer, _loader, _engines = _trainer(stage, target="target", scores=scores)

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert production.report["realized_optimizer_updates"] == 6
    assert production.report["checkpoint_selection"]["selected_update"] == 4
    assert production.report["checkpoint_view_selection"]["2"]["selected_update"] == 2
    assert production.report["checkpoint_view_selection"]["2"]["max_update"] == 2
    assert production.view_checkpoints[2].read_text(encoding="utf-8") == "2"
    assert production.report["metrics"]["val"]["score"] == 0.80
    assert production.report["zero-shot-metrics"]["test"]["score"] == 0.51
    assert production.report["checkpoint_view_metrics"]["2"]["metrics"]["val"]["score"] == 0.70


def test_early_stopping_waits_until_declared_prefix_cap(tmp_path: Path) -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=5,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
        views=(3,),
    )
    scores = {"target": {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.40, 5: 0.30}}
    common = CommonFineTuneConfig(early_stopping=True, patience=1, min_delta=0.0)
    trainer, _loader, _engines = _trainer(stage, target="target", scores=scores, common=common)

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert production.report["realized_optimizer_updates"] == 3
    assert production.report["checkpoint_selection"]["selected_update"] == 1
    assert production.report["checkpoint_view_selection"]["3"]["selected_update"] == 1
    assert production.view_checkpoints[3].read_text(encoding="utf-8") == "1"
    assert production.report["validation_cadence"]["minimum_updates_before_early_stopping"] == 3


def test_tabicl_engine_retries_vendor_episode_and_reload_smokes_icl(tmp_path: Path) -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=2,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
    )
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 64), "z": np.tile([0.0, 1.0], 32)})
    data = SimpleNamespace(
        task_type="binary",
        train_X=frame,
        train_y=np.tile([0, 1], 32),
        val_X=frame.iloc[:16].copy(),
        val_y=np.tile([0, 1], 8),
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    class SlowSettlingScaler:
        def __init__(self) -> None:
            self.attempts = 0
            self.scale_value = 65536.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            _ = optimizer

        def step(self, optimizer):
            self.attempts += 1
            if self.attempts > 9:
                optimizer.step()

        def update(self):
            if self.attempts <= 9:
                self.scale_value /= 2

        def get_scale(self):
            return self.scale_value

    class TinyEstimator:
        norm_methods = None
        feat_shuffle_method = "latin"
        class_shuffle_method = "shift"
        outlier_threshold = 4.0

        def __init__(self) -> None:
            self.skip_calls = 0

        def _resolve_device(self):
            return torch.device("cpu")

        def _load_pretrained(self, device):
            _ = device
            self.model_ = TinyModel()

        def _apply_freezing(self, model):
            _ = model
            return False

        def _make_amp(self, device):
            _ = device
            return True, SlowSettlingScaler(), contextlib.nullcontext

        def _skip_batch(self, batch, using_ddp, device):
            _ = batch, using_ddp, device
            self.skip_calls += 1
            return self.skip_calls == 1

        def _set_training_mode(self, training):
            self.model_.train(training)

        def _compute_batch_loss(self, batch, model):
            target = batch.y_query.float().mean()
            return (model.weight - target).square()

    class TinyAdapter:
        _preprocessing_protocol = "shared"

        def __init__(self) -> None:
            self.reload_context_labels = None
            self.reload_prediction_called = False
            self.reload_seed = None

        def resolve_finetune_estimator(self, *args, **kwargs):
            _ = args, kwargs
            return TinyEstimator()

        def clone_finetuned_state(self, estimator):
            return {key: value.detach().clone() for key, value in estimator.model_.state_dict().items()}

        def load_checkpoint_for_inference(self, *args, **kwargs):
            _ = args
            self.reload_seed = kwargs["seed"]
            return SimpleNamespace()

        def fit_inference_context(self, estimator, context_X, context_y):
            _ = context_X
            self.reload_context_labels = np.asarray(context_y)
            return estimator

        def predict_for_metrics(self, estimator, query_X, task_type):
            _ = estimator, query_X, task_type
            self.reload_prediction_called = True
            return np.array([[0.4, 0.6]])

    adapter = TinyAdapter()
    engine = TabICLTransferEngine(
        adapter=adapter,
        stage=stage,
        common_config=CommonFineTuneConfig(
            context_size=32,
            n_estimators_finetune=2,
            use_lr_scheduler=False,
        ),
        device="cpu",
        datasets={"target": data, "evaluation-only": data},
        initial_checkpoint=None,
    )
    before = engine.estimator.model_.weight.detach().clone()

    result = engine.train_update("target", rng=np.random.default_rng(123))

    assert result.examples == 32
    assert set(engine.datasets) == {"target", "evaluation-only"}
    assert not torch.equal(engine.estimator.model_.weight.detach(), before)
    assert engine.scaler.attempts == 10
    assert engine.meta_batch_diagnostics["amp_overflow_retry_count"] == 9
    assert engine.meta_batch_diagnostics["total_chunk_count"] == 2
    assert engine.meta_batch_diagnostics["rejected_episode_count"] == 1
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"weights")
    assert engine.validate_reload(checkpoint) is True
    assert set(adapter.reload_context_labels) == {0, 1}
    assert adapter.reload_prediction_called is True
    assert adapter.reload_seed == 7


def test_tabicl_engine_rejects_non_finite_loss_before_optimizer_step() -> None:
    stage = _stage(
        datasets=("target",),
        update_budget=1,
        role="terminal",
        selector=ValidationSelector("target_validation", ("target",)),
    )
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 64)})
    data = SimpleNamespace(
        task_type="binary",
        train_X=frame,
        train_y=np.tile([0, 1], 32),
        val_X=frame.iloc[:4].copy(),
    )

    class Estimator:
        norm_methods = None
        feat_shuffle_method = "latin"
        class_shuffle_method = "shift"
        outlier_threshold = 4.0

        def _resolve_device(self):
            return torch.device("cpu")

        def _load_pretrained(self, device):
            _ = device
            self.model_ = torch.nn.Linear(1, 1)

        def _apply_freezing(self, model):
            _ = model
            return False

        def _make_amp(self, device):
            _ = device
            return False, torch.GradScaler("cuda", enabled=False), contextlib.nullcontext

        def _skip_batch(self, batch, using_ddp, device):
            _ = batch, using_ddp, device
            return False

        def _set_training_mode(self, training):
            self.model_.train(training)

        def _compute_batch_loss(self, batch, model):
            _ = batch
            return model.weight.sum() * torch.tensor(float("nan"))

    class Adapter:
        _preprocessing_protocol = "shared"

        def resolve_finetune_estimator(self, *args, **kwargs):
            _ = args, kwargs
            return Estimator()

    engine = TabICLTransferEngine(
        adapter=Adapter(),
        stage=stage,
        common_config=CommonFineTuneConfig(context_size=32, use_lr_scheduler=False),
        device="cpu",
        datasets={"target": data},
        initial_checkpoint=None,
    )

    with pytest.raises(RuntimeError, match="non-finite transfer loss"):
        engine.train_update("target", rng=np.random.default_rng(9))
