from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tfm_finetuning.experiments.transfer.pooled import run_balanced_pooled_training
from tfm_finetuning.experiments.transfer.runner.training import ValidatedStageTrainer
from tfm_finetuning.experiments.transfer.spec import (
    TransferRouteSpec,
    TransferStageSpec,
    TransferTrainingRecipe,
    ValidationSelector,
)
from tfm_finetuning.finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferTrainerRequest,
)
from tfm_finetuning.finetuning.models.mitra.adapter import MitraAdapter
from tfm_finetuning.finetuning.models.mitra.chunking import LocalMitraFineTuneDiagnostics
from tfm_finetuning.finetuning.models.mitra.configured import _configured_mitra_class
from tfm_finetuning.finetuning.models.mitra.transfer import MitraDatasetTransferStageTrainer
from tfm_finetuning.finetuning.models.mitra.transfer_evaluation import (
    MitraTransferEvaluator,
)
from tfm_finetuning.finetuning.models.mitra.transfer_session import (
    MitraDatasetEpisodeSource,
    MitraPersistentStageSession,
    PreparedMitraDataset,
    assign_transfer_member_indices,
    initialize_stage_estimator,
)


class _FakeModel:
    def train(self) -> None:
        pass


class _PickleableMitraBase:
    pass


def test_configured_mitra_class_is_stable_for_pickle_identity() -> None:
    first = _configured_mitra_class(_PickleableMitraBase)
    second = _configured_mitra_class(_PickleableMitraBase)

    assert second is first
    restored = pickle.loads(pickle.dumps(object.__new__(first)))
    assert type(restored) is first


class _FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def get_last_lr(self) -> list[float]:
        return [1e-5]


class _FakeOptimizer:
    def __init__(self) -> None:
        self.state = {0: {"step": 0}}


class _FakeMember:
    def __init__(self, *, advance_optimizer: bool = True) -> None:
        self.optimizer = _FakeOptimizer()
        self.model = _FakeModel()
        self.device = "cpu"
        self.preprocessor = object()
        self.steps = 0
        self.advance_optimizer = advance_optimizer
        self.cfg = SimpleNamespace(
            task="classification",
            hyperparams={"max_samples_support": 8, "max_samples_query": 4},
        )

    def make_loader(self, dataset, training):
        _ = dataset, training
        return [{}]

    def _train_batch(self, batch, *, tracker, diagnostics):
        import torch

        _ = batch, tracker, diagnostics
        self.steps += 1
        if self.advance_optimizer:
            self.optimizer.state[0]["step"] += 1
        return torch.tensor(float(self.steps))

    def _log_step(self, **kwargs) -> None:
        _ = kwargs


def _prepared(name: str, *, rows: int, features: int, members: int) -> PreparedMitraDataset:
    labels = np.arange(rows, dtype=np.int64) % 2
    return PreparedMitraDataset(
        name=name,
        data=SimpleNamespace(task_type="binary"),
        classes=np.array([0, 1]),
        member_features=tuple(
            np.arange(rows * features, dtype=np.float32).reshape(rows, features) + member for member in range(members)
        ),
        member_labels=tuple(labels.copy() for _ in range(members)),
        member_preprocessors=tuple(object() for _ in range(members)),
    )


def test_mitra_pooled_session_counts_logical_and_member_optimizer_steps() -> None:
    trainers = tuple(_FakeMember() for _ in range(2))
    assign_transfer_member_indices(trainers)
    datasets = {
        "narrow": _prepared("narrow", rows=12, features=2, members=2),
        "wide": _prepared("wide", rows=15, features=5, members=2),
    }
    schedulers = tuple(_FakeScheduler() for _ in trainers)
    session = MitraPersistentStageSession(
        trainers=trainers,
        datasets=datasets,
        schedulers=schedulers,
        diagnostics=tuple(LocalMitraFineTuneDiagnostics() for _ in trainers),
        seed=7,
    )
    sources = {
        name: MitraDatasetEpisodeSource(dataset, context_size=8, query_ratio=0.25)
        for name, dataset in datasets.items()
    }

    result = run_balanced_pooled_training(
        session,
        sources,
        updates_per_dataset=3,
        seed=7,
    )

    assert result.total_updates == 6
    assert dict(result.update_counts) == {"narrow": 3, "wide": 3}
    assert session.member_optimizer_steps == [6, 6]
    assert [trainer.steps for trainer in trainers] == [6, 6]
    assert [scheduler.steps for scheduler in schedulers] == [6, 6]
    assert session.member_examples == session.logical_examples * 2
    assert {datasets[record.dataset_name].member_features[0].shape[1] for record in result.records} == {2, 5}


def test_mitra_persistent_session_rejects_optimizer_replacement() -> None:
    trainer = _FakeMember()
    assign_transfer_member_indices((trainer,))
    dataset = _prepared("data", rows=8, features=2, members=1)
    session = MitraPersistentStageSession(
        trainers=(trainer,),
        datasets={"data": dataset},
        schedulers=(None,),
        diagnostics=(LocalMitraFineTuneDiagnostics(),),
        seed=0,
    )
    episode = MitraDatasetEpisodeSource(dataset, context_size=8, query_ratio=0.25).sample_episode(
        rng=np.random.default_rng(0)
    )
    session.train_episode(
        episode,
        dataset_name="data",
        global_update_index=0,
        dataset_update_index=0,
    )
    trainer.optimizer = object()

    with pytest.raises(RuntimeError, match="optimizer was replaced"):
        session.train_episode(
            episode,
            dataset_name="data",
            global_update_index=1,
            dataset_update_index=1,
        )


def test_mitra_persistent_session_rejects_amp_skipped_member_step() -> None:
    trainer = _FakeMember(advance_optimizer=False)
    assign_transfer_member_indices((trainer,))
    dataset = _prepared("data", rows=8, features=2, members=1)
    session = MitraPersistentStageSession(
        trainers=(trainer,),
        datasets={"data": dataset},
        schedulers=(None,),
        diagnostics=(LocalMitraFineTuneDiagnostics(),),
        seed=0,
    )
    episode = MitraDatasetEpisodeSource(dataset, context_size=8, query_ratio=0.25).sample_episode(
        rng=np.random.default_rng(0)
    )

    with pytest.raises(RuntimeError, match="optimizer did not advance"):
        session.train_episode(
            episode,
            dataset_name="data",
            global_update_index=0,
            dataset_update_index=0,
        )
    assert session.member_optimizer_steps == [0]


def test_mitra_adapter_exposes_exact_transfer_factory() -> None:
    route = TransferRouteSpec(
        route_id="target",
        family="dataset_transfer",
        target_dataset="target",
        execution_mode="train",
        stages=(
            TransferStageSpec(
                name="target",
                role="terminal",
                training_mode="single_dataset",
                datasets=("target",),
                row_budget_per_dataset=None,
                update_budget=1,
                sampling_weights=(1.0,),
                selector=ValidationSelector("target_validation", ("target",)),
                model_id="mitra",
                base_checkpoint="autogluon/mitra-classifier",
            ),
        ),
    )
    adapter = MitraAdapter(SimpleNamespace())

    trainer = adapter.create_dataset_transfer_stage_trainer(
        DatasetTransferTrainerRequest(
            route=route,
            config={},
            common_config=CommonFineTuneConfig(
                n_estimators_finetune=2,
                n_estimators_validation=2,
                n_estimators_final_inference=2,
            ),
            device="cpu",
            dataset_loader=lambda name: name,
        )
    )

    assert isinstance(trainer, MitraDatasetTransferStageTrainer)
    assert trainer.contract.persistent_optimizer_within_stage is True
    assert trainer.contract.logical_update_unit == "dataset_episode_all_ensemble_members"


def test_mitra_stage_initializer_sets_portable_task_and_architecture_metadata(monkeypatch) -> None:
    from autogluon.tabular.models.mitra._internal.config.enums import Task

    observed = {}

    class Estimator:
        tfm_mitra_finetune_interface = "local_chunked"
        _tfm_initial_trainer_states = [{"weight": 1}, {"weight": 2}]
        trainers = []

        def _create_config(self, task, dim_output, budget):
            observed["create"] = (task, dim_output, budget)
            return SimpleNamespace(task=None, hyperparams={"dim_output": dim_output}), object

    estimator = Estimator()

    class Adapter:
        def resolve_finetune_estimator(self, *args, **kwargs):
            observed["resolve"] = (args, kwargs)
            return estimator

    def fresh(resolved, *, cfg, model_cls, task, dim_output, n_classes, states):
        observed["fresh"] = (resolved, cfg, model_cls, task, dim_output, n_classes, states)
        return [
            SimpleNamespace(cfg=SimpleNamespace(hyperparams={"dim_output": 10}), n_classes=10),
            SimpleNamespace(cfg=SimpleNamespace(hyperparams={"dim_output": 10}), n_classes=10),
        ]

    monkeypatch.setattr("tfm_finetuning.finetuning.models.mitra.transfer_session._fresh_local_trainers", fresh)
    common = CommonFineTuneConfig(
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_final_inference=2,
    )

    initialized = initialize_stage_estimator(
        Adapter(),
        task_type="multiclass",
        common_config=common,
        seed=3,
        device="cpu",
        initial_checkpoint=Path("parent.ckpt"),
    )

    assert initialized is estimator
    assert estimator.task is Task.CLASSIFICATION
    assert observed["create"] == (Task.CLASSIFICATION, 10, None)
    assert observed["fresh"][3:6] == (Task.CLASSIFICATION, 10, 10)
    assert observed["fresh"][6] == [{"weight": 1}, {"weight": 2}]
    assert estimator._tfm_initial_trainer_states is None


def test_mitra_stage_initializer_uses_regression_architecture(monkeypatch) -> None:
    from autogluon.tabular.models.mitra._internal.config.enums import Task

    observed = {}

    class Estimator:
        tfm_mitra_finetune_interface = "local_chunked"
        _tfm_initial_trainer_states = None
        trainers = []

        def _create_config(self, task, dim_output, budget):
            observed["create"] = (task, dim_output, budget)
            return SimpleNamespace(task=None, hyperparams={"dim_output": dim_output}), object

    estimator = Estimator()
    adapter = SimpleNamespace(resolve_finetune_estimator=lambda *args, **kwargs: estimator)

    def fresh(resolved, *, cfg, model_cls, task, dim_output, n_classes, states):
        observed["fresh"] = (task, dim_output, n_classes, states)
        return [SimpleNamespace(cfg=SimpleNamespace(hyperparams={"dim_output": 1}), n_classes=0)]

    monkeypatch.setattr("tfm_finetuning.finetuning.models.mitra.transfer_session._fresh_local_trainers", fresh)

    initialized = initialize_stage_estimator(
        adapter,
        task_type="regression",
        common_config=CommonFineTuneConfig(
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_final_inference=1,
        ),
        seed=3,
        device="cpu",
        initial_checkpoint=None,
    )

    assert initialized.task is Task.REGRESSION
    assert observed["create"] == (Task.REGRESSION, 1, None)
    assert observed["fresh"] == (Task.REGRESSION, 1, 0, None)


@pytest.mark.parametrize(
    ("sampling_weights", "strict_resources", "message"),
    [
        ((1.0, 2.0), True, "uniform dataset sampling"),
        ((1.0, float("nan")), True, "uniform dataset sampling"),
        ((1.0, 1.0), False, "strict_resource_semantics"),
    ],
)
def test_mitra_stage_rejects_nonconfirmatory_sampling_or_resource_semantics(
    sampling_weights,
    strict_resources,
    message,
    tmp_path: Path,
) -> None:
    stage = TransferStageSpec(
        name="pool",
        role="terminal",
        training_mode="pooled_episodic",
        datasets=("target", "source"),
        row_budget_per_dataset=None,
        update_budget=2,
        sampling_weights=sampling_weights,
        selector=ValidationSelector("target_validation", ("target",)),
        model_id="mitra",
        base_checkpoint="autogluon/mitra-classifier",
        strict_resource_semantics=strict_resources,
    )
    route = TransferRouteSpec(
        route_id="pool",
        family="dataset_transfer",
        target_dataset="target",
        execution_mode="train",
        stages=(stage,),
    )
    trainer = MitraDatasetTransferStageTrainer(
        SimpleNamespace(),
        route=route,
        config={},
        common_config=CommonFineTuneConfig(),
        device="cpu",
        dataset_loader=lambda *args, **kwargs: pytest.fail("data loaded before protocol validation"),
    )

    with pytest.raises(ValueError, match=message):
        trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)


def test_mitra_stage_early_stopping_waits_for_checkpoint_view_boundary(monkeypatch, tmp_path: Path) -> None:
    stage = TransferStageSpec(
        name="target",
        role="terminal",
        training_mode="single_dataset",
        datasets=("target",),
        row_budget_per_dataset=10,
        update_budget=4,
        sampling_weights=(1.0,),
        selector=ValidationSelector("target_validation", ("target",)),
        model_id="mitra",
        base_checkpoint="autogluon/mitra-classifier",
        view_boundaries=(3,),
    )
    route = TransferRouteSpec(
        route_id="target_compute_matched",
        family="dataset_transfer",
        target_dataset="target",
        execution_mode="train",
        stages=(stage,),
    )
    data = SimpleNamespace(
        task_type="multiclass",
        metric_name="log_loss",
        train_X=pd.DataFrame(
            {
                "x": np.arange(10, dtype=np.float32),
                "new_y": np.arange(10, dtype=np.float32) + 1,
                "new_z": np.arange(10, dtype=np.float32) + 2,
            }
        ),
        train_y=np.arange(10) % 3,
        val_context_X=pd.DataFrame(
            {
                "x": np.arange(6, dtype=np.float32),
                "new_y": np.arange(6, dtype=np.float32) + 1,
                "new_z": np.arange(6, dtype=np.float32) + 2,
            }
        ),
        val_context_y=np.arange(6) % 3,
        val_X=pd.DataFrame({"x": [7.0], "new_y": [8.0], "new_z": [9.0]}),
    )
    loader_calls = []

    def loader(name, **kwargs):
        loader_calls.append((name, kwargs))
        return data

    members = tuple(_FakeMember() for _ in range(2))
    estimator = SimpleNamespace(trainers=list(members))
    monkeypatch.setattr(
        "tfm_finetuning.finetuning.models.mitra.transfer.initialize_stage_estimator",
        lambda *args, **kwargs: estimator,
    )
    monkeypatch.setattr(
        "tfm_finetuning.finetuning.models.mitra.transfer.assign_transfer_member_indices",
        lambda trainers: None,
    )
    prepared = _prepared("target", rows=10, features=1, members=2)
    monkeypatch.setattr(
        "tfm_finetuning.finetuning.models.mitra.transfer.prepare_stage_datasets",
        lambda *args, **kwargs: {"target": prepared},
    )
    monkeypatch.setattr(
        "tfm_finetuning.finetuning.models.mitra.transfer.exact_stage_schedulers",
        lambda *args, **kwargs: ((None, None), (LocalMitraFineTuneDiagnostics(),) * 2),
    )

    class FakeSession:
        def __init__(self, **kwargs):
            self.trainers = kwargs["trainers"]
            self.member_optimizer_steps = [0, 0]
            self.logical_examples = 0
            self.member_examples = 0

        def train_episode(self, episode, **kwargs):
            _ = kwargs
            self.member_optimizer_steps = [value + 1 for value in self.member_optimizer_steps]
            self.logical_examples += episode.example_count
            self.member_examples += episode.example_count * 2
            return {"loss": 1.0, "member_optimizer_steps": 2}

    monkeypatch.setattr("tfm_finetuning.finetuning.models.mitra.transfer.MitraPersistentStageSession", FakeSession)

    def fake_write(estimator, path):
        _ = estimator
        update = path.stem.removeprefix("candidate-update-")
        path.write_text(update, encoding="utf-8")

    monkeypatch.setattr(
        "tfm_finetuning.finetuning.models.mitra.transfer_evaluation."
        "write_checkpoint_package",
        fake_write,
    )

    class FakeAdapter:
        fitted_context_width = None
        inference_seeds: list[int] = []

        def load_checkpoint_for_inference(self, path, task_type, device, ensemble_size, seed):
            _ = path, task_type, device
            self.inference_seeds.append(seed)
            return SimpleNamespace(trainers=[object() for _ in range(ensemble_size)])

        def fit_inference_context(self, estimator, context_X, context_y):
            self.fitted_context_width = context_X.shape[1]
            estimator.classes_ = np.unique(context_y)
            estimator._tfm_mitra_feature_preprocessor = {"target_width": context_X.shape[1]}
            return estimator

        def predict_for_metrics(self, estimator, query_X, task_type):
            _ = estimator, task_type
            return np.full((len(query_X), 3), 1.0 / 3.0)

    common = CommonFineTuneConfig(
        context_size=8,
        query_ratio=0.25,
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_final_inference=2,
        early_stopping=True,
        patience=1,
        min_delta=0.0,
    )
    stage = replace(stage, training_recipe=TransferTrainingRecipe.from_common_config(common))
    route = replace(route, stages=(stage,))
    trainer = MitraDatasetTransferStageTrainer(
        FakeAdapter(),
        route=route,
        config={},
        common_config=common,
        device="cpu",
        dataset_loader=loader,
    )
    monkeypatch.setattr(
        MitraTransferEvaluator,
        "evaluate_baseline",
        lambda self, *args, **kwargs: {
            "val": {"score": 0.5},
            "test": {"score": 0.5},
        },
    )

    def fake_checkpoint(
        self,
        path,
        data,
        *,
        splits,
        inference_seed,
        isolation_seed,
    ):
        _ = self, data, isolation_seed
        assert inference_seed == stage.seed
        update = int(path.read_text(encoding="utf-8"))
        scores = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
        return {split: {"score": scores[update]} for split in splits}

    monkeypatch.setattr(
        MitraTransferEvaluator,
        "evaluate_checkpoint",
        fake_checkpoint,
    )

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    class Delegate:
        common_config = trainer.common_config

        def train_stage(self, **kwargs):
            _ = kwargs
            return production

    with pytest.raises(ValueError, match="exact update budget"):
        ValidatedStageTrainer(
            Delegate(),
            recipe=stage.training_recipe,
        ).train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert production.report["realized_optimizer_updates"] == 3
    assert production.report["realized_member_optimizer_steps"] == 6
    assert production.report["checkpoint_selection"]["selected_update"] == 1
    assert production.report["checkpoint_view_selection"]["3"]["selected_update"] == 1
    assert production.report["early_stopped"] is True
    assert production.checkpoint_path.name == "candidate-update-1.ckpt"
    assert production.view_checkpoints[3].name == "view-boundary-3.ckpt"
    assert production.reload_validator(production.checkpoint_path) is True
    assert trainer.adapter.fitted_context_width == 3
    assert trainer.adapter.inference_seeds == [stage.seed]
    assert not (tmp_path / "candidate-update-2.ckpt").exists()
    assert not (tmp_path / "candidate-update-3.ckpt").exists()
    assert loader_calls == [
        (
            "target",
            {"row_budget": 10, "row_budget_policy": "cap", "row_subset_seed": 0},
        )
    ]
