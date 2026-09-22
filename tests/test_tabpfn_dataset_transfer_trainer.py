from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.dataset_transfer.spec import (
    TransferRouteSpec,
    TransferStageSpec,
    ValidationSelector,
)
from finetuning.models import tabpfn_transfer, tabpfn_transfer_runtime
from finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferTrainerRequest,
)
from finetuning.models.tabpfn import TabPFNAdapter


def _data(name: str) -> SimpleNamespace:
    train_X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    train_y = np.array([0, 1, 0, 1])
    return SimpleNamespace(
        name=name,
        task_type="binary",
        metric_name="roc_auc",
        train_X=train_X,
        train_y=train_y,
        val_context_X=train_X,
        val_context_y=train_y,
        val_X=pd.DataFrame({"x": [5.0, 6.0]}),
        val_y=np.array([0, 1]),
        test_context_X=train_X,
        test_context_y=train_y,
        test_X=pd.DataFrame({"x": [7.0, 8.0]}),
        test_y=np.array([0, 1]),
    )


def _stage(*, role: str = "terminal", view_boundaries: tuple[int, ...] = (2,)) -> TransferStageSpec:
    return TransferStageSpec(
        name="pooled",
        role=role,
        training_mode="pooled_episodic",
        datasets=("target", "source"),
        row_budget_per_dataset=4,
        update_budget=4,
        sampling_weights=(0.5, 0.5),
        selector=ValidationSelector("target_validation", ("target",)),
        preprocessing="shared",
        model_id="tabpfn_v2_5",
        base_checkpoint="base.ckpt",
        seed=7,
        row_subset_seed=11,
        view_boundaries=view_boundaries,
    )


def _route(stage: TransferStageSpec, *, target: str = "target") -> TransferRouteSpec:
    return TransferRouteSpec(
        route_id="pooled_target_sources",
        family="dataset_transfer",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
    )


class _FakeRuntime:
    instances: list[_FakeRuntime] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.step = 0
        self.wrapper = self
        self.sampled: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def synchronize(self) -> None:
        return None

    def sample_episode(self, data, *, rng):
        _ = rng
        self.sampled.append(data.name)
        return SimpleNamespace(example_count=10)

    def train_episode(self, episode) -> float:
        _ = episode
        self.step += 1
        return float(self.step)

    def clone_state(self):
        return {"step": self.step}

    def export_state(self, state, output_dir, *, update, score, checkpoint_mode):
        _ = update, score, checkpoint_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "weights.pth"
        path.write_text(str(state["step"]), encoding="utf-8")
        return path

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    model_id = "tabpfn_v2_5"
    inference_seeds: list[int] = []

    def load_checkpoint_for_inference(self, path, task_type, device, n_estimators, seed):
        _ = task_type, device, n_estimators
        self.inference_seeds.append(seed)
        return SimpleNamespace(path=path)

    def fit_inference_context(self, estimator, context_X, context_y):
        _ = context_X, context_y
        return estimator

    def predict_for_metrics(self, estimator, query_X, task_type):
        _ = estimator, query_X, task_type
        return np.array([[0.4, 0.6]])


def _install_fake_evaluation(monkeypatch) -> None:
    def zero_shot(**kwargs):
        return {split: {"score": 0.5} for split in kwargs["splits"]}

    def live(**kwargs):
        step = kwargs["estimator"].step
        scores = {1: 0.51, 2: 0.56, 3: 0.52, 4: 0.54}
        return {"val": {"score": scores[step]}}

    def checkpoint(**kwargs):
        assert kwargs["seed"] == 7
        step = int(kwargs["checkpoint_path"].read_text(encoding="utf-8"))
        return {split: {"score": 0.5 + step / 100} for split in kwargs["splits"]}

    monkeypatch.setattr(tabpfn_transfer, "_evaluate_zero_shot", zero_shot)
    monkeypatch.setattr(tabpfn_transfer, "_evaluate_in_memory_estimator_on_data", live)
    monkeypatch.setattr(tabpfn_transfer, "_evaluate_checkpoint", checkpoint)
    monkeypatch.setattr(
        tabpfn_transfer,
        "transfer_package_versions",
        lambda model_id: {"model": model_id},
    )


def test_tabpfn_transfer_trainer_runs_one_balanced_persistent_stage(monkeypatch, tmp_path: Path) -> None:
    _install_fake_evaluation(monkeypatch)
    _FakeRuntime.instances.clear()
    _FakeAdapter.inference_seeds.clear()
    stage = _stage()
    route = _route(stage)
    calls: list[tuple[str, int | None, str, int | None]] = []

    def loader(name, *, row_budget, row_budget_policy, row_subset_seed):
        calls.append((name, row_budget, row_budget_policy, row_subset_seed))
        return _data(name)

    trainer = tabpfn_transfer.TabPFNDatasetTransferStageTrainer(
        adapter=_FakeAdapter(),
        route=route,
        config={},
        common_config=CommonFineTuneConfig(early_stopping=False),
        device="cpu",
        dataset_loader=loader,
        runtime_factory=_FakeRuntime,
    )
    initial = tmp_path / "initial.pth"
    initial.write_bytes(b"initial")

    production = trainer.train_stage(stage=stage, initial_checkpoint=initial, output_dir=tmp_path / "out")

    assert len(_FakeRuntime.instances) == 1
    runtime = _FakeRuntime.instances[0]
    assert runtime.kwargs["initial_checkpoint"] == initial
    assert runtime.kwargs["total_updates"] == 4
    assert runtime.closed
    assert sorted(runtime.sampled) == ["source", "source", "target", "target"]
    assert calls == [
        ("target", 4, "cap", 11),
        ("source", 4, "cap", 11),
    ]
    assert production.checkpoint_path.read_text(encoding="utf-8") == "2"
    assert production.view_checkpoints[2].read_text(encoding="utf-8") == "2"
    assert production.report["realized_optimizer_updates"] == 4
    assert production.report["dataset_optimizer_updates"] == {"target": 2, "source": 2}
    assert production.report["realized_training_examples"] == 40
    assert production.report["validation_cadence_updates"] == 2
    assert production.report["validation_candidate_updates"] == [2, 4]
    assert production.report["checkpoint_selection"]["selected_update"] == 2
    assert production.report["checkpoint_selection"]["max_update"] == 4
    assert production.report["checkpoint_view_selection"]["2"]["selected_update"] == 2
    assert production.report["metrics"]["test"]["score"] == 0.52
    assert production.package_versions == {"model": "tabpfn_v2_5"}
    assert production.reload_validator is not None
    assert production.reload_validator(production.checkpoint_path) is True
    assert _FakeAdapter.inference_seeds == [7]


def test_tabpfn_transfer_reusable_source_boundary_does_not_require_target_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_evaluation(monkeypatch)
    stage = _stage(role="reusable_boundary", view_boundaries=())
    route = _route(stage, target="later_target")
    trainer = tabpfn_transfer.TabPFNDatasetTransferStageTrainer(
        adapter=_FakeAdapter(),
        route=route,
        config={},
        common_config=CommonFineTuneConfig(early_stopping=False),
        device="cpu",
        dataset_loader=lambda name, **kwargs: _data(name),
        runtime_factory=_FakeRuntime,
    )

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert production.report["metrics"] == {}
    assert production.report["zero-shot-metrics"] == {}
    assert production.reload_validator is not None


def test_tabpfn_source_only_terminal_loads_target_for_evaluation_not_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_evaluation(monkeypatch)
    stage = replace(
        _stage(view_boundaries=()),
        datasets=("source-a", "source-b"),
        selector=ValidationSelector(
            "equal_mean_validation_improvement",
            ("source-a", "source-b"),
        ),
    )
    calls: list[str] = []

    def loader(name, **kwargs):
        _ = kwargs
        calls.append(name)
        return _data(name)

    trainer = tabpfn_transfer.TabPFNDatasetTransferStageTrainer(
        adapter=_FakeAdapter(),
        route=_route(stage, target="target"),
        config={},
        common_config=CommonFineTuneConfig(early_stopping=False),
        device="cpu",
        dataset_loader=loader,
        runtime_factory=_FakeRuntime,
    )

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert calls == ["source-a", "source-b", "target"]
    assert set(_FakeRuntime.instances[-1].sampled) == {"source-a", "source-b"}
    assert production.report["metrics"]["test"]["score"] > 0.5


def test_tabpfn_early_stopping_cannot_precede_required_prefix(monkeypatch, tmp_path: Path) -> None:
    _install_fake_evaluation(monkeypatch)
    stage = replace(
        _stage(),
        training_mode="single_dataset",
        datasets=("target",),
        update_budget=4,
        sampling_weights=(1.0,),
        view_boundaries=(3,),
    )
    trainer = tabpfn_transfer.TabPFNDatasetTransferStageTrainer(
        adapter=_FakeAdapter(),
        route=_route(stage),
        config={},
        common_config=CommonFineTuneConfig(early_stopping=True, patience=1),
        device="cpu",
        dataset_loader=lambda name, **kwargs: _data(name),
        runtime_factory=_FakeRuntime,
    )

    production = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)

    assert production.report["realized_optimizer_updates"] == 3
    assert production.report["validation_candidate_updates"] == [1, 2, 3]
    assert production.report["checkpoint_selection"]["selected_update"] == 2
    assert production.report["checkpoint_selection"]["max_update"] == 3
    view = production.report["checkpoint_view_selection"]["3"]
    assert view["kind"] == "target_validation"
    assert view["datasets"] == ["target"]
    assert view["max_update"] == 3
    assert view["selected_update"] == 2


def test_tabpfn_validation_baselines_are_cached_across_stage_calls(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def zero_shot(**kwargs):
        calls.append(kwargs["splits"])
        return {split: {"score": 0.5} for split in kwargs["splits"]}

    _install_fake_evaluation(monkeypatch)
    monkeypatch.setattr(tabpfn_transfer, "_evaluate_zero_shot", zero_shot)
    stage = _stage()
    data = {name: _data(name) for name in stage.datasets}
    trainer = tabpfn_transfer.TabPFNDatasetTransferStageTrainer(
        adapter=_FakeAdapter(),
        route=_route(stage),
        config={},
        common_config=CommonFineTuneConfig(early_stopping=False),
        device="cpu",
        dataset_loader=lambda name, **kwargs: data[name],
        runtime_factory=_FakeRuntime,
    )

    first = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path / "first")
    second = trainer.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path / "second")

    assert calls == [("val",), ("test",)]
    assert first.report["validation_baseline_cache_hits"] == 1
    assert first.report["validation_baseline_cache_misses"] == 2
    assert second.report["validation_baseline_cache_hits"] == 3
    assert second.report["validation_baseline_cache_misses"] == 0


def test_tabpfn_runtime_initializes_one_fresh_optimizer_and_fitted_wrapper(monkeypatch) -> None:
    class Model:
        def parameters(self):
            return ("weights",)

        def to(self, device):
            self.device = device

    class Inner:
        def __init__(self) -> None:
            self.model_ = Model()

        def _initialize_model_variables(self):
            self.initialized = True

    inner = Inner()
    wrapper = SimpleNamespace(
        device="cpu",
        random_state=3,
        n_estimators_finetune=1,
        _estimator_kwargs={"model_path": "initial.pth", "inference_config": {}},
        _build_estimator_config=lambda base, count: {**base, "n_estimators": count},
        _create_estimator=lambda config: inner,
        _setup_estimator=lambda: None,
        _forward_with_loss=lambda batch: None,
        _get_valid_finetuning_query_size=lambda **kwargs: 1,
        _setup_batch=lambda batch: None,
        _should_skip_batch=lambda batch: False,
    )
    optimizer_calls: list[dict] = []

    class Optimizer:
        def zero_grad(self, **kwargs):
            self.zero_grad_kwargs = kwargs

    optimizer = Optimizer()

    def optimizer_factory(**kwargs):
        optimizer_calls.append(kwargs)
        return optimizer

    bindings = SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        PerformanceOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        GradScaler=lambda: None,
        get_and_init_optimizer=optimizer_factory,
        get_cosine_schedule_with_warmup=lambda **kwargs: None,
        LambdaLR=lambda *args, **kwargs: None,
    )
    adapter = SimpleNamespace(
        _vendor=lambda: None,
        resolve_finetune_estimator=lambda *args, **kwargs: wrapper,
    )
    monkeypatch.setattr(tabpfn_transfer_runtime, "_load_runtime_bindings", lambda: bindings)

    runtime = tabpfn_transfer_runtime.TabPFNTransferRuntime(
        adapter=adapter,
        common_config=CommonFineTuneConfig(use_lr_scheduler=False),
        task_type="binary",
        seed=3,
        device="cpu",
        initial_checkpoint=Path("initial.pth"),
        total_updates=4,
    )

    assert wrapper.is_fitted_ is True
    assert wrapper.finetuned_estimator_ is inner
    assert inner.initialized is True
    assert len(optimizer_calls) == 1
    assert optimizer_calls[0]["checkpoint_path"] is None
    assert optimizer_calls[0]["model_parameters"] == ("weights",)
    runtime.close()
    assert optimizer.zero_grad_kwargs == {"set_to_none": True}


def test_tabpfn_amp_overflow_is_identified_as_a_skipped_update() -> None:
    assert tabpfn_transfer_runtime._amp_step_was_skipped(scale_before=1024.0, scale_after=512.0)
    assert not tabpfn_transfer_runtime._amp_step_was_skipped(scale_before=1024.0, scale_after=1024.0)
    assert not tabpfn_transfer_runtime._amp_step_was_skipped(scale_before=1024.0, scale_after=2048.0)


def test_tabpfn_episode_uses_unstratified_split_only_when_required() -> None:
    assert tabpfn_transfer_runtime._requires_unstratified_episode_split(
        np.array([0, 0, 1, 1, 2]),
        episode_rows=5,
        query_rows=2,
    )
    assert tabpfn_transfer_runtime._requires_unstratified_episode_split(
        np.array([0, 0, 1, 1, 2, 2]),
        episode_rows=6,
        query_rows=2,
    )
    assert not tabpfn_transfer_runtime._requires_unstratified_episode_split(
        np.array([0, 0, 1, 1, 2, 2]),
        episode_rows=6,
        query_rows=3,
    )


def test_tabpfn_runtime_disables_vendor_stratification_for_singleton_class() -> None:
    observed: dict = {}
    item = SimpleNamespace(
        y_context=[np.array([0, 1, 2])],
        y_query=np.array([0, 1]),
    )

    def get_chunks(**kwargs):
        observed.update(kwargs)
        return [item]

    runtime = object.__new__(tabpfn_transfer_runtime.TabPFNTransferRuntime)
    runtime.common_config = CommonFineTuneConfig(context_size=5)
    runtime.unstratified_episode_count = 0
    runtime.wrapper = SimpleNamespace(
        _get_valid_finetuning_query_size=lambda **kwargs: 2,
        finetuned_estimator_=SimpleNamespace(),
    )
    runtime.bindings = SimpleNamespace(
        get_preprocessed_dataset_chunks=get_chunks,
        meta_dataset_collator=lambda items: items[0],
    )
    data = SimpleNamespace(
        train_X=pd.DataFrame({"x": range(5)}),
        train_y=np.array([0, 0, 1, 1, 2]),
    )

    episode = runtime._sample_episode_once(data, seed=7)

    assert observed["force_no_stratify"] is True
    assert runtime.unstratified_episode_count == 1
    assert episode.example_count == 5


def test_tabpfn_regression_episode_uses_regressor_preprocessing_without_stratification() -> None:
    observed: dict = {}
    item = SimpleNamespace(y_context=[np.array([0.1, 0.2, 0.3])], y_query=np.array([0.4, 0.5]))

    runtime = object.__new__(tabpfn_transfer_runtime.TabPFNTransferRuntime)
    runtime.task_type = "regression"
    runtime.common_config = CommonFineTuneConfig(context_size=5)
    runtime.unstratified_episode_count = 0
    runtime.wrapper = SimpleNamespace(
        _get_valid_finetuning_query_size=lambda **kwargs: 2,
        finetuned_estimator_=SimpleNamespace(),
    )
    runtime.bindings = SimpleNamespace(
        get_preprocessed_dataset_chunks=lambda **kwargs: observed.update(kwargs) or [item],
        meta_dataset_collator=lambda items: items[0],
    )
    data = SimpleNamespace(
        train_X=pd.DataFrame({"x": range(5)}),
        train_y=np.linspace(0.0, 1.0, 5),
    )

    runtime._sample_episode_once(data, seed=7)

    assert observed["model_type"] == "regressor"
    assert observed["force_no_stratify"] is True


def test_tabpfn_amp_skipped_step_retries_before_advancing_scheduler() -> None:
    class Scalar:
        def detach(self):
            return self

        def item(self):
            return 1.0

        def backward(self):
            return None

    class Scaler:
        scale_value = 1024.0
        step_calls = 0
        update_calls = 0

        def get_scale(self):
            return self.scale_value

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            return None

        def step(self, optimizer):
            self.step_calls += 1

        def update(self):
            self.update_calls += 1
            if self.update_calls == 1:
                self.scale_value = 512.0

    class Counter:
        calls = 0

        def step(self):
            self.calls += 1

    model = SimpleNamespace(parameters=lambda: ())
    inner = SimpleNamespace(
        model_=model,
        fit_from_preprocessed=lambda *args, **kwargs: None,
    )
    wrapper = SimpleNamespace(
        finetuned_estimator_=inner,
        _setup_batch=lambda batch: None,
        _forward_with_loss=lambda batch: Scalar(),
    )
    scheduler = Counter()
    runtime = object.__new__(tabpfn_transfer_runtime.TabPFNTransferRuntime)
    runtime.wrapper = wrapper
    runtime.optimizer = SimpleNamespace(zero_grad=lambda: None)
    runtime.scheduler = scheduler
    runtime.scaler = Scaler()
    runtime.use_amp = True
    runtime.common_config = CommonFineTuneConfig(grad_clip=None)
    runtime.performance_options = SimpleNamespace()
    runtime.amp_overflow_retry_count = 0
    runtime.bindings = SimpleNamespace(
        torch=SimpleNamespace(isfinite=lambda value: SimpleNamespace(item=lambda: True)),
        autocast=lambda **kwargs: contextlib.nullcontext(),
        sdpa_kernel_context=lambda: contextlib.nullcontext(),
    )
    batch = SimpleNamespace(X_context=[], y_context=[], cat_indices=[], configs=[])

    loss = runtime.train_episode(tabpfn_transfer_runtime.TabPFNTransferEpisode(batch=batch, example_count=1))

    assert loss == 1.0
    assert runtime.scaler.step_calls == 2
    assert runtime.amp_overflow_retry_count == 1
    assert scheduler.calls == 1


def test_tabpfn_adapter_exposes_exact_transfer_factory() -> None:
    adapter = TabPFNAdapter(SimpleNamespace(model_id="tabpfn_v2_5"))
    stage = _stage()
    trainer = adapter.create_dataset_transfer_stage_trainer(
        DatasetTransferTrainerRequest(
            route=_route(stage),
            config={},
            common_config=CommonFineTuneConfig(),
            device="cpu",
            dataset_loader=lambda *args, **kwargs: None,
        )
    )

    assert isinstance(trainer, tabpfn_transfer.TabPFNDatasetTransferStageTrainer)
    assert trainer.contract.persistent_optimizer_within_stage is True
    assert trainer.contract.raw_row_concatenation is False
