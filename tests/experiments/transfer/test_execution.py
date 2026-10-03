"""Fresh transfer execution preserves budgets, dependency weights and no-update evaluation."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tfm_finetuning.experiments.transfer.dependencies import producer_config, resolve_checkpoint_dependency
from tfm_finetuning.experiments.transfer.routes.resolve import resolve_transfer_route
from tfm_finetuning.experiments.transfer.runner.executor import StageProduction, execute_route
from tfm_finetuning.experiments.transfer.runner.reporting import training_report
from tfm_finetuning.experiments.transfer.runner.training import ValidatedStageTrainer


def config(route="pooled_sources_then_target", **transfer):
    """Small exact-update recipe without any model downloads."""
    return {
        "seed": 2,
        "data": {"path": "data/tabarena/credit-g", "split": "fold0_repeat0_sample0"},
        "finetune": {
            "model_id": "tabpfn_v2_5",
            "classifier_checkpoint": "base.ckpt",
            "use_lr_scheduler": False,
            "early_stopping": False,
            "test_context_includes_validation_data": True,
        },
        "dataset_transfer": {
            "enabled": True,
            "family": "dataset_transfer",
            "route_id": route,
            "updates_per_dataset": 2,
            "row_budget_per_dataset": 100,
            "context_policy": "target_train_plus_validation",
            **transfer,
        },
    }


def report(stage, checkpoint):
    selection = {
        "kind": stage.selector.kind,
        "datasets": list(stage.selector.datasets),
        "selected_update": 1,
        "max_update": stage.update_budget,
    }
    scores = {split: {"score": 0.8, "roc_auc": 0.8} for split in ("val", "test")}
    return StageProduction(
        checkpoint,
        {
            "allocated_optimizer_updates": stage.update_budget,
            "realized_optimizer_updates": stage.update_budget,
            "dataset_optimizer_updates": {name: stage.update_budget // len(stage.datasets) for name in stage.datasets},
            "checkpoint_selection": selection,
            "metrics": scores,
            "zero-shot-metrics": scores,
            "checkpoint_view_metrics": {
                str(boundary): {"metrics": scores, "baseline_metrics": scores} for boundary in stage.view_boundaries
            },
            "checkpoint_view_selection": {
                str(boundary): {**selection, "max_update": boundary} for boundary in stage.view_boundaries
            },
        },
        view_checkpoints={boundary: checkpoint for boundary in stage.view_boundaries},
    )


def test_execution_loads_selected_stage_weights_and_never_replays_reports(tmp_path):
    route = resolve_transfer_route(config())
    calls = []

    class Trainer:
        def train_stage(self, *, stage, initial_checkpoint, output_dir):
            calls.append((stage.name, initial_checkpoint))
            checkpoint = output_dir / "selected.ckpt"
            checkpoint.write_text(stage.name)
            return report(stage, checkpoint)

    results = execute_route(route, trainer=Trainer(), output=tmp_path)
    assert calls == [("source_pool", None), ("target", results[0].production.checkpoint_path)]
    assert results[0].production.checkpoint_path.is_file()
    assert results[1].production.checkpoint_path.read_text() == "target"
    execute_route(route, trainer=Trainer(), output=tmp_path)
    assert len(calls) == 4


def test_missing_weights_are_not_replaced_with_training(tmp_path):
    route = resolve_transfer_route(config())
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        execute_route(route, trainer=None, output=tmp_path, initial_checkpoint=tmp_path / "missing")


def test_shorter_budget_control_retains_own_validation_selection(tmp_path):
    route = resolve_transfer_route(config("target_compute_matched"))
    stage = route.stages[0]
    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_text("selected")
    delegate = SimpleNamespace(
        common_config=stage.training_recipe, train_stage=lambda **kwargs: report(stage, checkpoint)
    )
    validated = ValidatedStageTrainer(delegate, stage.training_recipe)
    result = validated.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)
    from tfm_finetuning.experiments.transfer.runner.executor import StageResult

    payload = training_report(route, (StageResult(stage, result),))
    assert payload["target_update_metrics"]["2"]["selection"]["max_update"] == 2
    assert payload["target_update_metrics"]["6"]["selection"]["max_update"] == 6
    assert "resultViews" not in payload
    bad = dict(result.report)
    bad["checkpoint_view_selection"] = {"2": {**bad["checkpoint_view_selection"]["2"], "selected_update": 3}}
    delegate.train_stage = lambda **kwargs: replace(result, report=bad)
    with pytest.raises(ValueError, match="outside its 2-update"):
        validated.train_stage(stage=stage, initial_checkpoint=None, output_dir=tmp_path)


def consumer_config():
    return config(
        "frozen_probe",
        family="dataset_transfer_source_only_eval",
        execution_mode="checkpoint_eval",
        evaluation_target="credit-g",
        evaluation_targets=["satimage"],
        checkpoint_origin="credit-g",
        checkpoint_route="pooled_sources_then_target",
        checkpoint_role="reusable_boundary",
        dependency_updates_per_dataset=2,
        dependency_row_budget_per_dataset=100,
    )


def write_producer(root, consumer, *, name="producer", status="complete", changed_seed=None):
    route = resolve_transfer_route(consumer)
    producer = producer_config(consumer, route)
    if changed_seed is not None:
        producer["seed"] = changed_seed
    directory = root / "transfer" / name
    directory.mkdir(parents=True)
    checkpoint = directory / "source.ckpt"
    checkpoint.write_text("weights")
    (directory / "run.json").write_text(json.dumps({"status": status, "config": producer}))
    (directory / "finetune_report.json").write_text(
        json.dumps(
            {
                "config": producer,
                "dataset_transfer": {"family": "dataset_transfer", "route": "pooled_sources_then_target"},
                "stage_checkpoints": {"source_pool": str(checkpoint)},
            }
        )
    )
    return checkpoint


def test_dependency_requires_matching_completed_producer_and_actual_weights(tmp_path):
    consumer = consumer_config()
    route = resolve_transfer_route(consumer)
    write_producer(tmp_path, consumer, name="wrong-seed", changed_seed=99)
    write_producer(tmp_path, consumer, name="unfinished", status="running")
    with pytest.raises(FileNotFoundError, match="Run producer"):
        resolve_checkpoint_dependency(consumer, route, results_root=tmp_path, repo_root=tmp_path)
    checkpoint = write_producer(tmp_path, consumer)
    assert resolve_checkpoint_dependency(consumer, route, results_root=tmp_path, repo_root=tmp_path) == checkpoint
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        resolve_checkpoint_dependency(consumer, route, results_root=tmp_path, repo_root=tmp_path)


def test_dependency_ambiguity_requires_an_explicit_choice(tmp_path):
    consumer = consumer_config()
    route = resolve_transfer_route(consumer)
    checkpoint = write_producer(tmp_path, consumer, name="first")
    write_producer(tmp_path, consumer, name="second")
    with pytest.raises(ValueError, match="Multiple completed"):
        resolve_checkpoint_dependency(consumer, route, results_root=tmp_path, repo_root=tmp_path)
    consumer["dataset_transfer"]["checkpoint_path"] = str(checkpoint)
    assert resolve_checkpoint_dependency(consumer, route, results_root=tmp_path, repo_root=tmp_path) == checkpoint


def test_frozen_evaluation_has_no_training_path_and_uses_identical_context(monkeypatch, tmp_path):
    from tfm_finetuning.experiments.transfer.runner import evaluation

    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_text("weights")
    active = []

    @contextmanager
    def inference_mode():
        active.append(True)
        yield
        active.pop()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=inference_mode))
    contexts = []

    def evaluate(**kwargs):
        assert active == [True]
        assert kwargs["splits"] == ("test",)
        contexts.append(kwargs["data"])
        return {"test": {"score": 0.8}}

    monkeypatch.setattr(evaluation, "_evaluate_zero_shot", evaluate)
    monkeypatch.setattr(evaluation, "_evaluate_checkpoint", evaluate)
    route = resolve_transfer_route(consumer_config())
    data = SimpleNamespace(metric_name="roc_auc")
    views = evaluation.evaluate_frozen_checkpoint_bundle(
        route,
        checkpoint_path=checkpoint,
        target_data={"satimage": data},
        adapter_factory=lambda _: object(),
        common_config=None,
        seed=2,
        device="cpu",
    )
    assert contexts == [data, data]
    assert views[0]["target"] == "satimage"
    assert views[0]["route"] == "pooled_sources_then_target"


def test_all_final_transfer_consumers_have_one_matching_producer_recipe():
    """The published scripts can be run in dependency order without manual weight paths."""
    from collections import defaultdict
    from pathlib import Path

    from tfm_finetuning.experiments.configuration import expand_runs
    from tfm_finetuning.experiments.transfer.dependencies import _stage_recipe
    from tfm_finetuning.utils.config import read_yaml_mapping

    root = Path(__file__).resolve().parents[3]
    producers, consumers = defaultdict(list), []
    for path in sorted((root / "configs/experiments").glob("dataset_transfer*.yaml")):
        for run in expand_runs(read_yaml_mapping(path), repo_root=root):
            route = resolve_transfer_route(run["config"])
            if route.execution_mode == "train":
                producers[(route.family, route.route_id, route.target_dataset)].append(route)
            if route.dependency is not None:
                consumers.append((run, route))
    assert len(consumers) == 675
    for run, route in consumers:
        dependency = route.dependency
        expected = resolve_transfer_route(
            producer_config(run["config"], route), target_dataset=dependency.checkpoint_origin
        )
        matches = [
            actual
            for actual in producers[(dependency.family, dependency.route_id, dependency.checkpoint_origin)]
            if _stage_recipe(actual, dependency.checkpoint_role) == _stage_recipe(expected, dependency.checkpoint_role)
        ]
        assert len(matches) == 1, (run["study"], run["run_id"], len(matches))
