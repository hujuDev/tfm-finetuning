"""Observable logging and on-disk guarantees for fresh reproduction runs."""

import json
import sys
from types import SimpleNamespace

import pytest

from finetuning.reporting.report import _init_wandb_run, finish_run
from finetuning.reporting.wandb_logging import flatten_report_metrics_for_wandb
from finetuning.run.runtime import prepare_output


def test_report_metrics_keep_selected_and_final_separate():
    flat = flatten_report_metrics_for_wandb(
        {
            "zero_shot_metrics": {"test": {"score": -3.0, "rmse": 3.0}},
            "selected_checkpoint_metrics": {"test": {"score": -1.0, "rmse": 1.0}},
            "final_checkpoint_metrics": {"test": {"score": -2.0, "rmse": 2.0, "mae": float("nan")}},
        }
    )
    assert flat["selected/test/score"] == -1
    assert flat["final/test/score"] == -2
    assert flat["core/final/test/rmse"] == 2
    assert flat["core/final/test/mse"] == 4
    assert flat["delta/test/score"] == 1
    assert flat["delta/core/test/rmse"] == -1
    assert "core/final/test/mae" not in flat


def test_class_metrics_and_paired_test_views_survive_logging():
    metrics = {"test": {"score": 0.8, "accuracy": 0.7, "classification_report": {"0": {"recall": 0.4, "support": 10}}}}
    flat = flatten_report_metrics_for_wandb(
        {
            "selected_checkpoint_metrics": metrics,
            "distribution_shift_test_views": {"selected_checkpoint_metrics": {"raw": metrics}},
        }
    )
    assert flat["details/class_report/selected/test/0/recall"] == 0.4
    assert flat["details/distribution_shift_test_views/selected/raw/test/score"] == 0.8


def test_explicit_output_ignores_legacy_artifact_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TABPFN_USE_ARTIFACTS", "1")
    monkeypatch.setenv("TABPFN_ARTIFACTS_ROOT", str(tmp_path / "cache"))
    output = prepare_output({"seed": 7}, tmp_path / "run", force=False)
    assert output == tmp_path / "run"
    assert (output / "resolved_config.toml").read_text() == "seed = 7\n"
    assert not (tmp_path / "cache").exists()
    with pytest.raises(FileExistsError):
        prepare_output({"seed": 8}, output, force=False)
    (output / "checkpoint.ckpt").touch()
    prepare_output({"seed": 8}, output, force=True)
    assert not (output / "checkpoint.ckpt").exists()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="Refusing"):
        prepare_output({}, unrelated, force=True)
    assert (unrelated / "notes.txt").read_text() == "keep"


def test_wandb_environment_and_resolved_seed_are_preserved(tmp_path, monkeypatch):
    calls = {}
    run = SimpleNamespace(define_metric=lambda *_a, **_kw: None)

    def initialize(**kwargs):
        calls.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "environment-project")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-log"))
    assert _init_wandb_run({"seed": 7, "wandb": {"project": "repro", "mode": "online"}}, output=tmp_path) is run
    assert calls["mode"] == "offline"
    assert calls["project"] == "environment-project"
    assert calls["config"]["seed"] == 7
    assert calls["dir"] == str(tmp_path / "wandb-log")


def test_completed_report_is_saved_even_when_tracking_fails(tmp_path):
    class BrokenRun:
        summary = {}
        finished = False

        def log(self, payload):
            raise RuntimeError("tracking unavailable")

        def finish(self):
            self.finished = True

    tracker = BrokenRun()
    setup = SimpleNamespace(config={"seed": 7}, output=tmp_path, wandb_run=tracker)
    with pytest.raises(RuntimeError, match="tracking unavailable"):
        finish_run(setup, {"train_history": [{"train/epoch": 1, "train/loss": 2.0}]})
    result = json.loads((tmp_path / "finetune_report.json").read_text())
    assert result["schema_version"] == 1
    assert result["config"] == {"seed": 7}
    assert tracker.finished
    assert sorted(p.name for p in tmp_path.iterdir()) == ["finetune_report.json"]


def test_experiment_runner_and_training_setup_share_output_without_replacing_lock(tmp_path):
    from experiments.run_experiment import execute_run
    from finetuning.run.setup import initialize_foundation_run

    spec = SimpleNamespace(ensure_source_available=lambda: None, adapter_cls=lambda _: SimpleNamespace())
    run = {
        "schema_version": 1,
        "study": "example",
        "run_id": "example-seed7",
        "dataset": "example",
        "config": {"seed": 7, "foundation_model": {"model_id": "fake"}, "wandb": {"mode": "disabled"}},
    }
    observed_inodes = []

    def train(config, output, *, force):
        inode = (output / ".lock").stat().st_ino
        setup = initialize_foundation_run(config, output, force=force, model_spec_resolver=lambda *a, **kw: spec)
        assert (output / ".lock").stat().st_ino == inode
        assert json.loads((output / "run.json").read_text())["status"] == "running"
        observed_inodes.append(inode)
        return finish_run(setup, {})

    output = execute_run(run, tmp_path, repo_root=tmp_path, trainer=train)
    (output / "old.ckpt").touch()
    execute_run(run, tmp_path, repo_root=tmp_path, force=True, trainer=train)
    assert len(observed_inodes) == 2 and observed_inodes[0] == observed_inodes[1]
    assert not (output / "old.ckpt").exists()
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    assert (output / "finetune_report.json").exists()


def test_retry_after_failure_before_training_snapshot(tmp_path):
    output = tmp_path / "failed"
    output.mkdir()
    state = {"schema_version": 1, "study": "example", "run_id": "one", "config": {}, "status": "failed"}
    (output / "run.json").write_text(json.dumps(state))
    (output / ".lock").touch()
    (output / "partial.tmp").touch()
    prepare_output({"seed": 1}, output, force=True)
    assert (output / "run.json").exists() and (output / ".lock").exists()
    assert not (output / "partial.tmp").exists()
