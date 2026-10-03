"""Execute the notebook workflow with mocked training and no downloads."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tfm_finetuning.experiments.configuration import expand_runs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/introductory.ipynb"


def notebook_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


def execute(source, namespace):
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)


@pytest.fixture
def setup_notebook(tmp_path, monkeypatch):
    cells = notebook_cells()
    monkeypatch.chdir(ROOT / "notebooks")
    for key in list(os.environ):
        if key.startswith("WANDB_") or key in {"RESULTS_ROOT", "REPO_ROOT", "PYTHONPATH"}:
            monkeypatch.delenv(key)
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("REPO_ROOT", str(ROOT))
    namespace = {}
    old_path = list(sys.path)
    execute(cells["environment"], namespace)
    assert namespace["root"] == ROOT
    assert Path.cwd() == ROOT / "notebooks"
    assert sys.path == old_path
    assert "PYTHONPATH" not in os.environ
    namespace["display"] = lambda *args: None
    execute(cells["settings"], namespace)
    return cells, namespace


def test_notebook_has_clean_outputs_and_valid_cell_syntax():
    from IPython.core.inputtransformer2 import TransformerManager

    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile(TransformerManager().transform_cell("".join(cell["source"])), str(NOTEBOOK), "exec")


@pytest.mark.parametrize("seeds", [[0], [0, 1, 2]])
def test_example_expands_tabpfn_preprocessing_comparison_for_each_seed(seeds):
    recipe = yaml.safe_load((ROOT / "configs/examples/introductory.yaml").read_text())
    recipe["split"]["seeds"] = seeds
    runs = expand_runs(recipe, repo_root=ROOT)
    assert len(runs) == 2 * len(seeds)
    assert {run["preprocessing"] for run in runs} == {"native", "shared"}
    assert {run["model"] for run in runs} == {"tabpfnv2_5"}
    for run in runs:
        assert run["fold"] == run["repeat"] == run["sample"] == 0
        assert run["config"]["finetune"]["model_id"] == "tabpfn_v2_5"
        assert run["config"]["finetune"]["epochs"] == 1
        assert run["config"]["finetune"]["context_size"] == 1024


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_tracking_and_current_report_comparison(setup_notebook, monkeypatch, mode):
    cells, namespace = setup_notebook
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_PROJECT", "example-project")
    monkeypatch.setenv("WANDB_ENTITY", "example-team")
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-test-credential")
    execute(cells["tracking-settings"], namespace)
    execute(cells["save-config"], namespace)
    execute(cells["preview"], namespace)
    saved = yaml.safe_load(namespace["config_path"].read_text())
    assert saved["wandb"] == {"mode": mode, "project": "example-project", "entity": "example-team"}
    assert "not-a-real-test-credential" not in namespace["config_path"].read_text()
    assert len(namespace["runs"]) == 2
    calls, logins = [], []

    def fake_runner(argv):
        calls.append(argv)
        assert argv[0] == sys.executable
        assert argv[argv.index("--config") + 1] == str(namespace["config_path"])
        assert argv[argv.index("--results-root") + 1] == str(namespace["results"])
        for run in expand_runs(saved, repo_root=ROOT):
            folder = namespace["results"] / run["study"] / run["run_id"]
            folder.mkdir(parents=True)
            (folder / "run.json").write_text(json.dumps({**run, "status": "complete"}))
            (folder / "finetune_report.json").write_text(json.dumps({
                "schema_version": 1,
                "score_metric": "roc_auc",
                "evaluation": {"selected_epoch": 0},
                "zero_shot_metrics": {"val": {"roc_auc": .8}, "test": {"roc_auc": .81}},
                "selected_checkpoint_metrics": {"val": {"roc_auc": .8}, "test": {"roc_auc": .81}},
                "final_checkpoint_metrics": {"val": {"roc_auc": .7}, "test": {"roc_auc": .72}},
            }))
        # Unrelated earlier runs must not enter this notebook's comparison.
        stale = namespace["results"] / saved["study"] / "stale"
        stale.mkdir(parents=True)
        (stale / "finetune_report.json").write_text("{}")

    namespace["run_command"] = fake_runner
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=lambda: logins.append(True) or True))
    execute(cells["run"], namespace)
    assert len(calls) == 1
    assert len(logins) == (1 if mode == "online" else 0)
    execute(cells["results"], namespace)
    table = namespace["comparison"]
    assert len(table) == 4
    assert set(table["preprocessing"]) == {"native", "shared"}
    assert (table["validation-selected"] == table["ICL"]).all()
    assert (table["last checkpoint"] < table["ICL"]).all()


def test_results_require_completed_runs(setup_notebook):
    cells, namespace = setup_notebook
    execute(cells["preview"], namespace)
    with pytest.raises(RuntimeError, match="Run training first"):
        execute(cells["results"], namespace)
    run = namespace["runs"][0]
    folder = namespace["results"] / run["study"] / run["run_id"]
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"status": "failed"}))
    (folder / "finetune_report.json").write_text("{}")
    with pytest.raises(RuntimeError, match="has not completed"):
        execute(cells["results"], namespace)


def test_notebook_streams_cli_output_and_propagates_failure(setup_notebook, capsys):
    _, namespace = setup_notebook
    with pytest.raises(subprocess.CalledProcessError):
        namespace["run_command"]([sys.executable, "-c", "from pathlib import Path; print(Path.cwd()); raise SystemExit(7)"])
    assert capsys.readouterr().out.strip() == str(ROOT)
