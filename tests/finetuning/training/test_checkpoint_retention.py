from __future__ import annotations

from pathlib import Path

from tfm_finetuning.finetuning.training.checkpoint_retention import checkpoint_retention_payload


def test_checkpoint_retention_deletes_weight_files_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_RETAIN_CHECKPOINTS", raising=False)
    monkeypatch.delenv("TABPFN_KEEP_CHECKPOINTS", raising=False)
    (tmp_path / "model.ckpt").write_text("weights", encoding="utf-8")
    (tmp_path / "epoch1.pth").write_text("weights", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    payload = checkpoint_retention_payload(tmp_path)

    assert payload["retained"] is False
    assert payload["deleted_files"] == 2
    assert not (tmp_path / "model.ckpt").exists()
    assert not (tmp_path / "epoch1.pth").exists()
    assert (tmp_path / "summary.json").exists()


def test_checkpoint_retention_env_keeps_weight_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_RETAIN_CHECKPOINTS", "1")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights", encoding="utf-8")

    payload = checkpoint_retention_payload(tmp_path)

    assert payload["retained"] is True
    assert payload["deleted_files"] == 0
    assert checkpoint.exists()


def test_checkpoint_retention_keeps_explicit_checkpoint_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_RETAIN_CHECKPOINTS", raising=False)
    best = tmp_path / "best.ckpt"
    final = tmp_path / "epoch2.ckpt"
    best.write_text("best", encoding="utf-8")
    final.write_text("final", encoding="utf-8")

    payload = checkpoint_retention_payload(tmp_path, keep_paths=[best])

    assert payload["retained"] is False
    assert payload["kept_files"] == 1
    assert payload["deleted_files"] == 1
    assert best.exists()
    assert not final.exists()
