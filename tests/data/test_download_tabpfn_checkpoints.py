"""Model setup must reproduce the four configured thesis weights without hidden defaults."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from data.download import tabpfn_checkpoints as downloader
from finetuning.models.tabpfn.versions import TABPFN_VERSIONS


def test_checkpoint_files_match_model_config_and_original_v2_weights():
    configured = {
        Path(path).name
        for version in TABPFN_VERSIONS
        for path in (version.classifier_checkpoint, version.regressor_checkpoint)
    }
    assert {spec.filename for spec in downloader.CHECKPOINTS} == configured
    assert len(downloader.CHECKPOINTS) == 4
    classifier = downloader.CHECKPOINTS[0]
    assert classifier.repo_id == "Prior-Labs/TabPFN-v2-clf"
    assert classifier.filename == "tabpfn-v2-classifier.ckpt"
    assert classifier.revision == "a7e9a4696dd0efbd0e64a705645e0d137ca2de72"
    assert classifier.sha256 == "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"
    assert all(len(spec.revision) == 40 and len(spec.sha256) == 64 for spec in downloader.CHECKPOINTS)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    payload = b"verified checkpoint"
    cached = tmp_path / "cached.ckpt"
    cached.write_bytes(payload)
    spec = replace(downloader.CHECKPOINTS[0], sha256=hashlib.sha256(payload).hexdigest())
    fetch = Mock(return_value=str(cached))
    monkeypatch.setattr(downloader, "hf_hub_download", fetch)
    return spec, cached, tmp_path / "models", fetch


def test_download_uses_pinned_filename_and_native_auth_then_skips_verified_file(checkpoint):
    spec, cached, models, fetch = checkpoint
    assert downloader.download_checkpoint(spec, models) == "downloaded"
    assert (models / spec.filename).read_bytes() == cached.read_bytes()
    fetch.assert_called_once_with(
        repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision, force_download=False,
    )
    assert downloader.download_checkpoint(spec, models).startswith("skipped")
    assert fetch.call_count == 1
    assert not list(models.glob("*.part"))


def test_corrupt_existing_checkpoint_requires_explicit_overwrite(checkpoint):
    spec, cached, models, fetch = checkpoint
    models.mkdir()
    destination = models / spec.filename
    destination.write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.download_checkpoint(spec, models)
    fetch.assert_not_called()
    assert downloader.download_checkpoint(spec, models, overwrite=True) == "downloaded"
    assert destination.read_bytes() == cached.read_bytes()
    assert fetch.call_args.kwargs["force_download"] is True


def test_failed_or_corrupt_download_does_not_replace_existing_weights(checkpoint):
    spec, cached, models, fetch = checkpoint
    models.mkdir()
    destination = models / spec.filename
    destination.write_bytes(b"existing")
    fetch.side_effect = RuntimeError("access denied")
    with pytest.raises(RuntimeError, match="access denied"):
        downloader.download_checkpoint(spec, models, overwrite=True)
    assert destination.read_bytes() == b"existing"
    fetch.side_effect = None
    cached.write_bytes(b"unexpected upstream content")
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.download_checkpoint(spec, models, overwrite=True)
    assert destination.read_bytes() == b"existing"
    assert not list(models.glob("*.part"))


def test_cli_filters_versions_and_returns_failure_after_attempting_group(tmp_path, monkeypatch):
    fetch = Mock(side_effect=[RuntimeError("access denied"), "downloaded"])
    monkeypatch.setattr(downloader, "download_checkpoint", fetch)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--models-root", str(tmp_path), "--group", "v2.5"])
    assert error.value.code == 1
    assert fetch.call_count == 2
    assert all(call.args[0].version == "v2.5" for call in fetch.call_args_list)


def test_cli_default_downloads_all_four_weights(tmp_path, monkeypatch):
    fetch = Mock(return_value="downloaded")
    monkeypatch.setattr(downloader, "download_checkpoint", fetch)
    downloader.main(["--models-root", str(tmp_path)])
    assert [call.args[0] for call in fetch.call_args_list] == list(downloader.CHECKPOINTS)
