"""Model setup must reproduce the configured TabPFN weights without hidden defaults."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from tfm_finetuning.data.download import tabpfn_checkpoints as downloader
from tfm_finetuning.finetuning.models.tabpfn.versions import TABPFN_VERSIONS


def test_checkpoint_files_match_model_config_and_original_v2_weights():
    configured = {
        Path(path).name
        for version in TABPFN_VERSIONS
        for path in (version.classifier_checkpoint, version.regressor_checkpoint)
    }
    assert {spec.filename for spec in downloader.CHECKPOINTS} == configured
    assert len(downloader.CHECKPOINTS) == 8
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
        repo_id=spec.repo_id,
        filename=spec.filename,
        revision=spec.revision,
        force_download=False,
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


def test_cli_default_downloads_all_weights(tmp_path, monkeypatch):
    fetch = Mock(return_value="downloaded")
    monkeypatch.setattr(downloader, "download_checkpoint", fetch)
    downloader.main(["--models-root", str(tmp_path)])
    assert [call.args[0] for call in fetch.call_args_list] == list(downloader.CHECKPOINTS)


def test_v3_checkpoints_are_pinned_to_the_published_hugging_face_files():
    by_name = {spec.filename: spec for spec in downloader.CHECKPOINTS}
    v3_classifier = by_name["tabpfn-v3-classifier-v3_default.ckpt"]
    assert (v3_classifier.version, v3_classifier.repo_id) == ("v3", "Prior-Labs/tabpfn_3")
    assert v3_classifier.revision == "24a16a89d245878b846555110985634aa2e656d7"
    assert by_name["tabpfn-v3-regressor-v3_default.ckpt"].repo_id == "Prior-Labs/tabpfn_3"
    for filename in ("tabpfn-v3.5-20260909.safetensors", "tabpfn-v3.5-fast-20260909.safetensors"):
        assert (by_name[filename].version, by_name[filename].repo_id) == ("v3.5", "Prior-Labs/tabpfn_3_5")
        assert by_name[filename].revision == "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"


@pytest.mark.parametrize(("group", "expected"), [("v3", 2), ("v3.5", 2), ("v2", 2), ("v2.5", 2)])
def test_cli_group_selects_only_that_version(tmp_path, monkeypatch, group, expected):
    fetch = Mock(return_value="downloaded")
    monkeypatch.setattr(downloader, "download_checkpoint", fetch)
    downloader.main(["--models-root", str(tmp_path), "--group", group])
    assert fetch.call_count == expected
    assert all(call.args[0].version == group for call in fetch.call_args_list)


def test_ensure_checkpoint_fetches_only_missing_pinned_files(tmp_path, monkeypatch):
    fetch = Mock(return_value="downloaded")
    monkeypatch.setattr(downloader, "download_checkpoint", fetch)
    spec = downloader.CHECKPOINTS[2]
    downloader.ensure_checkpoint(tmp_path / spec.filename)
    fetch.assert_called_once_with(spec, tmp_path)
    (tmp_path / spec.filename).write_bytes(b"present")
    downloader.ensure_checkpoint(tmp_path / spec.filename)
    downloader.ensure_checkpoint(tmp_path / "my-own-weights.ckpt")
    assert fetch.call_count == 1
