"""Dataset descriptors come from available exports, without a historical catalog."""

import json

import pytest

from data import build_base_meta_cache as cache_builder


def test_build_cache_includes_every_local_export(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_builder, "local_dataset_ids", lambda root: ("credit-g", "satimage"))
    calls = []

    def descriptors(*, repo_root, dataset_id, task_family):
        assert repo_root == tmp_path
        assert task_family is None
        calls.append(dataset_id)
        return {"baseTrainRows": 10, "nFeaturesTotal": 4, "trainingTableCells": 40}

    monkeypatch.setattr(cache_builder, "load_local_dataset_base_meta_features", descriptors)
    output = tmp_path / "cache.json"
    result = cache_builder.build_cache(data_repo_root=tmp_path, output_path=output)
    cache = json.loads(output.read_text())["datasets"]
    assert result["datasetCount"] == 2
    assert set(cache) == {"credit_g", "satimage"}
    assert calls == ["credit-g", "satimage"]
    assert cache["satimage"]["trainingTableCells"] == 40


@pytest.mark.parametrize("datasets", [(), ("incomplete",)])
def test_missing_training_data_does_not_create_a_cache(tmp_path, monkeypatch, datasets):
    monkeypatch.setattr(cache_builder, "local_dataset_ids", lambda root: datasets)
    monkeypatch.setattr(cache_builder, "load_local_dataset_base_meta_features", lambda **kwargs: {})
    output = tmp_path / "cache.json"
    with pytest.raises(ValueError, match="No exported datasets|Cannot compute training characteristics"):
        cache_builder.build_cache(data_repo_root=tmp_path, output_path=output)
    assert not output.exists()
