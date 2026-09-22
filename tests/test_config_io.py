from __future__ import annotations

from pathlib import Path

from src.utils.config_io import read_json_mapping, read_yaml_mapping, write_json


def test_read_yaml_mapping_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_yaml_mapping(tmp_path / "missing.yaml") == {}


def test_read_yaml_mapping_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    try:
        read_yaml_mapping(path)
    except ValueError as exc:
        assert "Expected YAML mapping" in str(exc)
    else:
        raise AssertionError("read_yaml_mapping should reject non-mapping payloads")


def test_read_yaml_mapping_rejects_empty_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    try:
        read_yaml_mapping(path)
    except ValueError as exc:
        assert "Expected YAML mapping" in str(exc)
    else:
        raise AssertionError("read_yaml_mapping should reject empty existing files")


def test_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    write_json(path, {"b": 2, "a": 1})

    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert read_json_mapping(path) == {"a": 1, "b": 2}
