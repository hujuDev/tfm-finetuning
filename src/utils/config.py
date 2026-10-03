from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping in {path}")
    return dict(payload)


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return dict(payload)


def read_toml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected TOML mapping in {path}")
    return dict(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def write_toml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import tomli_w
    except ImportError as exc:
        raise RuntimeError("tomli_w is required to write TOML files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(dict(payload)), encoding="utf-8")


def set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a dotted-key value on a nested mapping in-place."""
    cursor: dict[str, Any] = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = cursor.get(part)
        if not isinstance(node, dict):
            node = {}
            cursor[part] = node
        cursor = node
    cursor[parts[-1]] = value


def get_nested(target: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get a dotted-key value from a nested mapping, returning default if missing."""
    cursor: Any = target
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def pop_nested(target: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Pop a dotted-key value from a nested mapping, pruning empty mappings."""
    parts = dotted_key.split(".")
    cursor: Any = target
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        parents.append((cursor, part))
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        return default
    value = cursor.pop(parts[-1], default)
    if value is default:
        return default
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
            continue
        break
    return value


def unflatten_dot_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand dot-delimited keys (e.g., data.subsample_size) into nested dict entries."""
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if "." not in key:
            result[key] = value
    for key, value in cfg.items():
        if "." in key:
            set_nested(result, key, value)
    return result


def merge_nested_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-ish merged mapping without mutating inputs."""
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
