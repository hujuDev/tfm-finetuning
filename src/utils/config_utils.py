from __future__ import annotations

from typing import Any


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
