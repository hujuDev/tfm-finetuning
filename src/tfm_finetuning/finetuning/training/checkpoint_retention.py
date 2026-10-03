from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

CHECKPOINT_WEIGHT_SUFFIXES = {".ckpt", ".pth", ".pt", ".bin"}


def retain_checkpoints_enabled() -> bool:
    """Return whether fine-tune checkpoint files should remain on disk.

    Example:
        >>> isinstance(retain_checkpoints_enabled(), bool)
        True
    """
    value = os.environ.get("TABPFN_RETAIN_CHECKPOINTS")
    if value is None:
        value = os.environ.get("TABPFN_KEEP_CHECKPOINTS")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def checkpoint_retention_payload(root: Path, *, keep_paths: list[Path] | None = None) -> dict[str, Any]:
    """Delete checkpoint weight files unless retention is explicitly enabled.

    Example:
        >>> payload = checkpoint_retention_payload(Path('/tmp/missing-retention-root'))
        >>> payload["retained"]
        False
    """
    retained = retain_checkpoints_enabled()
    payload: dict[str, Any] = {
        "retained": retained,
        "env_var": "TABPFN_RETAIN_CHECKPOINTS",
        "deleted_files": 0,
        "deleted_bytes": 0,
        "kept_files": 0,
    }
    if retained or not root.exists():
        return payload
    kept = {path.resolve() for path in keep_paths or [] if path is not None}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_WEIGHT_SUFFIXES:
            continue
        if path.resolve() in kept:
            payload["kept_files"] += 1
            continue
        size = path.stat().st_size
        path.unlink()
        payload["deleted_files"] += 1
        payload["deleted_bytes"] += size
    return payload
