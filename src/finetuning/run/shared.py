from __future__ import annotations

import re
from typing import Any


def parse_split_tuple_name(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"fold(\d+)_repeat(\d+)_sample(\d+)", str(value).strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def normalize_split_seed_mode(value: Any) -> str:
    if value is None:
        return "constant"
    cleaned = str(value).strip().lower()
    if cleaned in {"constant", "same", "fixed", "none"}:
        return "constant"
    if cleaned in {"fold", "per_fold", "per-fold"}:
        return "fold"
    if cleaned in {"repeat", "per_repeat", "per-repeat"}:
        return "repeat"
    if cleaned in {"fold_repeat", "fold-repeat", "combination", "combo"}:
        return "fold_repeat"
    raise SystemExit(f"Unknown split.seed_mode={value!r}. Expected one of: constant, fold, repeat, fold_repeat.")


def split_seed_delta(*, fold: int, repeat: int, mode: str, stride: int) -> int:
    if mode == "constant":
        return 0
    if mode == "fold":
        return fold
    if mode == "repeat":
        return repeat
    if mode == "fold_repeat":
        return repeat * stride + fold
    raise SystemExit(f"Unsupported split seed mode: {mode!r}")


def apply_split_seed(config: dict[str, Any]) -> tuple[int, int, int, str | None]:
    seed_raw = config.get("seed")
    try:
        base_seed = int(seed_raw)
    except Exception as exc:
        raise SystemExit(f"Expected integer seed, got {seed_raw!r}.") from exc
    split_cfg = config.get("split")
    if not isinstance(split_cfg, dict):
        return base_seed, base_seed, 0, None
    mode = normalize_split_seed_mode(split_cfg.get("seed_mode"))
    stride_raw = split_cfg.get("seed_stride", 1000)
    try:
        stride = int(stride_raw)
    except Exception as exc:
        raise SystemExit(f"Expected integer split.seed_stride, got {stride_raw!r}.") from exc
    data_cfg = config.get("data")
    split_name = data_cfg.get("split") if isinstance(data_cfg, dict) else None
    split_tuple = parse_split_tuple_name(split_name)
    if split_tuple is None:
        return base_seed, base_seed, 0, str(split_name) if split_name is not None else None
    fold, repeat, _ = split_tuple
    seed_delta = split_seed_delta(fold=fold, repeat=repeat, mode=mode, stride=stride)
    resolved_seed = base_seed + seed_delta
    config["seed"] = resolved_seed
    return base_seed, resolved_seed, seed_delta, str(split_name)
