"""Keep every submitted artifact reachable from the public reproduction catalogue."""

from __future__ import annotations

import importlib
import inspect
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_final_catalogue_covers_all_submitted_assets_with_callable_producers():
    catalogue = tomllib.loads((ROOT / "configs/plots/final_figures.toml").read_text())
    assets = [asset for group in catalogue["groups"].values() for asset in group["assets"]]
    assert len(assets) == len(set(assets)) == 35
    assert sum(asset.endswith(".pdf") for asset in assets) == 31
    assert sum(asset.endswith(".tex") for asset in assets) == 4
    for group in catalogue["groups"].values():
        config = ROOT / group["config"]
        assert config.is_file()
        assert tomllib.loads(config.read_text())["input"]
        module, function = group["entrypoint"].split(":")
        producer = getattr(importlib.import_module(module), function)
        inspect.signature(producer).bind(config_path=config)
