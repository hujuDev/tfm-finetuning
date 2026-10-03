"""Discover the standalone experiment recipes, e.g. ``load_catalog(root)``."""
from pathlib import Path

from tfm_finetuning.utils.config import read_yaml_mapping


def load_catalog(root: Path) -> dict:
    """List checked-in experiment files without a separate version catalog."""
    return {p.stem: read_yaml_mapping(p) for p in sorted((root / 'configs/experiments').glob('*.yaml'))}
