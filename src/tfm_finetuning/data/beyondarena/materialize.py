"""CLI for the pinned BeyondArena natural-domain materialization.

Usage:
    uv run python -m tfm_finetuning.data.beyondarena.materialize \
      --download-dir /path/to/beyondarena \
      --output-root data/beyondarena
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfm_finetuning.data.beyondarena.natural_domains import (
    download_selected_snapshot,
    materialize_natural_domain_datasets,
)


def main() -> None:
    """Download when requested and write all logical dataset exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/beyondarena"))
    args = parser.parse_args()
    if (args.snapshot_root is None) == (args.download_dir is None):
        parser.error("provide exactly one of --snapshot-root or --download-dir")
    snapshot_root = (
        args.snapshot_root
        if args.snapshot_root is not None
        else download_selected_snapshot(local_dir=args.download_dir)
    )
    manifest = materialize_natural_domain_datasets(
        snapshot_root=snapshot_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
