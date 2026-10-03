"""Compute the dataset characteristics used in the association analysis.

Example: ``python -m tfm_finetuning.data.build_base_meta_cache`` reads local exports;
only outer training rows (training plus inner validation) enter the descriptors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfm_finetuning.data.dataset_meta_features import load_local_dataset_base_meta_features, local_dataset_ids
from tfm_finetuning.experiments.configuration import dataset_key

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'results/dataset_characteristics.json'


def build_cache(*, output_path: Path = DEFAULT_OUTPUT, data_repo_root: Path | None = None) -> dict:
    """Compute current descriptors for all exported datasets, without historical fallbacks."""
    root = data_repo_root or REPO_ROOT
    values = {}
    for dataset in sorted(local_dataset_ids(root)):
        characteristics = load_local_dataset_base_meta_features(repo_root=root, dataset_id=dataset, task_family=None)
        if characteristics.get('baseTrainRows') is None:
            raise ValueError(f'Cannot compute training characteristics for {dataset}; check its fold0 export')
        values[dataset_key(dataset)] = characteristics
    if not values:
        raise ValueError('No exported datasets found; run the overview experiments first')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({'datasets': values}, indent=2) + '\n')
    return {'output': str(output_path), 'datasetCount': len(values)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-repo-root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(build_cache(output_path=args.output, data_repo_root=args.data_repo_root)))


if __name__ == '__main__':
    main()
