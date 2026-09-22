"""Resolve experiment runs and their resource requirements before submission.

Example: ``build_plan(root, 'distribution_shift_feature_order', None, output)``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from experiments.configuration import expand_runs, recipe_groups
from experiments.execution.catalog import load_catalog
from experiments.execution.resources import resolve_resources
from utils.config_io import read_yaml_mapping
from utils.paths import get_paths


def build_plan(root: Path, name: str, cluster: dict | None, output: Path, *, datasets=None,
               models=None, preprocessing=None, scope='full', force=False, results_root=None,
               config_path=None) -> dict:
    """Keep each scientific run intact; Submitit receives one run per job."""
    if scope not in {'full', 'smoke'}:
        raise ValueError(f'Unknown scope: {scope}')
    root, output = root.resolve(), output.resolve()
    if config_path is None:
        catalog = load_catalog(root)
        if name not in catalog:
            raise ValueError(f'Unknown experiment: {name}; use --list')
        recipe = catalog[name]
        source = root / 'configs/experiments' / f'{name}.yaml'
    else:
        source = (root / config_path).resolve()
        recipe = read_yaml_mapping(source)
        name = source.stem
    runs = expand_runs(recipe, repo_root=root, datasets=datasets, models=models,
                       preprocessing=preprocessing, smoke=scope == 'smoke')
    resources = read_yaml_mapping(root / 'configs/resources.yaml')
    jobs = []
    for group in recipe_groups(recipe):
        selected = [r for r in runs if r['group'] == group['name']]
        if not selected:
            continue
        jobs.append({'id': group['name'], 'datasets': list(dict.fromkeys(r['dataset'] for r in selected)),
                     'result_datasets': group.get('targets', list(dict.fromkeys(r['dataset'] for r in selected))),
                     'models': list(dict.fromkeys(r['model'] for r in selected)),
                     'resource_rule': f'{source}:{group["name"]}',
                     'reason': group.get('reason', 'Thesis dataset resource allocation'),
                     'resources': resolve_resources(group['resource'], resources, cluster),
                     'runs': selected})
    python = [sys.executable] if cluster is None else cluster.get('python', [sys.executable])
    if not isinstance(python, list) or not python or any(not isinstance(v, str) or not v for v in python):
        raise ValueError('cluster.python must be a nonempty list of command arguments')
    results = (root / (results_root or get_paths(repo_root=root).results_root)).resolve()
    if scope == 'smoke':
        results /= 'smoke'
    return {'experiment': name, 'config_path': str(source), 'experiment_config': recipe,
            'repo_root': str(root), 'output': str(output), 'results_root': str(results),
            'note': recipe.get('note'), 'scope': scope, 'force': force,
            'backend': 'local' if cluster is None else 'slurm', 'python': python, 'jobs': jobs}
