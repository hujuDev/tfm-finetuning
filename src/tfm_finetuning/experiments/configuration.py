"""Expand the fixed research recipes into independent runs.

Example: ``expand_runs(read_yaml_mapping(path), repo_root=root)`` needs no GPU or
network connection. The same expansion drives local execution and SLURM arrays.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from tfm_finetuning.utils.config import read_yaml_mapping, set_nested


def dataset_key(value: str) -> str:
    """Normalize spelling differences in dataset labels, e.g. ``credit-g``."""
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def recipe_groups(recipe: dict) -> list[dict]:
    """Combine shared recipe settings with each explicit resource group."""
    common = {k: v for k, v in recipe.items() if k != 'groups'}
    groups = recipe.get('groups', [{}])
    if not groups:
        raise ValueError('An experiment needs at least one group')
    result = []
    for group in groups:
        merged = copy.deepcopy(common)
        for key, value in group.items():
            if key in {'settings', 'split'}:
                merged[key] = {**merged.get(key, {}), **copy.deepcopy(value)}
            else:
                merged[key] = copy.deepcopy(value)
        merged.setdefault('name', 'default')
        result.append(merged)
    return result


def _values(value: Any) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f'Expected a nonempty list, got {value!r}')
    return value


def _split_grid(split: dict, dataset: str, facts: dict) -> list[tuple[int, int, int, int]]:
    protocol = split.get('protocol', 'fixed_split_repeated_seeds')
    if protocol == 'cv_auto':
        rows = facts.get(dataset_key(dataset), {}).get('n_rows')
        if rows is None:
            raise ValueError(f'Missing n_rows for {dataset} in configs/datasets.yaml; specify folds explicitly')
        protocol = 'cross_validation_repeated_seeds' if rows < 2500 else 'fixed_split_repeated_seeds'
    if protocol not in {'fixed_split_repeated_seeds', 'cross_validation_repeated_seeds'}:
        raise ValueError(f'Unknown split protocol: {protocol}')
    folds = split.get('folds', [0, 1, 2] if protocol == 'cross_validation_repeated_seeds' else [0])
    axes = [folds, split.get('repeats', [0]), split.get('samples', [0]), split.get('seeds', [0, 1, 2])]
    for values in axes:
        if any(type(v) is not int or v < 0 for v in _values(values)) or len(values) != len(set(values)):
            raise ValueError(f'Split values must be distinct nonnegative integers: {values}')
    if protocol == 'fixed_split_repeated_seeds' and any(len(x) != 1 for x in axes[:3]):
        raise ValueError('A fixed split needs exactly one fold, repeat, and sample')
    return list(itertools.product(*axes))


def _parameter_values(group: dict, dataset: str, facts: dict) -> dict[str, list]:
    values = {k: list(_values(v['values'])) for k, v in group.get('parameters', {}).items()}
    if group.get('deduplicate_context_sizes') and 'context_size' in values:
        outer_rows = facts.get(dataset_key(dataset), {}).get('outer_train_rows')
        if outer_rows is None:
            raise ValueError(f'Missing outer_train_rows for context-size deduplication: {dataset}')
        fraction = group.get('settings', {}).get('foundation_model.validation_split_ratio', 0.1)
        effective = int(outer_rows * (1 - float(fraction)))
        sizes = sorted(set(int(v) for v in values['context_size']))
        saturating = next((v for v in sizes if v >= effective), None)
        values['context_size'] = [v for v in sizes if v < effective or v == saturating]
    return values


def expand_runs(
    recipe: dict, *, repo_root: Path, datasets: list[str] | None = None,
    models: list[str] | None = None, preprocessing: list[str] | None = None,
    group_name: str | None = None, smoke: bool = False,
) -> list[dict]:
    """Return complete run configurations; reject unknown selections and duplicate runs."""
    selection = recipe.get("selection", {})
    datasets = datasets or selection.get("datasets")
    models = models or selection.get("models")
    preprocessing = preprocessing or selection.get("preprocessing")
    study = str(recipe.get('study', ''))
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', study):
        raise ValueError('study must be a simple directory name')
    facts = {dataset_key(k): v for k, v in read_yaml_mapping(repo_root / 'configs/datasets.yaml').items()}
    groups = recipe_groups(recipe)
    selected = {dataset_key(d) for d in datasets or []}
    known_datasets = {dataset_key(d) for g in groups for d in g['datasets']}
    known_models = {m['name'] for g in groups for m in g['models']}
    if selected - known_datasets:
        raise ValueError(f'Unknown datasets: {sorted(selected - known_datasets)}')
    if set(models or []) - known_models:
        raise ValueError(f'Unknown models: {sorted(set(models) - known_models)}')
    if set(preprocessing or []) - {'native', 'shared'}:
        raise ValueError('Preprocessing must be native or shared')
    if group_name and group_name not in {g['name'] for g in groups}:
        raise ValueError(f'Unknown group: {group_name}')
    runs, seen = [], set()
    for group in groups:
        if group_name and group['name'] != group_name:
            continue
        axes = group.get('axes', {})
        for dataset in group['datasets']:
            if selected and dataset_key(dataset) not in selected:
                continue
            parameters = _parameter_values(group, dataset, facts)
            splits = _split_grid(group.get('split', {}), dataset, facts)
            if smoke:
                splits = [(0, 0, 0, 0)]
            for model in group['models']:
                if models and model['name'] not in models:
                    continue
                base = tomllib.loads((repo_root / model['config']).read_text())
                for axis_entries in itertools.product(*(_values(v) for v in axes.values())):
                    for param_values in itertools.product(*parameters.values()):
                        varied = {k: e['value'] for k, e in zip(axes, axis_entries, strict=True)}
                        varied.update(zip(parameters, param_values, strict=True))
                        config = copy.deepcopy(base)
                        overrides = dict(model.get('settings', {}))
                        for entry in axis_entries:
                            overrides.update(entry['settings'])
                        overrides.update(group.get('settings', {}))
                        for key, value in overrides.items():
                            set_nested(config, key, value)
                        for key, value in zip(parameters, param_values, strict=True):
                            paths = group['parameters'][key]['path']
                            for path in paths if isinstance(paths, list) else [paths]:
                                set_nested(config, path, value)
                        protocol = config['foundation_model'].get('preprocessing_protocol', 'native')
                        if preprocessing and protocol not in preprocessing:
                            continue
                        for fold, repeat, sample, seed in splits:
                            resolved = copy.deepcopy(config)
                            resolved['seed'] = seed
                            resolved['data'].update(seed=seed, path=f':data/tabarena/{dataset_key(dataset)}',
                                                    split=f'fold{fold}_repeat{repeat}_sample{sample}')
                            resolved['split'] = {'policy': 'tabarena', 'seed_mode': 'constant'}
                            resolved['foundation_model']['validation_split_seed'] = group.get('split', {}).get('validation_seed', 0)
                            resolved['experiment'] = {'study': study, 'key_param': {**varied, 'model': model['name']}}
                            # W&B records the same run metadata as local reports; environment variables may override it.
                            resolved['wandb'] = {'project': 'tfm-finetuning', **group.get('wandb', {})}
                            if smoke and not resolved.get('dataset_transfer', {}).get('enabled'):
                                resolved['foundation_model']['epochs'] = 2
                                resolved['foundation_model']['test_eval_interval_epochs'] = 1
                            identity = {'dataset': dataset_key(dataset), 'model': model['name'], 'config': resolved}
                            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
                            run_id = f'{dataset_key(dataset)}-{model["name"]}-{digest}'
                            if run_id in seen:
                                raise ValueError(f'Duplicate scientific run in recipe: {run_id}')
                            seen.add(run_id)
                            runs.append({'schema_version': 1, 'study': study, 'run_id': run_id,
                                         'dataset': dataset, 'model': model['name'], 'preprocessing': protocol,
                                         'fold': fold, 'repeat': repeat, 'sample': sample, 'seed': seed,
                                         'parameters': varied, 'config': resolved, 'group': group['name']})
    if not runs:
        raise ValueError('No runs match the selected experiment settings')
    return runs
