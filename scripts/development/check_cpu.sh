#!/usr/bin/env bash
# Run the same CPU fixture checks as CI: uv run bash scripts/development/check_cpu.sh.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg
exec "${TFM_PYTHON:-python3}" -m pytest -q \
  tests/utils/test_config.py tests/utils/test_paths.py tests/data/test_dataset_preparation.py \
  tests/experiments/test_reproduction_runner.py tests/experiments/test_run_experiment_dry_run.py \
  tests/experiments/execution/test_local_launch.py tests/experiments/execution/test_slurm_launch.py \
  tests/test_introductory_notebook.py tests/plots/test_final_reproduction.py tests/plots/test_final_tables.py \
  tests/plots/test_fresh_analysis.py tests/plots/test_remaining_artifacts.py \
  tests/plots/overview/test_pattern_analysis.py tests/plots/overview/test_hpo_analysis.py \
  tests/plots/overview/test_mitra_pipeline_analysis.py tests/plots/transformations/test_transformation_data.py \
  tests/plots/transformations/test_transformation_figures.py tests/plots/transformations/test_transformation_overviews.py \
  tests/plots/transfer/test_transfer_analysis.py tests/plots/transfer/test_natural_domain_analysis.py \
  tests/experiments/transformations/test_calibration.py tests/experiments/transformations/test_final_configs.py \
  tests/data/test_build_base_meta_cache.py tests/plots/test_metrics.py \
  "$@"
