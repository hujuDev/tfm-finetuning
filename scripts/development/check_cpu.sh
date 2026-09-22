#!/usr/bin/env bash
# Run the same CPU fixture checks as CI: uv run bash scripts/development/check_cpu.sh.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg
exec "${TFM_PYTHON:-python3}" -m pytest -q \
  tests/test_config_io.py tests/test_paths.py tests/test_dataset_preparation.py \
  tests/test_reproduction_runner.py tests/test_run_experiment_dry_run.py \
  tests/test_local_launch.py tests/test_slurm_launch.py \
  tests/test_introductory_notebook.py tests/test_final_reproduction.py tests/test_final_tables.py \
  tests/test_fresh_analysis.py tests/test_remaining_artifacts.py \
  tests/test_section1_pattern_analysis.py tests/test_section1_hpo_analysis.py \
  tests/test_section1_mitra_pipeline_analysis.py tests/test_section2_transformation_data.py \
  tests/test_section2_transformation_figures.py tests/test_section2_transformation_overviews.py \
  tests/test_section3_transfer_analysis.py tests/test_section3_natural_domain_analysis.py \
  tests/test_distribution_shift_calibration.py tests/test_distribution_shift_final_configs.py \
  tests/test_build_section1_base_meta_cache.py tests/test_metrics.py \
  "$@"
