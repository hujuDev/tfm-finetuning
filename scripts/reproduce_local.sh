#!/usr/bin/env bash
# Full sequential reproduction; requires the installed project environment and a suitable GPU.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# Put all producers before their checkpoint consumers.
studies=(
  unified_foundation_baseline foundation_regular_hpo_representative
  mitra_original_pipeline mitra_adapted_pipeline
  distribution_shift_subsample distribution_shift_label_noise distribution_shift_spurious_correlation
  distribution_shift_feature_order distribution_shift_numeric_scale
  distribution_shift_feature_order_alignment distribution_shift_numeric_scale_alignment
  distribution_shift_class_prior_majority_downsampling distribution_shift_class_prior_context_only_downsampling
  distribution_shift_contextual_scale_calibration distribution_shift_contextual_scale_feature_count_calibration
  distribution_shift_contextual_scale_coil_screen distribution_shift_heavy_tail_calibration
  distribution_shift_contextual_scale distribution_shift_heavy_tail distribution_shift_heavy_tail_diamonds
  dataset_transfer dataset_transfer_target_controls dataset_transfer_houses_source_checkpoint
  dataset_transfer_data_budget dataset_transfer_balanced_budget_classification
  dataset_transfer_balanced_budget_regression dataset_transfer_source_only_eval_classification
  dataset_transfer_source_only_eval_regression
  dataset_transfer_natural_domain_training dataset_transfer_natural_domain
  dataset_transfer_learned_retention_train dataset_transfer_learned_retention_eval
)
for study in "${studies[@]}"; do
  bash scripts/launch_experiment.sh "$study" --local --run "$@"
done
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"${TFM_PYTHON:-python3}" -m data.build_base_meta_cache
"${TFM_PYTHON:-python3}" -m plots.final_figures
