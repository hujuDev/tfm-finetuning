#!/usr/bin/env bash
# Example: run this script with --dataset DATASET --dry-run. TabPFN-2.5 with tabpfn 9.2 against overview_baseline.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/../../launch_experiment.sh" preprint_tabpfn_drift_check "$@"
