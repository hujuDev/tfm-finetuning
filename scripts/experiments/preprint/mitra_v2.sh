#!/usr/bin/env bash
# Example: run this script with --dataset DATASET --dry-run.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/../../launch_experiment.sh" preprint_mitra_v2 "$@"
