#!/bin/bash
#SBATCH --job-name=build-tfm-image
#SBATCH --partition=cpu-5h
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=scripts/apptainer/img/build-%j.out
# Builds scripts/apptainer/img/tfm_finetuning.sif. Submit from the repository root: sbatch scripts/apptainer/build.sbatch
# Set FORCE=1 to rebuild; FAKEROOT=1 if the cluster requires --fakeroot.
set -euo pipefail
mkdir -p scripts/apptainer/img
if command -v module &>/dev/null; then module load apptainer &>/dev/null || true; fi
args=()
[[ -n "${FAKEROOT:-}" ]] && args+=(--fakeroot)
[[ -n "${FORCE:-}" ]] && args+=(--force)
apptainer build "${args[@]}" scripts/apptainer/img/tfm_finetuning.sif scripts/apptainer/tfm_finetuning.def
