#!/usr/bin/env bash
# Preview by default; execute with --local --run or --cluster CONFIG --submit.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
launcher_python="${TFM_PYTHON:-}"
if [[ -z "$launcher_python" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    launcher_python="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    launcher_python="$repo_root/.venv/bin/python"
  else
    launcher_python=python3
  fi
fi
exec "$launcher_python" -m tfm_finetuning.experiments.execution.launch --repo-root "$repo_root" "$@"
