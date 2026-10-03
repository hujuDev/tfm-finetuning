"""Notebook support for the same experiment CLI used by local and cluster runs."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence

from tfm_finetuning.utils.paths import get_repo_root


def run_command(argv: Sequence[str]) -> None:
    """Stream a CLI command from the checkout and stop it when interrupted.

    For example, ``run_command([sys.executable, '-m',
    'tfm_finetuning.experiments.run_experiment', '--config', str(config_path), '--dry-run'])``
    previews an experiment using the notebook's Python environment.
    """
    with subprocess.Popen(
        argv,
        cwd=get_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, argv)
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    # Also stop child training processes on Unix.
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
            raise
