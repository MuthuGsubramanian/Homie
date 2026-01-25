from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: Optional[int]
    error: Optional[str]


def _apply_limits(proc: subprocess.Popen, cpu_percent: int, mem_mb: int) -> None:
    """Best-effort resource limiting (POSIX only)."""
    if os.name != "posix":
        return
    try:
        import psutil  # type: ignore

        p = psutil.Process(proc.pid)
        p.nice(psutil.IDLE_PRIORITY_CLASS if sys.platform.startswith("win") else 19)
        if mem_mb:
            p.rlimit(psutil.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    except Exception:  # noqa: BLE001
        logging.debug("Could not set resource limits", exc_info=True)


def run_command(
    command: str,
    workdir: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: int = 120,
    cpu_percent: int = 50,
    mem_mb: int = 512,
    dry_run: bool = False,
) -> ExecutionResult:
    if dry_run:
        return ExecutionResult(stdout="", stderr="", exit_code=None, error=None)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=workdir,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _apply_limits(proc, cpu_percent, mem_mb)

        timer = threading.Timer(timeout_sec, proc.kill)
        try:
            timer.start()
            stdout, stderr = proc.communicate()
        finally:
            timer.cancel()
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode, error=None)
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(stdout="", stderr="", exit_code=None, error=str(exc))


__all__ = ["run_command", "ExecutionResult"]
