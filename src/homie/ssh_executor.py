from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

import paramiko

from homie.config import HomieConfig, cfg_get
from homie.utils import ensure_ip_literal


@dataclass
class SSHResult:
    target: str
    stdout: str
    stderr: str
    exit_status: Optional[int]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.exit_status is None or self.exit_status == 0)


def _get_target_config(cfg: HomieConfig, target: str) -> Dict:
    targets = cfg_get(cfg, "ssh", "targets", default={}) or {}
    if target not in targets:
        raise ValueError(f"Unknown target '{target}'. Known targets: {list(targets.keys())}")
    return targets[target]


def _extract_ip(tcfg: Dict, target_name: str) -> str:
    """Return IP for the target, enforcing IP-only requirement."""
    host = tcfg.get("ip") or tcfg.get("host") or target_name
    return ensure_ip_literal(host)


def run_ssh_command(cfg: HomieConfig, target: str, command: str) -> SSHResult:
    tcfg = _get_target_config(cfg, target)
    try:
        host = _extract_ip(tcfg, target)
    except ValueError as exc:
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))

    user = tcfg.get("user", cfg_get(cfg, "ssh", "default_user"))
    port = int(tcfg.get("port", 22))
    timeout = int(cfg_get(cfg, "ssh", "connect_timeout_sec", default=10))
    key_filename = tcfg.get("key_filename")
    password = tcfg.get("password")
    method = (tcfg.get("method") or cfg_get(cfg, "ssh", "method", default="openssh_ip")).lower()

    # tailscale ssh method
    if method in {"tailscale_ssh_ip", "tailscale_ssh"}:
        return _run_tailscale_ssh(target, user, host, command, timeout)

    # openssh method
    if method in {"openssh_ip", "openssh"}:
        return _run_openssh(target, user, host, port, key_filename, command, timeout)

    # paramiko with fallback to openssh on resolution failure
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            look_for_keys=True,
            allow_agent=True,
            key_filename=key_filename,
            password=password,
        )
        stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="ignore")
        err = stderr.read().decode(errors="ignore")
        return SSHResult(target=target, stdout=out, stderr=err, exit_status=exit_status)
    except Exception as exc:
        logging.error("SSH command failed on %s: %s", target, exc)
        # fallback to openssh using same params if available
        return _run_openssh(target, user, host, port, key_filename, command, timeout, error_hint=str(exc))
    finally:
        try:
            client.close()
        except Exception:
            pass


def copy_file(cfg: HomieConfig, target: str, local_path: str, remote_path: str) -> SSHResult:
    tcfg = _get_target_config(cfg, target)
    try:
        host = _extract_ip(tcfg, target)
    except ValueError as exc:
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))
    user = tcfg.get("user", cfg_get(cfg, "ssh", "default_user"))
    port = int(tcfg.get("port", 22))
    timeout = int(cfg_get(cfg, "ssh", "connect_timeout_sec", default=10))
    key_filename = tcfg.get("key_filename")
    password = tcfg.get("password")
    method = (tcfg.get("method") or cfg_get(cfg, "ssh", "method", default="openssh_ip")).lower()

    if method in {"openssh_ip", "openssh"}:
        return _run_scp(target, user, host, port, key_filename, local_path, remote_path, timeout)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            look_for_keys=True,
            allow_agent=True,
            key_filename=key_filename,
            password=password,
        )
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        return SSHResult(target=target, stdout="file copied", stderr="", exit_status=0)
    except Exception as exc:
        logging.error("SSH copy failed on %s: %s", target, exc)
        return SSHResult(
            target=target,
            stdout="",
            stderr="",
            exit_status=None,
            error=str(exc),
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


def _run_openssh(
    target: str,
    user: str,
    host: str,
    port: int,
    key_filename: Optional[str],
    command: str,
    timeout: int,
    error_hint: Optional[str] = None,
) -> SSHResult:
    try:
        host = ensure_ip_literal(host)
    except ValueError as exc:
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))

    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "CheckHostIP=yes",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "PasswordAuthentication=no",
    ]
    if port != 22:
        ssh_cmd += ["-p", str(port)]
    if key_filename:
        ssh_cmd += ["-i", key_filename]
    ssh_cmd.append(f"{user}@{host}")
    # Use explicit command argument to avoid shell interpretation of hostnames.
    ssh_cmd.append(command)
    try:
        completed = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SSHResult(
            target=target,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_status=completed.returncode,
            error=None if completed.returncode == 0 else error_hint or completed.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))


def _run_tailscale_ssh(
    target: str,
    user: str,
    host: str,
    command: str,
    timeout: int,
) -> SSHResult:
    try:
        host = ensure_ip_literal(host)
    except ValueError as exc:
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))

    # Must be IP-only per policy.
    ts_cmd = ["tailscale", "ssh", f"{user}@{host}", command]
    try:
        completed = subprocess.run(
            ts_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SSHResult(
            target=target,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_status=completed.returncode,
            error=None if completed.returncode == 0 else completed.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))


def _run_scp(
    target: str,
    user: str,
    host: str,
    port: int,
    key_filename: Optional[str],
    local_path: str,
    remote_path: str,
    timeout: int,
) -> SSHResult:
    scp_cmd = [
        "scp",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "PasswordAuthentication=no",
    ]
    if port != 22:
        scp_cmd += ["-P", str(port)]
    if key_filename:
        scp_cmd += ["-i", key_filename]
    scp_cmd += [local_path, f"{user}@{host}:{remote_path}"]
    try:
        completed = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SSHResult(
            target=target,
            stdout=completed.stdout or "file copied",
            stderr=completed.stderr,
            exit_status=completed.returncode,
            error=None if completed.returncode == 0 else completed.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        return SSHResult(target=target, stdout="", stderr="", exit_status=None, error=str(exc))


__all__ = ["SSHResult", "run_ssh_command", "copy_file"]
