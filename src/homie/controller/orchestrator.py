from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from homie.config import HomieConfig, cfg_get
from homie.controller.node_api import NodeApiClient, NodeApiError
from homie.safety import validate_plan
from homie.ssh_executor import SSHResult, copy_file, run_ssh_command
from homie.utils import timestamp, ensure_ip_literal


@dataclass
class OrchestratorResult:
    ok: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class Orchestrator:
    """Routes tasks to node API or SSH backends with safety + logging."""

    def __init__(self, cfg: HomieConfig, storage=None):
        self.cfg = cfg
        self.storage = storage  # optional storage adapter

    def _get_node_client(self, target: str) -> Optional[NodeApiClient]:
        node_cfg = cfg_get(self.cfg, "node_api", "targets", default={}) or {}
        if target not in node_cfg:
            return None
        base_url = node_cfg[target]["base_url"]
        secret = node_cfg[target]["shared_secret"]
        timeout = int(cfg_get(self.cfg, "node_api", "timeout_sec", default=10))
        return NodeApiClient(base_url, secret, timeout=timeout)

    def _target_ip(self, target: str) -> str:
        ssh_cfg = cfg_get(self.cfg, "ssh", "targets", default={}) or {}
        if target not in ssh_cfg:
            raise ValueError(f"Unknown target '{target}'. Configure ssh.targets with IPs.")
        host = ssh_cfg[target].get("ip") or ssh_cfg[target].get("host") or target
        return ensure_ip_literal(host)

    def _record(self, kind: str, target: str, payload: Dict[str, Any], result: Dict[str, Any]):
        if not self.storage:
            return
        try:
            run_id = result.get("run_id") or payload.get("run_id")
            if run_id:
                self.storage.complete_run(
                    run_id=run_id,
                    status="ok" if not result.get("error") else "failed",
                    exit_status=result.get("exit_status"),
                    stdout_path=None,
                    stderr_path=None,
                    finished_at=timestamp(),
                    error=result.get("error"),
                )
        except Exception:  # noqa: BLE001
            logging.exception("Failed to persist task record")

    def dispatch(self, plan_dict: Dict[str, Any], dry_run: bool = False) -> OrchestratorResult:
        is_safe, reason = validate_plan(plan_dict, self.cfg)
        if not is_safe:
            return OrchestratorResult(ok=False, data={}, error=reason)

        action = plan_dict.get("action")
        target = plan_dict.get("target")
        command = plan_dict.get("command", "")
        args = plan_dict.get("args") or {}
        backend_pref = plan_dict.get("backend") or cfg_get(self.cfg, "orchestrator", "backend", default="node")
        risk_class = plan_dict.get("risk_class", "medium")
        autonomy_level = plan_dict.get("autonomy_level", "suggest")
        policy = {
            "blocked_substrings": cfg_get(self.cfg, "safety", "blocked_substrings", default=[]),
            "max_command_len": cfg_get(self.cfg, "safety", "max_command_len", default=300),
        }

        if dry_run:
            logging.info("DRY-RUN %s on %s cmd=%s", action, target, command)
            return OrchestratorResult(ok=True, data={"dry_run": True, "ts": timestamp()})

        try:
            target_ip = self._target_ip(target)
        except ValueError as exc:
            return OrchestratorResult(ok=False, data={}, error=str(exc))

        run_id = None
        if self.storage:
            try:
                run_id = self.storage.record_run(
                    machine_ip=target_ip,
                    command=command,
                    user=cfg_get(self.cfg, "ssh", "default_user"),
                    reason=plan_dict.get("reason", ""),
                    status="pending",
                    autonomy_level=autonomy_level,
                    risk_class=risk_class,
                    rollback_plan=plan_dict.get("rollback_plan"),
                    started_at=timestamp(),
                )
            except Exception:  # noqa: BLE001
                logging.exception("Failed to record run")
                run_id = None

        client = self._get_node_client(target) if backend_pref == "node" else None

        result: Dict[str, Any] = {}
        if action == "run_command":
            if client:
                try:
                    result = client.run_task(
                        {
                            "run_id": run_id,
                            "command": command,
                            "reason": plan_dict.get("reason"),
                            "policy": policy,
                            "timeout_s": cfg_get(self.cfg, "ssh", "command_timeout_sec", default=120),
                        }
                    )
                except NodeApiError as exc:
                    logging.warning("Node API failed on %s (%s); falling back to SSH", target, exc)
                    client = None  # trigger SSH fallback below
            if not client:
                ssh_res = run_ssh_command(self.cfg, target, command)
                result = {
                    "stdout": ssh_res.stdout,
                    "stderr": ssh_res.stderr,
                    "exit_status": ssh_res.exit_status,
                    "error": ssh_res.error,
                    "run_id": run_id,
                    "target_ip": target_ip,
                }
        elif action == "check_status":
            if client:
                try:
                    result = client.get_status()
                except NodeApiError as exc:
                    logging.warning("Node API status failed on %s (%s); falling back to SSH", target, exc)
                    client = None
            if not client:
                ssh_res = run_ssh_command(self.cfg, target, plan_dict.get("command") or "uptime")
                result = {"stdout": ssh_res.stdout, "stderr": ssh_res.stderr, "error": ssh_res.error}
        elif action == "copy_file":
            src = args.get("src") or args.get("source")
            dest = args.get("dest") or args.get("destination") or args.get("target")
            if not src or not dest:
                return OrchestratorResult(ok=False, data={}, error="missing src/dest")
            ssh_res: SSHResult = copy_file(self.cfg, target, src, dest)
            result = {"stdout": ssh_res.stdout, "stderr": ssh_res.stderr, "error": ssh_res.error}
        else:
            return OrchestratorResult(ok=False, data={}, error=f"Unsupported action {action}")

        self._record(kind=action, target=target, payload=plan_dict, result=result)
        if self.storage:
            try:
                self.storage.record_ledger(
                    ts=timestamp(),
                    actor="controller",
                    machine_ip=target_ip,
                    entry_type="run" if action == "run_command" else action,
                    ref_id=str(run_id) if run_id else "",
                    what=action,
                    why=plan_dict.get("reason", ""),
                    signals={"backend": "node" if client else "ssh"},
                    confidence=None,
                    outcome="ok" if not result.get("error") else "error",
                    rollback_available=bool(plan_dict.get("rollback_plan")),
                    autonomy_level=autonomy_level,
                    risk_class=risk_class,
                )
            except Exception:  # noqa: BLE001
                logging.exception("Failed to record ledger")

        error = result.get("error") if isinstance(result, dict) else None
        ok = not error
        return OrchestratorResult(ok=ok, data=result, error=error)


__all__ = ["Orchestrator", "OrchestratorResult"]
