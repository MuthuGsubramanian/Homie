from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from homie.node.auth import HMACAuthMiddleware
from homie.node.executor import run_command
from homie.node.safety import is_command_safe
from homie.node.status import collect_status
from homie.node.workflows import WorkflowRecorder
from homie.utils import timestamp

SHARED_SECRET = os.getenv("HOMIE_SHARED_SECRET", "changeme")
DATA_DIR = Path(os.getenv("HOMIE_DATA_DIR", "~/.homie/node")).expanduser()
MAX_LOG_BYTES = 256_000  # per file safeguard

app = FastAPI(title="HOMIE Node Agent", docs_url=None, redoc_url=None)
app.add_middleware(HMACAuthMiddleware, shared_secret=SHARED_SECRET)

status_cache: Dict = collect_status()
recorder = WorkflowRecorder(DATA_DIR / "workflows")


@app.get("/health")
def health():
    return {"status": "ok", "ts": timestamp()}


# --- v1 API (spec aligned) ---
@app.post("/v1/hello")
def v1_hello(payload: dict):
    node_info = collect_status()
    node_info.update({"status": "online"})
    return {
        "node_id": payload.get("node_id") or node_info.get("tailscale_ip") or "unknown",
        "os": node_info.get("os"),
        "capabilities": {"gpu": node_info.get("gpu"), "docker": node_info.get("docker")},
        "tailscale_ip": node_info.get("tailscale_ip"),
        "version": os.getenv("HOMIE_VERSION", "vnext"),
    }


@app.get("/v1/metrics")
def v1_metrics():
    global status_cache  # noqa: PLW0603
    status_cache = collect_status()
    status_cache.update({"status": "online"})
    return status_cache


@app.post("/v1/run")
def v1_run(payload: dict):
    policy = payload.get("policy", {"blocked_substrings": [], "max_command_len": 300})
    reason = payload.get("reason")
    if not reason:
        raise HTTPException(status_code=400, detail="reason required")

    cmd = (payload.get("command") or "").strip()
    safe, why = is_command_safe(cmd, policy)
    if not safe:
        raise HTTPException(status_code=400, detail=why)

    result = run_command(
        cmd,
        workdir=Path(payload.get("workdir")) if payload.get("workdir") else None,
        env=payload.get("env"),
        timeout_sec=int(payload.get("timeout_s", 120)),
        cpu_percent=int(payload.get("cpu_percent", 50)),
        mem_mb=int(payload.get("mem_mb", 512)),
        dry_run=bool(payload.get("dry_run", False)),
    )
    return {
        "run_id": payload.get("run_id") or timestamp(),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_status": result.exit_code,
        "error": result.error,
    }


@app.post("/v1/fetch_logs")
def v1_fetch_logs(payload: dict):
    paths = payload.get("paths") or []
    tail_lines = int(payload.get("tail_lines") or 0)
    files = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            continue
        content: str
        data = p.read_bytes()
        if len(data) > MAX_LOG_BYTES:
            data = data[-MAX_LOG_BYTES:]
        content = data.decode("utf-8", errors="ignore")
        if tail_lines > 0:
            content = "\n".join(content.splitlines()[-tail_lines:])
        files.append({"path": str(p), "content": content})
    return {"files": files}


@app.post("/v1/capabilities")
def v1_capabilities():
    info = collect_status()
    return {"gpu": info.get("gpu"), "docker": info.get("docker"), "os": info.get("os")}


@app.post("/v1/rollback")
def v1_rollback(payload: dict):
    # Placeholder: rollback requires previously stored plan; keep explicit ask-only.
    return {"run_id": payload.get("run_id"), "rollback": "not_implemented", "ok": False}


@app.post("/v1/recordings/upload")
def v1_recordings_upload(payload: dict):
    name = payload.get("name") or f"recording_{timestamp()}"
    data = payload.get("payload") or {}
    rec_dir = DATA_DIR / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    dest = rec_dir / f"{name}.json"
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"path": str(dest), "bytes": dest.stat().st_size}


# --- legacy endpoints (kept for compatibility) ---
@app.get("/status")
def status():
    global status_cache  # noqa: PLW0603
    status_cache = collect_status()
    status_cache.update({"status": "online"})
    return status_cache


@app.post("/commands")
def commands(payload: dict):
    policy = payload.get("policy", {"blocked_substrings": [], "max_command_len": 300})
    reason = payload.get("reason")
    if not reason:
        raise HTTPException(status_code=400, detail="reason required")

    cmd = (payload.get("command") or "").strip()
    safe, why = is_command_safe(cmd, policy)
    if not safe:
        raise HTTPException(status_code=400, detail=why)

    result = run_command(
        cmd,
        workdir=Path(payload.get("cwd")) if payload.get("cwd") else None,
        env=payload.get("env"),
        timeout_sec=int(payload.get("timeout_sec", 120)),
        cpu_percent=int(payload.get("cpu_percent", 50)),
        mem_mb=int(payload.get("mem_mb", 512)),
        dry_run=bool(payload.get("dry_run", False)),
    )
    return {
        "run_id": payload.get("id") or timestamp(),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "error": result.error,
    }


@app.post("/tasks/run")
def tasks_run(payload: dict):
    # Alias to /commands for now
    return commands(payload)


@app.post("/workflows/start")
def workflows_start(payload: dict):
    if not payload.get("permissions"):
        raise HTTPException(status_code=400, detail="permissions required (explicit consent)")
    session = recorder.start(payload.get("name", "workflow"), payload["permissions"])
    return {"workflow_id": session.id, "indicator": "Recording ON"}


@app.post("/workflows/{workflow_id}/checkpoint")
def workflows_checkpoint(workflow_id: str, payload: dict):
    if not recorder.active or recorder.active.id != workflow_id:
        raise HTTPException(status_code=404, detail="no active workflow")
    recorder.checkpoint(payload.get("note", ""), payload.get("payload"))
    return {"ok": True}


@app.post("/workflows/{workflow_id}/stop")
def workflows_stop(workflow_id: str):
    if not recorder.active or recorder.active.id != workflow_id:
        raise HTTPException(status_code=404, detail="no active workflow")
    session = recorder.stop()
    return {
        "workflow_id": workflow_id,
        "stored_at": str(session.storage_path) if session else None,
        "steps": len(session.steps) if session else 0,
    }


@app.get("/workflows")
def workflows_list():
    files = (recorder.root.glob("workflow_*.json") if recorder else [])
    return [{"path": str(p), "size": p.stat().st_size} for p in files]


@app.post("/workflows/{workflow_id}/replay")
def workflows_replay(workflow_id: str, payload: dict):
    # Placeholder: actual replay not implemented to avoid unsafe automation defaults.
    return {"workflow_id": workflow_id, "accepted": True, "note": "Replay stub; implement with explicit actions."}


@app.post("/clear")
def clear(payload: dict):
    cleared = []
    if payload.get("logs"):
        log_dir = DATA_DIR / "logs"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        cleared.append("logs")
    if payload.get("cache"):
        cache_dir = DATA_DIR / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cleared.append("cache")
    if payload.get("workflows"):
        wf_dir = recorder.root
        if wf_dir.exists():
            shutil.rmtree(wf_dir)
            wf_dir.mkdir(parents=True, exist_ok=True)
        cleared.append("workflows")
    return {"cleared": cleared}


__all__ = ["app"]
