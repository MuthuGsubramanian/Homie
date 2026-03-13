"""Handler for /daemon slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_daemon_start(args: str, **ctx) -> str:
    try:
        import subprocess
        import sys
        config_path = ctx.get("config_path", "")
        cmd = [sys.executable, "-m", "homie_app.daemon"]
        if config_path:
            cmd.extend(["--config", config_path])
        proc = subprocess.Popen(cmd, start_new_session=True)

        from pathlib import Path
        pid_file = Path.home() / ".homie" / "daemon.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(proc.pid))

        return f"Daemon started (PID: {proc.pid})"
    except Exception as e:
        return f"Could not start daemon: {e}"


def _handle_daemon_stop(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        import signal, os
        pid_file = Path.home() / ".homie" / "daemon.pid"
        if not pid_file.exists():
            return "No daemon PID file found. Is the daemon running?"
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink(missing_ok=True)
            return f"Daemon stopped (PID: {pid})"
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return "Daemon was not running (stale PID file cleaned up)."
    except Exception as e:
        return f"Could not stop daemon: {e}"


def _handle_daemon_status(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        pid_file = Path.home() / ".homie" / "daemon.pid"
        if not pid_file.exists():
            return "Daemon: not running"
        pid = int(pid_file.read_text().strip())
        try:
            import psutil
            proc = psutil.Process(pid)
            if proc.is_running():
                import time
                uptime = time.time() - proc.create_time()
                hours = int(uptime // 3600)
                mins = int((uptime % 3600) // 60)
                return f"Daemon: running (PID: {pid}, uptime: {hours}h {mins}m)"
            else:
                return "Daemon: not running (stale PID)"
        except (ImportError, Exception):
            return f"Daemon: PID file exists (PID: {pid}), status check requires psutil."
    except Exception as e:
        return f"Could not check daemon status: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="daemon",
        description="Manage background service",
        args_spec="start|stop|status",
        subcommands={
            "start": SlashCommand(name="start", description="Start background daemon", handler_fn=_handle_daemon_start),
            "stop": SlashCommand(name="stop", description="Stop daemon", handler_fn=_handle_daemon_stop),
            "status": SlashCommand(name="status", description="Show daemon status", handler_fn=_handle_daemon_status),
        },
    ))
