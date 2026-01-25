from __future__ import annotations

# Allow running this file directly (python src/homie/homie.py) by ensuring
# the repository's src/ directory is on sys.path before absolute imports.
import pathlib
import sys

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import logging
from typing import Dict, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from homie.config import HomieConfig

from homie.ai_planner import plan, plan_to_dict
from pathlib import Path

from homie.config import cfg_get, list_targets, load_config
from homie.controller.orchestrator import Orchestrator
from homie.controller.storage import Storage
from homie.controller.scheduler import AutomationScheduler
from homie.controller.automations import register_automation_jobs
from homie.llm_ollama import LLMError
from homie.safety import validate_plan
from homie.ssh_executor import SSHResult, copy_file, run_ssh_command
from homie.utils import host_header, pretty_json, setup_logging


STATUS_COMMAND = (
    "uptime && df -h / && free -h && "
    "(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true)"
)


def _print_banner(cfg: HomieConfig) -> None:
    model = cfg_get(cfg, "llm", "model")
    provider = cfg_get(cfg, "llm", "provider")
    base_url = cfg_get(cfg, "llm", "base_url")
    targets = ", ".join(list_targets(cfg)) or "none"
    print(f"HOMIE ready | provider={provider} model={model} base_url={base_url}")
    print(f"Targets: {targets}")


def _handle_meta_command(cmd: str, cfg: HomieConfig, dry_run: bool) -> bool:
    """Handle meta commands. Return updated dry_run flag."""
    lower = cmd.lower()
    if lower.startswith(":dryrun"):
        parts = lower.split()
        if len(parts) == 2 and parts[1] in {"on", "off"}:
            dry_run = parts[1] == "on"
            print(f"dry_run set to {dry_run}")
        else:
            print("Usage: :dryrun on|off")
    elif lower == ":targets":
        print("Targets:", ", ".join(list_targets(cfg)) or "none")
    elif lower == ":model":
        print(
            f"Model: {cfg_get(cfg,'llm','model')} "
            f"provider={cfg_get(cfg,'llm','provider')} "
            f"base_url={cfg_get(cfg,'llm','base_url')}"
        )
    elif lower == ":help":
        print(
            "Meta commands:\n"
            "  :targets       list configured targets\n"
            "  :model         show current model/base_url\n"
            "  :dryrun on|off toggle execution\n"
            "  :help          show this help\n"
            "  :exit          quit"
        )
    elif lower == ":exit":
        raise SystemExit
    else:
        print("Unknown command. Type :help for options.")
    return dry_run


def _execute_plan(cfg: HomieConfig, plan_dict: Dict, dry_run: bool, orchestrator: Orchestrator | None = None) -> None:
    action = plan_dict.get("action")
    target = plan_dict.get("target")

    if dry_run:
        print("[DRY RUN] Planned action skipped.")
        return

    targets: Iterable[str]
    if target == "all":
        targets = list_targets(cfg)
    else:
        targets = [target]

    # Prefer orchestrator (Node API / SSH selection); fallback to legacy SSH if absent.
    if orchestrator:
        for t in targets:
            plan_with_target = dict(plan_dict, target=t)
            res = orchestrator.dispatch(plan_with_target, dry_run=dry_run)
            if res.ok:
                print(host_header(t))
                print(pretty_json(res.data))
            else:
                print(host_header(t))
                print(f"ERROR: {res.error}")
        return

    if action == "run_command":
        command = plan_dict.get("command", "")
        for t in targets:
            result = run_ssh_command(cfg, t, command)
            _print_result(result)
    elif action == "check_status":
        for t in targets:
            result = run_ssh_command(cfg, t, STATUS_COMMAND)
            _print_result(result)
    elif action == "copy_file":
        args = plan_dict.get("args") or {}
        src = args.get("src") or args.get("source")
        dest = args.get("dest") or args.get("destination") or args.get("target")
        if not src or not dest:
            print("copy_file requires args.src and args.dest; skipping.")
            return
        for t in targets:
            result = copy_file(cfg, t, src, dest)
            _print_result(result)
    else:
        print(f"Unsupported action: {action}")


def _print_result(result: SSHResult) -> None:
    print(host_header(result.target))
    if result.error:
        print(f"ERROR: {result.error}")
        return
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.exit_status not in (0, None):
        print(f"[exit_status={result.exit_status}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOMIE CLI")
    parser.add_argument("-c", "--config", help="Path to homie.config.yaml", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    cfg = load_config(args.config)
    storage_path = Path(cfg_get(cfg, "storage", "path", default="~/.homie/controller.db")).expanduser()
    storage = Storage(storage_path)
    orchestrator = Orchestrator(cfg, storage=storage)
    scheduler = AutomationScheduler()
    register_automation_jobs(scheduler, orchestrator, cfg)
    scheduler.start()
    _print_banner(cfg)
    dry_run = bool(cfg_get(cfg, "orchestrator", "dry_run", default=False))
    show_plan = bool(cfg_get(cfg, "orchestrator", "show_plan", default=True))

    try:
        while True:
            try:
                user_input = input("HOMIE> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith(":"):
                dry_run = _handle_meta_command(user_input, cfg, dry_run)
                continue

            try:
                planned = plan(cfg, user_input)
                plan_dict = plan_to_dict(planned)
            except (ValueError, LLMError) as exc:
                logging.error("Planning failed: %s", exc)
                print(f"Planning failed: {exc}")
                continue
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Unexpected planning error")
                print(f"Unexpected error: {exc}")
                continue

            is_safe, reason = validate_plan(plan_dict, cfg)
            if not is_safe:
                print(f"Safety check failed: {reason}")
                continue

            if show_plan:
                print("Planned action:")
                print(pretty_json(plan_dict))

            _execute_plan(cfg, plan_dict, dry_run, orchestrator)
    finally:
        try:
            scheduler.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
