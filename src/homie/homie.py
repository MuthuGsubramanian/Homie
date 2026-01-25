from config import load_config, cfg_get
from ai_planner import plan
from ssh_executor import run_ssh

def main():
    cfg = load_config()

    print(f"HOMIE ready | model={cfg_get(cfg,'llm','model')} | provider={cfg_get(cfg,'llm','provider')}")
    while True:
        user = input("HOMIE> ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        task = plan(cfg, user)

        dry_run = cfg_get(cfg, "orchestrator", "dry_run", default=False)
        if dry_run:
            print("[DRY RUN]", task)
            continue

        action = task.get("action")
        target = task.get("target")

        if action == "run_command":
            cmd = task.get("command", "")
            if target == "all":
                for t in cfg_get(cfg, "ssh", "targets", default={}).keys():
                    res = run_ssh(cfg, t, cmd)
                    print(f"\n== {t} ==\n{res['stdout']}{res['stderr']}")
            else:
                res = run_ssh(cfg, target, cmd)
                print(res["stdout"] or res["stderr"])

        elif action == "check_status":
            # simplest: uptime + disk + memory
            cmd = "uptime && df -h / && free -h"
            res = run_ssh(cfg, target, cmd)
            print(res["stdout"] or res["stderr"])

        else:
            print("Unsupported action:", action, task)

if __name__ == "__main__":
    main()
