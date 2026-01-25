import paramiko
from config import HomieConfig, cfg_get

def run_ssh(cfg: HomieConfig, target: str, command: str):
    targets = cfg_get(cfg, "ssh", "targets", default={})
    if target not in targets:
        raise ValueError(f"Unknown target: {target}. Known: {list(targets.keys())}")

    t = targets[target]
    host = t["host"]
    user = t.get("user", cfg_get(cfg, "ssh", "default_user"))
    port = int(t.get("port", 22))
    timeout = int(cfg_get(cfg, "ssh", "connect_timeout_sec", default=10))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=user, timeout=timeout)

    stdin, stdout, stderr = ssh.exec_command(command)
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    ssh.close()
    return {"stdout": out, "stderr": err}
