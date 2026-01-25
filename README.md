# HOMIE (Home Orchestrated Machine Intelligence Engine)

Local-first personal assistant that runs only on your machines and Tailnet. Controller orchestrates tasks via Node Agents or SSH, with a localhost dashboard + tray, strict safety policy, and Ollama-based planning (glm-4.7-flash by default).

## Quickstart
- Create and activate a virtual environment.
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Pull the model: `ollama pull glm-4.7-flash`
- Configure controller + nodes: edit `homie.config.yaml` (or set `HOMIE_CONFIG`).
- Run the CLI (from repo root, with `src` on PYTHONPATH):
  - Windows PowerShell: `set PYTHONPATH=src; python -m homie.homie -c homie.config.yaml`
  - Linux/macOS: `PYTHONPATH=src python -m homie.homie -c homie.config.yaml`
- Run node agent (per machine, Tailnet-bound): `PYTHONPATH=src HOMIE_SHARED_SECRET=... uvicorn homie.node.app:app --host 100.x.y.z --port 8443 --ssl-keyfile ... --ssl-certfile ...`
- Run dashboard (controller): `PYTHONPATH=src uvicorn homie.controller.dashboard:create_app --factory --port 8080`

## Configuration (`homie.config.yaml`)
- `llm`: provider (`ollama`), `base_url`, `model`, `temperature`, `max_tokens`, `timeout_sec`.
- `orchestrator`: `mode`, `backend=node|tailscale_ssh|openssh`, `dry_run`, `show_plan`, `allowed_actions`.
- `ssh`: `default_user`, `connect_timeout_sec`, `method` (paramiko|openssh|tailscale_ssh), per-target `host`/`port`/`user` (plus optional `key_filename`/`password`). Use `method: openssh` if you rely on your local `~/.ssh/config`/Tailnet hostnames (e.g., `ssh -o PreferredAuthentications=publickey -o PasswordAuthentication=no msi`).
- `node_api`: per-node `base_url` (Tailnet IP) + `shared_secret`, `timeout_sec`.
- `privacy`: `data_retention_days`, `max_storage_mb`, permissions flags for signals/screenshots.
- `safety`: `blocked_substrings`, `max_command_len`, `require_reason`.
- Env overrides: `HOMIE_MODEL` and `HOMIE_OLLAMA_URL` take precedence.

## CLI usage
- Prompt: `HOMIE> `
- Natural language requests are planned by the LLM and validated against safety rules.
- Meta commands (no LLM):
  - `:targets` - list configured hosts
  - `:model` - show active model/provider/base_url
  - `:dryrun on|off` - toggle execution
  - `:help`, `:exit`
- Planned JSON is printed before execution (unless `show_plan=false`).

## Actions
- `run_command`: executes the planned `command` on one target or `all`.
- `check_status`: runs `uptime && df -h / && free -h` and, if available, `nvidia-smi`.
- `copy_file`: uses SFTP; expects `args.src` and `args.dest`.

## Examples
- `check status on msi`
- `run docker ps on all`
- `copy the file C:\\tmp\\bundle.tar.gz to /tmp on lenovo`
- `show gpu usage on msi`

## Components (high level)
- Controller: orchestrator, dashboard (FastAPI localhost), scheduler, notifier, storage (SQLite).
- Node Agent: lightweight FastAPI bound to Tailnet IP/localhost, runs tasks locally with HMAC auth.
- Tray: start/stop services, pause suggestions, record workflow, open dashboard, clear data.

Privacy defaults: no outbound data, no passive screenshots/keystrokes, recording is opt-in and visibly indicated.

## Notes
- If JSON from the model is malformed, HOMIE performs one repair pass; if parsing still fails, nothing is executed.
- Safety guardrails block dangerous substrings/patterns and overlong commands.
- Dry run mode honors config and runtime toggles; no SSH commands are issued when enabled.
