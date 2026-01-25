# HOMIE Architecture (local-first, Tailnet-only)

## Components (text diagram)
Controller (primary)
- CLI + Tray + Web Dashboard (localhost only)
- Orchestrator (task router) -> Node API client (Tailnet HTTPS/HMAC)
- Planner (Ollama glm-4.7-flash, JSON contract) + Safety Engine
- Scheduler (APScheduler) + Suggestion Engine (throttled)
- Storage: SQLite (memory, tasks, workflows) + encrypted file vault for artifacts
- Notifier: Windows toast / Linux notify-send (local only)
- Pairing Manager: generates & rotates shared secret; distributes via tailscale ssh

Node Agent (per machine)
- FastAPI service bound to 100.x.y.z (Tailnet) and/or localhost; mutual-auth/HMAC
- Auth middleware + Safety gate
- Status collector (psutil, disk/GPU probes) -> cached heartbeat
- Task executor (local subprocess); optional ssh fallback disabled by default
- Workflow recorder (manual, user-triggered, visible indicator)
- Logs + cache stored locally with “clear” endpoint
- Assistive signals (opt-in): active window, process summary

Data never leaves Tailnet; no cloud calls. LLM is local (Ollama).

## Request/Response Flows
- get_status (Controller -> Node): GET /status (signed). Node returns cached metrics + capabilities.
- run_task (Controller -> Node): POST /tasks/run {task_id?, command?, args}. Node validates via safety + policy, executes locally, streams logs (SSE/WebSocket) or returns final result.
- run_command (Controller -> Node): POST /commands (same payload as run_task, type=adhoc).
- start_recording (Controller -> Node): POST /workflows/start {name, permissions}. Node starts recording with visible indicator, only allowed signals; returns session_id.
- stop_recording (Controller -> Node): POST /workflows/{id}/stop. Node finalizes, stores locally, returns summary pointer; Controller ingests metadata only.
- list_workflows (Controller -> Node): GET /workflows. Returns metadata (no payload data unless requested).
- replay_workflow (Controller -> Node): POST /workflows/{id}/replay {speed?, targets?}. Node replays locally; Controller shows status.

Heartbeat model: Controller pull (GET /status every 15–30s, jitter). Reason: keeps Controller authoritative, avoids inbound to Controller, simpler firewall story. Optional future push `/heartbeat` available but off by default.

## API Contracts (FastAPI)
### Node Agent (all responses JSON, authenticated)
- `GET /health` -> {status:"ok", version}
- `GET /status` -> {node_id, ts, cpu, ram, disk, gpu:bool, docker:bool, os, tailscale_ip, workloads:{running, queued}, permissions, assistive:{window?, process_summary?}}
- `POST /tasks/run` body: {id?, command, args?, env?, timeout_sec?, reason, require_approval?:bool, dry_run?:bool}
  - returns {run_id, accepted:bool, queued:bool}
- `POST /commands` alias of /tasks/run for adhoc shell; response same
- `GET /tasks/{run_id}` -> {run_id, state:queued|running|succeeded|failed|cancelled, exit_code?, stdout_tail, stderr_tail, started_at, finished_at}
- `POST /workflows/start` body: {name, permissions:{screenshots:bool, window_titles:bool, process_list:bool}, snapshot_interval_sec?, indicators:{tray:true}} -> {workflow_id, indicator_hint}
- `POST /workflows/{workflow_id}/checkpoint` body: {note?, take_screenshot?:bool} -> {ok:true, path}
- `POST /workflows/{workflow_id}/stop` -> {workflow_id, steps, stored_at}
- `GET /workflows` -> [{workflow_id, name, started_at, duration_sec, steps, size_bytes}]
- `POST /workflows/{workflow_id}/replay` body: {speed?:float} -> {accepted:true}
- `POST /clear` body: {logs?:bool, cache?:bool, workflows?:bool} -> {cleared:["logs",...]}
- `POST /auth/rotate` body: {new_key?} -> {rotated_at} (requires local auth header)
- Error format: {error:{code,int,message}}

### Controller Dashboard API (localhost only)
- `GET /api/machines` -> list of machines with status + caps
- `POST /api/machines/pair` {node_name, tailscale_ip, secret?} -> {node_id}
- `GET /api/tasks` -> recent task runs (joined across nodes)
- `POST /api/tasks/run` {target, command, reason, dry_run?, backend:"node|tailscale_ssh|openssh"} -> {run_id}
- `GET /api/workflows` / `GET /api/workflows/{id}`
- `POST /api/workflows/{id}/replay`
- `GET /api/suggestions` -> [{id, text, why, data_used, actions}]
- `POST /api/privacy/clear` {scope:["memory","workflows","logs"]} -> {cleared}
- `POST /api/settings/reload` -> {reloaded:true} (hot reload)

## Data Model (SQLite)
- machines(id PK, name, tailnet_ip, os, has_gpu bool, docker bool, status, last_seen_ts, shared_secret_hash, allow_connections bool)
- task_runs(id PK, machine_id FK, type, command, args_json, reason, backend, state, exit_code, stdout_path, stderr_path, created_ts, started_ts, finished_ts, error)
- suggestions(id PK, machine_id FK nullable, text, why, data_used, state:shown|dismissed|accepted|blocked, created_ts, acted_ts, action)
- workflows(id PK, machine_id FK, name, started_ts, ended_ts, permissions_json, storage_path, size_bytes, redacted bool)
- workflow_steps(id PK, workflow_id FK, ts, kind:checkpoint|screenshot|action, payload_json, redacted bool)
- permissions(id PK, user, key, value_json, updated_ts)  // user preferences incl. allowlist apps, signals
- privacy_prefs(id PK, retention_days, max_storage_mb, allowlist_apps_json, flags_json, updated_ts)

## Minimal Code Skeletons (key modules)
- Controller
  - `src/homie/controller/orchestrator.py`: routes tasks to node/ssh backend, enforces safety.
  - `src/homie/controller/node_api.py`: signed client for Node API (HMAC + Tailnet binding).
  - `src/homie/controller/scheduler.py`: APScheduler wrapper for automations.
  - `src/homie/controller/storage.py`: SQLite layer (machines, task_runs, workflows, suggestions).
  - `src/homie/controller/notifier.py`: local notifications (Windows toast, Linux notify-send).
  - `src/homie/controller/dashboard.py`: FastAPI app serving UI + REST (localhost).
  - `src/homie/controller/pairing.py`: generate/distribute shared secret via tailscale ssh.
  - `src/homie/controller/safety.py`: central policies shared with Node.
- Node Agent
  - `src/homie/node/app.py`: FastAPI service, startup hooks, routers.
  - `src/homie/node/auth.py`: HMAC middleware, Tailnet/localhost binding check.
  - `src/homie/node/status.py`: psutil probes cached in-memory.
  - `src/homie/node/executor.py`: subprocess runner with resource caps and DRY-RUN.
  - `src/homie/node/workflows.py`: manual recorder, checkpoints, redaction helpers.
  - `src/homie/node/signals.py`: opt-in assistive signals (window title/process summary).
  - `src/homie/node/safety.py`: reuses central patterns, enforces blocked commands.
- Tray
  - `src/homie/tray/app.py`: lightweight tray menu for start/stop, pause suggestions, record toggle, clear data, open dashboard.

## Integration (tailscale ssh bootstrap)
1) On Controller: `tailscale up ...` ensure nodes visible.
2) Generate secret: `python -m homie.controller.pairing --create-key > ~/.homie/secret`.
3) Copy agent to node via tailscale ssh: `tailscale ssh user@node "mkdir -p ~/homie && exit"` then `tailscale ssh user@node 'cat > ~/homie/.env'` with `HOMIE_SHARED_SECRET=...`.
4) Install node service: `tailscale ssh user@node "python -m pip install homie && homie-node install-service --env ~/homie/.env"`.
5) Register node: Controller `homie-cli register --name laptop --ip 100.x.y.z --secret ~/.homie/secret`.
6) Verify: Controller `homie-cli status --target laptop` (calls GET /status); dashboard shows online.

## Tests & Validation Checklist
- Safety tests: blocked substrings/patterns reject; max command len enforced; dry-run mode no execution.
- Auth tests: HMAC required, bad signature rejected; Tailnet IP requirement enforced; secret rotation works.
- Offline nodes: status timeout returns cached/“offline”; task submission refused with clear error; scheduler skips offline nodes.
- Resource budgets: executor kills tasks exceeding CPU/RAM caps; sampling interval obeyed; backoff on overload.
- Privacy audit: zero outbound HTTP except Ollama local; verify recording requires consent flag; screenshots only during recording; no keystroke collection.
- Notifications: suggestion payload always includes why + data_used + actions.
- Config reload: hot reload applies new model/targets without restart (or emits “restart required” flag).

All capture/suggestion features remain opt-in, visible, and pausable. No covert logging, no cloud calls.
