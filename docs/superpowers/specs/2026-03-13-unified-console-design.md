# Unified Console Design

**Date:** 2026-03-13
**Status:** Approved

## Overview

Replace all separate CLI subcommands with a single `homie start` entry point. All actions — service connections, settings, plugins, email, daemon management, and local intelligence — happen inside the interactive console via slash commands. Users never leave the Homie console to perform any task.

## Architecture

### Single Entry Point

`homie` (no subcommand) or `homie start` enters the console. These are the only CLI entry points.

- **No config detected on first run** → auto-enters init wizard inside the console
- **Incomplete config detected** → resumes wizard from where it left off until all required config is in place
- **Config exists** → enters interactive chat mode

**Argparse change:** `main()` changes from `parser.print_help()` on no command to launching the console directly. `homie start` remains as an explicit alias. `homie --version` still works.

**CLI flags carried over to console:**
- `--config <path>` — custom config file path
- `--no-voice` — disable voice pipeline
- `--no-tray` — disable system tray

These become top-level arguments on the parser (not on a subparser), so `homie --no-voice` and `homie start --no-voice` both work.

### Console Loop

```
1. Read input
2. If input starts with "/" → parse command name + args → dispatch to SlashCommandRouter
3. If input is bare "/" → show all available commands (autocomplete)
4. Otherwise → send to Brain as chat message
```

### Migration from Existing `_handle_meta_command`

The current `cli.py` already has a slash-command handler at `_handle_meta_command()` (line 548) supporting `/status`, `/learn`, `/facts`, `/remember`, `/forget`, `/clear`, `/insights`, `/schedule`, `/skills`, `/connections`, `/consent-log`, `/vault`, `/connect`, `/email`, `/disconnect`, `/help`.

**Migration path:** Each handler block in `_handle_meta_command()` moves into its corresponding `console/commands/*.py` module. The monolithic function is replaced by the `SlashCommandRouter` registry. Existing commands like `/connect gmail` that currently say "Use the CLI command: homie connect gmail" will be updated to run the OAuth flow inline.

### File Structure

New module `src/homie_app/console/`:

```
console/
├── __init__.py          # Console class (main loop)
├── router.py            # SlashCommandRouter + SlashCommand dataclass
├── commands/
│   ├── __init__.py      # register_all_commands()
│   ├── help.py          # /help
│   ├── settings.py      # /settings (reuses _step_* functions)
│   ├── connect.py       # /connect, /disconnect, /connections
│   ├── email.py         # /email summary|sync|config
│   ├── plugins.py       # /plugins list|enable|disable
│   ├── daemon.py        # /daemon start|stop|status
│   ├── consent.py       # /consent-log
│   ├── vault.py         # /vault
│   ├── model.py         # /model list|download|add|remove|switch|benchmark
│   ├── memory.py        # /status, /learn, /facts, /remember, /forget, /clear
│   ├── insights.py      # /insights
│   ├── schedule.py      # /schedule add|list|remove
│   ├── skills.py        # /skills
│   ├── folder.py        # /folder watch|list|scan|unwatch
│   ├── social.py        # /social channels|recent
│   ├── sm.py            # /sm feed|profile|scan|publish|dms|send-dm
│   ├── browser.py       # /browser enable|disable|config|history|scan|patterns
│   ├── voice.py         # /voice status|enable|disable|mode|tts
│   ├── backup.py        # /backup, /restore
│   ├── location.py      # /location
│   ├── weather.py       # /weather (NEW — implementation required)
│   ├── news.py          # /news (NEW — implementation required)
│   └── briefing.py      # /briefing (NEW — implementation required)
└── autocomplete.py      # Tab-completion / "/" listing logic
```

### SlashCommandRouter

- Registry (dict) mapping command names to handler functions
- Each handler is a `SlashCommand` dataclass: `name`, `description`, `args_spec`, `handler_fn`, `subcommands`, `autocomplete_fn`
- Handlers receive console I/O context for interactive prompts (OAuth flows, menus)
- Handlers receive app state (config, vault, services)
- **Subcommand dispatch:** If a command has subcommands and user types just the command (e.g., `/daemon`), show subcommand help. If a subcommand is provided, dispatch to it.

### What Happens to cli.py

- Shrinks to: argument parsing for `homie [start]` with `--config`, `--no-voice`, `--no-tray`, `--version`
- `main()` defaults to launching the console when no command is given
- All `cmd_*` handler functions move into `console/commands/`
- `_handle_meta_command()` is deleted — replaced by `SlashCommandRouter`
- Helper functions (`_step_email()`, `_step_social_connections()`, OAuth flows) reused with console I/O context instead of raw `input()`

## Slash Commands

### Complete Command List

All existing CLI subcommands and in-chat meta-commands, unified:

| Command | Args | Description | Source |
|---------|------|-------------|--------|
| `/help` | `[command]` | List all commands or detail for one | existing meta-cmd |
| `/status` | — | Show system status (memory, facts, user) | existing meta-cmd |
| `/settings` | `[category]` | Interactive settings (LLM, voice, privacy, notifications, location) | from `homie settings` |
| `/connect` | `<provider>` | Run OAuth/API-key setup inline | from `homie connect` + meta-cmd |
| `/disconnect` | `<provider>` | Revoke and disconnect | from `homie disconnect` |
| `/connections` | — | Show all providers with status | existing both |
| `/email` | `summary\|sync\|config` | Email operations | from `homie email` |
| `/plugins` | `list\|enable\|disable` | Plugin management | from `homie plugin` |
| `/daemon` | `start\|stop\|status` | Manage background service | from `homie daemon` + `homie stop/status` |
| `/consent-log` | `<provider>` | Consent audit trail | existing both |
| `/vault` | `[status]` | Vault health and status | existing both |
| `/model` | `list\|download\|add\|remove\|switch\|benchmark` | Model management | from `homie model` |
| `/backup` | `<path>` | Create encrypted backup | from `homie backup` |
| `/restore` | `<path>` | Restore from backup | from `homie restore` |
| `/insights` | `[--days N]` | Usage analytics and stats | existing both |
| `/schedule` | `add\|list\|remove` | Manage scheduled tasks | existing both |
| `/skills` | — | List installed skills | existing both |
| `/folder` | `watch\|list\|scan\|unwatch` | Folder awareness management | from `homie folder` |
| `/social` | `channels\|recent` | Social messaging (Slack) | from `homie social` |
| `/sm` | `feed\|profile\|scan\|publish\|dms\|send-dm` | Social media operations | from `homie sm` |
| `/browser` | `enable\|disable\|config\|history\|scan\|patterns` | Browser history management | from `homie browser` |
| `/voice` | `status\|enable\|disable\|mode\|tts` | Voice pipeline control | from `homie voice` |
| `/learn` | — | Show session learning stats | existing meta-cmd |
| `/facts` | — | Show stored facts about user | existing meta-cmd |
| `/remember` | `<fact>` | Store a fact explicitly | existing meta-cmd |
| `/forget` | `<topic>` | Forget facts about a topic | existing meta-cmd |
| `/clear` | — | Clear conversation history | existing meta-cmd |
| `/location` | `[set <city>]` | View or set location | **NEW** |
| `/weather` | `[forecast]` | Current conditions or forecast | **NEW — requires implementation** |
| `/news` | `[topic]` | Personalized headlines | **NEW — requires implementation** |
| `/briefing` | — | Full personalized briefing on demand | **NEW — requires implementation** |
| `/quit` | — | Exit Homie | replaces `quit` |

### Provider Types in `/connect`

`/connect` handles two types of providers:

1. **OAuth providers** (gmail, linkedin, twitter, reddit, slack, facebook, instagram): Opens browser, captures token via local redirect server, stores in vault
2. **API-key providers** (weather, news): Prompts for API key, validates it with a test call, stores in vault

The handler detects provider type and runs the appropriate flow.

### Autocomplete Behavior

- Type `/` → show all top-level commands with descriptions
- Type `/con` → filter to `/connect`, `/connections`, `/consent-log`
- Type `/connect ` → show available providers (gmail, linkedin, twitter, etc.)
- Type `/daemon ` → show `start`, `stop`, `status`
- Unknown command → "Unknown command. Type `/help` to see available commands."

**Implementation:** Use `prompt_toolkit` for rich input handling (tab-completion, history, syntax highlighting). This replaces the current raw `input()` calls in `cmd_chat()`. `prompt_toolkit` is a pure-Python library with no native dependencies.

## Complete CLI Migration Table

Every existing CLI command and its new console equivalent:

| Old CLI Command | New Slash Command |
|---|---|
| `homie` (no args) | Launches console directly |
| `homie start` | Launches console (alias) |
| `homie init` | Auto first-run detection in console |
| `homie chat` | Default mode inside console |
| `homie connect <provider>` | `/connect <provider>` |
| `homie disconnect <provider>` | `/disconnect <provider>` |
| `homie connections` | `/connections` |
| `homie settings` | `/settings` |
| `homie email summary\|sync\|config` | `/email summary\|sync\|config` |
| `homie consent-log <provider>` | `/consent-log <provider>` |
| `homie daemon` | `/daemon start\|stop\|status` |
| `homie stop` | `/daemon stop` |
| `homie status` | `/daemon status` (for daemon) or `/status` (for session) |
| `homie model list\|download\|add\|remove\|switch\|benchmark` | `/model list\|download\|add\|remove\|switch\|benchmark` |
| `homie plugin list\|enable\|disable` | `/plugins list\|enable\|disable` |
| `homie backup --to <path>` | `/backup <path>` |
| `homie restore --from <path>` | `/restore <path>` |
| `homie insights` | `/insights` |
| `homie schedule add\|list\|remove` | `/schedule add\|list\|remove` |
| `homie skills` | `/skills` |
| `homie folder watch\|list\|scan\|unwatch` | `/folder watch\|list\|scan\|unwatch` |
| `homie social channels\|recent` | `/social channels\|recent` |
| `homie sm feed\|profile\|scan\|publish\|dms\|send-dm` | `/sm feed\|profile\|scan\|publish\|dms\|send-dm` |
| `homie browser enable\|disable\|config\|history\|scan\|patterns` | `/browser enable\|disable\|config\|history\|scan\|patterns` |
| `homie voice status\|enable\|disable` | `/voice status\|enable\|disable` |
| `homie vault status` | `/vault status` |

## First-Run Detection & Inline Wizard

- Check for `homie.config.yaml` (or `~/.homie/` directory)
- No config → run init wizard inside console
- Incomplete config → resume wizard from where it left off until all required config is in place
- After wizard completes → seamlessly transition to chat mode (no restart)
- User can re-run wizard steps later via `/settings`

## Daemon Management

- `/daemon start` — spawns daemon as background subprocess; console stays interactive
- `/daemon stop` — sends stop signal via PID file (replaces old `homie stop`)
- `/daemon status` — running state, uptime, last sync times (replaces old `homie status`)
- After `/connect <provider>`, prompt: "Start background sync now? (y/n)" → auto-start daemon if yes
- On `/quit`: daemon keeps running if started; user informed how to stop it later

## Location & Local Intelligence

### Setup

- Init wizard includes optional location step: "What's your city/region?"
- Also available via `/location set <city>` or `/settings > Location`
- Stored in config as new `location` top-level key (additive, no breaking change):

```yaml
# Existing config fields unchanged
user_name: Master

# New top-level location block (added by wizard or /location set)
location:
  city: "Chennai"
  region: "Tamil Nadu"
  country: "IN"
  timezone: "Asia/Kolkata"
```

**Config migration:** This is an additive change. Existing configs without a `location` key simply have no location set. The config loader treats missing `location` as `None`. No migration script needed.

### Data Sources

- **Weather:** Free API (OpenWeatherMap or similar) — user provides API key via `/connect weather`
- **News:** RSS feeds + news API (NewsAPI free tier or similar) — API key via `/connect news`
- **No location tracking** — purely based on user's configured city

### Brain Tools (Reactive) — NEW IMPLEMENTATION

These 6 tools do not exist yet and must be built from scratch:

- `get_weather(location?)` — current conditions, forecast; defaults to configured location. Calls weather API, formats response.
- `get_news(query?, location?)` — headlines filterable by topic; defaults to local news. Calls news API.
- `get_local_conditions(location?)` — weather + air quality + alerts combined. Aggregates multiple API calls.
- `get_personalized_briefing()` — weather + news + calendar + email + financial in one contextual summary. Orchestrates existing services + new weather/news tools. `/briefing` slash command calls this tool.
- `get_local_events(interests?)` — events matching user interests in their area. Scrapes/APIs for local events.
- `get_commute_update(route?)` — traffic for saved routes. Maps/traffic API.

**Note:** The existing `BriefingGenerator` in `intelligence/briefing.py` generates session-resumption summaries only. `get_personalized_briefing()` is a new, distinct tool that composes a daily life briefing.

### Daemon Integration (Proactive)

- Morning briefing notification (configurable time): weather, top news, alerts, email summary, bills due
- Severe weather alerts pushed immediately
- Configurable via `/settings > Notifications`

### Personalization

Information is filtered and prioritized based on user profile from working memory:

| User context | Personalized output |
|---|---|
| Software engineer | Local tech meetups, hiring trends, relevant company news |
| Has a commute | Traffic on their specific route |
| Financial goals | Watchlist updates, bills due, local fuel prices |
| Health-conscious | Air quality + pollen + UV with exercise recommendations |
| Family | School closures, local family events |

### Privacy

- All personalization happens locally — profile stays in vault, not sent to APIs
- Brain composes personalized queries and summaries; APIs provide raw data only
- User controls via `/settings > Privacy`

## Error Handling

### OAuth Failures Inside Console

- Browser fails to open → show manual URL for copy/paste (existing fallback)
- Token capture times out → clear error, offer retry
- Invalid client ID/secret → show error, prompt to re-enter

### API-Key Provider Failures

- Invalid API key → test call fails → show error, prompt to re-enter
- API rate limit → show retry-after time, cache last successful response

### Daemon Conflicts

- `/daemon start` when already running → show status instead of double-start
- Daemon crashes → `/daemon status` detects missing process, reports cleanly

### Plugin Errors

- `/plugins enable` fails → show error, plugin stays disabled
- Plugin crashes at runtime → sandbox isolates, notify in console

### Console I/O During Long Operations

- OAuth waiting for callback → spinner/waiting message, Ctrl+C to cancel
- `/email sync` on large mailbox → progress indicator, cancellable
- `/model download` → progress bar with size/speed

## Testing Strategy

### Unit Tests

- `SlashCommandRouter` — registration, dispatch, unknown command handling, subcommand dispatch
- Each command handler — mock vault/config/services, verify actions
- Autocomplete — partial matching, subcommand completion, provider listing
- First-run detection — config exists, missing, incomplete resume
- Provider type detection in `/connect` — OAuth vs API-key routing

### Integration Tests

- Full console loop: `/` input → command list output
- `/connect gmail` with mocked OAuth (no real browser)
- `/settings` menu navigation within console
- First-run → wizard → chat mode transition
- Wizard resume on incomplete config
- `/daemon start` → `/daemon status` → `/daemon stop` lifecycle
- `/model list` → `/model switch` flow

### Migration of Existing Tests

Specific test files to migrate:
- Tests for `cmd_connect`, `cmd_disconnect` → `test_connect_command.py`
- Tests for `cmd_settings` → `test_settings_command.py`
- Tests for `cmd_model` → `test_model_command.py`
- Tests for `cmd_plugin` → `test_plugins_command.py`
- Tests for `cmd_email_*` → `test_email_command.py`
- Tests for `cmd_folder` → `test_folder_command.py`
- Tests for `cmd_social`, `cmd_sm` → `test_social_command.py`, `test_sm_command.py`
- Tests for `cmd_browser` → `test_browser_command.py`
- Tests for `cmd_voice` → `test_voice_command.py`
- Integration tests from commit `b8c698e` (init wizard, component integration) → update to test console-embedded wizard flow
- Tests for `_handle_meta_command` → delete after all handlers migrated to `SlashCommandRouter`

### Deprecation Strategy

No deprecation period — this is a clean cut. All old subcommands are removed in one release. Rationale:
- Homie is pre-1.0, no backwards-compatibility obligation
- The old commands were never publicly documented for scripting
- Users who type old commands get a clear message: "Did you mean `/connect gmail`? All commands now run inside the Homie console. Just run `homie` to start."
