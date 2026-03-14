"""Handler for /model slash command — model management."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_model_list(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        from homie_core.model.registry import ModelRegistry
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
        registry.initialize()
        models = registry.list_models()
        if not models:
            return "No models installed. Run '/model download <repo_id>' or 'homie init'."
        lines = ["**Installed Models:**"]
        for m in models:
            active = " [ACTIVE]" if m.active else ""
            lines.append(f"  {m.name} ({m.params}, {m.format}){active}")
            if m.format == "cloud":
                lines.append("    Endpoint: cloud API")
            else:
                lines.append(f"    Path: {m.path}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not list models: {e}"


def _handle_model_download(args: str, **ctx) -> str:
    parts = args.strip().split()
    if not parts:
        return "Usage: /model download <repo_id> [--filename <file>]"
    try:
        repo_id = parts[0]
        filename = ""
        if "--filename" in parts:
            idx = parts.index("--filename")
            if idx + 1 < len(parts):
                filename = parts[idx + 1]
        from homie_core.model.downloader import ModelDownloader
        downloader = ModelDownloader()
        path = downloader.download(repo_id, filename=filename or None)
        return f"Downloaded: {path}"
    except Exception as e:
        return f"Could not download model: {e}"


def _handle_model_add(args: str, **ctx) -> str:
    parts = args.strip().split()
    if not parts or "--name" not in parts:
        return "Usage: /model add <path> --name <name> [--format gguf|safetensors|cloud|hf] [--params <params>]"
    try:
        from pathlib import Path
        from homie_core.model.registry import ModelRegistry
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."

        # Parse args
        path_str = parts[0]
        name = ""
        fmt = "gguf"
        params = "unknown"
        if "--name" in parts:
            name = parts[parts.index("--name") + 1]
        if "--format" in parts:
            fmt = parts[parts.index("--format") + 1]
        if "--params" in parts:
            params = parts[parts.index("--params") + 1]

        path = Path(path_str)
        if not path.exists():
            return f"Error: File not found: {path}"

        registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
        registry.initialize()
        registry.register(name, path, format=fmt, params=params)
        return f"Registered model '{name}' from {path}"
    except Exception as e:
        return f"Could not add model: {e}"


def _handle_model_remove(args: str, **ctx) -> str:
    name = args.strip()
    if not name:
        return "Usage: /model remove <name>"
    try:
        from pathlib import Path
        from homie_core.model.registry import ModelRegistry
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
        registry.initialize()
        registry.remove(name)
        return f"Removed model '{name}'"
    except Exception as e:
        return f"Could not remove model: {e}"


def _handle_model_switch(args: str, **ctx) -> str:
    name = args.strip()
    if not name:
        return "Usage: /model switch <name>"
    try:
        from pathlib import Path
        from homie_core.model.registry import ModelRegistry
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
        registry.initialize()
        if not registry.get(name):
            return f"Error: Model '{name}' not found"
        registry.set_active(name)
        return f"Switched to model '{name}'"
    except Exception as e:
        return f"Could not switch model: {e}"


def _handle_model_benchmark(args: str, **ctx) -> str:
    try:
        return "Benchmark not yet implemented in slash command mode. Use 'homie model benchmark' from the CLI."
    except Exception as e:
        return f"Could not benchmark: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="model",
        description="Model management (list, download, add, remove, switch, benchmark)",
        args_spec="list|download|add|remove|switch|benchmark",
        subcommands={
            "list": SlashCommand(name="list", description="List installed models", handler_fn=_handle_model_list),
            "download": SlashCommand(name="download", description="Download a model from HuggingFace", args_spec="<repo_id> [--filename <file>]", handler_fn=_handle_model_download),
            "add": SlashCommand(name="add", description="Register a local model file", args_spec="<path> --name <name>", handler_fn=_handle_model_add),
            "remove": SlashCommand(name="remove", description="Remove a model", args_spec="<name>", handler_fn=_handle_model_remove),
            "switch": SlashCommand(name="switch", description="Switch active model", args_spec="<name>", handler_fn=_handle_model_switch),
            "benchmark": SlashCommand(name="benchmark", description="Benchmark current model", handler_fn=_handle_model_benchmark),
        },
    ))
