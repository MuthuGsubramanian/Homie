from __future__ import annotations
import yaml
from pathlib import Path
from homie_core.plugins.base import HomiePlugin, PluginResult


class WorkflowsPlugin(HomiePlugin):
    name = "workflows"
    description = "Multi-step automation workflows"
    permissions = ["execute_workflows"]

    def __init__(self):
        self._workflows: dict[str, dict] = {}

    def on_activate(self, config):
        workflow_dir = config.get("workflow_dir")
        if workflow_dir:
            self._load_workflows(Path(workflow_dir))

    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "list":
            return PluginResult(success=True, data=list(self._workflows.keys()))
        if intent == "get":
            name = params.get("name", "")
            wf = self._workflows.get(name)
            if wf:
                return PluginResult(success=True, data=wf)
            return PluginResult(success=False, error=f"Workflow '{name}' not found")
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        if action == "add":
            name = params.get("name", "")
            steps = params.get("steps", [])
            if name and steps:
                self._workflows[name] = {"name": name, "steps": steps}
                return PluginResult(success=True, data=f"Workflow '{name}' added")
            return PluginResult(success=False, error="Missing name or steps")
        return PluginResult(success=False, error=f"Unknown action: {action}")

    def _load_workflows(self, directory: Path) -> None:
        if not directory.exists():
            return
        for f in directory.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text())
                if isinstance(data, dict) and "name" in data:
                    self._workflows[data["name"]] = data
            except Exception:
                pass
