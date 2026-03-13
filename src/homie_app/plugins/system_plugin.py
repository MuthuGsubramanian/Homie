from __future__ import annotations
from homie_core.plugins.base import HomiePlugin, PluginResult
import psutil


class SystemPlugin(HomiePlugin):
    name = "system"
    description = "System info: CPU, RAM, disk, running processes"
    permissions = ["read_system_info"]

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_context(self):
        return {
            "cpu_percent": psutil.cpu_percent(interval=0),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent if hasattr(psutil.disk_usage("/"), "percent") else 0,
        }

    def on_query(self, intent, params):
        if intent == "status":
            return PluginResult(success=True, data={
                "cpu": psutil.cpu_percent(interval=1),
                "ram": psutil.virtual_memory()._asdict(),
                "disk": psutil.disk_usage("/")._asdict() if hasattr(psutil, "disk_usage") else {},
            })
        if intent == "processes":
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    procs.append(p.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs.sort(key=lambda x: x.get("cpu_percent", 0) or 0, reverse=True)
            return PluginResult(success=True, data=procs[:20])
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")
