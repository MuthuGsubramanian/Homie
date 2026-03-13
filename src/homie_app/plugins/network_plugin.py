from __future__ import annotations
import psutil
from homie_core.plugins.base import HomiePlugin, PluginResult


class NetworkPlugin(HomiePlugin):
    name = "network"
    description = "Network status and connection info"
    permissions = ["read_network"]

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_context(self):
        stats = psutil.net_io_counters()
        return {"bytes_sent": stats.bytes_sent, "bytes_recv": stats.bytes_recv}

    def on_query(self, intent, params):
        if intent == "status":
            addrs = {}
            for iface, addr_list in psutil.net_if_addrs().items():
                for addr in addr_list:
                    if addr.family.name == "AF_INET":
                        addrs[iface] = addr.address
            return PluginResult(success=True, data={"interfaces": addrs})
        if intent == "connections":
            conns = []
            for c in psutil.net_connections(kind="inet")[:20]:
                conns.append({"local": str(c.laddr), "remote": str(c.raddr), "status": c.status})
            return PluginResult(success=True, data=conns)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")
