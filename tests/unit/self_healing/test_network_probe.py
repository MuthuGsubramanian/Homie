# tests/unit/self_healing/test_network_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.network_probe import NetworkProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestNetworkProbe:
    def test_healthy_when_lan_discovery_works(self):
        net = MagicMock()
        net.is_running = True
        net.peer_count = 2
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_no_peers(self):
        net = MagicMock()
        net.is_running = True
        net.peer_count = 0
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_not_running(self):
        net = MagicMock()
        net.is_running = False
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_manager(self):
        probe = NetworkProbe(network_manager=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
