"""Tests for system_status module."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch

import pytest

from homie_core.context.system_status import SystemStatus, get_system_status


def _make_mock_psutil(
    disk_used=50 * 1024**3,
    disk_total=100 * 1024**3,
    disk_pct=50.0,
    batt_pct=80.0,
    batt_plugged=True,
    net_sent=1024**2,
    net_recv=2 * 1024**2,
    cpu_pct=25.0,
    ram_pct=60.0,
):
    mock_psutil = MagicMock()

    disk = MagicMock()
    disk.used = disk_used
    disk.total = disk_total
    disk.percent = disk_pct
    mock_psutil.disk_usage.return_value = disk

    batt = MagicMock()
    batt.percent = batt_pct
    batt.power_plugged = batt_plugged
    mock_psutil.sensors_battery.return_value = batt

    net = MagicMock()
    net.bytes_sent = net_sent
    net.bytes_recv = net_recv
    mock_psutil.net_io_counters.return_value = net

    mem = MagicMock()
    mem.percent = ram_pct
    mock_psutil.virtual_memory.return_value = mem

    mock_psutil.cpu_percent.return_value = cpu_pct

    return mock_psutil


class TestGetSystemStatus:
    def test_returns_system_status_instance(self):
        mock_psutil = _make_mock_psutil()
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert isinstance(result, SystemStatus)

    def test_disk_pct_between_0_and_100(self):
        mock_psutil = _make_mock_psutil(disk_pct=65.3)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert 0 <= result.disk_pct <= 100

    def test_disk_values_converted_to_gb(self):
        mock_psutil = _make_mock_psutil(disk_used=50 * 1024**3, disk_total=100 * 1024**3)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert result.disk_used_gb == pytest.approx(50.0, abs=0.1)
        assert result.disk_total_gb == pytest.approx(100.0, abs=0.1)

    def test_battery_present(self):
        mock_psutil = _make_mock_psutil(batt_pct=75.0, batt_plugged=False)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert result.battery_pct == pytest.approx(75.0)
        assert result.battery_plugged is False

    def test_battery_absent_returns_none(self):
        mock_psutil = _make_mock_psutil()
        mock_psutil.sensors_battery.return_value = None
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert result.battery_pct is None
        assert result.battery_plugged is None

    def test_network_converted_to_mb(self):
        mock_psutil = _make_mock_psutil(net_sent=10 * 1024**2, net_recv=20 * 1024**2)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert result.net_sent_mb == pytest.approx(10.0, abs=0.1)
        assert result.net_recv_mb == pytest.approx(20.0, abs=0.1)

    def test_cpu_and_ram_pct(self):
        mock_psutil = _make_mock_psutil(cpu_pct=33.0, ram_pct=70.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_status("/")
        assert result.cpu_pct == pytest.approx(33.0)
        assert result.ram_pct == pytest.approx(70.0)

    def test_handles_psutil_not_installed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            result = get_system_status()
        assert isinstance(result, SystemStatus)
        assert result.disk_used_gb == 0
        assert result.disk_total_gb == 0
        assert result.disk_pct == 0
        assert result.battery_pct is None
        assert result.battery_plugged is None
        assert result.net_sent_mb == 0
        assert result.net_recv_mb == 0
        assert result.cpu_pct == 0
        assert result.ram_pct == 0
