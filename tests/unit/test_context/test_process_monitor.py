"""Tests for process_monitor module."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch

import pytest

from homie_core.context.process_monitor import ProcessSnapshot, get_top_processes


def _make_mock_proc(pid: int, name: str, cpu: float, mem_rss: int, status: str):
    p = MagicMock()
    p.pid = pid
    p.name.return_value = name
    p.cpu_percent.return_value = cpu
    mem_info = MagicMock()
    mem_info.rss = mem_rss
    p.memory_info.return_value = mem_info
    p.status.return_value = status
    return p


class TestGetTopProcesses:
    def test_returns_list_of_process_snapshots(self):
        mock_procs = [
            _make_mock_proc(1, "python.exe", 5.0, 50 * 1024**2, "running"),
            _make_mock_proc(2, "chrome.exe", 10.0, 200 * 1024**2, "running"),
        ]
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = mock_procs
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = PermissionError

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_top_processes(n=15)

        assert isinstance(result, list)
        assert all(isinstance(p, ProcessSnapshot) for p in result)

    def test_returns_top_n_by_cpu(self):
        mock_procs = [
            _make_mock_proc(i, f"proc{i}.exe", float(i), i * 1024**2, "running")
            for i in range(10)
        ]
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = mock_procs
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = PermissionError

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_top_processes(n=3)

        assert len(result) == 3
        # Should be sorted descending by cpu_pct
        assert result[0].cpu_pct >= result[1].cpu_pct >= result[2].cpu_pct

    def test_respects_n_parameter_when_fewer_procs(self):
        mock_procs = [
            _make_mock_proc(1, "only.exe", 1.0, 1024**2, "running"),
        ]
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = mock_procs
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = PermissionError

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_top_processes(n=15)

        assert len(result) == 1

    def test_handles_psutil_not_installed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            result = get_top_processes()
        assert result == []

    def test_skips_no_such_process_errors(self):
        class FakeNoSuchProcess(Exception):
            pass

        good_proc = _make_mock_proc(1, "good.exe", 2.0, 10 * 1024**2, "running")
        bad_proc = MagicMock()
        bad_proc.cpu_percent.side_effect = FakeNoSuchProcess("gone")

        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [bad_proc, good_proc]
        mock_psutil.NoSuchProcess = FakeNoSuchProcess
        mock_psutil.AccessDenied = PermissionError

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_top_processes()

        assert len(result) == 1
        assert result[0].name == "good.exe"

    def test_snapshot_fields(self):
        mock_procs = [
            _make_mock_proc(42, "test.exe", 7.5, 128 * 1024**2, "sleeping"),
        ]
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = mock_procs
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = PermissionError

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_top_processes()

        snap = result[0]
        assert snap.pid == 42
        assert snap.name == "test.exe"
        assert snap.cpu_pct == 7.5
        assert snap.mem_mb == pytest.approx(128.0, abs=0.1)
        assert snap.status == "sleeping"
