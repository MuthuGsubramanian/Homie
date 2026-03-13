from unittest.mock import MagicMock, patch
from pathlib import Path

from homie_app.daemon import HomieDaemon


def test_daemon_init():
    daemon = HomieDaemon.__new__(HomieDaemon)
    daemon._config = MagicMock()
    daemon._running = False
    assert not daemon._running


def test_daemon_components_created(tmp_path):
    with patch("homie_app.daemon.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.storage.log_dir = "logs"
        cfg.user_name = "Test"
        cfg.voice.enabled = False
        cfg.llm.backend = "cloud"
        cfg.llm.api_key = "test"
        cfg.llm.api_base_url = "http://localhost"
        mock_cfg.return_value = cfg

        with patch("homie_app.daemon.load_enterprise_policy", return_value=None):
            daemon = HomieDaemon(config_path=None)

    assert daemon._task_graph is not None
    assert daemon._observer is not None
    assert daemon._briefing is not None


def test_daemon_on_hotkey_triggers_overlay(tmp_path):
    with patch("homie_app.daemon.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.storage.log_dir = "logs"
        cfg.user_name = "Test"
        cfg.voice.enabled = False
        cfg.llm.backend = "cloud"
        cfg.llm.api_key = "test"
        cfg.llm.api_base_url = "http://localhost"
        mock_cfg.return_value = cfg

        with patch("homie_app.daemon.load_enterprise_policy", return_value=None):
            daemon = HomieDaemon(config_path=None)

    daemon._overlay = MagicMock()
    daemon._on_hotkey()
    daemon._overlay.toggle.assert_called_once()
