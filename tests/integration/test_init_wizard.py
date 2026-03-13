"""End-to-end test of the init wizard step functions with simulated input."""
from unittest.mock import patch, MagicMock
from homie_core.config import HomieConfig


class TestInitWizardSteps:
    """Test each step function in isolation to verify the full wizard flow."""

    @patch("homie_app.init.input", return_value="TestUser")
    def test_user_profile_step(self, mock_input):
        from homie_app.init import _step_user_profile
        cfg = HomieConfig()
        _step_user_profile(cfg)
        assert cfg.user.name == "TestUser"

    @patch("homie_app.init._ask_choice", return_value=1)  # level 2
    def test_screen_reader_level2(self, mock_choice):
        from homie_app.init import _step_screen_reader
        cfg = HomieConfig()
        _step_screen_reader(cfg)
        assert cfg.screen_reader.enabled is True
        assert cfg.screen_reader.level == 2
        assert cfg.privacy.screen_reader_consent is True

    @patch("homie_app.init._ask_choice", return_value=2)  # level 3
    def test_screen_reader_level3(self, mock_choice):
        from homie_app.init import _step_screen_reader
        cfg = HomieConfig()
        _step_screen_reader(cfg)
        assert cfg.screen_reader.enabled is True
        assert cfg.screen_reader.level == 3

    @patch("homie_app.init.input", return_value="y")
    def test_email_connect(self, mock_input):
        from homie_app.init import _step_email
        cfg = HomieConfig()
        _step_email(cfg)
        assert cfg.connections.gmail.connected is True

    @patch("homie_app.init._ask_choice", return_value=1)  # windows service
    def test_service_mode_windows(self, mock_choice):
        from homie_app.init import _step_service_mode
        cfg = HomieConfig()
        _step_service_mode(cfg)
        assert cfg.service.mode == "windows_service"

    def test_save_and_load_config(self, tmp_path):
        """Test that config can be saved and loaded back."""
        from homie_app.init import _save_config
        from homie_core.config import HomieConfig
        import yaml

        cfg = HomieConfig()
        cfg.user.name = "IntegrationTest"
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = 2
        cfg.service.mode = "windows_service"
        cfg.voice.enabled = True

        path = str(tmp_path / "test_config.yaml")
        _save_config(cfg, path)

        # Verify file exists and is valid YAML
        from pathlib import Path
        assert Path(path).exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert data["user"]["name"] == "IntegrationTest"
        assert data["screen_reader"]["enabled"] is True
        assert data["screen_reader"]["level"] == 2
        assert data["service"]["mode"] == "windows_service"

    def test_detect_existing_config(self, tmp_path):
        """Test existing config detection."""
        import yaml
        config_file = tmp_path / "homie.config.yaml"
        config_file.write_text(yaml.dump({"user": {"name": "ExistingUser"}}))
        from homie_app.init import _detect_existing_config
        exists, data = _detect_existing_config(str(config_file))
        assert exists is True
        assert data["user"]["name"] == "ExistingUser"

    def test_detect_missing_config(self, tmp_path):
        from homie_app.init import _detect_existing_config
        exists, data = _detect_existing_config(str(tmp_path / "nonexistent.yaml"))
        assert exists is False
        assert data == {}

    def test_all_config_models_have_defaults(self):
        """Verify HomieConfig can be created with all defaults (backwards compat)."""
        cfg = HomieConfig()
        # All new sections should have sane defaults
        assert cfg.user.name == "Master"
        assert cfg.screen_reader.enabled is False
        assert cfg.screen_reader.level == 1
        assert cfg.service.mode == "on_demand"
        assert cfg.notifications.enabled is True
        assert "email_digest" in cfg.notifications.categories
        assert cfg.voice.enabled is False

    def test_notification_router_with_config(self):
        """Test notification router works with config defaults."""
        from homie_core.notifications.router import NotificationRouter, Notification
        cfg = HomieConfig()
        router = NotificationRouter(config=cfg.notifications)
        n = Notification(category="email_digest", title="Test", body="Test body")
        assert router.should_deliver(n) is True

    def test_pii_filter_pipeline(self):
        """Test PII filter works end-to-end."""
        from homie_core.screen_reader.pii_filter import PIIFilter
        f = PIIFilter()
        text = "Email: user@test.com, Phone: 555-123-4567, SSN: 123-45-6789"
        result = f.filter(text)
        assert "user@test.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert "[EMAIL]" in result
        assert "[PHONE]" in result

    def test_window_tracker_blocklist(self):
        """Test window tracker blocklist works."""
        from homie_core.screen_reader.window_tracker import WindowTracker
        tracker = WindowTracker(blocklist=["*password*", "*banking*"])
        assert tracker.is_blocked("1Password - Vault") is True
        assert tracker.is_blocked("VS Code - main.py") is False

    def test_tray_menu_builds(self):
        """Test tray menu can be built with config."""
        from homie_app.service.tray_menu import build_tray_menu
        cfg = MagicMock()
        cfg.voice.enabled = True
        cfg.voice.mode = "push_to_talk"
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = 2
        cfg.screen_reader.dnd = False
        cfg.notifications.dnd_schedule_enabled = False
        items = build_tray_menu(cfg)
        assert len(items) > 0
        labels = [i["label"] for i in items]
        assert "Open Chat" in labels
        assert "Stop Homie" in labels
