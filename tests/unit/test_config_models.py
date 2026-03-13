import pytest
from homie_core.config import UserProfileConfig, HomieConfig


class TestUserProfileConfig:
    def test_defaults(self):
        cfg = UserProfileConfig()
        assert cfg.name == "Master"
        assert cfg.language == "en"
        assert cfg.timezone == "auto"
        assert cfg.work_hours_start == "09:00"
        assert cfg.work_hours_end == "18:00"
        assert cfg.work_days == ["mon", "tue", "wed", "thu", "fri"]

    def test_custom_values(self):
        cfg = UserProfileConfig(
            name="Muthu",
            language="ta",
            timezone="Asia/Kolkata",
            work_hours_start="10:00",
            work_hours_end="19:00",
            work_days=["mon", "tue", "wed", "thu"],
        )
        assert cfg.name == "Muthu"
        assert cfg.timezone == "Asia/Kolkata"

    def test_homie_config_includes_user(self):
        cfg = HomieConfig()
        assert hasattr(cfg, "user")
        assert cfg.user.name == "Master"
