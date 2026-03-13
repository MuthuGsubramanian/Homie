from unittest.mock import patch
from homie_app.cli import create_parser


class TestSettingsCommand:
    def test_parser_has_settings(self):
        parser = create_parser()
        args = parser.parse_args(["settings"])
        assert args.command == "settings"

    def test_parser_connect_still_works(self):
        """homie connect is deprecated but still parses."""
        parser = create_parser()
        args = parser.parse_args(["connect", "gmail"])
        assert args.command == "connect"
