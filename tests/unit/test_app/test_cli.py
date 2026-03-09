from homie_core.config import HomieConfig
from homie_app.cli import create_parser, main


def test_parser_creation():
    parser = create_parser()
    assert parser is not None


def test_parse_model_list():
    parser = create_parser()
    args = parser.parse_args(["model", "list"])
    assert args.command == "model"
    assert args.model_command == "list"


def test_parse_plugin_enable():
    parser = create_parser()
    args = parser.parse_args(["plugin", "enable", "email"])
    assert args.command == "plugin"
    assert args.plugin_command == "enable"
    assert args.name == "email"


def test_parse_backup():
    parser = create_parser()
    args = parser.parse_args(["backup", "--to", "/tmp/backup"])
    assert args.command == "backup"
    assert args.backup_path == "/tmp/backup"


def test_parse_chat():
    parser = create_parser()
    args = parser.parse_args(["chat"])
    assert args.command == "chat"


def test_no_command_prints_help(capsys):
    main([])
    captured = capsys.readouterr()
    assert "homie" in captured.out.lower() or "usage" in captured.out.lower()
