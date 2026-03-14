"""Tests for Qubrid cloud inference client."""
from unittest.mock import patch, MagicMock
import json
import pytest

from homie_core.inference.qubrid import QubridClient


def test_qubrid_client_init():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=30,
    )
    assert client.model == "Qwen/Qwen3.5-Flash"
    assert client.is_available is False


def test_qubrid_generate_success():
    client = QubridClient(api_key="test-key", model="Qwen/Qwen3.5-Flash")
    mock_response = {"choices": [{"message": {"content": "Hello!"}}]}
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = client.generate("Hi", max_tokens=100)
    assert result == "Hello!"


def test_qubrid_generate_network_error():
    client = QubridClient(api_key="test-key", model="Qwen/Qwen3.5-Flash")
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("timeout")
        with pytest.raises(ConnectionError):
            client.generate("Hi")


def test_qubrid_generate_api_error():
    client = QubridClient(api_key="test-key", model="Qwen/Qwen3.5-Flash")
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        from urllib.error import HTTPError
        from io import BytesIO
        error = HTTPError(
            url="https://platform.qubrid.com/v1/chat/completions",
            code=429, msg="Too Many Requests", hdrs=MagicMock(),
            fp=BytesIO(json.dumps({"error": {"message": "Rate limited"}}).encode()),
        )
        mock_urlopen.side_effect = error
        with pytest.raises(RuntimeError, match="Rate limited"):
            client.generate("Hi")


def test_qubrid_check_available_success():
    client = QubridClient(api_key="test-key", model="Qwen/Qwen3.5-Flash")
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert client.check_available() is True
        assert client.is_available is True


def test_qubrid_check_available_unreachable():
    client = QubridClient(api_key="test-key", model="Qwen/Qwen3.5-Flash")
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("unreachable")
        assert client.check_available() is False
        assert client.is_available is False
