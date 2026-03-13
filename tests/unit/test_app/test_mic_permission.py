from unittest.mock import patch, MagicMock
from homie_app.mic_permission import request_microphone_access


class TestMicPermission:
    @patch("homie_app.mic_permission.sd")
    def test_mic_already_available(self, mock_sd):
        mock_sd.query_devices.return_value = [{"max_input_channels": 1}]
        result = request_microphone_access()
        assert result is True

    @patch("homie_app.mic_permission.sd")
    def test_no_mic_at_all(self, mock_sd):
        mock_sd.query_devices.return_value = [{"max_input_channels": 0}]
        mock_sd.InputStream.side_effect = Exception("No mic")
        result = request_microphone_access(interactive=False)
        assert result is False

    @patch("homie_app.mic_permission.sd")
    def test_triggers_os_prompt(self, mock_sd):
        # First call: no mic. Second call (after OS prompt): mic found
        mock_sd.query_devices.side_effect = [
            [{"max_input_channels": 0}],
            [{"max_input_channels": 1}],
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        result = request_microphone_access(interactive=False)
        mock_sd.InputStream.assert_called()  # Triggered OS prompt
