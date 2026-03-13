import queue
import struct
import threading
from unittest.mock import MagicMock, patch

from homie_core.voice.audio_io import AudioInThread, AudioOutThread


def test_audio_in_pushes_chunks():
    out_q = queue.Queue()
    stop = threading.Event()
    fake_chunk = struct.pack("<512h", *([1000] * 512))
    call_count = 0

    def read_with_stop(*a, **kw):
        nonlocal call_count
        if call_count >= 2:
            stop.set()
            raise Exception("stopped")
        call_count += 1
        return (fake_chunk, False)

    mock_stream = MagicMock()
    mock_stream.read.side_effect = read_with_stop
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("homie_core.voice.audio_io.sd") as mock_sd:
        mock_sd.RawInputStream.return_value = mock_stream
        audio_in = AudioInThread(out_q, stop, sample_rate=16000, chunk_size=512)
        t = threading.Thread(target=audio_in.run)
        t.start()
        t.join(timeout=3)

    assert out_q.qsize() >= 2


def test_audio_out_plays():
    in_q = queue.Queue()
    stop = threading.Event()
    fake_chunk = struct.pack("<512h", *([500] * 512))
    in_q.put(fake_chunk)

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    def stop_after_write(*a, **kw):
        stop.set()

    mock_stream.write.side_effect = stop_after_write

    with patch("homie_core.voice.audio_io.sd") as mock_sd:
        mock_sd.RawOutputStream.return_value = mock_stream
        audio_out = AudioOutThread(in_q, stop, sample_rate=16000, chunk_size=512)
        t = threading.Thread(target=audio_out.run)
        t.start()
        t.join(timeout=3)

    mock_stream.write.assert_called_once_with(fake_chunk)
