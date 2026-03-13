from homie_core.voice.voice_prompts import get_voice_hint, VOICE_SYSTEM_HINT

def test_voice_hint_is_string():
    assert isinstance(VOICE_SYSTEM_HINT, str)
    assert "voice" in VOICE_SYSTEM_HINT.lower()

def test_get_voice_hint():
    hint = get_voice_hint()
    assert "concise" in hint.lower()
    assert "markdown" in hint.lower()
