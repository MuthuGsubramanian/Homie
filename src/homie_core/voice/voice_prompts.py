from __future__ import annotations

VOICE_SYSTEM_HINT = (
    "User is speaking via voice. Keep responses concise and conversational. "
    "Avoid markdown, code blocks, or visual formatting — the response will be read aloud."
)
EXIT_CONFIRMATION = "Would you like to end our conversation?"
EXIT_AUTO_MESSAGE = "Ending the conversation since you seem to be away. Talk to you later!"

def get_voice_hint() -> str:
    return VOICE_SYSTEM_HINT
