ONBOARDING_QUESTIONS = [
    {"question": "What's your name?", "field": "identity.name", "type": "open"},
    {"question": "What do you do for work?", "field": "work.role", "type": "open"},
    {"question": "What programming languages do you use most?", "field": "work.languages", "type": "open"},
    {"question": "What IDE or editor do you prefer?", "field": "work.ide", "type": "choice",
     "choices": ["VS Code", "PyCharm", "IntelliJ", "Vim/Neovim", "Other"]},
    {"question": "What's your typical work schedule?", "field": "routine.work_hours", "type": "open"},
    {"question": "How often would you like break reminders?", "field": "health.break_preference", "type": "choice",
     "choices": ["Every 30 min", "Every 60 min", "Every 90 min", "No reminders"]},
    {"question": "What kind of music do you enjoy while working?", "field": "music.work_genre", "type": "open"},
    {"question": "What are your current learning goals?", "field": "interests.learning_goals", "type": "open"},
]


def get_onboarding_prompt(question_index: int) -> dict | None:
    if question_index >= len(ONBOARDING_QUESTIONS):
        return None
    return ONBOARDING_QUESTIONS[question_index]
