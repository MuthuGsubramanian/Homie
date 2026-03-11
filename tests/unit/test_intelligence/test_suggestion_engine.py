from homie_core.intelligence.suggestion_engine import SuggestionEngine, Suggestion, SuggestionType


def test_create_suggestion():
    s = Suggestion(
        type=SuggestionType.BREAK,
        title="Take a break",
        body="You've been coding for 2 hours straight.",
        confidence=0.8,
        source="flow_detector",
        evidence={"hours_worked": 2.0, "flow_score": 0.3},
    )
    assert s.type == SuggestionType.BREAK
    assert s.confidence == 0.8


def test_generate_break_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "flow_score": 0.3,
        "minutes_in_task": 120,
        "in_flow": False,
    }
    suggestions = engine.generate(context)
    break_suggestions = [s for s in suggestions if s.type == SuggestionType.BREAK]
    assert len(break_suggestions) > 0


def test_no_break_during_flow():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "flow_score": 0.9,
        "minutes_in_task": 120,
        "in_flow": True,
    }
    suggestions = engine.generate(context)
    break_suggestions = [s for s in suggestions if s.type == SuggestionType.BREAK]
    assert len(break_suggestions) == 0


def test_generate_workflow_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "predicted_next": [("testing", 0.8), ("committing", 0.6)],
    }
    suggestions = engine.generate(context)
    workflow_sugs = [s for s in suggestions if s.type == SuggestionType.WORKFLOW]
    assert len(workflow_sugs) > 0


def test_generate_anomaly_alert():
    engine = SuggestionEngine()
    context = {
        "activity_type": "browsing",
        "anomaly_score": 0.85,
    }
    suggestions = engine.generate(context)
    anomaly_sugs = [s for s in suggestions if s.type == SuggestionType.ANOMALY]
    assert len(anomaly_sugs) > 0


def test_no_anomaly_for_normal():
    engine = SuggestionEngine()
    context = {"activity_type": "coding", "anomaly_score": 0.3}
    suggestions = engine.generate(context)
    anomaly_sugs = [s for s in suggestions if s.type == SuggestionType.ANOMALY]
    assert len(anomaly_sugs) == 0


def test_generate_rhythm_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "optimal_windows": [{"hour": 10, "predicted_score": 0.9}],
        "current_hour": 10,
    }
    suggestions = engine.generate(context)
    rhythm_sugs = [s for s in suggestions if s.type == SuggestionType.RHYTHM]
    assert len(rhythm_sugs) > 0


def test_preference_shift_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "preference_shifts": [{"domain": "tool", "key": "vim", "new_value": 0.9}],
    }
    suggestions = engine.generate(context)
    pref_sugs = [s for s in suggestions if s.type == SuggestionType.INSIGHT]
    assert len(pref_sugs) > 0


def test_context_switch_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "email",
        "context_shift": True,
        "previous_activity": "coding",
    }
    suggestions = engine.generate(context)
    ctx_sugs = [s for s in suggestions if s.type == SuggestionType.CONTEXT]
    assert len(ctx_sugs) > 0


def test_stuck_task_suggestion():
    engine = SuggestionEngine()
    context = {
        "activity_type": "coding",
        "stuck_tasks": [{"id": "t1", "apps": ["Code.exe"], "duration_minutes": 45}],
    }
    suggestions = engine.generate(context)
    stuck_sugs = [s for s in suggestions if s.type == SuggestionType.HELP]
    assert len(stuck_sugs) > 0


def test_empty_context_returns_empty():
    engine = SuggestionEngine()
    suggestions = engine.generate({})
    assert suggestions == []


def test_throttle_cooldown():
    engine = SuggestionEngine(cooldown_seconds=60)
    context = {
        "activity_type": "coding",
        "flow_score": 0.2,
        "minutes_in_task": 150,
        "in_flow": False,
    }
    s1 = engine.generate(context)
    s2 = engine.generate(context)
    # Second call within cooldown should return fewer or no suggestions of same type
    break1 = [s for s in s1 if s.type == SuggestionType.BREAK]
    break2 = [s for s in s2 if s.type == SuggestionType.BREAK]
    if break1:
        assert len(break2) < len(break1)
