from homie_core.intelligence.interruption_model import InterruptionModel


def test_initial_prediction_is_conservative():
    model = InterruptionModel()
    prob = model.predict(
        minutes_in_task=5, switch_freq_10min=2,
        minutes_since_interaction=10, category="health",
    )
    assert 0.3 <= prob <= 0.7


def test_should_interrupt_above_threshold():
    model = InterruptionModel(threshold=0.7)
    for _ in range(20):
        model.record_feedback(
            accepted=True, minutes_in_task=30, switch_freq_10min=5,
            minutes_since_interaction=60, category="health",
        )
    assert model.should_interrupt(
        minutes_in_task=30, switch_freq_10min=5,
        minutes_since_interaction=60, category="health",
    )


def test_should_not_interrupt_after_dismissals():
    model = InterruptionModel(threshold=0.7)
    for _ in range(20):
        model.record_feedback(
            accepted=False, minutes_in_task=5, switch_freq_10min=1,
            minutes_since_interaction=2, category="health",
        )
    assert not model.should_interrupt(
        minutes_in_task=5, switch_freq_10min=1,
        minutes_since_interaction=2, category="health",
    )


def test_serialize_and_restore():
    model = InterruptionModel()
    for _ in range(5):
        model.record_feedback(True, 10, 3, 20, "health")
    data = model.serialize()
    model2 = InterruptionModel.deserialize(data)
    p1 = model.predict(10, 3, 20, "health")
    p2 = model2.predict(10, 3, 20, "health")
    assert abs(p1 - p2) < 0.01


def test_category_encoding():
    model = InterruptionModel()
    for _ in range(15):
        model.record_feedback(True, 10, 3, 20, "health")
        model.record_feedback(False, 10, 3, 20, "calendar")
    p_health = model.predict(10, 3, 20, "health")
    p_calendar = model.predict(10, 3, 20, "calendar")
    assert p_health > p_calendar
