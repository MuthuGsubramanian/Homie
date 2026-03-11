from homie_core.intelligence.action_pipeline import ActionPipeline, PipelineConfig


def test_create_pipeline():
    pipeline = ActionPipeline()
    assert pipeline is not None


def test_process_context_generates_suggestions():
    pipeline = ActionPipeline()
    context = {
        "activity_type": "coding",
        "flow_score": 0.2,
        "minutes_in_task": 150,
        "in_flow": False,
    }
    result = pipeline.process(context)
    assert "suggestions" in result
    assert len(result["suggestions"]) > 0


def test_process_includes_explanations():
    pipeline = ActionPipeline()
    context = {
        "activity_type": "coding",
        "flow_score": 0.2,
        "minutes_in_task": 150,
        "in_flow": False,
    }
    result = pipeline.process(context)
    for sug in result["suggestions"]:
        assert "explanation" in sug
        assert "short" in sug["explanation"]


def test_process_ranks_suggestions():
    pipeline = ActionPipeline()
    context = {
        "activity_type": "coding",
        "flow_score": 0.2,
        "minutes_in_task": 150,
        "in_flow": False,
        "predicted_next": [("testing", 0.8)],
        "anomaly_score": 0.85,
    }
    result = pipeline.process(context)
    # Should have multiple suggestions, ranked
    assert len(result["suggestions"]) >= 2


def test_record_feedback():
    pipeline = ActionPipeline()
    pipeline.record_feedback("sug_001", "break", accepted=True)
    summary = pipeline.get_feedback_summary()
    assert summary["total"] == 1


def test_empty_context():
    pipeline = ActionPipeline()
    result = pipeline.process({})
    assert result["suggestions"] == []


def test_max_suggestions_config():
    pipeline = ActionPipeline(config=PipelineConfig(max_suggestions=1))
    context = {
        "activity_type": "coding",
        "flow_score": 0.2,
        "minutes_in_task": 150,
        "in_flow": False,
        "predicted_next": [("testing", 0.8)],
    }
    result = pipeline.process(context)
    assert len(result["suggestions"]) <= 1


def test_pipeline_state_persists():
    pipeline = ActionPipeline()
    pipeline.record_feedback("s1", "break", accepted=True)
    pipeline.record_feedback("s2", "break", accepted=True)
    data = pipeline.serialize()
    pipeline2 = ActionPipeline.deserialize(data)
    summary = pipeline2.get_feedback_summary()
    assert summary["total"] == 2


def test_get_intelligence_report():
    pipeline = ActionPipeline()
    context = {
        "activity_type": "coding",
        "flow_score": 0.8,
        "minutes_in_task": 30,
        "in_flow": True,
    }
    pipeline.process(context)
    report = pipeline.get_intelligence_report(context)
    assert "activity" in report
    assert "flow" in report
