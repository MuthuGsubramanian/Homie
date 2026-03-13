from homie_core.intelligence.explanation_chain import ExplanationChain, ExplanationNode


def test_create_node():
    node = ExplanationNode(
        source="flow_detector",
        claim="User focus is declining",
        evidence={"flow_score": 0.3, "minutes_in_task": 120},
        confidence=0.8,
    )
    assert node.source == "flow_detector"
    assert node.confidence == 0.8


def test_build_chain():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(
        source="flow_detector",
        claim="Focus is declining",
        evidence={"flow_score": 0.3},
        confidence=0.8,
    ))
    chain.add_node(ExplanationNode(
        source="rhythm_model",
        claim="Past peak productivity window",
        evidence={"current_hour": 16, "peak_hour": 10},
        confidence=0.7,
    ))
    chain.set_conclusion("Take a break — you're past peak focus and your flow score is low.")
    assert len(chain.nodes) == 2
    assert chain.conclusion is not None


def test_explain_short():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(
        source="flow_detector",
        claim="Focus is declining",
        evidence={"flow_score": 0.3},
        confidence=0.8,
    ))
    chain.set_conclusion("Take a break.")
    short = chain.explain_short()
    assert "flow_detector" in short or "Focus" in short


def test_explain_detailed():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(
        source="flow_detector",
        claim="Focus declining",
        evidence={"flow_score": 0.3},
        confidence=0.8,
    ))
    chain.add_node(ExplanationNode(
        source="rhythm_model",
        claim="Past peak",
        evidence={"current_hour": 16},
        confidence=0.7,
    ))
    chain.set_conclusion("Take a break.")
    detailed = chain.explain_detailed()
    assert "flow_detector" in detailed
    assert "rhythm_model" in detailed
    assert "0.3" in detailed or "flow_score" in detailed


def test_chain_confidence():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(
        source="a", claim="c1", evidence={}, confidence=0.8
    ))
    chain.add_node(ExplanationNode(
        source="b", claim="c2", evidence={}, confidence=0.6
    ))
    # Overall confidence should reflect combined evidence
    overall = chain.overall_confidence()
    assert 0.0 < overall < 1.0
    # Should be between min and max of individual confidences
    assert 0.6 <= overall <= 0.8


def test_empty_chain():
    chain = ExplanationChain()
    assert chain.explain_short() == ""
    assert chain.explain_detailed() == ""
    assert chain.overall_confidence() == 0.0


def test_get_sources():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(source="flow_detector", claim="a", evidence={}, confidence=0.5))
    chain.add_node(ExplanationNode(source="rhythm_model", claim="b", evidence={}, confidence=0.6))
    sources = chain.get_sources()
    assert "flow_detector" in sources
    assert "rhythm_model" in sources


def test_to_dict():
    chain = ExplanationChain()
    chain.add_node(ExplanationNode(
        source="flow_detector",
        claim="Focus declining",
        evidence={"flow_score": 0.3},
        confidence=0.8,
    ))
    chain.set_conclusion("Take a break.")
    d = chain.to_dict()
    assert "nodes" in d
    assert "conclusion" in d
    assert d["conclusion"] == "Take a break."
    assert len(d["nodes"]) == 1


def test_from_suggestion_evidence():
    """Build explanation chain from suggestion evidence dict."""
    chain = ExplanationChain.from_evidence(
        source="flow_detector",
        conclusion="Take a break",
        evidence={"flow_score": 0.3, "minutes_in_task": 120},
        confidence=0.8,
    )
    assert len(chain.nodes) == 1
    assert chain.conclusion == "Take a break"
