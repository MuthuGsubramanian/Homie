# tests/unit/self_optimizer/test_pipeline_gate.py
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.pipeline_gate import PipelineGate


class TestPipelineGate:
    def test_trivial_overrides_to_trivial(self):
        gate = PipelineGate()
        result = gate.apply("trivial")
        assert result == "trivial"

    def test_complex_stays_complex(self):
        gate = PipelineGate()
        result = gate.apply("complex")
        assert result == "complex"

    def test_promotes_after_clarifications(self):
        gate = PipelineGate(promotion_threshold=2)
        gate.record_clarification("trivial")
        gate.record_clarification("trivial")
        # After 2 clarifications, trivial should be promoted
        result = gate.apply("trivial")
        assert result == "simple"  # promoted one tier

    def test_no_promote_below_threshold(self):
        gate = PipelineGate(promotion_threshold=3)
        gate.record_clarification("trivial")
        result = gate.apply("trivial")
        assert result == "trivial"  # not enough clarifications yet

    def test_double_promotion(self):
        gate = PipelineGate(promotion_threshold=2)
        # Promote trivial → simple
        gate.record_clarification("trivial")
        gate.record_clarification("trivial")
        # Then promote simple → moderate
        gate.record_clarification("simple")
        gate.record_clarification("simple")
        result = gate.apply("simple")
        assert result == "moderate"

    def test_deep_never_promotes(self):
        gate = PipelineGate(promotion_threshold=1)
        gate.record_clarification("deep")
        result = gate.apply("deep")
        assert result == "deep"  # can't promote beyond deep

    def test_get_skip_report(self):
        gate = PipelineGate()
        gate.apply("trivial")
        report = gate.get_stats()
        assert "trivial" in report
