# tests/unit/self_healing/test_metrics.py
import pytest
from homie_core.self_healing.metrics import MetricsCollector, AnomalyAlert


class TestMetricsCollector:
    def test_record_and_get_latest(self):
        mc = MetricsCollector(window_size=100)
        mc.record("inference", "latency_ms", 42.0)
        mc.record("inference", "latency_ms", 55.0)
        assert mc.get_latest("inference", "latency_ms") == 55.0

    def test_get_average(self):
        mc = MetricsCollector(window_size=100)
        mc.record("storage", "query_ms", 10.0)
        mc.record("storage", "query_ms", 20.0)
        mc.record("storage", "query_ms", 30.0)
        assert mc.get_average("storage", "query_ms") == pytest.approx(20.0)

    def test_window_size_respected(self):
        mc = MetricsCollector(window_size=3)
        for v in [10, 20, 30, 40, 50]:
            mc.record("m", "v", float(v))
        # Only last 3 values retained
        assert mc.get_average("m", "v") == pytest.approx(40.0)

    def test_unknown_metric_returns_none(self):
        mc = MetricsCollector(window_size=100)
        assert mc.get_latest("unknown", "metric") is None
        assert mc.get_average("unknown", "metric") is None

    def test_detect_anomaly_spike(self):
        mc = MetricsCollector(window_size=100, anomaly_std_threshold=2.0)
        # Establish baseline
        for _ in range(20):
            mc.record("inference", "latency_ms", 50.0)
        # Spike
        alert = mc.record("inference", "latency_ms", 200.0)
        assert alert is not None
        assert alert.metric_name == "latency_ms"
        assert alert.module == "inference"

    def test_no_anomaly_for_normal_values(self):
        mc = MetricsCollector(window_size=100, anomaly_std_threshold=2.0)
        for _ in range(20):
            alert = mc.record("m", "v", 50.0)
        assert alert is None  # last record returns no anomaly

    def test_snapshot(self):
        mc = MetricsCollector(window_size=100)
        mc.record("inference", "latency_ms", 42.0)
        mc.record("storage", "query_ms", 10.0)
        snap = mc.snapshot()
        assert "inference" in snap
        assert "storage" in snap
        assert "latency_ms" in snap["inference"]
