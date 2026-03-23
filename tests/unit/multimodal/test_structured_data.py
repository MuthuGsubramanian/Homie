from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from homie_core.multimodal.structured_data import StructuredDataAnalyzer


@pytest.fixture
def csv_file(tmp_path):
    """Create a simple CSV file."""
    p = tmp_path / "data.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age", "score"])
        writer.writerow(["Alice", "30", "88.5"])
        writer.writerow(["Bob", "25", "92.0"])
        writer.writerow(["Charlie", "35", "76.3"])
        writer.writerow(["Diana", "28", "95.1"])
    return str(p)


@pytest.fixture
def empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,age,score\n")
    return str(p)


@pytest.fixture
def sample_data():
    return [
        {"name": "Alice", "age": "30", "score": "88.5"},
        {"name": "Bob", "age": "25", "score": "92.0"},
        {"name": "Charlie", "age": "35", "score": "76.3"},
        {"name": "Diana", "age": "28", "score": "95.1"},
    ]


class TestStructuredDataAnalyzer:
    def test_init_no_inference(self):
        analyzer = StructuredDataAnalyzer()
        assert analyzer._infer is None

    def test_analyze_csv_basic(self, csv_file):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.analyze_csv(csv_file)
        assert result["columns"] == ["name", "age", "score"]
        assert result["row_count"] == 4
        assert len(result["sample_rows"]) == 4

    def test_analyze_csv_types(self, csv_file):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.analyze_csv(csv_file)
        assert result["column_types"]["name"] == "string"
        assert result["column_types"]["age"] in ("integer", "float")
        assert result["column_types"]["score"] == "float"

    def test_analyze_csv_statistics(self, csv_file):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.analyze_csv(csv_file)
        stats = result["statistics"]
        assert "age" in stats
        assert stats["age"]["mean"] == 29.5
        assert stats["age"]["min"] == 25.0
        assert stats["age"]["max"] == 35.0

    def test_analyze_csv_empty(self, empty_csv):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.analyze_csv(empty_csv)
        assert result["row_count"] == 0

    def test_analyze_json_dict(self):
        analyzer = StructuredDataAnalyzer()
        data = {"a": 1, "b": {"c": 2}}
        result = analyzer.analyze_json(data)
        assert result["type"] == "dict"
        assert result["size"] == 2
        assert result["depth"] >= 2

    def test_analyze_json_list_of_dicts(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.analyze_json(sample_data)
        assert result["type"] == "list"
        assert result["size"] == 4
        assert "columns" in result
        assert result["columns"] == ["name", "age", "score"]

    def test_analyze_json_nested_depth(self):
        analyzer = StructuredDataAnalyzer()
        data = {"a": {"b": {"c": {"d": 1}}}}
        result = analyzer.analyze_json(data)
        assert result["depth"] == 5  # top-level dict + 3 nested dicts + leaf

    def test_query_data_count(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data(sample_data, "How many rows are there?")
        assert "4" in result

    def test_query_data_average(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data(sample_data, "What is the average age?")
        assert "29.5" in result

    def test_query_data_max(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data(sample_data, "What is the max score?")
        assert "95.1" in result

    def test_query_data_min(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data(sample_data, "What is the min age?")
        assert "25" in result

    def test_query_data_columns(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data(sample_data, "What columns are there?")
        assert "name" in result
        assert "age" in result

    def test_query_data_empty(self):
        analyzer = StructuredDataAnalyzer()
        result = analyzer.query_data([], "How many?")
        assert "No data" in result

    def test_query_data_with_inference(self, sample_data):
        fn = lambda prompt: "The average age is 29.5"
        analyzer = StructuredDataAnalyzer(inference_fn=fn)
        result = analyzer.query_data(sample_data, "average age?")
        assert "29.5" in result

    def test_generate_summary_stats(self, sample_data):
        analyzer = StructuredDataAnalyzer()
        stats = analyzer.generate_summary_stats(sample_data)
        assert "age" in stats
        assert "score" in stats
        assert stats["age"]["count"] == 4

    def test_generate_summary_stats_empty(self):
        analyzer = StructuredDataAnalyzer()
        assert analyzer.generate_summary_stats([]) == {}
