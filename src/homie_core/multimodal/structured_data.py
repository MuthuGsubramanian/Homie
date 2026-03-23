from __future__ import annotations

import csv
import io
import json
import logging
import statistics
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StructuredDataAnalyzer:
    """Analyzes spreadsheets, CSVs, databases, and JSON — stdlib only."""

    def __init__(self, inference_fn: Optional[Callable[..., str]] = None):
        self._infer = inference_fn

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def analyze_csv(self, path: str) -> dict:
        """Analyze a CSV file: columns, types, row count, and basic statistics.

        Returns a dict with ``columns``, ``row_count``, ``column_types``,
        ``sample_rows``, and ``statistics`` (for numeric columns).
        """
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows: list[dict[str, str]] = list(reader)

        if not rows:
            return {"columns": [], "row_count": 0, "column_types": {}, "sample_rows": [], "statistics": {}}

        columns = list(rows[0].keys())
        col_types = self._infer_column_types(rows, columns)
        stats = self._compute_statistics(rows, columns, col_types)

        return {
            "columns": columns,
            "row_count": len(rows),
            "column_types": col_types,
            "sample_rows": rows[:5],
            "statistics": stats,
        }

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def analyze_json(self, data: dict | list) -> dict:
        """Analyze a JSON structure: type, schema overview, size, nesting depth."""
        result: dict[str, Any] = {
            "type": type(data).__name__,
            "size": len(data) if isinstance(data, (list, dict)) else 1,
            "depth": self._json_depth(data),
            "schema": self._json_schema(data),
        }

        if isinstance(data, list) and data and isinstance(data[0], dict):
            result["columns"] = list(data[0].keys())
            result["sample_rows"] = data[:5]

        return result

    # ------------------------------------------------------------------
    # Natural-language queries
    # ------------------------------------------------------------------

    def query_data(self, data: list[dict], question: str) -> str:
        """Answer a natural language question about tabular data.

        Uses the LLM when available; otherwise falls back to simple
        keyword-based heuristics.
        """
        if self._infer is not None:
            try:
                snippet = json.dumps(data[:20], default=str, indent=2)
                prompt = (
                    f"Given this data:\n```json\n{snippet}\n```\n"
                    f"Answer the question: {question}"
                )
                return self._infer(prompt)
            except Exception:
                logger.debug("LLM query_data failed, using fallback", exc_info=True)

        return self._heuristic_query(data, question)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def generate_summary_stats(self, data: list[dict]) -> dict:
        """Generate a statistical summary of tabular data (list of dicts)."""
        if not data:
            return {}

        columns = list(data[0].keys())
        col_types = self._infer_column_types(data, columns)
        return self._compute_statistics(data, columns, col_types)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_column_types(rows: list[dict], columns: list[str]) -> dict[str, str]:
        """Infer column types by inspecting values."""
        col_types: dict[str, str] = {}
        for col in columns:
            sample_values = [r.get(col, "") for r in rows[:100] if r.get(col, "")]
            if not sample_values:
                col_types[col] = "empty"
                continue

            numeric_count = 0
            int_count = 0
            for v in sample_values:
                v_stripped = str(v).strip()
                try:
                    float(v_stripped)
                    numeric_count += 1
                    if v_stripped.isdigit() or (v_stripped.startswith("-") and v_stripped[1:].isdigit()):
                        int_count += 1
                except (ValueError, TypeError):
                    pass

            ratio = numeric_count / len(sample_values)
            if ratio > 0.8:
                col_types[col] = "integer" if int_count == numeric_count else "float"
            else:
                col_types[col] = "string"

        return col_types

    @staticmethod
    def _compute_statistics(
        rows: list[dict], columns: list[str], col_types: dict[str, str]
    ) -> dict[str, dict]:
        """Compute mean/median/stdev/min/max for numeric columns."""
        stats: dict[str, dict] = {}
        for col in columns:
            if col_types.get(col) not in ("integer", "float"):
                continue
            values: list[float] = []
            for r in rows:
                try:
                    values.append(float(r[col]))
                except (ValueError, TypeError, KeyError):
                    pass
            if not values:
                continue
            stats[col] = {
                "count": len(values),
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": min(values),
                "max": max(values),
            }
            if len(values) >= 2:
                stats[col]["stdev"] = round(statistics.stdev(values), 4)
        return stats

    @staticmethod
    def _json_depth(obj: Any, current: int = 1) -> int:
        if isinstance(obj, dict):
            if not obj:
                return current
            return max(
                StructuredDataAnalyzer._json_depth(v, current + 1) for v in obj.values()
            )
        if isinstance(obj, list):
            if not obj:
                return current
            return max(
                StructuredDataAnalyzer._json_depth(v, current + 1) for v in obj
            )
        return current

    @staticmethod
    def _json_schema(obj: Any, max_depth: int = 4, depth: int = 0) -> Any:
        """Return a simplified schema description."""
        if depth >= max_depth:
            return "..."
        if isinstance(obj, dict):
            return {k: StructuredDataAnalyzer._json_schema(v, max_depth, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if not obj:
                return ["empty"]
            return [StructuredDataAnalyzer._json_schema(obj[0], max_depth, depth + 1)]
        return type(obj).__name__

    @staticmethod
    def _heuristic_query(data: list[dict], question: str) -> str:
        """Simple keyword-based query fallback."""
        if not data:
            return "No data provided."

        q = question.lower()
        columns = list(data[0].keys())

        # "how many" / "count" -> row count
        if any(kw in q for kw in ("how many", "count", "total rows", "number of")):
            return f"The dataset has {len(data)} rows."

        # "average" / "mean" of a column
        for col in columns:
            if col.lower() in q and any(kw in q for kw in ("average", "mean")):
                try:
                    vals = [float(r[col]) for r in data if r.get(col)]
                    return f"The average {col} is {round(statistics.mean(vals), 4)}."
                except (ValueError, TypeError):
                    return f"Could not compute average for column '{col}'."

        # "max" / "highest" of a column
        for col in columns:
            if col.lower() in q and any(kw in q for kw in ("max", "highest", "largest", "biggest")):
                try:
                    vals = [float(r[col]) for r in data if r.get(col)]
                    return f"The maximum {col} is {max(vals)}."
                except (ValueError, TypeError):
                    return f"Could not compute max for column '{col}'."

        # "min" / "lowest"
        for col in columns:
            if col.lower() in q and any(kw in q for kw in ("min", "lowest", "smallest")):
                try:
                    vals = [float(r[col]) for r in data if r.get(col)]
                    return f"The minimum {col} is {min(vals)}."
                except (ValueError, TypeError):
                    return f"Could not compute min for column '{col}'."

        # Columns listing
        if "column" in q or "field" in q:
            return f"Columns: {', '.join(columns)}"

        return f"The dataset has {len(data)} rows and columns: {', '.join(columns)}."
