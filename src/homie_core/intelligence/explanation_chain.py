"""Traceable explanation chains for suggestion provenance.

Every suggestion Homie makes must be explainable. This module provides
a chain-of-reasoning structure that traces suggestions back to their
source signals, evidence data, and confidence scores. Users can ask
"Why did you suggest this?" and get a clear, multi-level answer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExplanationNode:
    """Single piece of evidence in an explanation chain."""
    source: str           # e.g., "flow_detector", "rhythm_model"
    claim: str            # what this evidence asserts
    evidence: dict        # supporting data
    confidence: float     # how confident this claim is (0-1)


class ExplanationChain:
    """Chain of reasoning linking a suggestion to its evidence.

    Provides short (one-line) and detailed (multi-line) explanations,
    plus an overall confidence score combining all evidence nodes.
    """

    def __init__(self):
        self.nodes: list[ExplanationNode] = []
        self.conclusion: Optional[str] = None

    def add_node(self, node: ExplanationNode) -> None:
        self.nodes.append(node)

    def set_conclusion(self, conclusion: str) -> None:
        self.conclusion = conclusion

    def overall_confidence(self) -> float:
        """Weighted average confidence across all evidence nodes."""
        if not self.nodes:
            return 0.0
        total = sum(n.confidence for n in self.nodes)
        return total / len(self.nodes)

    def get_sources(self) -> list[str]:
        return [n.source for n in self.nodes]

    def explain_short(self) -> str:
        """One-line explanation."""
        if not self.nodes:
            return ""
        claims = [n.claim for n in self.nodes]
        return "; ".join(claims)

    def explain_detailed(self) -> str:
        """Multi-line detailed explanation with evidence."""
        if not self.nodes:
            return ""
        lines = []
        if self.conclusion:
            lines.append(f"Conclusion: {self.conclusion}")
            lines.append("")
        lines.append("Evidence:")
        for i, node in enumerate(self.nodes, 1):
            lines.append(f"  {i}. [{node.source}] {node.claim} (confidence: {node.confidence:.0%})")
            for k, v in node.evidence.items():
                lines.append(f"     - {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "overall_confidence": self.overall_confidence(),
            "nodes": [
                {
                    "source": n.source,
                    "claim": n.claim,
                    "evidence": n.evidence,
                    "confidence": n.confidence,
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_evidence(
        cls,
        source: str,
        conclusion: str,
        evidence: dict,
        confidence: float,
    ) -> ExplanationChain:
        """Quick builder from a single evidence source."""
        chain = cls()
        chain.add_node(ExplanationNode(
            source=source,
            claim=conclusion,
            evidence=evidence,
            confidence=confidence,
        ))
        chain.set_conclusion(conclusion)
        return chain
