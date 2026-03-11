# Suggestion & Action Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge Homie's neural intelligence layer to user-visible suggestions, explanations, feedback loops, and proactive actions — making Homie a truly interactive intelligent assistant.

**Architecture:** A pipeline that flows from neural signals → suggestion generation → ranking/throttling → explanation tracing → delivery via overlay → feedback collection → online learning. Each component is modular and testable independently.

**Tech Stack:** Pure Python, integrates with existing ObserverLoop, WorkingMemory, and overlay UI.

---

## File Structure

### Suggestion Engine
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/suggestion_engine.py` | Core suggestion generation from neural signals |
| `src/homie_core/intelligence/suggestion_ranker.py` | Multi-signal ranking with Thompson sampling |
| `src/homie_core/intelligence/explanation_chain.py` | Traceable reasoning for every suggestion |
| `tests/unit/test_intelligence/test_suggestion_engine.py` | Tests |
| `tests/unit/test_intelligence/test_suggestion_ranker.py` | Tests |
| `tests/unit/test_intelligence/test_explanation_chain.py` | Tests |

### Feedback & Learning
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/feedback_loop.py` | Collects user responses, updates models |
| `tests/unit/test_intelligence/test_feedback_loop.py` | Tests |

### Orchestrator
| File | Responsibility |
|------|---------------|
| `src/homie_core/intelligence/action_pipeline.py` | End-to-end pipeline orchestrating all components |
| `tests/unit/test_intelligence/test_action_pipeline.py` | Tests |

---

## Task 1: Suggestion Engine

## Task 2: Thompson Sampling Suggestion Ranker

## Task 3: Explanation Chain with Provenance Tracking

## Task 4: Feedback Loop with Online Learning

## Task 5: Action Pipeline Orchestrator

## Task 6: Integration with ObserverLoop
