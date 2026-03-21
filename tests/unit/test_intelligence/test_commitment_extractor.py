"""Tests for commitment_extractor module."""
from __future__ import annotations

import pytest
from homie_core.intelligence.commitment_extractor import extract_commitments, Commitment


class TestExtractCommitments:
    def test_ill_reply_by_tomorrow(self):
        results = extract_commitments("I'll reply by tomorrow")
        assert len(results) >= 1
        c = results[0]
        assert "reply" in c.text.lower()
        assert c.due_by == "tomorrow"

    def test_i_will_finish_pr_by_friday(self):
        text = "I will finish the PR review by Friday"
        results = extract_commitments(text)
        assert len(results) >= 1
        assert any("PR review" in r.text or "finish" in r.text.lower() for r in results)
        assert any(r.due_by and "friday" in r.due_by.lower() for r in results)

    def test_remind_me_to_buy_groceries(self):
        results = extract_commitments("remind me to buy groceries")
        assert len(results) >= 1
        assert any("buy groceries" in r.text.lower() for r in results)

    def test_dont_let_me_forget(self):
        results = extract_commitments("don't let me forget to send the invoice")
        assert len(results) >= 1
        assert any("send the invoice" in r.text.lower() for r in results)

    def test_todo_colon_finish_pr_review(self):
        results = extract_commitments("todo: finish the PR review")
        assert len(results) >= 1
        assert any("finish the PR review" in r.text for r in results)

    def test_action_item_prefix(self):
        results = extract_commitments("action item: update the deployment docs")
        assert len(results) >= 1
        assert any("update the deployment docs" in r.text for r in results)

    def test_short_text_no_commitment(self):
        # A text that is too short to be a meaningful commitment
        results = extract_commitments("ok")
        assert results == []

    def test_plain_statement_no_commitment(self):
        results = extract_commitments("The weather is nice today.")
        assert results == []

    def test_multiple_commitments_in_one_text(self):
        text = (
            "I'll send the report by end of day. "
            "remind me to call the client. "
            "todo: update the changelog"
        )
        results = extract_commitments(text)
        assert len(results) >= 3

    def test_source_is_passed_through(self):
        results = extract_commitments("I'll finish the draft by tomorrow", source="slack")
        assert all(r.source == "slack" for r in results)

    def test_commitment_dataclass_defaults(self):
        c = Commitment(text="finish something")
        assert c.source == "conversation"
        assert c.confidence == 0.7
        assert c.due_by is None

    def test_i_need_to_pattern(self):
        results = extract_commitments("I need to update the config file by Friday")
        assert len(results) >= 1

    def test_i_have_to_pattern(self):
        results = extract_commitments("I have to submit the form before noon")
        assert len(results) >= 1

    def test_im_going_to_pattern(self):
        results = extract_commitments("I'm going to deploy the fix by end of week")
        assert len(results) >= 1
