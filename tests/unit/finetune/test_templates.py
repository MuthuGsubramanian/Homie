"""Tests for domain scenario templates."""

import pytest

from homie_core.finetune.synthetic.templates import (
    DOMAIN_TEMPLATES,
    Domain,
    ScenarioTemplate,
    get_all_templates,
    get_templates_for_domain,
)


class TestDomainTemplates:
    def test_six_domains(self):
        assert len(Domain) == 6

    def test_total_template_count(self):
        all_t = get_all_templates()
        assert len(all_t) >= 300

    def test_each_domain_has_50_templates(self):
        for domain in Domain:
            templates = get_templates_for_domain(domain)
            assert len(templates) >= 50, f"{domain.value} has only {len(templates)}"

    def test_template_structure(self):
        t = get_all_templates()[0]
        assert isinstance(t, ScenarioTemplate)
        assert t.domain in Domain
        assert t.name
        assert t.system_context
        assert t.user_prompt_template
        assert t.good_behavior
        assert t.bad_behavior
        assert t.difficulty in (1, 2, 3, 4)

    def test_difficulty_distribution(self):
        for domain in Domain:
            templates = get_templates_for_domain(domain)
            tiers = {t.difficulty for t in templates}
            assert 1 in tiers, f"{domain.value} missing tier 1"
            assert 2 in tiers, f"{domain.value} missing tier 2"

    def test_difficulty_tier_counts(self):
        for domain in Domain:
            templates = get_templates_for_domain(domain)
            counts = {}
            for t in templates:
                counts[t.difficulty] = counts.get(t.difficulty, 0) + 1
            assert counts.get(1, 0) >= 20, f"{domain.value} tier 1: {counts.get(1, 0)}"
            assert counts.get(2, 0) >= 15, f"{domain.value} tier 2: {counts.get(2, 0)}"
            assert counts.get(3, 0) >= 10, f"{domain.value} tier 3: {counts.get(3, 0)}"
            assert counts.get(4, 0) >= 5, f"{domain.value} tier 4: {counts.get(4, 0)}"

    def test_proactive_templates_exist(self):
        all_t = get_all_templates()
        proactive = [t for t in all_t if t.proactive]
        assert len(proactive) >= 30

    def test_filter_by_max_difficulty(self):
        for domain in Domain:
            easy = get_templates_for_domain(domain, max_difficulty=1)
            assert all(t.difficulty == 1 for t in easy)
            mid = get_templates_for_domain(domain, max_difficulty=2)
            assert all(t.difficulty <= 2 for t in mid)

    def test_all_domains_present_in_dict(self):
        for domain in Domain:
            assert domain in DOMAIN_TEMPLATES

    def test_unique_template_names_per_domain(self):
        for domain in Domain:
            templates = DOMAIN_TEMPLATES[domain]
            names = [t.name for t in templates]
            assert len(names) == len(set(names)), (
                f"{domain.value} has duplicate template names"
            )
