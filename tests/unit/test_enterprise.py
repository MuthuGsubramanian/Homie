import yaml
from pathlib import Path

from homie_core.enterprise import EnterprisePolicy, load_enterprise_policy, apply_policy
from homie_core.config import HomieConfig


def test_load_policy_from_file(tmp_path):
    policy_data = {
        "org_name": "Acme Corp",
        "model_policy": {
            "allowed_backends": ["cloud"],
            "endpoint": "https://models.acme.internal/v1",
            "api_key_env": "ACME_AI_KEY",
            "allowed_models": ["llama-3.1-70b"],
        },
        "plugins": {
            "disabled": ["browser_plugin"],
            "required": ["git_plugin"],
        },
        "privacy": {
            "data_retention_days": 90,
            "disable_observers": ["browsing", "social"],
            "audit_log": True,
        },
    }
    policy_file = tmp_path / "homie.enterprise.yaml"
    policy_file.write_text(yaml.dump(policy_data))

    policy = load_enterprise_policy(tmp_path)
    assert policy is not None
    assert policy.org_name == "Acme Corp"
    assert "cloud" in policy.model_policy.allowed_backends
    assert policy.privacy.audit_log is True


def test_no_policy_returns_none(tmp_path):
    policy = load_enterprise_policy(tmp_path)
    assert policy is None


def test_apply_policy_overrides_config(tmp_path):
    policy_data = {
        "org_name": "TestCorp",
        "model_policy": {
            "allowed_backends": ["cloud"],
            "endpoint": "https://internal.api/v1",
        },
        "privacy": {
            "data_retention_days": 90,
            "disable_observers": ["browsing"],
        },
    }
    policy_file = tmp_path / "homie.enterprise.yaml"
    policy_file.write_text(yaml.dump(policy_data))
    policy = load_enterprise_policy(tmp_path)

    cfg = HomieConfig()
    cfg.llm.backend = "gguf"
    cfg.privacy.data_retention_days = 30

    applied = apply_policy(cfg, policy)
    assert applied.llm.api_base_url == "https://internal.api/v1"
    assert applied.privacy.data_retention_days == 90
    assert applied.privacy.observers["browsing"] is False


def test_plugin_restrictions():
    from homie_core.enterprise import EnterprisePolicy, PluginPolicy

    policy = EnterprisePolicy(
        org_name="Test",
        plugins=PluginPolicy(disabled=["browser_plugin"], required=["git_plugin"]),
    )
    assert policy.is_plugin_disabled("browser_plugin")
    assert not policy.is_plugin_disabled("git_plugin")
    assert policy.is_plugin_required("git_plugin")


def test_model_allowed():
    from homie_core.enterprise import EnterprisePolicy, ModelPolicy

    policy = EnterprisePolicy(
        org_name="Test",
        model_policy=ModelPolicy(
            allowed_backends=["cloud"],
            allowed_models=["llama-3.1-70b"],
        ),
    )
    assert policy.is_backend_allowed("cloud")
    assert not policy.is_backend_allowed("gguf")
    assert policy.is_model_allowed("llama-3.1-70b")
    assert not policy.is_model_allowed("gpt-4o")
