from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from homie_core.config import HomieConfig


@dataclass
class ModelPolicy:
    allowed_backends: list[str] = field(default_factory=list)
    endpoint: str = ""
    api_key_env: str = ""
    allowed_models: list[str] = field(default_factory=list)


@dataclass
class PluginPolicy:
    disabled: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)


@dataclass
class PrivacyPolicy:
    data_retention_days: int = 0
    disable_observers: list[str] = field(default_factory=list)
    audit_log: bool = False


@dataclass
class EnterprisePolicy:
    org_name: str = ""
    model_policy: ModelPolicy = field(default_factory=ModelPolicy)
    plugins: PluginPolicy = field(default_factory=PluginPolicy)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    policy_url: str = ""
    org_user_id: str = ""

    def is_plugin_disabled(self, name: str) -> bool:
        return name in self.plugins.disabled

    def is_plugin_required(self, name: str) -> bool:
        return name in self.plugins.required

    def is_backend_allowed(self, backend: str) -> bool:
        if not self.model_policy.allowed_backends:
            return True
        return backend in self.model_policy.allowed_backends

    def is_model_allowed(self, model: str) -> bool:
        if not self.model_policy.allowed_models:
            return True
        return model in self.model_policy.allowed_models


def load_enterprise_policy(config_dir: Path | str) -> Optional[EnterprisePolicy]:
    path = Path(config_dir) / "homie.enterprise.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None

    mp = data.get("model_policy", {})
    pp = data.get("plugins", {})
    pr = data.get("privacy", {})

    return EnterprisePolicy(
        org_name=data.get("org_name", ""),
        model_policy=ModelPolicy(
            allowed_backends=mp.get("allowed_backends", []),
            endpoint=mp.get("endpoint", ""),
            api_key_env=mp.get("api_key_env", ""),
            allowed_models=mp.get("allowed_models", []),
        ),
        plugins=PluginPolicy(
            disabled=pp.get("disabled", []),
            required=pp.get("required", []),
        ),
        privacy=PrivacyPolicy(
            data_retention_days=pr.get("data_retention_days", 0),
            disable_observers=pr.get("disable_observers", []),
            audit_log=pr.get("audit_log", False),
        ),
        policy_url=data.get("policy_url", ""),
    )


def apply_policy(cfg: HomieConfig, policy: EnterprisePolicy) -> HomieConfig:
    """Merge enterprise policy over personal config. Enterprise wins."""
    if policy.model_policy.endpoint:
        cfg.llm.api_base_url = policy.model_policy.endpoint

    if policy.model_policy.api_key_env:
        key = os.environ.get(policy.model_policy.api_key_env, "")
        if key:
            cfg.llm.api_key = key

    if policy.model_policy.allowed_backends:
        if cfg.llm.backend not in policy.model_policy.allowed_backends:
            cfg.llm.backend = policy.model_policy.allowed_backends[0]

    if policy.privacy.data_retention_days:
        cfg.privacy.data_retention_days = policy.privacy.data_retention_days

    for obs in policy.privacy.disable_observers:
        if obs in cfg.privacy.observers:
            cfg.privacy.observers[obs] = False

    return cfg
