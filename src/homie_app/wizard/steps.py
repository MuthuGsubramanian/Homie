from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WizardStep:
    title: str
    description: str
    step_type: str  # "info", "choice", "input", "progress"
    choices: list[str] | None = None
    default: str = ""
    result: Any = None


def get_setup_steps() -> list[WizardStep]:
    return [
        WizardStep(
            title="Welcome",
            description="Welcome to Homie AI! Let's set things up.",
            step_type="info",
        ),
        WizardStep(
            title="Hardware Detection",
            description="Detecting your hardware configuration...",
            step_type="progress",
        ),
        WizardStep(
            title="Model Selection",
            description="Select the AI model to use:",
            step_type="choice",
            choices=["Recommended (auto-detect)", "Small (fast, less accurate)", "Large (slower, more accurate)", "Custom"],
        ),
        WizardStep(
            title="Voice Setup",
            description="Would you like to enable voice control?",
            step_type="choice",
            choices=["Yes - Always listening", "Yes - Push to talk", "No - Text only"],
            default="No - Text only",
        ),
        WizardStep(
            title="Your Name",
            description="What should Homie call you?",
            step_type="input",
        ),
        WizardStep(
            title="Plugins",
            description="Select plugins to enable:",
            step_type="choice",
            choices=["Essential (system, clipboard, health)", "Development (+ IDE, git, terminal)", "Full (all available)"],
            default="Essential (system, clipboard, health)",
        ),
        WizardStep(
            title="Complete",
            description="Setup complete! Homie is ready to use.",
            step_type="info",
        ),
    ]
