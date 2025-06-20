"""
OnboardAI - Intelligent Developer Onboarding Platform

This module provides comprehensive AI-powered developer onboarding capabilities,
transforming the existing triage.flow codebase into a specialized onboarding assistant.
"""

from .onboarding_ai_core import OnboardingAICore
from .workflow_engine import OnboardingWorkflowEngine
from .developer_profile import DeveloperProfile
from .onboarding_prompts import OnboardingPrompts

__all__ = [
    "OnboardingAICore",
    "OnboardingWorkflowEngine", 
    "DeveloperProfile",
    "OnboardingPrompts"
]