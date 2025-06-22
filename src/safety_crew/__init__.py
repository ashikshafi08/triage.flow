"""
Safety Crew Module - Multi-Agent Safety Analysis Platform

This module implements a CrewAI-based multi-agent system for comprehensive
code safety analysis including security vulnerabilities, hallucination detection,
and code quality assessment.
"""

from .agents import (
    SecuritySpecialist,
    GroundingSpecialist,
    QualityArchitect,
    SafetyOrchestrator
)
from .crews import SafetyCrew, RealtimeSafetyCrew, EnterpriseSafetyCrew
from .models import (
    SafetyAnalysisRequest,
    SafetyAnalysisResponse,
    SecurityFinding,
    HallucinationFlag,
    QualityIssue,
    AgentRecommendation
)

__all__ = [
    # Agents
    "SecuritySpecialist",
    "GroundingSpecialist", 
    "QualityArchitect",
    "SafetyOrchestrator",
    # Crews
    "SafetyCrew",
    "RealtimeSafetyCrew",
    "EnterpriseSafetyCrew",
    # Models
    "SafetyAnalysisRequest",
    "SafetyAnalysisResponse",
    "SecurityFinding",
    "HallucinationFlag",
    "QualityIssue",
    "AgentRecommendation"
]