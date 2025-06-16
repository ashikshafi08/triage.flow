"""
Predictive Issue Resolution System

This module implements a multi-agent system for predicting potential issues
before they occur by analyzing code patterns, deployment history, and team behavior.
"""

from .data_models.prediction_models import (
    BugPattern,
    TeamPattern,
    RiskFactor,
    PredictionReport
)

from .agents.orchestrator import PredictionOrchestrator
from .tools.data_collector import PredictiveDataCollector
from .tools.pattern_detector import PatternDetectionEngine
from .tools.semantic_analyzer import SemanticCodeAnalyzer
from .tools.risk_scorer import HybridRiskScorer
from .api import prediction_api

__version__ = "1.0.0"
__all__ = [
    "BugPattern",
    "TeamPattern", 
    "RiskFactor",
    "PredictionReport",
    "PredictionOrchestrator",
    "PredictiveDataCollector",
    "PatternDetectionEngine",
    "SemanticCodeAnalyzer",
    "HybridRiskScorer",
    "prediction_api"
]
