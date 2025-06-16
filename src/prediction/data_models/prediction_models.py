"""
Core data models for the predictive issue resolution system
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class BugPattern:
    """Represents a detected bug pattern"""
    pattern_id: str
    pattern_type: str  # "code_complexity", "file_hotspot", "team_velocity"
    description: str
    confidence: float
    historical_occurrences: int
    files_affected: List[str]
    risk_score: float
    prevention_strategies: List[str]

@dataclass
class TeamPattern:
    """Represents team behavior patterns that correlate with issues"""
    pattern_id: str
    team_members: List[str]
    pattern_type: str  # "review_rush", "workload_spike", "knowledge_gap"
    time_period: str
    correlation_strength: float
    issue_count: int
    prevention_recommendations: List[str]

@dataclass
class RiskFactor:
    """Individual risk factor for prediction"""
    factor_type: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    description: str
    affected_files: List[str]
    mitigation_actions: List[str]

class PredictionReport(BaseModel):
    """Complete prediction analysis report"""
    repo_owner: str
    repo_name: str
    analysis_timestamp: datetime
    prediction_horizon_days: int = 14
    
    # Core predictions
    predicted_issues: List[Dict[str, Any]]
    risk_factors: List[RiskFactor]
    confidence_score: float
    
    # Pattern analysis
    detected_patterns: List[BugPattern]
    team_patterns: List[TeamPattern]
    
    # Prevention strategies
    immediate_actions: List[str]
    long_term_strategies: List[str]
    
    # Metadata
    analysis_duration_seconds: float
    data_sources_used: List[str]

    class Config:
        arbitrary_types_allowed = True

# Additional models for API requests/responses
class PredictionRequest(BaseModel):
    repo_owner: str
    repo_name: str
    repo_path: Optional[str] = None
    prediction_horizon_days: int = 14
    include_team_analysis: bool = True
    include_code_analysis: bool = True

class PreventionRequest(BaseModel):
    repo_owner: str
    repo_name: str
    context_description: Optional[str] = None
    focus_areas: List[str] = []

class PredictionResponse(BaseModel):
    status: str
    prediction_report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    analysis_duration_seconds: Optional[float] = None

class PreventionResponse(BaseModel):
    status: str
    prevention_strategies: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DashboardData(BaseModel):
    summary: Dict[str, Any]
    risk_breakdown: Dict[str, int]
    pattern_insights: Dict[str, Any]
    action_items: Dict[str, List[str]]
    recent_predictions: List[Dict[str, Any]]
    trend_data: Dict[str, Any]

# Metrics and monitoring models
class PredictionMetrics(BaseModel):
    """Metrics for tracking prediction accuracy"""
    prediction_id: str
    repo_owner: str
    repo_name: str
    prediction_timestamp: datetime
    predicted_issues: List[Dict[str, Any]]
    actual_outcomes: Optional[List[Dict[str, Any]]] = None
    accuracy_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None

class AlertConfiguration(BaseModel):
    """Configuration for prediction-based alerts"""
    repo_owner: str
    repo_name: str
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        'critical_risk': 0.8,
        'high_risk': 0.6,
        'pattern_confidence': 0.7
    })
    notification_channels: List[str] = []
    alert_frequency: str = "daily"  # daily, weekly, immediate
    enabled: bool = True

class PredictionCache(BaseModel):
    """Cache model for prediction results"""
    cache_key: str
    repo_owner: str
    repo_name: str
    prediction_data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    cache_version: str = "1.0" 