"""Crew Performance Metrics Models"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AgentEffectivenessScore(BaseModel):
    """Individual agent effectiveness metrics"""
    agent_name: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time_ms: float = 0.0
    average_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    tool_usage_efficiency: float = Field(ge=0.0, le=1.0, default=0.0)
    findings_accuracy: float = Field(ge=0.0, le=1.0, default=0.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    user_feedback_score: Optional[float] = Field(None, ge=0.0, le=5.0)


class TaskCompletionMetrics(BaseModel):
    """Metrics for task completion"""
    task_name: str
    total_executions: int = 0
    successful_completions: int = 0
    partial_completions: int = 0
    failures: int = 0
    average_duration_ms: float = 0.0
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    error_types: Dict[str, int] = Field(default_factory=dict)


class CrewPerformanceMetrics(BaseModel):
    """Overall crew performance metrics"""
    crew_name: str
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    average_analysis_time_ms: float = 0.0
    
    # Agent metrics
    agent_effectiveness: Dict[str, AgentEffectivenessScore] = Field(default_factory=dict)
    
    # Task metrics
    task_metrics: Dict[str, TaskCompletionMetrics] = Field(default_factory=dict)
    
    # Resource usage
    total_llm_calls: int = 0
    total_tool_invocations: int = 0
    cache_hit_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Quality metrics
    findings_per_analysis: float = 0.0
    auto_fix_success_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    user_satisfaction_score: Optional[float] = Field(None, ge=0.0, le=5.0)
    
    # Time series data
    hourly_load: Dict[int, int] = Field(default_factory=dict)
    daily_trends: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def update_from_analysis(self, analysis_response: Dict[str, Any], duration_ms: int):
        """Update metrics from a completed analysis"""
        self.total_analyses += 1
        
        if analysis_response.get("status") == "success":
            self.successful_analyses += 1
        else:
            self.failed_analyses += 1
            
        # Update rolling average
        self.average_analysis_time_ms = (
            (self.average_analysis_time_ms * (self.total_analyses - 1) + duration_ms) 
            / self.total_analyses
        )
        
        # Update agent effectiveness
        for agent_response in analysis_response.get("agent_responses", []):
            agent_name = agent_response.get("agent_name")
            if agent_name and agent_name in self.agent_effectiveness:
                agent_metrics = self.agent_effectiveness[agent_name]
                agent_metrics.tasks_completed += 1 if agent_response.get("status") == "success" else 0
                agent_metrics.tasks_failed += 1 if agent_response.get("status") == "failed" else 0
                
        self.last_updated = datetime.now()