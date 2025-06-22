"""Agent Response Models for CrewAI Integration"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AgentResponse(BaseModel):
    """Base response from any safety agent"""
    agent_name: str
    task_name: str
    status: str = Field(description="success, partial, or failed")
    execution_time_ms: int
    llm_calls_made: int = 0
    tools_used: List[str] = Field(default_factory=list)
    raw_output: str
    structured_output: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SecurityAgentResponse(AgentResponse):
    """Response from Security Specialist agent"""
    vulnerabilities_found: int = 0
    semgrep_rules_applied: List[str] = Field(default_factory=list)
    custom_patterns_matched: List[str] = Field(default_factory=list)
    security_context: Dict[str, Any] = Field(default_factory=dict)
    owasp_coverage: List[str] = Field(default_factory=list)
    cwe_mappings: Dict[str, List[str]] = Field(default_factory=dict)


class GroundingAgentResponse(AgentResponse):
    """Response from Grounding Specialist agent"""
    hallucinations_detected: int = 0
    apis_verified: int = 0
    imports_checked: int = 0
    grounding_sources: List[Dict[str, Any]] = Field(default_factory=list)
    rag_queries_performed: int = 0
    false_positive_corrections: int = 0
    codebase_context: Dict[str, Any] = Field(default_factory=dict)


class QualityAgentResponse(AgentResponse):
    """Response from Quality Architect agent"""
    quality_issues_found: int = 0
    complexity_metrics: Dict[str, float] = Field(default_factory=dict)
    maintainability_score: Optional[float] = None
    test_coverage_estimate: Optional[float] = None
    architectural_violations: List[str] = Field(default_factory=list)
    best_practices_violations: List[str] = Field(default_factory=list)
    improvement_opportunities: List[Dict[str, Any]] = Field(default_factory=list)


class OrchestratorResponse(AgentResponse):
    """Response from Safety Orchestrator agent"""
    synthesis_complete: bool = False
    agents_coordinated: List[str] = Field(default_factory=list)
    conflicts_resolved: List[Dict[str, Any]] = Field(default_factory=list)
    prioritized_findings: List[str] = Field(default_factory=list)
    overall_assessment: str = ""
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)
    cross_agent_insights: List[str] = Field(default_factory=list)