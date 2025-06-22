"""Safety Analysis Data Models"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityFindingType(str, Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    AUTHENTICATION_ISSUE = "authentication_issue"
    AUTHORIZATION_ISSUE = "authorization_issue"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    OTHER = "other"


class HallucinationType(str, Enum):
    """Types of AI hallucinations"""
    NON_EXISTENT_API = "non_existent_api"
    INCORRECT_PARAMETERS = "incorrect_parameters"
    WRONG_IMPORT = "wrong_import"
    FICTIONAL_LIBRARY = "fictional_library"
    INCORRECT_SYNTAX = "incorrect_syntax"
    WRONG_PATTERN = "wrong_pattern"


class QualityIssueType(str, Enum):
    """Types of code quality issues"""
    HIGH_COMPLEXITY = "high_complexity"
    POOR_NAMING = "poor_naming"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    PERFORMANCE_ISSUE = "performance_issue"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    ARCHITECTURAL_VIOLATION = "architectural_violation"
    POOR_DOCUMENTATION = "poor_documentation"
    MISSING_TYPE_HINTS = "missing_type_hints"


class SecurityFinding(BaseModel):
    """Security vulnerability finding"""
    id: str = Field(description="Unique identifier for the finding")
    type: SecurityFindingType
    severity: SeverityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    owasp_category: Optional[str] = None
    remediation: str = Field(description="Suggested fix for the vulnerability")
    references: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    semgrep_rule_id: Optional[str] = None


class HallucinationFlag(BaseModel):
    """AI hallucination detection result"""
    id: str
    type: HallucinationType
    severity: SeverityLevel
    description: str
    hallucinated_code: str
    suggested_correction: Optional[str] = None
    grounding_context: Optional[str] = Field(None, description="Context from codebase that proves hallucination")
    confidence: float = Field(ge=0.0, le=1.0)
    affected_lines: List[int] = Field(default_factory=list)


class QualityIssue(BaseModel):
    """Code quality issue"""
    id: str
    type: QualityIssueType
    severity: SeverityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_range: Optional[tuple[int, int]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quality metrics (e.g., cyclomatic complexity)")
    improvement_suggestion: str
    impact: str = Field(description="Business/technical impact of the issue")
    effort_estimate: Optional[str] = None


class AutoFixSuggestion(BaseModel):
    """Automated fix suggestion"""
    finding_id: str = Field(description="ID of the finding this fix addresses")
    fix_type: str = Field(description="Type of fix: security, hallucination, or quality")
    original_code: str
    suggested_code: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool = True
    potential_side_effects: List[str] = Field(default_factory=list)


class AgentRecommendation(BaseModel):
    """High-level recommendation from an agent"""
    agent_name: str
    recommendation: str
    priority: str = Field(description="high, medium, or low")
    rationale: str
    action_items: List[str] = Field(default_factory=list)
    related_findings: List[str] = Field(default_factory=list, description="IDs of related findings")


class SafetyMetrics(BaseModel):
    """Aggregated safety metrics"""
    overall_risk_score: float = Field(ge=0.0, le=10.0, description="Overall risk score 0-10")
    security_score: float = Field(ge=0.0, le=10.0)
    grounding_score: float = Field(ge=0.0, le=10.0)
    quality_score: float = Field(ge=0.0, le=10.0)
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    auto_fixable_count: int
    estimated_remediation_effort: Optional[str] = None


class SafetyAnalysisRequest(BaseModel):
    """Request for safety analysis"""
    session_id: str
    code: str
    file_path: Optional[str] = None
    language: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    analysis_depth: str = Field(default="standard", description="quick, standard, or deep")
    include_auto_fix: bool = True
    custom_rules: Optional[List[str]] = Field(default_factory=list, description="Custom Semgrep rule IDs")
    target_compliance: Optional[List[str]] = Field(default_factory=list, description="SOC2, ISO, etc.")


class SafetyAnalysisResponse(BaseModel):
    """Complete safety analysis response"""
    request_id: str
    session_id: str
    timestamp: datetime
    analysis_duration_ms: int
    
    # Findings
    security_findings: List[SecurityFinding] = Field(default_factory=list)
    hallucination_flags: List[HallucinationFlag] = Field(default_factory=list)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    
    # Recommendations and fixes
    agent_recommendations: List[AgentRecommendation] = Field(default_factory=list)
    auto_fix_suggestions: List[AutoFixSuggestion] = Field(default_factory=list)
    
    # Metrics
    safety_metrics: SafetyMetrics
    
    # Crew metadata
    crew_type: str = Field(description="safety_crew, realtime_crew, or enterprise_crew")
    agents_involved: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_crew_results(cls, crew_output: Dict[str, Any], request: SafetyAnalysisRequest, duration_ms: int) -> "SafetyAnalysisResponse":
        """Create response from CrewAI output"""
        import uuid
        
        # Extract findings from crew output
        security_findings = crew_output.get("security_findings", [])
        hallucination_flags = crew_output.get("hallucination_flags", [])
        quality_issues = crew_output.get("quality_issues", [])
        recommendations = crew_output.get("recommendations", [])
        auto_fixes = crew_output.get("auto_fixes", [])
        
        # Calculate metrics
        total_findings = len(security_findings) + len(hallucination_flags) + len(quality_issues)
        critical_count = sum(1 for f in security_findings if f.get("severity") == "critical")
        high_count = sum(1 for f in security_findings if f.get("severity") == "high")
        medium_count = sum(1 for f in security_findings if f.get("severity") == "medium")
        low_count = sum(1 for f in security_findings if f.get("severity") == "low")
        
        # Calculate scores (10 = perfect, 0 = worst)
        security_score = max(0, 10 - (critical_count * 3 + high_count * 2 + medium_count * 1 + low_count * 0.5))
        grounding_score = max(0, 10 - len(hallucination_flags) * 2)
        quality_score = max(0, 10 - len(quality_issues) * 1.5)
        overall_score = (security_score + grounding_score + quality_score) / 3
        
        metrics = SafetyMetrics(
            overall_risk_score=overall_score,
            security_score=security_score,
            grounding_score=grounding_score,
            quality_score=quality_score,
            total_findings=total_findings,
            critical_findings=critical_count,
            high_findings=high_count,
            medium_findings=medium_count,
            low_findings=low_count,
            auto_fixable_count=len(auto_fixes)
        )
        
        return cls(
            request_id=str(uuid.uuid4()),
            session_id=request.session_id,
            timestamp=datetime.now(),
            analysis_duration_ms=duration_ms,
            security_findings=[SecurityFinding(**f) for f in security_findings],
            hallucination_flags=[HallucinationFlag(**f) for f in hallucination_flags],
            quality_issues=[QualityIssue(**f) for f in quality_issues],
            agent_recommendations=[AgentRecommendation(**r) for r in recommendations],
            auto_fix_suggestions=[AutoFixSuggestion(**f) for f in auto_fixes],
            safety_metrics=metrics,
            crew_type=crew_output.get("crew_type", "safety_crew"),
            agents_involved=crew_output.get("agents_involved", [])
        )