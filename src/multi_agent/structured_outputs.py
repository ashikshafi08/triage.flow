"""
Pydantic models for structured outputs in the multi-agent system.
Uses LlamaIndex's structured output capabilities for reliable parsing.
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(str, Enum):
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_INVESTIGATION = "bug_investigation" 
    CODE_IMPROVEMENT = "code_improvement"
    ARCHITECTURE_DECISION = "architecture_decision"
    RESEARCH_QUESTION = "research_question"


class Scope(str, Enum):
    SINGLE_FILE = "single_file"
    MULTIPLE_FILES = "multiple_files"
    ENTIRE_CODEBASE = "entire_codebase"
    EXTERNAL_DEPENDENCIES = "external_dependencies"


class UrgencyLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QueryAnalysis(BaseModel):
    """Structured analysis of a user query for research planning"""
    
    query_type: QueryType = Field(description="Type of development query")
    scope: Scope = Field(description="Scope of changes required")
    required_knowledge: List[str] = Field(
        default=["code_patterns"], 
        description="Types of knowledge needed"
    )
    technical_domains: List[str] = Field(
        default=["general"], 
        description="Technical domains involved"
    )
    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.NORMAL, 
        description="Priority level"
    )
    estimated_complexity: int = Field(
        default=5, 
        ge=1, 
        le=10, 
        description="Complexity score from 1-10"
    )


class ImplementationStrategy(BaseModel):
    """Structured implementation strategy for code generation"""
    
    high_level_approach: str = Field(description="Brief description of the approach")
    technology_choices: List[str] = Field(
        default=["python"], 
        description="Technologies to use"
    )
    file_organization: List[str] = Field(
        default=["main.py"], 
        description="Files to create or modify"
    )
    integration_points: List[str] = Field(
        default=[], 
        description="Existing systems to integrate with"
    )
    testing_strategy: str = Field(
        default="unit_tests", 
        description="Testing approach"
    )
    additional_components: List[Dict[str, Any]] = Field(
        default=[], 
        description="Additional components to generate"
    )


# === RAG Integration Models ===

class RAGSource(BaseModel):
    """Source document from RAG retrieval"""
    
    file: str = Field(description="File path")
    language: str = Field(description="Programming language")
    content: str = Field(description="Relevant content")
    description: Optional[str] = Field(default=None, description="File description")
    match_reasons: List[str] = Field(default=[], description="Why this source was selected")


class IssueContext(BaseModel):
    """Related GitHub issue context"""
    
    number: int = Field(description="Issue number")
    title: str = Field(description="Issue title")
    state: str = Field(description="Issue state (open/closed)")
    url: str = Field(description="GitHub URL")
    similarity: float = Field(description="Similarity score")
    labels: List[str] = Field(default=[], description="Issue labels")
    body_preview: str = Field(description="Issue body preview")


class RAGContext(BaseModel):
    """Enhanced RAG context from AgenticRAGSystem"""
    
    sources: List[RAGSource] = Field(description="Source documents")
    related_issues: List[IssueContext] = Field(default=[], description="Related GitHub issues")
    search_type: str = Field(description="Type of search performed")
    complexity: int = Field(default=5, description="Query complexity score")
    processing_time: float = Field(description="Time taken to retrieve context")
    repo_info: Dict[str, Any] = Field(default={}, description="Repository information")


class ToolExecutionResult(BaseModel):
    """Result from executing an agentic tool"""
    
    tool_name: str = Field(description="Name of the tool executed")
    success: bool = Field(description="Whether execution was successful")
    result: str = Field(description="Tool execution result")
    execution_time: float = Field(description="Time taken to execute")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AgenticAnalysis(BaseModel):
    """Analysis results from agentic tools"""
    
    tools_used: List[str] = Field(description="List of tools that were executed")
    tool_results: List[ToolExecutionResult] = Field(description="Results from tool executions")
    insights: List[str] = Field(description="Key insights discovered")
    recommendations: List[str] = Field(description="Recommended actions")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")


class MultiRAGResearchResult(BaseModel):
    """Combined results from multi-RAG research"""
    
    rag_context: RAGContext = Field(description="RAG retrieval context")
    agentic_analysis: Optional[AgenticAnalysis] = Field(default=None, description="Agentic tool analysis")
    combined_insights: List[str] = Field(description="Insights from combining RAG and agentic analysis")
    research_quality_score: float = Field(ge=0.0, le=10.0, description="Quality of research performed")


# === Implementation Models ===

class CodeValidationResult(BaseModel):
    """Results from code validation"""
    
    valid: bool = Field(description="Whether code passed validation")
    errors: List[str] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Security/quality warnings")
    suggestions: List[str] = Field(default=[], description="Improvement suggestions")
    language: str = Field(default="python", description="Code language")
    lines_of_code: int = Field(default=0, description="Lines of code analyzed")


class RiskAssessment(BaseModel):
    """Risk assessment for implementation"""
    
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Overall risk level")
    risk_score: float = Field(ge=0.0, le=10.0, description="Numerical risk score")
    requires_approval: bool = Field(description="Whether human approval is needed")
    approval_reason: str = Field(description="Reason for approval requirement")
    risk_factors: Dict[str, float] = Field(default={}, description="Individual risk factors")
    mitigation_suggestions: Dict[str, List[str]] = Field(default={}, description="Risk mitigation strategies")


class ValidationFeedback(BaseModel):
    """Structured feedback from validation process"""
    
    overall_status: Literal["approved", "requires_review", "rejected"] = Field(description="Overall validation status")
    code_quality_score: float = Field(ge=0.0, le=10.0, default=5.0, description="Code quality score")
    security_score: float = Field(ge=0.0, le=10.0, default=5.0, description="Security assessment score") 
    feedback_items: List[str] = Field(description="Specific feedback points")
    required_changes: List[str] = Field(description="Changes that must be made")
    optional_improvements: List[str] = Field(default=[], description="Suggested improvements")
    validation_time_seconds: float = Field(description="Time taken for validation")


class ComponentMetadata(BaseModel):
    """Metadata for generated code components"""
    
    component_type: str = Field(description="Type of component (function, class, module)")
    name: str = Field(description="Component name")
    description: str = Field(description="Component description")
    language: str = Field(default="python", description="Programming language")
    dependencies: List[str] = Field(default=[], description="Required dependencies")
    estimated_lines: int = Field(default=0, description="Estimated lines of code")
    complexity_score: int = Field(default=3, ge=1, le=10, description="Implementation complexity")


# === Enhanced Multi-Agent Response ===

class EnhancedMultiAgentResult(BaseModel):
    """Complete result from enhanced multi-agent system"""
    
    query: str = Field(description="Original user query")
    query_analysis: QueryAnalysis = Field(description="Structured query analysis")
    research_results: MultiRAGResearchResult = Field(description="Multi-RAG research results")
    implementation_strategy: ImplementationStrategy = Field(description="Implementation strategy")
    validation_feedback: ValidationFeedback = Field(description="Validation results")
    performance_metrics: Dict[str, float] = Field(description="Performance timing metrics")
    session_id: str = Field(description="Session identifier")
    total_execution_time: float = Field(description="Total time for complete workflow")
    
    # Workflow status
    approved: bool = Field(description="Whether implementation is approved")
    next_steps: List[str] = Field(description="Recommended next steps")
    
    # Integration metadata
    rag_systems_used: List[str] = Field(description="Which RAG systems were used")
    tools_executed: List[str] = Field(description="Agentic tools that were executed")
    workflow_version: str = Field(default="2.0.0", description="Enhanced workflow version") 