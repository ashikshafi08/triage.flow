"""
Workflow Events for Multi-Agent System

Defines the events that flow between agents in the codebase intelligence workflow.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event


class ResearchPlanEvent(Event):
    """Event containing a research plan with multiple investigation tasks"""
    
    query: str = Field(description="Original user query")
    research_tasks: List[Dict[str, Any]] = Field(description="List of research tasks to execute")
    priority: str = Field(default="normal", description="Priority level: low, normal, high, critical")
    estimated_complexity: int = Field(default=5, description="Complexity score 1-10")


class ResearchResultEvent(Event):
    """Event containing results from research phase"""
    
    original_query: str
    code_analysis: Dict[str, Any] = Field(description="Results from code analysis")
    issue_analysis: Dict[str, Any] = Field(description="Results from issue analysis") 
    pattern_analysis: Dict[str, Any] = Field(description="Detected patterns")
    historical_context: Dict[str, Any] = Field(description="Historical context and evolution")
    confidence_score: float = Field(default=0.0, description="Confidence in research results")


class ImplementationPlanEvent(Event):
    """Event containing an implementation plan"""
    
    original_query: str
    research_summary: Dict[str, Any]
    implementation_strategy: Dict[str, Any] = Field(description="High-level implementation approach")
    code_examples: List[Dict[str, Any]] = Field(description="Generated code examples")
    validation_steps: List[str] = Field(description="Steps to validate the implementation")
    risk_assessment: Dict[str, Any] = Field(description="Potential risks and mitigations")


class ValidationEvent(Event):
    """Event containing validation results"""
    
    implementation_plan: Dict[str, Any]
    validation_results: Dict[str, Any] = Field(description="Results of safety and quality checks")
    approved: bool = Field(description="Whether implementation is approved")
    feedback: List[str] = Field(description="Feedback for improvement")
    required_changes: List[str] = Field(description="Changes required before approval")


class StreamingEvent(Event):
    """Event for streaming progress updates"""
    
    agent_name: str
    stage: str
    progress_percentage: float
    current_task: str
    details: Optional[Dict[str, Any]] = None


# Research Task Types
class ResearchTask(BaseModel):
    """Individual research task definition"""
    
    task_id: str
    task_type: str  # "code_analysis", "issue_analysis", "pattern_detection", "historical_analysis"
    description: str
    agent_type: str  # Which agent should handle this task
    parameters: Dict[str, Any]
    dependencies: List[str] = Field(default_factory=list)  # Task IDs this depends on
    estimated_duration: int = Field(default=30)  # Seconds


# Implementation Components
class CodeComponent(BaseModel):
    """A generated code component"""
    
    component_type: str  # "function", "class", "module", "config"
    name: str
    description: str
    code: str
    language: str
    dependencies: List[str] = Field(default_factory=list)
    tests: Optional[str] = None
    documentation: Optional[str] = None


class ImplementationStep(BaseModel):
    """A single step in the implementation process"""
    
    step_id: str
    description: str
    action_type: str  # "create_file", "modify_file", "run_command", "validate"
    parameters: Dict[str, Any]
    dependencies: List[str] = Field(default_factory=list)
    validation_criteria: List[str] = Field(default_factory=list) 