"""
Base Crew implementation following CrewAI documentation
https://docs.crewai.com/core-concepts/crews
"""

from typing import List, Dict, Any, Optional
from crewai import Crew, Task, Agent, Process
from crewai.project import CrewBase, agent, task, crew
import time
from src.agent_tools.llm_config import get_llm_instance
from src.config import settings


class SafetyAnalysisCrew(CrewBase):
    """Safety Analysis Crew following CrewAI best practices"""
    
    def __init__(self, inputs: Dict[str, Any]):
        super().__init__()
        self.inputs = inputs
        # Use the existing LLM configuration
        self.llm = inputs.get("llm") or get_llm_instance()
        self.tools = inputs.get("tools", [])
    
    @agent
    def security_specialist(self) -> Agent:
        """Security vulnerability detection expert"""
        return Agent(
            role="Senior Security Engineer",
            goal="Identify all security vulnerabilities using Semgrep and pattern analysis",
            backstory="""You are a security expert with 15+ years experience.
            Expert in OWASP Top 10, CWE classifications, and secure coding.
            You use Semgrep to scan for vulnerabilities systematically.""",
            tools=self.tools.get("security_tools", []),
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def grounding_specialist(self) -> Agent:
        """AI hallucination detection expert"""
        return Agent(
            role="AI Safety Specialist",
            goal="Detect hallucinations by verifying code against actual codebase",
            backstory="""You specialize in preventing AI hallucinations in code.
            Expert at verifying APIs, imports, and patterns exist in reality.
            You use RAG systems to ground all code elements.""",
            tools=self.tools.get("grounding_tools", []),
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def quality_architect(self) -> Agent:
        """Code quality and architecture expert"""
        return Agent(
            role="Principal Software Architect",
            goal="Ensure code quality, maintainability, and best practices",
            backstory="""You are a principal architect with 20+ years experience.
            Expert in design patterns, SOLID principles, and clean code.
            You identify technical debt and architectural issues.""",
            tools=self.tools.get("quality_tools", []),
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def safety_orchestrator(self) -> Agent:
        """Chief Safety Officer coordinating analysis"""
        return Agent(
            role="Chief Safety Officer",
            goal="Synthesize findings and provide actionable recommendations",
            backstory="""You coordinate safety assessments for Fortune 500 companies.
            Expert at synthesizing findings from multiple specialists.
            You prioritize risks and provide clear remediation guidance.""",
            tools=self.tools.get("orchestrator_tools", []),
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def security_analysis_task(self) -> Task:
        """Comprehensive security vulnerability analysis"""
        return Task(
            description="""
            Analyze the provided code for security vulnerabilities:
            1. Run Semgrep with OWASP, CWE, and security audit rules
            2. Detect patterns for SQL injection, XSS, command injection
            3. Check for weak cryptography and hardcoded secrets
            4. Assess supply chain security risks
            
            Code to analyze:
            {code}
            
            Provide detailed findings with severity, impact, and remediation.
            """,
            expected_output="""
            Security analysis report with:
            - List of vulnerabilities by severity
            - Technical details and exploit scenarios
            - Remediation steps with code examples
            - Risk assessment and prioritization
            """,
            agent=self.security_specialist()
        )
    
    @task
    def hallucination_check_task(self) -> Task:
        """AI hallucination detection"""
        return Task(
            description="""
            Verify all code elements are grounded in reality:
            1. Check all imports exist in the codebase/libraries
            2. Verify API calls and function signatures
            3. Validate patterns against actual codebase
            4. Identify any fictional or non-existent elements
            
            Code to verify:
            {code}
            
            Report all hallucinations with evidence and corrections.
            """,
            expected_output="""
            Hallucination report with:
            - List of detected hallucinations
            - Evidence why each is a hallucination
            - Suggested corrections
            - Grounding score (0-100%)
            """,
            agent=self.grounding_specialist()
        )
    
    @task
    def quality_review_task(self) -> Task:
        """Code quality and architecture review"""
        return Task(
            description="""
            Analyze code quality and architecture:
            1. Measure complexity and maintainability
            2. Check SOLID principles compliance
            3. Identify code smells and technical debt
            4. Assess testability and documentation
            
            Code to review:
            {code}
            
            Provide actionable improvement suggestions.
            """,
            expected_output="""
            Quality report with:
            - Metrics (complexity, maintainability)
            - Quality issues by severity
            - Improvement recommendations
            - Refactoring suggestions
            """,
            agent=self.quality_architect()
        )
    
    @task
    def synthesis_task(self) -> Task:
        """Synthesize all findings into actionable report"""
        return Task(
            description="""
            Synthesize findings from all specialists:
            1. Integrate security, hallucination, and quality findings
            2. Identify patterns and root causes
            3. Prioritize by risk and business impact
            4. Create actionable remediation plan
            
            Context: {analysis_context}
            
            Provide executive summary and implementation roadmap.
            """,
            expected_output="""
            Executive safety report with:
            - Overall safety score and risk assessment
            - Prioritized findings across all domains
            - Actionable remediation roadmap
            - Quick wins vs long-term improvements
            """,
            agent=self.safety_orchestrator(),
            context=[
                self.security_analysis_task(),
                self.hallucination_check_task(),
                self.quality_review_task()
            ]
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the safety analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )


def analyze_code_safety(
    code: str,
    llm: Any,
    tools: Dict[str, List[Any]],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze code safety using the crew
    
    Args:
        code: Code to analyze
        llm: Language model to use
        tools: Dictionary of tools by category
        context: Additional context
    
    Returns:
        Safety analysis results
    """
    
    start_time = time.time()
    
    # Prepare inputs
    inputs = {
        "code": code,
        "llm": llm,
        "tools": tools,
        "analysis_context": context or {}
    }
    
    # Create and run crew
    crew = SafetyAnalysisCrew(inputs)
    results = crew.crew().kickoff(inputs=inputs)
    
    # Process results
    duration_ms = int((time.time() - start_time) * 1000)
    
    return {
        "status": "success",
        "duration_ms": duration_ms,
        "results": results,
        "crew_type": "safety_analysis_crew"
    }