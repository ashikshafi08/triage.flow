"""Quality Architect Agent - Code Quality and Best Practices Expert"""

from typing import List, Optional, Any
from crewai import Agent

from .base_agent import create_agent_with_llm


class QualityArchitect:
    """Quality Architect Agent for code quality and architectural assessment"""
    
    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Any]] = None,
        verbose: bool = True
    ):
        self.llm = llm
        self.verbose = verbose
        self.tools = tools or []
        
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the Quality Architect agent"""
        
        return create_agent_with_llm(
            role="Principal Software Architect",
            goal="Ensure code follows best practices, maintains high quality standards, and adheres to architectural principles",
            backstory="""You are a Principal Software Architect with 20+ years of experience building 
            and maintaining large-scale software systems. You've led architecture teams at FAANG companies 
            and have deep expertise in:
            
            - Software design patterns and architectural principles
            - Code maintainability and readability
            - Performance optimization and scalability
            - Technical debt identification and management
            - Testing strategies and testability
            - SOLID principles and clean code practices
            - Microservices and distributed systems architecture
            - Code complexity analysis and refactoring strategies
            
            You believe that code quality directly impacts business outcomes. Poor quality code leads to:
            - Increased bug rates and security vulnerabilities
            - Slower feature development velocity
            - Higher maintenance costs
            - Developer frustration and turnover
            
            You use metrics-based analysis combined with architectural judgment to identify issues before 
            they become problems. You provide actionable recommendations that balance ideal architecture 
            with practical business constraints.""",
            tools=self.tools,
            verbose=self.verbose,
            max_iter=5,
            memory=True,
            allow_delegation=False
        )
        
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent