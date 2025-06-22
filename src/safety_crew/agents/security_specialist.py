"""Security Specialist Agent - Vulnerability Detection Expert"""

from typing import List, Optional, Any
from crewai import Agent

from .base_agent import create_agent_with_llm


class SecuritySpecialist:
    """Security Specialist Agent for comprehensive vulnerability detection"""
    
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
        """Create the Security Specialist agent"""
        
        return create_agent_with_llm(
            role="Senior Security Engineer",
            goal="Identify all security vulnerabilities and potential attack vectors in the code using industry-standard tools and deep security expertise",
            backstory="""You are a seasoned security engineer with 15+ years of experience in application security. 
            You've worked at top tech companies and have discovered critical vulnerabilities in major software projects.
            You're an expert in:
            - OWASP Top 10 and CWE classifications
            - Static and dynamic security analysis
            - Secure coding practices across multiple languages
            - Threat modeling and attack surface analysis
            - Cryptographic best practices
            - Supply chain security
            
            You use tools like Semgrep with comprehensive rule sets to ensure no vulnerability goes undetected.
            You understand that security is not just about finding bugs, but understanding the full attack surface
            and potential exploit chains.""",
            tools=self.tools,
            verbose=self.verbose,
            max_iter=5,
            memory=True,
            allow_delegation=False
        )
        
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent