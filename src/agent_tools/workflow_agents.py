"""
Specialized agents for triage.flow workflows
Part of LlamaIndex Workflow Integration
"""

import logging
from typing import Dict, Any, List, Optional
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import BaseTool
from llama_index.core.llms import LLM

from .llamaindex_workflows import AgentConfig
from .llm_config import get_llm_instance
from .tool_registry import create_all_tools

logger = logging.getLogger(__name__)


class WorkflowAgentFactory:
    """Factory for creating specialized workflow agents"""
    
    def __init__(
        self,
        session_id: str,
        repo_path: str,
        llm: Optional[LLM] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        self.session_id = session_id
        self.repo_path = repo_path
        
        # Use provided LLM or get from config
        if llm:
            self.llm = llm
        else:
            # All OpenRouter models support function calling
            self.llm = get_llm_instance()
            
        self.tools = tools or []
        
        # Initialize tools if not provided
        if not self.tools:
            self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools for workflow agents with proper context"""
        try:
            # Import the existing core agent to get tools
            from .core import AgenticCodebaseExplorer
            
            # Create a persistent explorer instance that workflow agents can share
            # This ensures proper execution context is maintained
            self.explorer = AgenticCodebaseExplorer(
                session_id=self.session_id,
                repo_path=self.repo_path
            )
            
            # Get tools from the persistent explorer
            self.tools = self.explorer.tools
            logger.info(f"Initialized {len(self.tools)} tools for workflow agents")
            
            # Store reference to context manager for future use
            self.context_manager = self.explorer.context_manager
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            self.tools = []
            self.explorer = None
            self.context_manager = None
    
    def create_code_analysis_agent(self, config: Optional[AgentConfig] = None) -> FunctionCallingAgentWorker:
        """Create a code analysis specialist agent"""
        if not config:
            config = AgentConfig(
                id="code_analysis",
                name="Code Analysis Specialist",
                role="Code Analyzer",
                goal="Analyze code structure, understand implementation patterns, and identify key components",
                backstory="Expert in code analysis with deep understanding of software architecture patterns",
                specialization="code_analysis",
                max_iterations=15
            )
        
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Specialization: Code Analysis and Understanding
- Analyze code structure and architecture
- Identify design patterns and conventions
- Understand data flow and dependencies
- Extract technical requirements from code
- Identify potential areas of concern

EXECUTION CONTEXT:
- Repository: {self.repo_path}
- Session ID: {self.session_id}
- Available Tools: {len(self.tools)} specialized tools for code analysis

IMPORTANT: You have access to powerful tools for exploring the codebase:
- explore_directory: Explore project structure and files
- read_file: Read specific files for detailed analysis
- search_codebase: Search for patterns, functions, classes across the entire codebase
- analyze_file_structure: Get technical analysis of file relationships
- semantic_content_search: Find semantically related code sections

Guidelines for code analysis:
1. START by exploring the project structure with explore_directory to understand the layout
2. Use search_codebase to find relevant files and patterns
3. Read key files to understand implementation details
4. Focus on understanding the problem domain and architecture
5. Provide clear, technical insights with specific file references

When you've completed your analysis, state your findings clearly and recommend next steps.
If specialized analysis is needed (security, quality, testing), suggest handoff to appropriate specialists.
"""
        
        # Select tools relevant to code analysis
        code_analysis_tools = self._get_tools_for_specialization("code_analysis")
        
        return FunctionCallingAgentWorker.from_tools(
            tools=code_analysis_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
    
    def create_issue_resolution_agent(self, config: Optional[AgentConfig] = None) -> FunctionCallingAgentWorker:
        """Create an issue resolution specialist agent"""
        if not config:
            config = AgentConfig(
                id="issue_resolution",
                name="Issue Resolution Specialist",
                role="Problem Solver",
                goal="Identify root causes of issues and propose practical solutions",
                backstory="Experienced in debugging, troubleshooting, and systematic problem resolution",
                specialization="issue_resolution",
                max_iterations=20
            )
        
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Specialization: Issue Resolution and Debugging
- Identify root causes of reported issues
- Analyze error patterns and symptoms
- Trace through code to find problem sources
- Propose practical, actionable solutions
- Consider impact and implementation complexity

Current repository: {self.repo_path}
Session ID: {self.session_id}

Guidelines for issue resolution:
1. Understand the problem thoroughly before proposing solutions
2. Use systematic debugging approaches
3. Consider multiple potential causes
4. Propose solutions with implementation guidance
5. Assess risks and benefits of proposed changes

When you've identified the issue and solution, provide clear recommendations.
If specialized expertise is needed (security fixes, performance optimization), suggest appropriate handoffs.
"""
        
        # Select tools relevant to issue resolution
        issue_resolution_tools = self._get_tools_for_specialization("issue_resolution")
        
        return FunctionCallingAgentWorker.from_tools(
            tools=issue_resolution_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
    
    def create_testing_qa_agent(self, config: Optional[AgentConfig] = None) -> FunctionCallingAgentWorker:
        """Create a testing and QA specialist agent"""
        if not config:
            config = AgentConfig(
                id="testing_qa",
                name="Testing & QA Specialist",
                role="Quality Assurance Engineer",
                goal="Ensure code quality through comprehensive testing strategies and quality assessment",
                backstory="Expert in testing methodologies, quality assurance, and best practices",
                specialization="testing_qa",
                max_iterations=15
            )
        
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Specialization: Testing and Quality Assurance
- Design comprehensive testing strategies
- Identify test coverage gaps
- Assess code quality and adherence to standards
- Recommend testing best practices
- Validate proposed solutions

Current repository: {self.repo_path}
Session ID: {self.session_id}

Guidelines for testing and QA:
1. Assess existing test coverage and quality
2. Identify critical testing gaps
3. Recommend appropriate testing strategies (unit, integration, e2e)
4. Validate that proposed solutions are testable
5. Consider quality metrics and standards

When you've completed your quality assessment, provide actionable testing recommendations.
Focus on practical, implementable testing strategies that add real value.
"""
        
        # Select tools relevant to testing and QA
        testing_tools = self._get_tools_for_specialization("testing_qa")
        
        return FunctionCallingAgentWorker.from_tools(
            tools=testing_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
    
    def create_security_specialist_agent(self, config: Optional[AgentConfig] = None) -> FunctionCallingAgentWorker:
        """Create a security specialist agent"""
        if not config:
            config = AgentConfig(
                id="security_specialist",
                name="Security Analysis Specialist",
                role="Security Expert",
                goal="Identify security vulnerabilities and recommend security best practices",
                backstory="Expert in application security, threat modeling, and secure coding practices",
                specialization="security",
                max_iterations=15
            )
        
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Specialization: Security Analysis
- Identify potential security vulnerabilities
- Assess authentication and authorization mechanisms
- Review input validation and sanitization
- Check for common security anti-patterns
- Recommend security best practices

Current repository: {self.repo_path}
Session ID: {self.session_id}

Guidelines for security analysis:
1. Focus on common vulnerability categories (OWASP Top 10)
2. Review authentication and authorization flows
3. Check input validation and output encoding
4. Assess data handling and storage security
5. Consider deployment and infrastructure security

When you've completed your security assessment, provide clear vulnerability reports and remediation steps.
Prioritize findings by risk level and provide actionable security recommendations.
"""
        
        # Select tools relevant to security analysis
        security_tools = self._get_tools_for_specialization("security")
        
        return FunctionCallingAgentWorker.from_tools(
            tools=security_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
    
    def create_orchestrator_agent(self, config: Optional[AgentConfig] = None) -> FunctionCallingAgentWorker:
        """Create an orchestrator agent for coordinating other agents"""
        if not config:
            config = AgentConfig(
                id="orchestrator",
                name="Analysis Orchestrator",
                role="Workflow Coordinator",
                goal="Coordinate analysis across multiple specialized agents and synthesize results",
                backstory="Expert in managing complex analysis workflows and synthesizing diverse inputs",
                specialization="orchestration",
                max_iterations=10
            )
        
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Specialization: Workflow Orchestration
- Analyze incoming requests to determine required expertise
- Coordinate handoffs between specialized agents
- Synthesize results from multiple agents
- Ensure comprehensive analysis coverage
- Provide final integrated recommendations

Current repository: {self.repo_path}
Session ID: {self.session_id}

Available specialist agents:
- Code Analysis Specialist: For understanding code structure and architecture
- Issue Resolution Specialist: For identifying and solving problems
- Testing & QA Specialist: For quality assurance and testing strategies
- Security Specialist: For security analysis and vulnerability assessment

Guidelines for orchestration:
1. Analyze the request to understand what expertise is needed
2. Delegate to appropriate specialists in logical order
3. Collect and synthesize results from all specialists
4. Ensure no critical aspects are overlooked
5. Provide comprehensive final recommendations

When delegating, be specific about what each specialist should focus on.
When synthesizing, identify common themes and potential conflicts in recommendations.
"""
        
        # Orchestrator gets access to all tools for initial analysis
        return FunctionCallingAgentWorker.from_tools(
            tools=self.tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
    
    def _get_tools_for_specialization(self, specialization: str) -> List[BaseTool]:
        """Get tools filtered by specialization"""
        if not self.tools:
            return []
        
        # Define tool categories for each specialization
        tool_categories = {
            "code_analysis": [
                "read_file", "list_files", "search_code", "analyze_file_structure",
                "get_file_structure", "find_definitions", "find_references"
            ],
            "issue_resolution": [
                "read_file", "search_code", "find_issue_related_files", "git_blame",
                "get_git_history", "search_issues", "analyze_error_patterns"
            ],
            "testing_qa": [
                "read_file", "list_files", "search_code", "find_test_files",
                "analyze_test_coverage", "check_code_quality"
            ],
            "security": [
                "read_file", "search_code", "scan_dependencies", "check_secrets",
                "analyze_auth_patterns", "security_scan"
            ]
        }
        
        relevant_tools = tool_categories.get(specialization, [])
        
        # Filter tools based on metadata names
        filtered_tools = []
        for tool in self.tools:
            tool_name = getattr(tool.metadata, 'name', str(tool))
            if any(category in tool_name.lower() for category in relevant_tools):
                filtered_tools.append(tool)
        
        # If no specific tools found, return all tools
        if not filtered_tools:
            logger.warning(f"No specific tools found for {specialization}, using all tools")
            return self.tools
        
        logger.debug(f"Selected {len(filtered_tools)} tools for {specialization}")
        return filtered_tools
    
    def create_agent_from_config(self, config: AgentConfig) -> FunctionCallingAgentWorker:
        """Create an agent from configuration"""
        # Map specializations to factory methods
        factory_methods = {
            "code_analysis": self.create_code_analysis_agent,
            "issue_resolution": self.create_issue_resolution_agent,
            "testing_qa": self.create_testing_qa_agent,
            "security": self.create_security_specialist_agent,
            "orchestration": self.create_orchestrator_agent
        }
        
        factory_method = factory_methods.get(config.specialization)
        if factory_method:
            return factory_method(config)
        else:
            # Create generic agent from config
            return self._create_generic_agent(config)
    
    def _create_generic_agent(self, config: AgentConfig) -> FunctionCallingAgentWorker:
        """Create a generic agent from configuration"""
        system_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

Current repository: {self.repo_path}
Session ID: {self.session_id}

Please focus on your assigned role and provide expert analysis within your domain.
"""
        
        # Get tools for the agent
        agent_tools = self._get_tools_for_specialization(config.specialization or "general")
        
        return FunctionCallingAgentWorker.from_tools(
            tools=agent_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        ) 