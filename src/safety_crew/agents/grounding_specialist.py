"""Grounding Specialist Agent - AI Hallucination Detection Expert"""

from typing import List, Optional, Any
from crewai import Agent

from ..tools.tool_converter import create_safety_specific_tools
from .base_agent import create_agent_with_llm


class GroundingSpecialist:
    """Grounding Specialist Agent for preventing AI hallucinations"""
    
    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Any]] = None,
        rag_system: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        verbose: bool = True
    ):
        self.llm = llm
        self.verbose = verbose
        self.rag_system = rag_system
        self.context_manager = context_manager
        
        # Combine provided tools (which should already be LlamaIndexTools)
        self.tools = tools or []
        
        # If no tools provided, create safety tools
        if not self.tools:
            safety_tools = create_safety_specific_tools(
                rag_system=rag_system,
                context_manager=context_manager
            )
            # Filter for grounding-specific tools (hallucination detector and quality analyzer)
            grounding_tools = [tool for tool in safety_tools if 
                              'hallucination' in getattr(tool, 'name', str(tool)) or 
                              'quality' in getattr(tool, 'name', str(tool))]
            self.tools.extend(grounding_tools)
        
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the Grounding Specialist agent"""
        
        return create_agent_with_llm(
            role="AI Safety and Grounding Expert",
            goal="Prevent AI hallucinations by verifying all code elements against the actual codebase and ensuring accuracy",
            backstory="""You are an AI safety specialist with deep expertise in preventing hallucinations 
            in AI-generated code. You've spent years researching how LLMs can generate plausible but incorrect code.
            
            Your expertise includes:
            - Semantic code analysis and verification
            - API signature validation
            - Import and dependency verification
            - Pattern matching against real codebases
            - Understanding common LLM hallucination patterns
            - Cross-referencing with documentation and actual implementations
            
            You use advanced RAG (Retrieval Augmented Generation) systems to verify that every piece of 
            generated code corresponds to actual APIs, functions, and patterns that exist in the target codebase.
            You're meticulous about checking imports, function signatures, parameter types, and usage patterns.
            
            Your goal is zero tolerance for hallucinations - every line of code must be grounded in reality.""",
            tools=self.tools,
            verbose=self.verbose,
            max_iter=5,
            memory=True,
            allow_delegation=False
        )
        
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent