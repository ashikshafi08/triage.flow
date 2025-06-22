"""
LlamaIndex + CrewAI Integration following the cookbook pattern
https://docs.llamaindex.ai/en/stable/examples/cookbooks/crewai_llamaindex/
"""

from typing import List, Any, Optional
from crewai_tools import LlamaIndexTool
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent

# Import safety-specific tools
from .tool_converter import (
    SemgrepScanner,
    HallucinationDetector,
    SecurityPatternAnalyzer,
    CodeQualityAnalyzer
)


def create_llamaindex_tools() -> List[FunctionTool]:
    """Create LlamaIndex FunctionTools for safety analysis"""
    
    # Initialize safety-specific tools
    semgrep = SemgrepScanner()
    hallucination = HallucinationDetector()
    security_analyzer = SecurityPatternAnalyzer()
    quality_analyzer = CodeQualityAnalyzer()
    
    # Create LlamaIndex FunctionTools
    tools = [
        FunctionTool.from_defaults(
            fn=semgrep._run,
            name="semgrep_scanner",
            description="Scan code for security vulnerabilities using Semgrep with OWASP, CWE, and other rule sets"
        ),
        FunctionTool.from_defaults(
            fn=hallucination._run,
            name="hallucination_detector",
            description="Detect AI hallucinations by verifying APIs, imports, and patterns exist in the codebase"
        ),
        FunctionTool.from_defaults(
            fn=security_analyzer._run,
            name="security_pattern_analyzer",
            description="Analyze code for security anti-patterns like SQL injection, command injection, weak crypto"
        ),
        FunctionTool.from_defaults(
            fn=quality_analyzer._run,
            name="code_quality_analyzer",
            description="Analyze code quality metrics including complexity, maintainability, and best practices"
        )
    ]
    
    return tools


def convert_to_crewai_tools(
    llamaindex_tools: List[FunctionTool],
    existing_rag_system: Optional[Any] = None
) -> List[LlamaIndexTool]:
    """
    Convert LlamaIndex tools to CrewAI tools following the cookbook pattern
    """
    crewai_tools = []
    
    for tool in llamaindex_tools:
        # Create CrewAI tool from LlamaIndex tool
        crewai_tool = LlamaIndexTool.from_tool(tool)
        crewai_tools.append(crewai_tool)
    
    # If we have an existing RAG system, create additional tools
    if existing_rag_system:
        # Create a RAG query tool
        def query_rag(query: str) -> str:
            """Query the existing RAG system for code context"""
            try:
                results = existing_rag_system.query(query, k=5)
                return str(results)
            except Exception as e:
                return f"Error querying RAG: {str(e)}"
        
        rag_tool = FunctionTool.from_defaults(
            fn=query_rag,
            name="query_codebase_rag",
            description="Query the codebase using the existing RAG system for semantic search"
        )
        
        crewai_tools.append(LlamaIndexTool.from_tool(rag_tool))
    
    return crewai_tools


def integrate_existing_explorer_tools(explorer_instance: Any) -> List[LlamaIndexTool]:
    """
    Integrate existing AgenticCodebaseExplorer tools with CrewAI
    Following the pattern from the cookbook
    """
    crewai_tools = []
    
    if not explorer_instance:
        return crewai_tools
    
    # Import the tool creation function
    try:
        from src.agent_tools.tool_registry import create_all_tools
        
        # Get all existing LlamaIndex tools
        existing_tools = create_all_tools(explorer_instance)
        
        # Convert each to CrewAI tool
        for tool in existing_tools:
            crewai_tool = LlamaIndexTool.from_tool(tool)
            crewai_tools.append(crewai_tool)
            
    except ImportError:
        pass
    
    return crewai_tools


def create_safety_crew_tools(
    explorer_instance: Optional[Any] = None,
    rag_system: Optional[Any] = None
) -> tuple[List[LlamaIndexTool], List[LlamaIndexTool]]:
    """
    Create all tools needed for the safety crew
    Returns: (safety_specific_tools, existing_system_tools)
    """
    
    # Create safety-specific LlamaIndex tools
    safety_llamaindex_tools = create_llamaindex_tools()
    safety_crewai_tools = convert_to_crewai_tools(safety_llamaindex_tools, rag_system)
    
    # Integrate existing explorer tools if available
    existing_crewai_tools = []
    if explorer_instance:
        existing_crewai_tools = integrate_existing_explorer_tools(explorer_instance)
    
    return safety_crewai_tools, existing_crewai_tools