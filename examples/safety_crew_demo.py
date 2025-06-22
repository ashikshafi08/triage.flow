"""
Safety Crew Demo - Shows how to use the safety crew with existing infrastructure

This demonstrates:
1. Using CrewAI with LlamaIndex tools
2. Integrating with existing RAG system
3. Running safety analysis on code
"""

import asyncio
import os
from typing import Dict, Any

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool

# LlamaIndex imports
from llama_index.core.tools import FunctionTool

# Existing infrastructure imports (when available)
try:
    from src.config import settings
    from src.agent_tools.llm_config import get_llm_instance
    from src.llm_client import LLMClient
    from src.agentic_rag import CompositeAgenticRetriever
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    print("Note: Running without existing infrastructure integration")

# Safety crew imports
from src.safety_crew.tools.tool_converter import (
    SemgrepScanner,
    HallucinationDetector,
    SecurityPatternAnalyzer,
    CodeQualityAnalyzer
)


def create_safety_tools() -> Dict[str, list]:
    """Create safety analysis tools"""
    
    # Create LlamaIndex tools
    semgrep_tool = FunctionTool.from_defaults(
        fn=SemgrepScanner()._run,
        name="semgrep_scanner",
        description="Scan code for security vulnerabilities"
    )
    
    hallucination_tool = FunctionTool.from_defaults(
        fn=HallucinationDetector()._run,
        name="hallucination_detector",
        description="Detect AI hallucinations in code"
    )
    
    security_pattern_tool = FunctionTool.from_defaults(
        fn=SecurityPatternAnalyzer()._run,
        name="security_pattern_analyzer",
        description="Analyze security anti-patterns"
    )
    
    quality_tool = FunctionTool.from_defaults(
        fn=CodeQualityAnalyzer()._run,
        name="code_quality_analyzer",
        description="Analyze code quality metrics"
    )
    
    # Convert to CrewAI tools
    tools = {
        "security": [
            LlamaIndexTool.from_tool(semgrep_tool),
            LlamaIndexTool.from_tool(security_pattern_tool)
        ],
        "grounding": [
            LlamaIndexTool.from_tool(hallucination_tool)
        ],
        "quality": [
            LlamaIndexTool.from_tool(quality_tool)
        ]
    }
    
    return tools


def create_safety_agents(llm: Any, tools: Dict[str, list]) -> Dict[str, Agent]:
    """Create safety analysis agents"""
    
    agents = {
        "security": Agent(
            role="Security Expert",
            goal="Find all security vulnerabilities",
            backstory="You are a security expert specializing in code analysis.",
            tools=tools["security"],
            llm=llm,
            verbose=True
        ),
        
        "grounding": Agent(
            role="AI Safety Expert",
            goal="Detect hallucinations in AI-generated code",
            backstory="You verify that all code elements are real and grounded.",
            tools=tools["grounding"],
            llm=llm,
            verbose=True
        ),
        
        "quality": Agent(
            role="Code Quality Expert",
            goal="Ensure code quality and maintainability",
            backstory="You are an architect focused on clean, maintainable code.",
            tools=tools["quality"],
            llm=llm,
            verbose=True
        ),
        
        "orchestrator": Agent(
            role="Safety Coordinator",
            goal="Synthesize all findings into actionable insights",
            backstory="You coordinate safety analysis and prioritize findings.",
            tools=[],  # No specific tools, just synthesis
            llm=llm,
            verbose=True,
            allow_delegation=True
        )
    }
    
    return agents


def create_safety_tasks(agents: Dict[str, Agent], code: str) -> list:
    """Create tasks for safety analysis"""
    
    tasks = [
        Task(
            description=f"""
            Analyze this code for security vulnerabilities:
            
            ```python
            {code}
            ```
            
            Use Semgrep to scan for OWASP Top 10, CWE issues, and security patterns.
            Report all findings with severity and remediation.
            """,
            expected_output="Detailed security findings with remediation",
            agent=agents["security"]
        ),
        
        Task(
            description=f"""
            Check this code for AI hallucinations:
            
            ```python
            {code}
            ```
            
            Verify all imports, APIs, and patterns are real.
            Report any hallucinated elements.
            """,
            expected_output="List of hallucinations with evidence",
            agent=agents["grounding"]
        ),
        
        Task(
            description=f"""
            Review code quality:
            
            ```python
            {code}
            ```
            
            Analyze complexity, maintainability, and best practices.
            Suggest improvements.
            """,
            expected_output="Quality metrics and improvement suggestions",
            agent=agents["quality"]
        ),
        
        Task(
            description="""
            Synthesize all findings from security, grounding, and quality analysis.
            
            Create a prioritized action plan with:
            1. Critical issues to fix immediately
            2. Important improvements for this sprint
            3. Long-term quality enhancements
            
            Focus on practical, actionable recommendations.
            """,
            expected_output="Executive summary with prioritized action plan",
            agent=agents["orchestrator"],
            context=[tasks[0], tasks[1], tasks[2]]  # Depends on previous tasks
        )
    ]
    
    return tasks


async def analyze_code_safety(code: str, session_id: str = "demo"):
    """Run safety analysis on code"""
    
    print("🔍 Starting Safety Analysis...\n")
    
    # Initialize LLM using existing configuration
    if INFRASTRUCTURE_AVAILABLE:
        # Use existing LLM configuration
        llm = get_llm_instance()
        print(f"Using LLM: {settings.llm_provider} with model {settings.default_model}")
    else:
        print("Error: Infrastructure not available. Please ensure all dependencies are installed.")
        return
    
    # Create tools
    print("🛠️  Creating safety analysis tools...")
    tools = create_safety_tools()
    
    # Create agents
    print("👥 Creating specialized agents...")
    agents = create_safety_agents(llm, tools)
    
    # Create tasks
    print("📋 Creating analysis tasks...")
    tasks = create_safety_tasks(agents, code)
    
    # Create crew
    print("🚀 Assembling the crew...\n")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True
    )
    
    # Run analysis
    print("🔬 Running safety analysis...\n")
    print("-" * 50)
    
    try:
        result = crew.kickoff()
        
        print("\n" + "-" * 50)
        print("\n✅ Safety Analysis Complete!\n")
        print("📊 Results:")
        print(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return None


def main():
    """Demo the safety crew"""
    
    # Example vulnerable code
    vulnerable_code = '''
import os
import hashlib
import subprocess
from flask import request, render_template_string

def process_user_input(user_data):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
    
    # Command injection vulnerability
    filename = user_data.get('filename', 'default.txt')
    os.system(f"cat /tmp/{filename}")
    
    # Weak cryptography
    password = user_data['password']
    hashed = hashlib.md5(password.encode()).hexdigest()
    
    # XSS vulnerability
    template = f"<h1>Welcome {user_data['name']}</h1>"
    return render_template_string(template)

def execute_command(cmd):
    # Command injection via subprocess
    result = subprocess.call(cmd, shell=True)
    return result

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"

# Non-existent import (hallucination)
from super_secure_lib import MagicEncryptor

def encrypt_data(data):
    # Using hallucinated API
    encryptor = MagicEncryptor()
    return encryptor.ultra_secure_encrypt(data)
'''
    
    print("🎯 Safety Crew Demo - AI-Powered Code Safety Analysis")
    print("=" * 60)
    print("\n📄 Analyzing the following code for safety issues:\n")
    print(vulnerable_code)
    print("\n" + "=" * 60 + "\n")
    
    # Run analysis
    asyncio.run(analyze_code_safety(vulnerable_code))


if __name__ == "__main__":
    main()