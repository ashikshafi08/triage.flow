#!/usr/bin/env python3
"""
Debug script to isolate CrewAI execution issues
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from crewai import Agent, Task, Crew, Process
from src.safety_crew.agents.base_agent import get_crewai_compatible_llm, create_agent_with_llm
from src.safety_crew.tools.tool_converter import create_safety_specific_tools

def test_minimal_crew():
    """Test a minimal crew setup to isolate issues"""
    
    print("🔧 Testing minimal CrewAI setup...")
    
    try:
        # Get LLM instance
        llm = get_crewai_compatible_llm()
        print(f"✅ LLM created: {type(llm)} = {llm}")
        
        # Create a simple agent using the helper function
        agent = create_agent_with_llm(
            role="Test Security Analyst",
            goal="Analyze the provided code for basic security issues",
            backstory="You are a security expert who can identify vulnerabilities in code.",
            max_iter=3
        )
        print("✅ Agent created successfully")
        
        # Create task
        task = Task(
            description="""
            Analyze this Python code for security vulnerabilities:

            ```python
            import os
            def get_user_data(user_id):
                query = f"SELECT * FROM users WHERE id = {user_id}"
                os.system(f"echo Processing user {user_id}")
                return query
            ```
            
            Identify any security issues and provide recommendations.
            """,
            agent=agent,
            expected_output="A detailed security analysis with identified vulnerabilities and recommendations"
        )
        print("✅ Task created successfully")
        
        # Create crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        print("✅ Crew created successfully")
        
        print("🚀 Starting crew execution...")
        
        # Execute crew
        result = crew.kickoff()
        
        print("✅ Crew execution completed!")
        print("📋 Result:", result)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in minimal crew test: {e}")
        traceback.print_exc()
        return False


def test_with_tools():
    """Test crew with LlamaIndexTools"""
    
    print("\n" + "="*50)
    print("🛠️  Testing with LlamaIndexTools...")
    
    try:
        # Import safety tools  
        from crewai_tools import LlamaIndexTool
        from llama_index.core.tools import FunctionTool
        
        # Get LLM instance
        llm = get_crewai_compatible_llm()
        print(f"✅ LLM created: {type(llm)} = {llm}")
        
        # Create a simple security analysis function
        def simple_security_scan(code: str) -> str:
            """Simple security scanner for testing"""
            issues = []
            
            # Basic checks
            if "SELECT" in code and "{" in code:
                issues.append("Potential SQL injection vulnerability")
            if "os.system(" in code:
                issues.append("Command injection risk")
            if "f\"" in code or "f'" in code:
                issues.append("F-string usage - verify no injection")
                
            if issues:
                return f"Security issues found: {', '.join(issues)}"
            else:
                return "No obvious security issues detected"
        
        # Create LlamaIndex FunctionTool
        security_function_tool = FunctionTool.from_defaults(
            fn=simple_security_scan,
            name="simple_security_scanner",
            description="Basic security scanner for code analysis"
        )
        
        # Wrap with LlamaIndexTool
        security_tool = LlamaIndexTool.from_tool(security_function_tool)
        print(f"✅ LlamaIndexTool created: {type(security_tool)}")
        
        # Create agent with tool using helper function
        agent = create_agent_with_llm(
            role="Security Analyst",
            goal="Use tools to analyze code for security vulnerabilities",
            backstory="You are a security expert who uses automated tools to find vulnerabilities.",
            tools=[security_tool],
            max_iter=3
        )
        print("✅ Agent with tools created successfully")
        
        # Create task
        task = Task(
            description="""
            Use the simple_security_scanner tool to analyze this code:

            ```python
            import os
            def get_user_data(user_id):
                query = f"SELECT * FROM users WHERE id = {user_id}"
                os.system(f"echo Processing user {user_id}")
                return query
            ```
            
            Use the tool to scan for issues and provide a summary.
            """,
            agent=agent,
            expected_output="Tool analysis results and security summary"
        )
        print("✅ Task with tools created successfully")
        
        # Create crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        print("✅ Crew with tools created successfully")
        
        print("🚀 Starting crew execution with tools...")
        
        # Execute crew
        result = crew.kickoff()
        
        print("✅ Crew with tools execution completed!")
        print("📋 Result:", result)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in tools test: {e}")
        traceback.print_exc()
        return False


def test_safety_tools():
    """Test with actual safety crew tools"""
    
    print("\n" + "="*50)
    print("🔒 Testing with actual safety tools...")
    
    try:
        # Get LLM instance
        llm = get_crewai_compatible_llm()
        print(f"✅ LLM created: {type(llm)} = {llm}")
        
        # Create safety tools
        safety_tools = create_safety_specific_tools()
        print(f"✅ Created {len(safety_tools)} safety tools")
        
        # Create agent with safety tools using helper function
        agent = create_agent_with_llm(
            role="Senior Security Engineer",
            goal="Perform comprehensive security analysis using all available tools",
            backstory="You are an expert security engineer with access to advanced scanning tools.",
            tools=safety_tools,
            max_iter=5
        )
        print("✅ Agent with safety tools created successfully")
        
        # Create task
        task = Task(
            description="""
            Use your security tools to perform a comprehensive analysis of this code:

            ```python
            import os
            def get_user_data(user_id):
                query = f"SELECT * FROM users WHERE id = {user_id}"
                os.system(f"echo Processing user {user_id}")
                password = "hardcoded_secret_123"
                return query
            ```
            
            Use multiple tools to analyze security, quality, and potential hallucinations.
            """,
            agent=agent,
            expected_output="Comprehensive security analysis using all available tools"
        )
        print("✅ Task with safety tools created successfully")
        
        # Create crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        print("✅ Crew with safety tools created successfully")
        
        print("🚀 Starting crew execution with safety tools...")
        
        # Execute crew
        result = crew.kickoff()
        
        print("✅ Crew with safety tools execution completed!")
        print("📋 Result:", result)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in safety tools test: {e}")
        traceback.print_exc()
        return False


def test_environment_setup():
    """Test environment variable setup"""
    
    print("\n" + "="*50)
    print("🌍 Testing environment setup...")
    
    try:
        from src.config import settings
        
        print("Environment Variables:")
        print(f"  OPENROUTER_API_KEY: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET'}")
        print(f"  OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
        print(f"  OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE', 'NOT SET')}")
        
        print("\nSettings:")
        print(f"  LLM Provider: {settings.llm_provider}")
        print(f"  Default Model: {settings.default_model}")
        print(f"  OpenRouter Key: {'SET' if settings.openrouter_api_key else 'NOT SET'}")
        
        # Test LLM creation
        llm = get_crewai_compatible_llm()
        print(f"\nLLM Config: {type(llm)} = {llm}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🐛 CrewAI Debug Script")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Minimal Crew", test_minimal_crew),
        ("With Tools", test_with_tools),
        ("Safety Tools", test_safety_tools)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
            print(f"❌ Unexpected error in {test_name}: {e}")
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    
    all_passed = all("✅ PASSED" in result for result in results.values())
    if all_passed:
        print("\n🎉 All tests passed! CrewAI integration is working.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    return all_passed


if __name__ == "__main__":
    main() 