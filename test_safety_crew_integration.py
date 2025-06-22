#!/usr/bin/env python3
"""Test script to verify safety crew integration"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.safety_crew.models.safety_reports import SafetyAnalysisRequest
from src.safety_crew.crews.safety_crew import SafetyCrew
from src.safety_crew.integration import SafetyCrewIntegration
from src.session_manager import SessionManager
from src.config import settings

# Test code with obvious issues
TEST_CODE = """
import nonexistent_module
from fake_library import FakeClass

def process_user_input(user_data):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
    
    # Command injection
    import os
    os.system(f"echo {user_data['name']}")
    
    # Weak crypto
    import hashlib
    password_hash = hashlib.md5(user_data['password'].encode()).hexdigest()
    
    # Hardcoded secret
    api_key = "sk-1234567890abcdef"
    
    # Non-existent API call
    result = SuperAdvancedAI.process_with_magic(user_data)
    
    # Path traversal
    with open(f"../../{user_data['file']}", 'r') as f:
        content = f.read()
    
    return result
"""


async def test_safety_crew():
    """Test the safety crew implementation"""
    
    print("Testing Safety Crew Integration...\n")
    
    # Create a mock session manager
    session_manager = SessionManager()
    session_id = "test-session-123"
    
    # Test 1: Direct SafetyCrew usage
    print("Test 1: Direct SafetyCrew usage")
    print("-" * 50)
    
    safety_crew = SafetyCrew(
        existing_rag=None,  # No RAG for this test
        existing_tools=None,
        context_manager=None,
        cache_manager=None,
        verbose=True,
        use_cheap_model_for_analysis=True
    )
    
    request = SafetyAnalysisRequest(
        session_id=session_id,
        code=TEST_CODE,
        file_path="test_file.py",
        language="python",
        analysis_depth="standard",
        include_auto_fix=True
    )
    
    try:
        response = await safety_crew.analyze(request)
        
        print(f"\nAnalysis completed in {response.analysis_duration_ms}ms")
        print(f"Overall risk score: {response.safety_metrics.overall_risk_score}/10")
        print(f"Security findings: {len(response.security_findings)}")
        print(f"Hallucinations detected: {len(response.hallucination_flags)}")
        print(f"Quality issues: {len(response.quality_issues)}")
        
        # Show some findings
        if response.security_findings:
            print("\nTop Security Findings:")
            for finding in response.security_findings[:3]:
                print(f"- [{finding.severity}] {finding.title}: {finding.description}")
        
        if response.hallucination_flags:
            print("\nHallucinations Detected:")
            for hall in response.hallucination_flags[:3]:
                print(f"- {hall.type}: {hall.hallucinated_code}")
                
    except Exception as e:
        print(f"Error in test 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Integration module usage
    print("\n\nTest 2: SafetyCrewIntegration usage")
    print("-" * 50)
    
    integration = SafetyCrewIntegration(
        session_manager=session_manager,
        cache_manager=None
    )
    
    try:
        result = await integration.analyze_code(
            code=TEST_CODE,
            session_id=session_id,
            repository_path=os.getcwd(),
            file_path="test_file.py",
            analysis_depth="quick"
        )
        
        print(f"\nAnalysis status: {result.get('status')}")
        print(f"Findings: {result.get('findings', {})}")
        
    except Exception as e:
        print(f"Error in test 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Tool functionality
    print("\n\nTest 3: Individual tool testing")
    print("-" * 50)
    
    from src.safety_crew.tools.tool_converter import (
        SemgrepScanner,
        HallucinationDetector,
        SecurityPatternAnalyzer,
        CodeQualityAnalyzer
    )
    
    # Test Semgrep scanner
    print("\nTesting SemgrepScanner...")
    scanner = SemgrepScanner()
    try:
        result = scanner._run(TEST_CODE, "python")
        print(f"Semgrep result: {result[:200]}...")
    except Exception as e:
        print(f"Semgrep error: {e}")
    
    # Test hallucination detector
    print("\nTesting HallucinationDetector...")
    detector = HallucinationDetector(rag_system=None, context_manager=None)
    try:
        result = detector._run(TEST_CODE, session_id)
        print(f"Hallucination result: {result[:200]}...")
    except Exception as e:
        print(f"Hallucination error: {e}")
    
    # Test security pattern analyzer
    print("\nTesting SecurityPatternAnalyzer...")
    analyzer = SecurityPatternAnalyzer()
    try:
        result = analyzer._run(TEST_CODE)
        print(f"Security pattern result: {result[:200]}...")
    except Exception as e:
        print(f"Security pattern error: {e}")
    
    # Test code quality analyzer
    print("\nTesting CodeQualityAnalyzer...")
    quality = CodeQualityAnalyzer()
    try:
        result = quality._run(TEST_CODE)
        print(f"Quality result: {result[:200]}...")
    except Exception as e:
        print(f"Quality error: {e}")
    
    print("\n\nAll tests completed!")


if __name__ == "__main__":
    # Check if API keys are configured
    if not settings.default_model:
        print("Error: No LLM model configured. Please set OPENAI_API_KEY or OPENROUTER_API_KEY")
        sys.exit(1)
    
    asyncio.run(test_safety_crew())