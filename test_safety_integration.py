#!/usr/bin/env python3
"""
Test script to verify safety crew integration is working properly
"""

import sys
import os
import asyncio
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_safety_imports():
    """Test that all safety crew imports work"""
    print("Testing safety crew imports...")
    
    try:
        # Test basic imports
        from src.safety_crew import (
            SecuritySpecialist,
            GroundingSpecialist, 
            QualityArchitect,
            SafetyOrchestrator
        )
        print("✅ Agent imports successful")
        
        from src.safety_crew.crews import SafetyCrew
        print("✅ SafetyCrew import successful")
        
        from src.safety_crew.models import (
            SafetyAnalysisRequest,
            SafetyAnalysisResponse
        )
        print("✅ Model imports successful")
        
        from src.safety_crew.integration import SafetyCrewIntegration
        print("✅ Integration import successful")
        
        from src.safety_crew.tools.tool_converter import (
            SemgrepScanner,
            HallucinationDetector,
            SecurityPatternAnalyzer,
            CodeQualityAnalyzer
        )
        print("✅ Tool imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_safety_tools():
    """Test that safety tools can be instantiated"""
    print("\nTesting safety tool instantiation...")
    
    try:
        from src.safety_crew.tools.tool_converter import (
            SemgrepScanner,
            HallucinationDetector,
            SecurityPatternAnalyzer,
            CodeQualityAnalyzer
        )
        
        # Test tool creation (without actually running them)
        semgrep = SemgrepScanner()
        print("✅ SemgrepScanner created")
        
        hallucination = HallucinationDetector()
        print("✅ HallucinationDetector created")
        
        security_analyzer = SecurityPatternAnalyzer()
        print("✅ SecurityPatternAnalyzer created")
        
        quality_analyzer = CodeQualityAnalyzer()
        print("✅ CodeQualityAnalyzer created")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool instantiation error: {e}")
        return False

async def test_safety_models():
    """Test that safety models work correctly"""
    print("\nTesting safety models...")
    
    try:
        from src.safety_crew.models import (
            SafetyAnalysisRequest,
            SafetyAnalysisResponse,
            SecurityFinding,
            HallucinationFlag,
            QualityIssue,
            SafetyMetrics
        )
        
        # Test request model
        request = SafetyAnalysisRequest(
            session_id="test-session",
            code="print('Hello, world!')",
            language="python",
            analysis_depth="standard"
        )
        print("✅ SafetyAnalysisRequest created")
        
        # Test metrics model
        metrics = SafetyMetrics(
            overall_risk_score=7.5,
            security_score=8.0,
            grounding_score=9.0,
            quality_score=7.0,
            total_findings=3,
            critical_findings=0,
            high_findings=1,
            medium_findings=2,
            low_findings=0,
            auto_fixable_count=1
        )
        print("✅ SafetyMetrics created")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

async def test_api_dependencies():
    """Test that API dependencies work"""
    print("\nTesting API dependencies...")
    
    try:
        from src.api.dependencies import (
            get_session_manager,
            get_cache_manager,
            session_manager
        )
        
        # Test session manager
        sm = get_session_manager()
        print("✅ Session manager dependency working")
        
        # Test cache manager (may fail if Redis not available, that's OK)
        try:
            cache_mgr = await get_cache_manager()
            if cache_mgr:
                print("✅ Cache manager dependency working")
            else:
                print("⚠️ Cache manager not available (Redis might not be running)")
        except Exception as e:
            print(f"⚠️ Cache manager not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API dependency error: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting Safety Crew Integration Tests\n")
    
    tests = [
        test_safety_imports,
        test_safety_tools,
        test_safety_models,
        test_api_dependencies
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All safety crew integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install crewai>=0.28.0 crewai-tools>=0.8.0")
        print("2. Install Semgrep: pip install semgrep>=1.45.0")
        print("3. Test the API: python -m uvicorn src.main:app --reload")
        print("4. Try a safety analysis: POST /api/safety/{session_id}/analyze")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 