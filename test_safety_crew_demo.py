#!/usr/bin/env python3
"""
Demo script to test Safety Crew on actual triage.flow codebase
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_safety_crew_on_codebase():
    """Test safety crew on actual code files from the codebase"""
    
    # Test files from the codebase
    test_files = [
        "src/main.py",  # FastAPI main file
        "src/github_client.py",  # GitHub client with API calls
        "src/llm_client.py",  # LLM client with potential security concerns
        "src/agent_tools/core.py",  # Core agent tools
        "src/safety_crew/integration.py"  # Safety crew integration
    ]
    
    print("🔍 Testing Safety Crew on Triage.Flow Codebase\n")
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {file_path} (not found)")
            continue
            
        print(f"📁 Analyzing: {file_path}")
        
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue
        
        # Test the safety crew integration directly
        try:
            from src.safety_crew.integration import SafetyCrewIntegration
            from src.session_manager import SessionManager
            from src.cache.redis_cache_manager import EnhancedCacheManager
            
            # Initialize components
            session_manager = SessionManager()
            try:
                cache_manager = EnhancedCacheManager(namespace="safety_demo")
            except:
                cache_manager = None
                print("  ⚠️ Cache manager not available (Redis not running)")
            
            # Create safety crew integration
            safety_integration = SafetyCrewIntegration(
                session_manager=session_manager,
                cache_manager=cache_manager
            )
            
            # Create a test session
            session_id = f"safety_test_{Path(file_path).stem}"
            await session_manager.create_session(session_id, {
                "repo_path": "/Users/ash/Documents/ash_projects/triage.flow",
                "metadata": {"test": True}
            })
            
            # Determine programming language
            language = "python" if file_path.endswith('.py') else "javascript"
            
            print(f"  🤖 Running safety analysis...")
            
            # Create safety analysis request
            from src.safety_crew.models import SafetyAnalysisRequest
            
            request = SafetyAnalysisRequest(
                session_id=session_id,
                code=code_content,
                file_path=file_path,
                language=language,
                analysis_depth="standard"
            )
            
            # Run safety analysis using standalone method (fallback when CrewAI has issues)
            print(f"  ⚙️ Using standalone analysis (CrewAI tools need fixing)...")
            result = await safety_integration.analyze_code_standalone(
                request=request,
                repository_path="/Users/ash/Documents/ash_projects/triage.flow"
            )
            
            # Display results
            print(f"  📊 Analysis Results:")
            
            if hasattr(result, 'error') and result.error:
                print(f"    ❌ Error: {result.error}")
                continue
            
            # Check if result is a dict (error case) or SafetyAnalysisResponse
            if isinstance(result, dict):
                if result.get("error"):
                    print(f"    ❌ Error: {result['error']}")
                    continue
                else:
                    print("    ⚠️ Unexpected result format")
                    continue
                    
            metrics = result.safety_metrics
            print(f"    🔒 Security Score: {metrics.security_score}/10")
            print(f"    🎯 Grounding Score: {metrics.grounding_score}/10")
            print(f"    ⭐ Quality Score: {metrics.quality_score}/10")
            print(f"    🚨 Overall Risk: {metrics.overall_risk_score}/10")
            
            # Show findings
            security_findings = result.security_findings
            hallucination_flags = result.hallucination_flags  
            quality_issues = result.quality_issues
            
            if security_findings:
                print(f"    🛡️ Security Findings: {len(security_findings)}")
                for finding in security_findings[:3]:  # Show first 3
                    severity = finding.severity if hasattr(finding, 'severity') else finding.get('severity', 'unknown')
                    title = finding.title if hasattr(finding, 'title') else finding.get('title', 'Unknown issue')
                    print(f"      • [{severity.upper()}] {title}")
            
            if hallucination_flags:
                print(f"    🧠 Hallucination Flags: {len(hallucination_flags)}")
                for flag in hallucination_flags[:3]:  # Show first 3
                    description = flag.description if hasattr(flag, 'description') else flag.get('description', 'Unknown hallucination')
                    print(f"      • {description}")
            
            if quality_issues:
                print(f"    📈 Quality Issues: {len(quality_issues)}")
                for issue in quality_issues[:3]:  # Show first 3
                    title = issue.title if hasattr(issue, 'title') else issue.get('title', 'Unknown quality issue')
                    print(f"      • {title}")
            
            # Show recommendations
            recommendations = result.agent_recommendations
            if recommendations:
                print(f"    💡 Agent Recommendations: {len(recommendations)}")
                for rec in recommendations[:2]:  # Show first 2
                    agent = rec.agent_name if hasattr(rec, 'agent_name') else rec.get('agent_name', 'Unknown')
                    recommendation = rec.recommendation if hasattr(rec, 'recommendation') else rec.get('recommendation', 'No recommendation')
                    print(f"      • [{agent}] {recommendation[:100]}...")
            
            print()  # Empty line between files
            
        except Exception as e:
            print(f"    ❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("✅ Safety crew testing complete!")

if __name__ == "__main__":
    asyncio.run(test_safety_crew_on_codebase()) 