"""
Demo Script for Multi-Agent Codebase Intelligence System

This script demonstrates the capabilities of the multi-agent system
by processing various types of software development queries.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.multi_agent.core_workflow import CodebaseIntelligenceWorkflow
    from src.issue_rag import IssueAwareRAG
except ImportError:
    # Fallback for relative imports when run as module
    from .core_workflow import CodebaseIntelligenceWorkflow
    from ..issue_rag import IssueAwareRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_query(workflow: CodebaseIntelligenceWorkflow, query: str, description: str):
    """Demo a single query with the multi-agent system"""
    
    print(f"\n{'='*80}")
    print(f"DEMO: {description}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Process the query
        result = await workflow.process_query(query)
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 RESULTS (completed in {execution_time:.2f}s)")
        print(f"Status: {result.get('summary', {}).get('status', 'unknown')}")
        print(f"Risk Level: {result.get('summary', {}).get('risk_level', 'unknown')}")
        print(f"Code Valid: {result.get('summary', {}).get('code_valid', False)}")
        print(f"Approved: {result.get('approved', False)}")
        
        # Show implementation preview
        implementation = result.get('implementation_plan', {}).get('main_implementation', '')
        if implementation and isinstance(implementation, str):
            print(f"\n💻 GENERATED CODE PREVIEW:")
            lines = implementation.split('\n')[:10]  # First 10 lines
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line}")
            if len(implementation.split('\n')) > 10:
                print("    ... (more lines)")
        
        # Show feedback
        feedback = result.get('feedback', [])
        if feedback:
            print(f"\n📝 FEEDBACK:")
            for i, item in enumerate(feedback[:5], 1):  # First 5 feedback items
                print(f"{i}. {item}")
        
        # Show next steps
        next_steps = result.get('next_steps', [])
        if next_steps:
            print(f"\n📋 RECOMMENDED NEXT STEPS:")
            for i, step in enumerate(next_steps[:3], 1):  # First 3 steps
                print(f"{i}. {step}")
        
        # Show performance metrics
        stage_times = result.get('stage_times', {})
        if stage_times:
            print(f"\n⏱️  PERFORMANCE BREAKDOWN:")
            for stage, duration in stage_times.items():
                print(f"  {stage}: {duration:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None


async def run_comprehensive_demo():
    """Run a comprehensive demonstration of the multi-agent system"""
    
    print("🚀 Multi-Agent Codebase Intelligence System Demo")
    print("=" * 60)
    
    # Configuration
    session_id = f"demo_{int(time.time())}"
    repo_path = os.getcwd()  # Use current directory as repo
    
    print(f"Session ID: {session_id}")
    print(f"Repository: {repo_path}")
    
    # Initialize the workflow
    print("\n🔧 Initializing multi-agent system...")
    
    try:
        # Try to initialize with issue RAG if available
        issue_rag = None
        try:
            # This would need to be properly configured with repo details
            # issue_rag = IssueAwareRAG("owner", "repo")
            # await issue_rag.initialize()
            pass
        except Exception as e:
            logger.warning(f"Issue RAG initialization failed: {e}")
        
        workflow = CodebaseIntelligenceWorkflow(
            session_id=session_id,
            repo_path=repo_path,
            issue_rag_system=issue_rag,
            timeout=300
        )
        
        print("✅ Multi-agent system initialized successfully!")
        
        # Demo queries of different types and complexities
        demo_queries = [
            {
                "query": "Create a simple user authentication system",
                "description": "Feature Development - Authentication System",
                "expected_complexity": "medium"
            },
            {
                "query": "Add logging to the existing codebase",
                "description": "Code Improvement - Add Logging",
                "expected_complexity": "low"
            },
            {
                "query": "Fix the bug in the multi-agent workflow where validation fails",
                "description": "Bug Investigation - Validation Bug",
                "expected_complexity": "medium"
            },
            {
                "query": "Optimize the performance of the research execution step",
                "description": "Performance Optimization",
                "expected_complexity": "high"
            },
            {
                "query": "Write unit tests for the CodeValidator class",
                "description": "Testing - Unit Tests",
                "expected_complexity": "low"
            }
        ]
        
        results = []
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n🎯 Running Demo {i}/{len(demo_queries)}")
            
            result = await demo_query(
                workflow=workflow,
                query=demo["query"],
                description=demo["description"]
            )
            
            results.append({
                "query": demo["query"],
                "description": demo["description"],
                "expected_complexity": demo["expected_complexity"],
                "result": result,
                "success": result is not None
            })
            
            # Small delay between demos
            await asyncio.sleep(1)
        
        # Summary
        print(f"\n{'='*80}")
        print("📈 DEMO SUMMARY")
        print(f"{'='*80}")
        
        successful_demos = sum(1 for r in results if r["success"])
        total_demos = len(results)
        
        print(f"Total Demos Run: {total_demos}")
        print(f"Successful: {successful_demos}")
        print(f"Failed: {total_demos - successful_demos}")
        print(f"Success Rate: {(successful_demos/total_demos)*100:.1f}%")
        
        # Show results breakdown
        print(f"\n📊 RESULTS BREAKDOWN:")
        
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            query_short = result["query"][:50] + "..." if len(result["query"]) > 50 else result["query"]
            
            if result["success"] and result["result"]:
                approval_status = "✅ Approved" if result["result"].get("approved") else "⚠️  Needs Review"
                risk_level = result["result"].get("summary", {}).get("risk_level", "Unknown")
                exec_time = result["result"].get("total_execution_time", 0)
                
                print(f"{i}. {status} {query_short}")
                print(f"   Status: {approval_status} | Risk: {risk_level} | Time: {exec_time:.1f}s")
            else:
                print(f"{i}. {status} {query_short}")
                print(f"   Status: Failed")
        
        # Performance analysis
        if successful_demos > 0:
            total_time = sum(
                r["result"].get("total_execution_time", 0) 
                for r in results 
                if r["success"] and r["result"]
            )
            avg_time = total_time / successful_demos
            
            print(f"\n⏱️  PERFORMANCE ANALYSIS:")
            print(f"Total Execution Time: {total_time:.2f}s")
            print(f"Average Time per Query: {avg_time:.2f}s")
            
            # Stage analysis
            stage_totals = {}
            for result in results:
                if result["success"] and result["result"]:
                    stage_times = result["result"].get("stage_times", {})
                    for stage, duration in stage_times.items():
                        stage_totals[stage] = stage_totals.get(stage, 0) + duration
            
            if stage_totals:
                print(f"\nStage Performance (total across all queries):")
                for stage, total_duration in stage_totals.items():
                    avg_duration = total_duration / successful_demos
                    print(f"  {stage}: {avg_duration:.2f}s avg ({total_duration:.2f}s total)")
        
        print(f"\n🎉 Demo completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        logger.exception("Demo failed")
        return None


async def demo_with_streaming():
    """Demonstrate the streaming capabilities of the multi-agent system"""
    
    print("\n🌊 STREAMING DEMO")
    print("=" * 40)
    
    session_id = f"stream_demo_{int(time.time())}"
    repo_path = os.getcwd()
    
    workflow = CodebaseIntelligenceWorkflow(
        session_id=session_id,
        repo_path=repo_path,
        timeout=120
    )
    
    query = "Create a simple API endpoint for user registration"
    
    print(f"Query: {query}")
    print("\nStreaming progress updates:")
    print("-" * 40)
    
    try:
        async for event in workflow.stream_process_query(query):
            if event.get("type") == "progress":
                progress = event.get("progress", 0)
                agent = event.get("agent", "Unknown")
                task = event.get("task", "Processing...")
                
                print(f"[{progress:5.1f}%] {agent}: {task}")
                
            elif event.get("type") == "result":
                result = event.get("data")
                print(f"\n✅ Final Result:")
                print(f"Status: {result.get('summary', {}).get('status', 'unknown')}")
                print(f"Approved: {result.get('approved', False)}")
                print(f"Total Time: {result.get('total_execution_time', 0):.2f}s")
                
                return result
    
    except Exception as e:
        print(f"\n❌ Streaming demo failed: {str(e)}")
        return None


def run_quick_test():
    """Run a quick test of the multi-agent system"""
    
    print("🧪 Quick Test of Multi-Agent System")
    print("=" * 40)
    
    session_id = "quick_test"
    repo_path = os.getcwd()
    
    workflow = CodebaseIntelligenceWorkflow(
        session_id=session_id,
        repo_path=repo_path,
        timeout=60
    )
    
    query = "Add error handling to a function"
    
    async def run_test():
        try:
            result = await workflow.process_query(query)
            
            print(f"✅ Test completed successfully!")
            print(f"Query: {query}")
            print(f"Status: {result.get('summary', {}).get('status', 'unknown')}")
            print(f"Time: {result.get('total_execution_time', 0):.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return False
    
    return asyncio.run(run_test())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "stream":
        # Streaming demo mode
        asyncio.run(demo_with_streaming())
    
    else:
        # Full comprehensive demo
        asyncio.run(run_comprehensive_demo()) 