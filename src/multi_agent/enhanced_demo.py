"""
Enhanced Multi-Agent Demo
Demonstrates the complete integration of:
- LocalRepoContextExtractor (code RAG)
- IssueAwareRAG (issue/PR RAG) 
- AgenticCodebaseExplorer (100+ agentic tools)
- Multi-Agent Orchestration (planning → research → implementation → validation)
"""

import asyncio
import time
import json
import os
from pathlib import Path
from typing import Dict, Any

from ..agentic_rag import AgenticRAGSystem
from .enhanced_agents import EnhancedWorkflowOrchestrator
from .structured_outputs import EnhancedMultiAgentResult


class EnhancedMultiAgentDemo:
    """Comprehensive demo of the enhanced multi-agent system"""
    
    def __init__(self):
        self.agentic_rag = None
        self.orchestrator = None
        self.session_id = "enhanced_demo"
        
    async def initialize(self, repo_url: str = None, branch: str = "main") -> None:
        """Initialize the complete system"""
        print("🚀 Initializing Enhanced Multi-Agent System")
        print("=" * 60)
        
        # Use current repository if no URL provided
        if not repo_url:
            current_dir = Path.cwd()
            repo_url = f"file://{current_dir}"
            print(f"📁 Using current directory: {current_dir}")
        
        start_time = time.time()
        
        try:
            # Initialize AgenticRAGSystem (this combines all RAG systems)
            print("🧠 Initializing AgenticRAGSystem...")
            print("   ├── LocalRepoContextExtractor (code RAG)")
            print("   ├── IssueAwareRAG (issue/PR RAG)") 
            print("   └── AgenticCodebaseExplorer (100+ tools)")
            
            self.agentic_rag = AgenticRAGSystem(self.session_id)
            
            # Initialize core systems first (code RAG + agentic tools)
            await self.agentic_rag.initialize_core_systems(repo_url, branch)
            print("   ✅ Core systems initialized")
            
            # Initialize issue RAG asynchronously (this can take time)
            session_metadata = {"metadata": {}}
            await self.agentic_rag.initialize_issue_rag_async(session_metadata)
            print("   ✅ Issue RAG initialized")
            
            # Initialize enhanced workflow orchestrator
            self.orchestrator = EnhancedWorkflowOrchestrator(self.session_id, self.agentic_rag)
            
            init_time = time.time() - start_time
            print(f"✅ Enhanced Multi-Agent System initialized in {init_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def run_comprehensive_demo(self) -> None:
        """Run comprehensive demonstration of all capabilities"""
        if not self.orchestrator:
            print("❌ System not initialized. Call initialize() first.")
            return
        
        print("🎯 Running Comprehensive Enhanced Multi-Agent Demo")
        print("=" * 60)
        
        # Demo queries showcasing different capabilities
        demo_queries = [
            {
                "name": "🐛 Bug Investigation",
                "query": "[BUG] Memory leak in authentication system",
                "description": "Tests issue RAG + agentic tools for debugging"
            },
            {
                "name": "✨ Feature Development", 
                "query": "Create a rate limiting middleware for FastAPI",
                "description": "Tests code RAG + implementation planning"
            },
            {
                "name": "🏗️ Architecture Analysis",
                "query": "How is the multi-agent system structured and how can I extend it?",
                "description": "Tests comprehensive code analysis"
            },
            {
                "name": "🔍 Code Search & Understanding",
                "query": "Find files that handle error handling and show me examples",
                "description": "Tests file-oriented search + content analysis"
            }
        ]
        
        results = []
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n🎯 Demo {i}/4: {demo['name']}")
            print(f"Query: {demo['query']}")
            print(f"Purpose: {demo['description']}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = await self.orchestrator.execute_enhanced_workflow(demo["query"])
                execution_time = time.time() - start_time
                
                # Display comprehensive results
                await self._display_enhanced_result(result, demo["name"])
                results.append(result)
                
                print(f"⚡ Total execution time: {execution_time:.2f}s")
                print()
                
            except Exception as e:
                print(f"❌ Demo {i} failed: {e}")
                print()
        
        # Display summary
        await self._display_demo_summary(results)
    
    async def _display_enhanced_result(self, result: EnhancedMultiAgentResult, demo_name: str) -> None:
        """Display comprehensive results from enhanced workflow"""
        
        print("📊 ENHANCED WORKFLOW RESULTS")
        print("-" * 30)
        
        # Query Analysis
        print(f"🔍 Query Analysis:")
        print(f"   Type: {result.query_analysis.query_type}")
        print(f"   Scope: {result.query_analysis.scope}")
        print(f"   Complexity: {result.query_analysis.estimated_complexity}/10")
        print(f"   Domains: {', '.join(result.query_analysis.technical_domains)}")
        
        # Research Results
        print(f"\n🧠 Research Results (Quality: {result.research_results.research_quality_score:.1f}/10):")
        rag_context = result.research_results.rag_context
        
        print(f"   📁 Code Sources: {len(rag_context.sources)}")
        for source in rag_context.sources[:3]:  # Show top 3
            print(f"      • {source.file} ({source.language})")
        
        print(f"   🎫 Related Issues: {len(rag_context.related_issues)}")
        for issue in rag_context.related_issues[:2]:  # Show top 2
            print(f"      • #{issue.number}: {issue.title[:50]}... ({issue.state})")
        
        # Agentic Analysis
        if result.research_results.agentic_analysis:
            agentic = result.research_results.agentic_analysis
            print(f"   🔧 Agentic Tools Used: {len(agentic.tools_used)}")
            print(f"      {', '.join(agentic.tools_used)}")
            print(f"   🎯 Confidence: {agentic.confidence_score:.1f}")
        
        # Combined Insights
        print(f"\n💡 Key Insights:")
        for insight in result.research_results.combined_insights[:3]:
            print(f"   • {insight}")
        
        # Implementation Strategy
        print(f"\n🛠️ Implementation Strategy:")
        strategy = result.implementation_strategy
        print(f"   Approach: {strategy.high_level_approach}")
        print(f"   Technologies: {', '.join(strategy.technology_choices)}")
        print(f"   Files: {', '.join(strategy.file_organization[:3])}")
        
        # Validation Results
        print(f"\n✅ Validation Results:")
        validation = result.validation_feedback
        print(f"   Status: {validation.overall_status}")
        print(f"   Code Quality: {validation.code_quality_score:.1f}/10")
        print(f"   Security Score: {validation.security_score:.1f}/10")
        
        if validation.required_changes:
            print(f"   Required Changes:")
            for change in validation.required_changes[:2]:
                print(f"      • {change}")
        
        # Performance Metrics
        print(f"\n⚡ Performance Breakdown:")
        for step, duration in result.performance_metrics.items():
            print(f"   {step}: {duration:.2f}s")
        
        # Systems Used
        print(f"\n🔗 Systems Integration:")
        print(f"   RAG Systems: {', '.join(result.rag_systems_used)}")
        if result.tools_executed:
            print(f"   Tools Executed: {len(result.tools_executed)}")
        
        # Final Status
        status_emoji = "✅" if result.approved else "⚠️"
        print(f"\n{status_emoji} Final Status: {'APPROVED' if result.approved else 'REQUIRES REVIEW'}")
        
        # Next Steps
        print(f"\n📋 Next Steps:")
        for step in result.next_steps[:3]:
            print(f"   • {step}")
    
    async def _display_demo_summary(self, results: list) -> None:
        """Display overall demo summary and insights"""
        if not results:
            return
            
        print("🎊 ENHANCED MULTI-AGENT DEMO SUMMARY")
        print("=" * 50)
        
        # Overall statistics
        total_queries = len(results)
        avg_execution_time = sum(r.total_execution_time for r in results) / total_queries
        approved_count = sum(1 for r in results if r.approved)
        
        print(f"📈 Overall Performance:")
        print(f"   Total Queries: {total_queries}")
        print(f"   Average Execution Time: {avg_execution_time:.2f}s")
        print(f"   Approval Rate: {approved_count}/{total_queries} ({approved_count/total_queries*100:.1f}%)")
        
        # System utilization
        all_rag_systems = set()
        all_tools = set()
        
        for result in results:
            all_rag_systems.update(result.rag_systems_used)
            all_tools.update(result.tools_executed)
        
        print(f"\n🔧 System Utilization:")
        print(f"   RAG Systems Used: {', '.join(all_rag_systems)}")
        print(f"   Unique Tools Executed: {len(all_tools)}")
        
        # Quality metrics
        avg_research_quality = sum(r.research_results.research_quality_score for r in results) / total_queries
        avg_code_quality = sum(r.validation_feedback.code_quality_score for r in results) / total_queries
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   Average Research Quality: {avg_research_quality:.1f}/10")
        print(f"   Average Code Quality: {avg_code_quality:.1f}/10")
        
        # Performance by step
        step_times = {}
        for result in results:
            for step, duration in result.performance_metrics.items():
                if step not in step_times:
                    step_times[step] = []
                step_times[step].append(duration)
        
        print(f"\n⚡ Average Step Performance:")
        for step, times in step_times.items():
            avg_time = sum(times) / len(times)
            print(f"   {step}: {avg_time:.2f}s")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Enhanced Multi-Agent System is fully operational")
        print(f"   All RAG systems and agentic tools integrated seamlessly")


async def run_quick_test():
    """Quick test of the enhanced system"""
    print("🧪 Enhanced Multi-Agent Quick Test")
    print("=" * 40)
    
    demo = EnhancedMultiAgentDemo()
    
    try:
        # Initialize with current repo
        await demo.initialize()
        
        # Run a quick test query
        test_query = "Add error handling to a function"
        print(f"🎯 Test Query: {test_query}")
        print("-" * 30)
        
        start_time = time.time()
        result = await demo.orchestrator.execute_enhanced_workflow(test_query)
        execution_time = time.time() - start_time
        
        await demo._display_enhanced_result(result, "Quick Test")
        
        print(f"\n✅ Enhanced system test completed in {execution_time:.2f}s")
        print(f"Status: {'APPROVED' if result.approved else 'REQUIRES REVIEW'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def run_full_demo():
    """Run the complete comprehensive demo"""
    demo = EnhancedMultiAgentDemo()
    
    try:
        # Initialize with current repo
        await demo.initialize()
        
        # Run comprehensive demo
        await demo.run_comprehensive_demo()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_test())
    else:
        asyncio.run(run_full_demo()) 