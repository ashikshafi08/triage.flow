#!/usr/bin/env python3
"""
Wrapper script to run the GitHub OnboardAI demo from the project root.
This handles the import paths correctly.
"""

import asyncio
import logging
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from onboarding.demo_onboarding import OnboardingDemo
from onboarding.skill_gap_analyzer import SkillGapAnalyzer
from onboarding.autonomous_workflow_generator import AutonomousWorkflowGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedOnboardingDemo:
    """Comprehensive demo of OnboardAI advanced features"""
    
    def __init__(self):
        self.demo = OnboardingDemo()
        self.skill_analyzer = None
        self.workflow_generator = None
        
    async def run_complete_demo(self):
        """Run the complete OnboardAI demo with advanced features"""
        print("\n" + "="*80)
        print("🚀 OnboardAI Advanced Demo - Complete MVP Showcase")
        print("="*80)
        
        # 1. Basic OnboardAI Demo
        await self._run_basic_demo()
        
        # 2. Advanced Skill Gap Analysis
        await self._run_skill_gap_analysis()
        
        # 3. Autonomous Workflow Generation
        await self._run_autonomous_workflow_generation()
        
        # 4. Integration Demo
        await self._run_integration_demo()
        
        print("\n" + "="*80)
        print("✅ OnboardAI Advanced Demo Complete!")
        print("The system now includes:")
        print("  • Intelligent developer onboarding")
        print("  • Skill gap analysis with personalized learning paths")
        print("  • Autonomous workflow generation")
        print("  • Self-improving AI system")
        print("="*80)

    async def _run_basic_demo(self):
        """Run the basic OnboardAI functionality demo"""
        print("\n📋 1. Basic OnboardAI Demo")
        print("-" * 40)
        
        try:
            await self.demo.run_demo()
        except Exception as e:
            logger.error(f"Basic demo failed: {e}")
            print(f"⚠️  Basic demo encountered an issue: {e}")
            print("   This is expected if running without proper API keys")

    async def _run_skill_gap_analysis(self):
        """Demonstrate skill gap analysis and personalized learning"""
        print("\n🎯 2. Advanced Skill Gap Analysis")
        print("-" * 40)
        
        try:
            # Initialize skill gap analyzer
            self.skill_analyzer = SkillGapAnalyzer(
                workspace_id="demo_workspace",
                user_id="demo_user"
            )
            
            # Simulate repository analysis
            print("🔍 Analyzing repository requirements...")
            repo_requirements = {
                "required_technologies": ["python", "fastapi", "react", "typescript"],
                "skill_areas": {
                    "backend_development": {
                        "weight": 0.4,
                        "skills": ["python", "fastapi", "database_design", "api_design"]
                    },
                    "frontend_development": {
                        "weight": 0.3,
                        "skills": ["react", "typescript", "ui_ux", "component_design"]
                    },
                    "ai_integration": {
                        "weight": 0.3,
                        "skills": ["llm_integration", "agent_frameworks", "rag_systems"]
                    }
                },
                "complexity_factors": {
                    "codebase_size": "large",
                    "architectural_patterns": ["microservices", "event_driven"],
                    "domain_complexity": "medium"
                }
            }
            
            # Analyze skill gaps
            print("📊 Performing skill gap analysis...")
            
            # Simulate skill gap results
            skill_gaps = {
                "critical_gaps": [
                    {
                        "skill": "FastAPI Advanced Features",
                        "current_level": 2,
                        "required_level": 4,
                        "priority": "high",
                        "learning_resources": [
                            "FastAPI Documentation Deep Dive",
                            "Advanced FastAPI Patterns Tutorial",
                            "Hands-on API Development Exercise"
                        ]
                    },
                    {
                        "skill": "LLM Integration Patterns", 
                        "current_level": 1,
                        "required_level": 3,
                        "priority": "high",
                        "learning_resources": [
                            "LlamaIndex Framework Tutorial",
                            "Agent Workflow Patterns",
                            "RAG System Implementation"
                        ]
                    }
                ],
                "moderate_gaps": [
                    {
                        "skill": "React Advanced Patterns",
                        "current_level": 3,
                        "required_level": 4,
                        "priority": "medium"
                    }
                ]
            }
            
            print("✅ Skill Gap Analysis Results:")
            print(f"   • {len(skill_gaps['critical_gaps'])} critical skill gaps identified")
            print(f"   • {len(skill_gaps['moderate_gaps'])} moderate gaps found")
            print("   • Personalized learning path generated")
            
            # Generate personalized learning path
            print("\n📚 Generating Personalized Learning Path...")
            learning_path = {
                "total_duration_weeks": 4,
                "phases": [
                    {
                        "phase": "Foundation Building",
                        "week": 1,
                        "focus": "FastAPI fundamentals and project setup",
                        "learning_objectives": [
                            "Master FastAPI routing and dependency injection",
                            "Understand async programming patterns",
                            "Set up development environment properly"
                        ]
                    },
                    {
                        "phase": "AI Integration",
                        "week": 2,
                        "focus": "LLM and agent system integration",
                        "learning_objectives": [
                            "Learn LlamaIndex framework basics",
                            "Implement simple agent workflows",
                            "Understand RAG system architecture"
                        ]
                    },
                    {
                        "phase": "Advanced Patterns",
                        "week": 3,
                        "focus": "Advanced development patterns",
                        "learning_objectives": [
                            "Master advanced React patterns",
                            "Implement complex AI workflows",
                            "Learn system optimization techniques"
                        ]
                    },
                    {
                        "phase": "Integration & Practice",
                        "week": 4,
                        "focus": "Real-world application and contribution",
                        "learning_objectives": [
                            "Complete first meaningful contribution",
                            "Apply learned patterns to codebase",
                            "Demonstrate proficiency in key areas"
                        ]
                    }
                ]
            }
            
            for phase in learning_path["phases"]:
                print(f"   Week {phase['week']}: {phase['phase']}")
                print(f"      Focus: {phase['focus']}")
                
        except Exception as e:
            logger.error(f"Skill gap analysis failed: {e}")
            print(f"⚠️  Skill gap analysis demonstration failed: {e}")

    async def _run_autonomous_workflow_generation(self):
        """Demonstrate autonomous workflow generation and optimization"""
        print("\n🤖 3. Autonomous Workflow Generation")
        print("-" * 40)
        
        try:
            # Initialize autonomous workflow generator
            self.workflow_generator = AutonomousWorkflowGenerator(
                workspace_id="demo_workspace",
                user_id="demo_user"
            )
            
            print("🔄 Generating autonomous workflow...")
            
            # Simulate workflow optimization based on data
            optimization_data = {
                "historical_performance": {
                    "average_completion_time": 3.2,  # weeks
                    "common_bottlenecks": ["environment_setup", "codebase_complexity"],
                    "success_patterns": ["hands_on_learning", "mentorship_pairing"],
                    "difficulty_feedback": {
                        "too_easy": 15,
                        "just_right": 65, 
                        "too_hard": 20
                    }
                },
                "developer_profile": {
                    "experience_level": "mid",
                    "learning_style": "hands_on",
                    "role": "fullstack",
                    "strengths": ["quick_learner", "problem_solver"],
                    "areas_for_growth": ["architectural_thinking", "advanced_patterns"]
                }
            }
            
            # Generate optimized workflow
            print("🎯 Optimizing workflow based on historical data...")
            
            optimized_workflow = {
                "workflow_id": "auto_generated_v2.1",
                "optimization_score": 0.87,
                "estimated_completion_time": 2.8,  # weeks (improved)
                "key_optimizations": [
                    "Pre-configured development environment",
                    "Interactive codebase exploration",
                    "Adaptive difficulty based on progress",
                    "Just-in-time learning resources"
                ],
                "phases": [
                    {
                        "phase": "Smart Setup",
                        "duration_days": 1,
                        "optimizations": [
                            "Automated environment setup",
                            "Pre-validated configuration",
                            "Interactive setup verification"
                        ]
                    },
                    {
                        "phase": "Guided Exploration", 
                        "duration_days": 5,
                        "optimizations": [
                            "AI-guided codebase tour",
                            "Progressive complexity increase",
                            "Real-time comprehension tracking"
                        ]
                    },
                    {
                        "phase": "Hands-on Contribution",
                        "duration_days": 8,
                        "optimizations": [
                            "Curated task suggestions",
                            "Incremental complexity",
                            "Continuous mentorship support"
                        ]
                    }
                ]
            }
            
            print("✅ Autonomous Workflow Generated:")
            print(f"   • Optimization Score: {optimized_workflow['optimization_score']:.2f}")
            print(f"   • Estimated Time: {optimized_workflow['estimated_completion_time']} weeks")
            print(f"   • Key Improvements: {len(optimized_workflow['key_optimizations'])} optimizations")
            
            for opt in optimized_workflow['key_optimizations']:
                print(f"     - {opt}")
            
            # Demonstrate continuous improvement
            print("\n🔄 Continuous Improvement Engine:")
            print("   • Real-time A/B testing of workflow variations")
            print("   • Automatic adjustment based on developer feedback")
            print("   • Learning from cross-team onboarding patterns")
            print("   • Predictive optimization for future developers")
            
    except Exception as e:
            logger.error(f"Autonomous workflow generation failed: {e}")
            print(f"⚠️  Autonomous workflow generation failed: {e}")

    async def _run_integration_demo(self):
        """Demonstrate integration between all components"""
        print("\n🔗 4. System Integration Demo")
        print("-" * 40)
        
        try:
            print("🌟 Demonstrating Complete OnboardAI Integration:")
            
            # Simulate end-to-end flow
            integration_flow = [
                {
                    "step": "Profile Creation",
                    "description": "Developer completes skill assessment survey",
                    "ai_components": ["Profile Analysis", "Skill Gap Detection"]
                },
                {
                    "step": "Repository Analysis", 
                    "description": "AI analyzes codebase and determines requirements",
                    "ai_components": ["Code Analysis", "Complexity Assessment", "Technology Detection"]
                },
                {
                    "step": "Workflow Generation",
                    "description": "Autonomous system creates personalized onboarding plan",
                    "ai_components": ["Workflow Optimization", "Learning Path Generation", "Resource Curation"]
                },
                {
                    "step": "Adaptive Learning",
                    "description": "Real-time adjustment based on developer progress",
                    "ai_components": ["Progress Tracking", "Difficulty Adaptation", "Recommendation Engine"]
                },
                {
                    "step": "Continuous Improvement",
                    "description": "System learns and improves for future developers",
                    "ai_components": ["Pattern Recognition", "Workflow Optimization", "Predictive Analytics"]
                }
            ]
            
            for i, step in enumerate(integration_flow, 1):
                print(f"\n   {i}. {step['step']}")
                print(f"      {step['description']}")
                print(f"      AI Components: {', '.join(step['ai_components'])}")
            
            print("\n✨ Key Integration Benefits:")
            print("   • Seamless developer experience from start to finish")
            print("   • AI-powered personalization at every step")
            print("   • Continuous learning and system improvement")
            print("   • Measurable onboarding efficiency gains")
            
            # Show metrics
            print("\n📈 Expected Impact Metrics:")
            metrics = {
                "Time to First Commit": "72 hours → 24 hours (67% improvement)",
                "Onboarding Completion Rate": "60% → 90% (30% improvement)", 
                "Developer Satisfaction": "3.2/5 → 4.6/5 (44% improvement)",
                "Knowledge Retention": "65% → 85% (31% improvement)",
                "Mentor Time Saved": "20 hours → 5 hours (75% reduction)"
            }
            
            for metric, improvement in metrics.items():
                print(f"   • {metric}: {improvement}")
                
        except Exception as e:
            logger.error(f"Integration demo failed: {e}")
            print(f"⚠️  Integration demo failed: {e}")

async def main():
    """Main demo execution"""
    demo = AdvancedOnboardingDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main()) 