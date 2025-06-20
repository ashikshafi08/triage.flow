#!/usr/bin/env python3
"""
OnboardAI Demo Script

Demonstrates how to use the OnboardAI system for intelligent developer onboarding.
This script shows the complete flow from profile creation to interactive learning.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from .onboarding_ai_core import OnboardingAICore
from .onboarding_agentic_explorer import OnboardingAgenticExplorer
from .developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle, Role
from .workflow_engine import OnboardingWorkflowEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnboardingDemo:
    """Demo class showing OnboardAI capabilities"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.workspace_id = "demo_workspace"
        self.user_id = "demo_user"
        
        # Validate repo path
        if not Path(repo_path).exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
    
    async def run_complete_demo(self):
        """Run complete onboarding demo"""
        print("🎯 OnboardAI Complete Demo")
        print("=" * 50)
        
        # Step 1: Create developer profile
        print("\n1️⃣ Creating Developer Profile...")
        profile = await self.create_demo_profile()
        print(f"✅ Profile created for {profile.role.value} developer with {profile.experience_level.value} experience")
        
        # Step 2: Initialize onboarding system
        print("\n2️⃣ Initializing Onboarding System...")
        ai_core, explorer, workflow_engine = await self.initialize_onboarding_system(profile)
        print("✅ OnboardAI system initialized")
        
        # Step 3: Generate personalized workflow
        print("\n3️⃣ Generating Personalized Workflow...")
        workflow = await workflow_engine.create_personalized_workflow(profile)
        print(f"✅ Workflow created with {len(workflow.steps)} steps, estimated {workflow.estimated_total_time // 60} hours")
        
        # Step 4: Start onboarding session
        print("\n4️⃣ Starting Onboarding Session...")
        session_result = await self.start_onboarding_session(ai_core, profile)
        print("✅ Onboarding session started")
        
        # Step 5: Interactive learning examples
        print("\n5️⃣ Interactive Learning Examples...")
        await self.demonstrate_interactive_learning(ai_core, explorer)
        
        # Step 6: Progress tracking
        print("\n6️⃣ Progress Tracking Demo...")
        await self.demonstrate_progress_tracking(ai_core)
        
        # Step 7: Generate learning summary
        print("\n7️⃣ Learning Summary...")
        summary = await explorer.generate_learning_summary()
        self.print_learning_summary(summary)
        
        print("\n🎉 Demo completed successfully!")
        print("The OnboardAI system is ready for developer onboarding.")
    
    async def create_demo_profile(self) -> DeveloperProfile:
        """Create a sample developer profile"""
        survey_data = {
            "experience_level": "mid",
            "role": "fullstack", 
            "years_experience": 3,
            "programming_languages": ["python", "javascript", "typescript"],
            "frameworks": ["fastapi", "react", "node.js"],
            "learning_style": "hands_on",
            "preferred_pace": "normal",
            "goals": ["understand_architecture", "contribute_effectively", "learn_best_practices"],
            "timezone": "UTC",
            "prefers_mentorship": True,
            "comfortable_with_ambiguity": True,
            "prefers_structured_learning": True
        }
        
        profile = DeveloperProfile.from_survey(survey_data)
        profile.name = "Demo Developer"
        profile.email = "demo@example.com"
        
        return profile
    
    async def initialize_onboarding_system(self, profile: DeveloperProfile) -> tuple:
        """Initialize all onboarding components"""
        # Initialize AI Core
        ai_core = OnboardingAICore(
            workspace_id=self.workspace_id,
            user_id=self.user_id,
            repo_path=self.repo_path,
            developer_profile=profile
        )
        
        # Initialize Agentic Explorer
        explorer = OnboardingAgenticExplorer(
            session_id=f"onboarding_{self.workspace_id}_{self.user_id}",
            repo_path=self.repo_path,
            developer_profile=profile
        )
        
        # Initialize Workflow Engine
        workflow_engine = OnboardingWorkflowEngine(
            workspace_id=self.workspace_id,
            user_id=self.user_id
        )
        
        return ai_core, explorer, workflow_engine
    
    async def start_onboarding_session(self, ai_core: OnboardingAICore, profile: DeveloperProfile) -> Dict[str, Any]:
        """Start the onboarding session"""
        initial_survey = profile.get_personalization_context()
        return await ai_core.start_onboarding_session(initial_survey)
    
    async def demonstrate_interactive_learning(self, ai_core: OnboardingAICore, explorer: OnboardingAgenticExplorer):
        """Demonstrate interactive learning capabilities"""
        
        # Example questions that a new developer might ask
        demo_questions = [
            "What is the overall architecture of this codebase?",
            "How do I set up my development environment?",
            "What are the main components I should understand first?",
            "Can you explain the data flow in this application?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n  📝 Example Question {i}: {question}")
            
            try:
                # Use AI core for educational response
                result = await ai_core.ask_question(question)
                
                print(f"  ✅ Response generated (confidence: {result.get('confidence_score', 0.8):.2f})")
                print(f"  📚 Learning hints: {len(result.get('learning_hints', []))} provided")
                print(f"  🔗 Related concepts: {', '.join(result.get('related_concepts', [])[:3])}")
                
                # Show response preview (first 100 chars)
                response_preview = result.get('response', {}).get('response', '')[:100]
                print(f"  💬 Response preview: {response_preview}...")
                
            except Exception as e:
                print(f"  ❌ Error processing question: {e}")
    
    async def demonstrate_progress_tracking(self, ai_core: OnboardingAICore):
        """Demonstrate progress tracking capabilities"""
        
        # Simulate completing some learning steps
        demo_steps = [
            {"id": "environment_setup", "time": 45, "difficulty": "just_right"},
            {"id": "architecture_overview", "time": 30, "difficulty": "too_easy"},
            {"id": "first_code_exploration", "time": 60, "difficulty": "too_hard"}
        ]
        
        for step in demo_steps:
            print(f"\n  📈 Tracking progress for: {step['id']}")
            
            try:
                result = await ai_core.track_progress(
                    step["id"],
                    step["time"], 
                    step["difficulty"]
                )
                
                print(f"  ✅ Progress recorded")
                print(f"  🎯 Achievements: {len(result.get('achievements', []))} unlocked")
                print(f"  💡 Recommendations: {len(result.get('recommendations', []))} provided")
                
            except Exception as e:
                print(f"  ❌ Error tracking progress: {e}")
    
    def print_learning_summary(self, summary: Dict[str, Any]):
        """Print learning summary in a nice format"""
        if summary.get("error"):
            print(f"  ❌ Error generating summary: {summary['error']}")
            return
        
        print(f"  📊 Total interactions: {summary.get('total_interactions', 0)}")
        print(f"  🧠 Concepts learned: {len(summary.get('concepts_learned', []))}")
        print(f"  ⏱️  Total learning time: {summary.get('total_time_minutes', 0)} minutes")
        
        achievements = summary.get('achievements', [])
        if achievements:
            print(f"  🏆 Achievements: {', '.join(achievements)}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"  💡 Recommendations: {', '.join(recommendations[:2])}...")

async def run_quick_demo():
    """Run a quick demo of OnboardAI"""
    # Use current directory as demo repo
    repo_path = str(Path.cwd())
    
    print("🚀 OnboardAI Quick Demo")
    print(f"📁 Using repository: {repo_path}")
    
    try:
        demo = OnboardingDemo(repo_path)
        await demo.run_complete_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo error details:")

def main():
    """Main entry point for demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OnboardAI Demo Script")
    parser.add_argument("--repo-path", default=".", help="Path to repository for demo")
    args = parser.parse_args()
    
    try:
        # Run the demo
        asyncio.run(run_quick_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Main error details:")

if __name__ == "__main__":
    main()