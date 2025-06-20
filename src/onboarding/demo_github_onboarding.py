#!/usr/bin/env python3
"""
Demo script showing OnboardAI working with GitHub repository URLs

This demonstrates how to use OnboardAI to onboard developers to any GitHub repository,
including examples like the AI Hedge Fund project.
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

# Handle imports properly regardless of where script is run from
script_dir = Path(__file__).parent
src_dir = script_dir.parent

# Add src to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from onboarding.onboarding_ai_core import OnboardingAICore
    from onboarding.developer_profile import DeveloperProfile, ExperienceLevel, Role, LearningStyle
    from local_repo_loader import clone_repo_to_temp
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path}")
    print(f"💡 Try running from project root: python run_github_demo.py")
    sys.exit(1)


class GitHubOnboardingDemo:
    """Demo class for GitHub repository onboarding"""
    
    def __init__(self):
        self.demo_repos = {
            "ai_hedge_fund": {
                "url": "https://github.com/virattt/ai-hedge-fund/",
                "description": "AI-powered hedge fund with multiple trading agents",
                "complexity": "intermediate",
                "focus_areas": ["AI agents", "financial trading", "multi-agent systems"]
            },
            "triage_flow": {
                "url": "https://github.com/ashikshafi08/triage.flow/",  # Replace with actual URL
                "description": "AI-powered GitHub repo analysis and code review platform",
                "complexity": "advanced", 
                "focus_areas": ["FastAPI", "React", "LlamaIndex", "AI analysis"]
            }
        }
    
    def create_demo_profile(self, experience: str = "mid", role: str = "fullstack") -> DeveloperProfile:
        """Create a demo developer profile"""
        profile = DeveloperProfile()
        profile.experience_level = ExperienceLevel(experience)
        profile.role = Role(role)
        profile.programming_languages = ["python", "javascript", "typescript"]
        profile.frameworks = ["fastapi", "react", "pandas"]
        profile.learning_style = LearningStyle.HANDS_ON
        profile.goals = [
            "understand_architecture",
            "contribute_effectively", 
            "learn_best_practices"
        ]
        profile.assessment_completed = True
        return profile
    
    async def demo_ai_hedge_fund_onboarding(self):
        """Demo onboarding to the AI Hedge Fund repository"""
        print("🚀 OnboardAI Demo: AI Hedge Fund Repository")
        print("=" * 60)
        
        repo_info = self.demo_repos["ai_hedge_fund"]
        github_url = repo_info["url"]
        
        print(f"📦 Repository: {github_url}")
        print(f"📋 Description: {repo_info['description']}")
        print(f"🎯 Complexity: {repo_info['complexity']}")
        print(f"🔍 Focus Areas: {', '.join(repo_info['focus_areas'])}")
        print()
        
        # Create developer profile
        profile = self.create_demo_profile("mid", "backend") 
        print(f"👤 Developer Profile: {profile.experience_level.value} {profile.role.value}")
        print(f"📚 Learning Style: {profile.learning_style.value}")
        print()
        
        # Clone repository and start onboarding
        print("📥 Cloning repository...")
        with clone_repo_to_temp(github_url) as repo_path:
            print(f"✅ Repository cloned to: {repo_path}")
            
            # Initialize OnboardAI
            ai_core = OnboardingAICore(
                workspace_id="demo_company",
                user_id="demo_developer", 
                repo_path=repo_path,
                developer_profile=profile
            )
            
            print("\n🤖 Initializing OnboardAI...")
            await ai_core.start_onboarding_session({
                "experience_level": "mid",
                "role": "backend",
                "goals": ["understand_ai_agents", "learn_trading_system"]
            })
            
            print("✅ OnboardAI initialized successfully!")
            print()
            
            # Demo questions
            questions = [
                "What is this AI hedge fund system and how does it work?",
                "Explain the different AI agents and their roles",
                "What are the main components I should understand first?",
                "How do the trading agents make decisions?",
                "What would be a good first task for me to contribute?"
            ]
            
            for i, question in enumerate(questions, 1):
                print(f"❓ Question {i}: {question}")
                print("-" * 50)
                
                try:
                    response = await ai_core.ask_question(question)
                    print(f"🤖 OnboardAI Response:")
                    print(response.get("response", "No response available"))
                    print()
                    
                    # Show suggestions if available
                    if response.get("suggestions"):
                        print("💡 Suggestions:")
                        for suggestion in response["suggestions"]:
                            print(f"   • {suggestion}")
                        print()
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print()
                
                # Pause between questions for readability
                if i < len(questions):
                    input("Press Enter to continue...")
                    print()
            
            # Generate codebase tour
            print("🗺️ Generating personalized codebase tour...")
            try:
                tour = await ai_core.generate_codebase_tour()
                print("✅ Codebase tour generated!")
                print("Tour steps:")
                for step in tour:
                    print(f"   {step.get('order', '?')}. {step.get('title', 'Untitled')}")
                    print(f"      {step.get('description', 'No description')}")
                print()
            except Exception as e:
                print(f"❌ Error generating tour: {e}")
                print()
            
            # Get first task suggestions
            print("🎯 Getting first task suggestions...")
            try:
                tasks = await ai_core.suggest_first_tasks()
                print("✅ Task suggestions generated!")
                for i, task in enumerate(tasks, 1):
                    print(f"   {i}. {task.get('title', 'Untitled Task')}")
                    print(f"      Difficulty: {task.get('difficulty', 'Unknown')}")
                    print(f"      Time: {task.get('estimated_time', '?')} minutes")
                    print(f"      Description: {task.get('description', 'No description')}")
                print()
            except Exception as e:
                print(f"❌ Error getting task suggestions: {e}")
                print()
        
        print("✅ Demo completed! Repository cleaned up.")
    
    async def demo_api_usage(self):
        """Demo showing how to use the API with GitHub URLs"""
        print("\n🌐 API Usage Demo")
        print("=" * 40)
        
        api_examples = [
            {
                "endpoint": "/api/onboarding/session/start",
                "method": "POST", 
                "description": "Start onboarding session with GitHub URL",
                "example": {
                    "user_id": "new_developer",
                    "workspace_id": "my_company",
                    "repo_path": "https://github.com/virattt/ai-hedge-fund/"
                }
            },
            {
                "endpoint": "/api/onboarding/chat",
                "method": "POST",
                "description": "Ask questions about the GitHub repository",
                "example": {
                    "user_id": "new_developer", 
                    "workspace_id": "my_company",
                    "repo_path": "https://github.com/virattt/ai-hedge-fund/",
                    "question": "Explain how the trading agents work together"
                }
            },
            {
                "endpoint": "/api/onboarding/tour",
                "method": "GET",
                "description": "Get personalized codebase tour",
                "example": {
                    "user_id": "new_developer",
                    "workspace_id": "my_company", 
                    "repo_path": "https://github.com/virattt/ai-hedge-fund/"
                }
            }
        ]
        
        for example in api_examples:
            print(f"🔗 {example['method']} {example['endpoint']}")
            print(f"📝 {example['description']}")
            print("📋 Example request:")
            print("```json")
            for key, value in example["example"].items():
                if isinstance(value, str):
                    print(f'  "{key}": "{value}"')
                else:
                    print(f'  "{key}": {value}')
            print("```")
            print()
    
    def run_interactive_demo(self):
        """Interactive demo menu"""
        print("🤖 OnboardAI GitHub Repository Demo")
        print("=" * 50)
        print()
        print("Available demos:")
        print("1. 🏦 AI Hedge Fund Repository Onboarding")
        print("2. 🌐 API Usage Examples") 
        print("3. 🚀 Run Full Demo")
        print("0. Exit")
        print()
        
        while True:
            try:
                choice = input("Choose a demo (0-3): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    asyncio.run(self.demo_ai_hedge_fund_onboarding())
                elif choice == "2":
                    asyncio.run(self.demo_api_usage())
                elif choice == "3":
                    asyncio.run(self.demo_ai_hedge_fund_onboarding())
                    asyncio.run(self.demo_api_usage())
                else:
                    print("❌ Invalid choice. Please try again.")
                
                print("\n" + "=" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OnboardAI GitHub Repository Demo")
    parser.add_argument(
        "--repo-url", 
        default="https://github.com/virattt/ai-hedge-fund/",
        help="GitHub repository URL to use for demo"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run interactive demo menu"
    )
    
    args = parser.parse_args()
    
    demo = GitHubOnboardingDemo()
    
    if args.interactive:
        demo.run_interactive_demo()
    else:
        # Run the AI hedge fund demo directly
        asyncio.run(demo.demo_ai_hedge_fund_onboarding())


if __name__ == "__main__":
    main() 