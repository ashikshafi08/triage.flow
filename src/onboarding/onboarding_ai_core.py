"""
OnboardingAICore - Main AI engine for developer onboarding

This is the central orchestrator that coordinates all onboarding activities,
personalizes the experience based on developer profiles, and provides intelligent guidance.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from ..agent_tools.core import AgenticCodebaseExplorer
from ..response_formatter import ResponseFormatter, ResponseType
from .onboarding_prompts import OnboardingPrompts
from .developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle
from .workflow_engine import OnboardingWorkflowEngine

logger = logging.getLogger(__name__)

class OnboardingPhase(Enum):
    """Different phases of developer onboarding"""
    WELCOME = "welcome"
    ENVIRONMENT_SETUP = "environment_setup"
    CODEBASE_EXPLORATION = "codebase_exploration"
    FIRST_TASK = "first_task"
    TEAM_INTEGRATION = "team_integration"
    PRODUCTIVITY = "productivity"

@dataclass
class OnboardingContext:
    """Context for the current onboarding session"""
    developer_profile: DeveloperProfile
    current_phase: OnboardingPhase
    completed_steps: List[str]
    time_spent: Dict[str, int]  # minutes per phase
    help_requests: List[Dict]
    repo_familiarity: Dict[str, float]  # file_path -> familiarity_score
    learning_goals: List[str]
    mentor_notes: List[str]

class OnboardingAICore:
    """
    Central AI engine for intelligent developer onboarding
    
    This class orchestrates the entire onboarding experience, providing:
    - Personalized guidance based on developer profile
    - Interactive code exploration and learning
    - Adaptive workflow management
    - Educational content generation
    - Progress tracking and recommendations
    """
    
    def __init__(
        self,
        workspace_id: str,
        user_id: str,
        repo_path: str,
        developer_profile: Optional[DeveloperProfile] = None
    ):
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.repo_path = repo_path
        self.session_id = f"onboarding_{workspace_id}_{user_id}"
        
        # Initialize AI components
        self._setup_llm()
        self.prompts = OnboardingPrompts()
        self.formatter = ResponseFormatter(repo_path)
        
        # Initialize codebase explorer with onboarding context
        self.explorer = AgenticCodebaseExplorer(
            session_id=self.session_id,
            repo_path=repo_path,
            #custom_system_prompt=self.prompts.get_system_prompt()
        )
        
        # Initialize workflow engine
        self.workflow_engine = OnboardingWorkflowEngine(workspace_id, user_id)
        
        # Developer profile and context
        self.developer_profile = developer_profile or DeveloperProfile()
        self.context = OnboardingContext(
            developer_profile=self.developer_profile,
            current_phase=OnboardingPhase.WELCOME,
            completed_steps=[],
            time_spent={},
            help_requests=[],
            repo_familiarity={},
            learning_goals=[],
            mentor_notes=[]
        )
        
        # Learning state
        self.knowledge_graph = {}  # Track what developer has learned
        self.difficulty_adjustments = {}  # Track areas that need adjustment
        
    def _setup_llm(self):
        """Configure LLM for onboarding-specific tasks"""
        # Use the existing LLM configuration from the agent tools
        from ..agent_tools.llm_config import get_llm_instance
        
        # Get LLM instances using existing configuration
        self.llm = get_llm_instance()  # Use default high-quality model for educational responses
        
        # Don't override global Settings - let existing configuration handle it
        
    async def start_onboarding_session(self, initial_survey: Dict[str, Any]) -> Dict:
        """
        Initialize onboarding session with developer information
        
        Args:
            initial_survey: Survey responses about experience, goals, preferences
            
        Returns:
            Personalized welcome message and initial steps
        """
        logger.info(f"Starting onboarding session for user {self.user_id}")
        
        # Update developer profile from survey
        self._update_profile_from_survey(initial_survey)
        
        # Generate personalized welcome
        welcome_message = await self._generate_welcome_message()
        
        # Create initial workflow
        workflow = await self.workflow_engine.create_personalized_workflow(
            self.developer_profile
        )
        
        # Analyze codebase for personalized introduction
        codebase_overview = await self._generate_codebase_overview()
        
        # Set learning goals
        learning_goals = await self._determine_learning_goals()
        self.context.learning_goals = learning_goals
        
        return {
            "welcome_message": welcome_message,
            "workflow": workflow,
            "codebase_overview": codebase_overview,
            "learning_goals": learning_goals,
            "next_steps": await self._get_next_steps(),
            "estimated_timeline": self._calculate_timeline(),
            "personalization_notes": self._get_personalization_notes()
        }
    
    async def ask_question(self, question: str, context: Optional[Dict] = None) -> Dict:
        """
        Answer developer questions with onboarding-focused guidance
        
        This is the main interface for developer interactions during onboarding.
        """
        start_time = datetime.utcnow()
        
        # Classify question type for appropriate handling
        question_type = await self._classify_question(question)
        
        # Add onboarding context to the question
        contextualized_query = self._contextualize_question(question, question_type)
        
        # Get response from explorer with onboarding prompts
        raw_response = await self.explorer.query(contextualized_query)
        
        # Apply educational formatting and personalization
        formatted_response = await self._format_educational_response(
            raw_response, question, question_type
        )
        
        # Track interaction for learning analytics
        await self._track_interaction(question, question_type, formatted_response)
        
        # Update context and provide adaptive recommendations
        recommendations = await self._get_adaptive_recommendations(question, formatted_response)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "response": formatted_response,
            "recommendations": recommendations,
            "learning_hints": self._get_learning_hints(question_type),
            "related_concepts": await self._get_related_concepts(question),
            "next_actions": await self._suggest_next_actions(),
            "response_time": response_time,
            "confidence_score": self._calculate_confidence_score(formatted_response)
        }
    
    async def generate_codebase_tour(self) -> List[Dict]:
        """
        Generate an interactive, personalized codebase tour
        """
        logger.info("Generating personalized codebase tour")
        
        # Analyze codebase structure for tour planning
        structure_analysis = await self.explorer.query(
            self.prompts.get_codebase_analysis_prompt(self.developer_profile)
        )
        
        # Generate tour steps based on developer profile
        tour_prompt = self.prompts.get_tour_generation_prompt(
            self.developer_profile,
            structure_analysis
        )
        
        tour_response = await self.explorer.query(tour_prompt)
        
        # Parse and structure the tour
        tour_steps = await self._parse_tour_response(tour_response)
        
        # Add interactive elements and learning objectives
        enhanced_tour = await self._enhance_tour_with_interactivity(tour_steps)
        
        return enhanced_tour
    
    async def suggest_first_tasks(self) -> List[Dict]:
        """
        Suggest appropriate first tasks based on developer profile and codebase analysis
        """
        # Analyze codebase for suitable first tasks
        task_analysis_prompt = self.prompts.get_first_task_analysis_prompt(
            self.developer_profile
        )
        
        analysis_response = await self.explorer.query(task_analysis_prompt)
        
        # Generate specific task suggestions
        task_suggestions = await self._generate_task_suggestions(analysis_response)
        
        # Score and rank tasks
        ranked_tasks = await self._rank_tasks_by_suitability(task_suggestions)
        
        return ranked_tasks[:5]  # Return top 5 tasks
    
    async def explain_concept(self, concept: str, file_context: Optional[str] = None) -> Dict:
        """
        Provide educational explanations tailored to developer's experience level
        """
        explanation_prompt = self.prompts.get_concept_explanation_prompt(
            concept=concept,
            experience_level=self.developer_profile.experience_level,
            learning_style=self.developer_profile.learning_style,
            file_context=file_context
        )
        
        explanation = await self.explorer.query(explanation_prompt)
        
        # Add examples, analogies, and learning resources
        enhanced_explanation = await self._enhance_explanation(explanation, concept)
        
        return {
            "explanation": enhanced_explanation,
            "examples": await self._generate_examples(concept),
            "analogies": self._get_analogies(concept),
            "further_reading": self._get_learning_resources(concept),
            "practice_exercises": await self._suggest_practice_exercises(concept)
        }
    
    async def track_progress(self, step_id: str, time_spent: int, difficulty: str) -> Dict:
        """
        Track learning progress and adapt the experience
        """
        # Update context
        self.context.completed_steps.append(step_id)
        self.context.time_spent[step_id] = time_spent
        
        # Analyze progress patterns
        progress_analysis = self._analyze_progress_patterns()
        
        # Adjust difficulty if needed
        if difficulty in ["too_hard", "too_easy"]:
            await self._adjust_difficulty(step_id, difficulty)
        
        # Update knowledge graph
        await self._update_knowledge_graph(step_id)
        
        # Generate progress summary
        summary = self._generate_progress_summary()
        
        return {
            "progress_summary": summary,
            "achievements": self._get_recent_achievements(),
            "recommendations": await self._get_progress_recommendations(),
            "next_milestone": self._get_next_milestone(),
            "adaptive_adjustments": self.difficulty_adjustments
        }
    
    async def get_stuck_assistance(self, current_task: str, stuck_reason: str) -> Dict:
        """
        Provide intelligent assistance when developer is stuck
        """
        # Analyze the stuck situation
        assistance_prompt = self.prompts.get_stuck_assistance_prompt(
            current_task=current_task,
            stuck_reason=stuck_reason,
            developer_profile=self.developer_profile,
            recent_context=self._get_recent_context()
        )
        
        assistance = await self.explorer.query(assistance_prompt)
        
        # Generate multiple assistance strategies
        strategies = await self._generate_assistance_strategies(current_task, stuck_reason)
        
        return {
            "immediate_help": assistance,
            "strategies": strategies,
            "hints": self._generate_progressive_hints(current_task),
            "resources": await self._find_relevant_resources(current_task),
            "similar_examples": await self._find_similar_examples(current_task),
            "mentor_escalation": self._should_escalate_to_mentor(stuck_reason)
        }
    
    # Private helper methods
    
    def _update_profile_from_survey(self, survey: Dict[str, Any]):
        """Update developer profile from initial survey"""
        if "experience_level" in survey:
            self.developer_profile.experience_level = ExperienceLevel(survey["experience_level"])
        
        if "learning_style" in survey:
            self.developer_profile.learning_style = LearningStyle(survey["learning_style"])
        
        if "programming_languages" in survey:
            self.developer_profile.programming_languages = survey["programming_languages"]
        
        if "frameworks" in survey:
            self.developer_profile.frameworks = survey["frameworks"]
        
        if "goals" in survey:
            self.developer_profile.goals = survey["goals"]
    
    async def _generate_welcome_message(self) -> str:
        """Generate personalized welcome message"""
        prompt = self.prompts.get_welcome_prompt(self.developer_profile)
        return await self.explorer.query(prompt)
    
    async def _generate_codebase_overview(self) -> Dict:
        """Generate personalized codebase overview"""
        overview_prompt = self.prompts.get_codebase_overview_prompt(self.developer_profile)
        overview = await self.explorer.query(overview_prompt)
        
        return {
            "summary": overview,
            "key_files": await self._identify_key_files(),
            "architecture_highlights": await self._get_architecture_highlights(),
            "learning_path": await self._suggest_learning_path()
        }
    
    async def _determine_learning_goals(self) -> List[str]:
        """Determine personalized learning goals"""
        goals_prompt = self.prompts.get_learning_goals_prompt(self.developer_profile)
        goals_response = await self.explorer.query(goals_prompt)
        
        # Parse goals from response
        return self._parse_learning_goals(goals_response)
    
    def _contextualize_question(self, question: str, question_type: str) -> str:
        """Add onboarding context to developer questions"""
        return self.prompts.get_contextualized_question_prompt(
            question=question,
            question_type=question_type,
            developer_profile=self.developer_profile,
            current_phase=self.context.current_phase,
            recent_context=self._get_recent_context()
        )
    
    async def _classify_question(self, question: str) -> str:
        """Classify question type for appropriate handling"""
        classification_prompt = self.prompts.get_question_classification_prompt(question)
        return await self.explorer.query(classification_prompt)
    
    async def _format_educational_response(self, response: str, question: str, question_type: str) -> Dict:
        """Format response with educational enhancements"""
        formatted_response = self.formatter.format_response(
            raw_content=response,
            query=question,
            response_type=ResponseType.EDUCATIONAL_GUIDANCE
        )
        
        # Add learning-specific enhancements
        enhanced_response = await self._add_educational_enhancements(
            formatted_response, question_type
        )
        
        return enhanced_response
    
    async def _track_interaction(self, question: str, question_type: str, response: Dict):
        """Track interaction for learning analytics"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "question_type": question_type,
            "response_quality": self._assess_response_quality(response),
            "phase": self.context.current_phase.value
        }
        
        self.context.help_requests.append(interaction)
    
    def _get_recent_context(self) -> Dict:
        """Get recent interaction context"""
        return {
            "recent_questions": self.context.help_requests[-3:],
            "current_phase": self.context.current_phase.value,
            "completed_steps": self.context.completed_steps[-5:],
            "learning_goals": self.context.learning_goals
        }
    
    def _calculate_timeline(self) -> Dict:
        """Calculate estimated onboarding timeline"""
        base_hours = {
            ExperienceLevel.JUNIOR: 40,
            ExperienceLevel.MID: 25,
            ExperienceLevel.SENIOR: 15
        }
        
        estimated_hours = base_hours.get(self.developer_profile.experience_level, 25)
        
        return {
            "estimated_total_hours": estimated_hours,
            "estimated_days": estimated_hours // 6,  # 6 hours per day
            "phases": {
                "environment_setup": estimated_hours * 0.15,
                "codebase_exploration": estimated_hours * 0.35,
                "first_task": estimated_hours * 0.30,
                "team_integration": estimated_hours * 0.20
            }
        }
    
    def _get_personalization_notes(self) -> List[str]:
        """Get notes about how the experience is personalized"""
        notes = []
        
        if self.developer_profile.experience_level == ExperienceLevel.JUNIOR:
            notes.append("🎓 Extra learning resources and detailed explanations provided")
        elif self.developer_profile.experience_level == ExperienceLevel.SENIOR:
            notes.append("🚀 Fast-track approach with focus on architecture and patterns")
        
        if self.developer_profile.learning_style == LearningStyle.VISUAL:
            notes.append("👁️ Visual diagrams and code examples emphasized")
        elif self.developer_profile.learning_style == LearningStyle.HANDS_ON:
            notes.append("🔨 Interactive exercises and practical tasks prioritized")
        
        return notes
    
    # Placeholder methods that would be implemented based on specific needs
    async def _get_next_steps(self) -> List[str]: pass
    async def _get_adaptive_recommendations(self, question: str, response: Dict) -> List[Dict]: pass
    def _get_learning_hints(self, question_type: str) -> List[str]: pass
    async def _get_related_concepts(self, question: str) -> List[str]: pass
    async def _suggest_next_actions(self) -> List[Dict]: pass
    def _calculate_confidence_score(self, response: Dict) -> float: pass
    async def _parse_tour_response(self, response: str) -> List[Dict]: pass
    async def _enhance_tour_with_interactivity(self, tour_steps: List[Dict]) -> List[Dict]: pass
    async def _generate_task_suggestions(self, analysis: str) -> List[Dict]: pass
    async def _rank_tasks_by_suitability(self, tasks: List[Dict]) -> List[Dict]: pass
    async def _enhance_explanation(self, explanation: str, concept: str) -> Dict: pass
    async def _generate_examples(self, concept: str) -> List[Dict]: pass
    def _get_analogies(self, concept: str) -> List[str]: pass
    def _get_learning_resources(self, concept: str) -> List[Dict]: pass
    async def _suggest_practice_exercises(self, concept: str) -> List[Dict]: pass
    def _analyze_progress_patterns(self) -> Dict: pass
    async def _adjust_difficulty(self, step_id: str, difficulty: str): pass
    async def _update_knowledge_graph(self, step_id: str): pass
    def _generate_progress_summary(self) -> Dict: pass
    def _get_recent_achievements(self) -> List[Dict]: pass
    async def _get_progress_recommendations(self) -> List[Dict]: pass
    def _get_next_milestone(self) -> Dict: pass
    async def _generate_assistance_strategies(self, task: str, reason: str) -> List[Dict]: pass
    def _generate_progressive_hints(self, task: str) -> List[str]: pass
    async def _find_relevant_resources(self, task: str) -> List[Dict]: pass
    async def _find_similar_examples(self, task: str) -> List[Dict]: pass
    def _should_escalate_to_mentor(self, reason: str) -> bool: pass
    async def _identify_key_files(self) -> List[Dict]: pass
    async def _get_architecture_highlights(self) -> List[str]: pass
    async def _suggest_learning_path(self) -> List[Dict]: pass
    def _parse_learning_goals(self, response: str) -> List[str]: pass
    async def _add_educational_enhancements(self, response: Dict, question_type: str) -> Dict: pass
    def _assess_response_quality(self, response: Dict) -> float: pass