"""
Onboarding Workflow Engine

Manages personalized onboarding workflows, step progression, and adaptive learning paths.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle, Role

logger = logging.getLogger(__name__)

class StepType(Enum):
    """Types of onboarding steps"""
    WELCOME = "welcome"
    ASSESSMENT = "assessment"
    ENVIRONMENT_SETUP = "environment_setup"
    DOCUMENTATION_READING = "documentation_reading"
    CODEBASE_EXPLORATION = "codebase_exploration"
    CONCEPT_LEARNING = "concept_learning"
    HANDS_ON_EXERCISE = "hands_on_exercise"
    FIRST_TASK = "first_task"
    CODE_REVIEW = "code_review"
    TEAM_INTEGRATION = "team_integration"
    CHECKPOINT = "checkpoint"
    MENTORSHIP = "mentorship"

class StepDifficulty(Enum):
    """Difficulty levels for steps"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class OnboardingStep:
    """Individual step in the onboarding workflow"""
    id: str
    title: str
    description: str
    step_type: StepType
    difficulty: StepDifficulty
    estimated_time_minutes: int
    required: bool = True
    
    # Learning objectives
    learning_objectives: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Resources and guidance
    resources: List[Dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    hints: List[str] = field(default_factory=list)
    
    # Validation and completion
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    validation_method: str = "self_assessment"  # self_assessment, mentor_review, automated
    
    # Personalization
    role_relevance: Dict[Role, float] = field(default_factory=dict)
    experience_adjustments: Dict[ExperienceLevel, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_adjusted_time(self, profile: DeveloperProfile) -> int:
        """Get time estimate adjusted for developer profile"""
        base_time = self.estimated_time_minutes
        
        # Adjust based on experience level
        experience_multipliers = {
            ExperienceLevel.JUNIOR: 1.5,
            ExperienceLevel.MID: 1.0,
            ExperienceLevel.SENIOR: 0.7,
            ExperienceLevel.LEAD: 0.5
        }
        
        multiplier = experience_multipliers.get(profile.experience_level, 1.0)
        
        # Adjust based on learning style for certain step types
        if self.step_type == StepType.DOCUMENTATION_READING:
            if profile.learning_style == LearningStyle.READING:
                multiplier *= 0.8  # Faster for reading-preferenced learners
            elif profile.learning_style == LearningStyle.HANDS_ON:
                multiplier *= 1.3  # Slower for hands-on learners
        
        elif self.step_type == StepType.HANDS_ON_EXERCISE:
            if profile.learning_style == LearningStyle.HANDS_ON:
                multiplier *= 0.8  # Faster for hands-on learners
            elif profile.learning_style == LearningStyle.READING:
                multiplier *= 1.2  # Slower for reading-preferenced learners
        
        return int(base_time * multiplier)
    
    def is_relevant_for_role(self, role: Role) -> bool:
        """Check if step is relevant for the given role"""
        if not self.role_relevance:
            return True  # If no role relevance specified, assume relevant for all
        
        relevance_score = self.role_relevance.get(role, 0.5)
        return relevance_score >= 0.3  # Threshold for relevance
    
    def get_personalized_resources(self, profile: DeveloperProfile) -> List[Dict[str, Any]]:
        """Get resources personalized for the developer profile"""
        personalized_resources = []
        
        for resource in self.resources:
            # Filter resources based on learning style
            resource_type = resource.get("type", "")
            
            if profile.learning_style == LearningStyle.VISUAL and resource_type in ["diagram", "video", "screenshot"]:
                personalized_resources.append(resource)
            elif profile.learning_style == LearningStyle.HANDS_ON and resource_type in ["exercise", "sandbox", "tutorial"]:
                personalized_resources.append(resource)
            elif profile.learning_style == LearningStyle.READING and resource_type in ["documentation", "article", "guide"]:
                personalized_resources.append(resource)
            elif resource_type in ["essential", "core"]:  # Always include essential resources
                personalized_resources.append(resource)
        
        return personalized_resources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for API responses"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "step_type": self.step_type.value,
            "difficulty": self.difficulty.value,
            "estimated_time_minutes": self.estimated_time_minutes,
            "required": self.required,
            "learning_objectives": self.learning_objectives,
            "prerequisites": self.prerequisites,
            "resources": self.resources,
            "instructions": self.instructions,
            "hints": self.hints,
            "completion_criteria": self.completion_criteria,
            "validation_method": self.validation_method
        }

@dataclass
class OnboardingWorkflow:
    """Complete onboarding workflow"""
    id: str
    name: str
    description: str
    target_profile: DeveloperProfile
    steps: List[OnboardingStep] = field(default_factory=list)
    estimated_total_time: int = 0  # minutes
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_step(self, step: OnboardingStep):
        """Add a step to the workflow"""
        self.steps.append(step)
        self.estimated_total_time += step.get_adjusted_time(self.target_profile)
    
    def get_next_step(self, completed_step_ids: List[str]) -> Optional[OnboardingStep]:
        """Get the next step to work on"""
        for step in self.steps:
            if step.id not in completed_step_ids:
                # Check if prerequisites are met
                prerequisites_met = all(
                    prereq in completed_step_ids 
                    for prereq in step.prerequisites
                )
                
                if prerequisites_met:
                    return step
        
        return None  # All steps completed
    
    def get_progress_percentage(self, completed_step_ids: List[str]) -> float:
        """Calculate completion percentage"""
        if not self.steps:
            return 100.0
        
        completed_count = len([s for s in self.steps if s.id in completed_step_ids])
        return (completed_count / len(self.steps)) * 100
    
    def get_remaining_time(self, completed_step_ids: List[str]) -> int:
        """Get estimated remaining time in minutes"""
        remaining_steps = [s for s in self.steps if s.id not in completed_step_ids]
        return sum(step.get_adjusted_time(self.target_profile) for step in remaining_steps)

class OnboardingWorkflowEngine:
    """
    Engine for creating and managing personalized onboarding workflows
    """
    
    def __init__(self, workspace_id: str, user_id: str):
        self.workspace_id = workspace_id
        self.user_id = user_id
        
        # Step templates for different scenarios
        self.step_templates = self._initialize_step_templates()
        
        # Workflow templates for different roles/experience levels
        self.workflow_templates = self._initialize_workflow_templates()
    
    async def create_personalized_workflow(self, profile: DeveloperProfile) -> OnboardingWorkflow:
        """Create a personalized workflow based on developer profile"""
        logger.info(f"Creating personalized workflow for {profile.role.value} {profile.experience_level.value}")
        
        # Start with base template
        base_template = self._get_base_template(profile)
        
        # Create workflow instance
        workflow = OnboardingWorkflow(
            id=f"workflow_{self.user_id}_{datetime.utcnow().timestamp()}",
            name=f"Onboarding for {profile.name or 'Developer'}",
            description=f"Personalized onboarding workflow for {profile.role.value} developer",
            target_profile=profile
        )
        
        # Add personalized steps
        steps = await self._generate_personalized_steps(profile, base_template)
        
        for step in steps:
            workflow.add_step(step)
        
        # Optimize workflow order
        workflow.steps = self._optimize_step_order(workflow.steps, profile)
        
        logger.info(f"Created workflow with {len(workflow.steps)} steps, estimated {workflow.estimated_total_time // 60} hours")
        
        return workflow
    
    async def adapt_workflow(
        self, 
        workflow: OnboardingWorkflow, 
        feedback: Dict[str, Any]
    ) -> OnboardingWorkflow:
        """Adapt workflow based on progress feedback"""
        
        # Analyze feedback patterns
        adaptations = self._analyze_feedback_for_adaptations(feedback)
        
        # Apply adaptations
        for adaptation in adaptations:
            await self._apply_adaptation(workflow, adaptation)
        
        return workflow
    
    def _initialize_step_templates(self) -> Dict[str, OnboardingStep]:
        """Initialize library of step templates"""
        templates = {}
        
        # Welcome and Assessment
        templates["welcome"] = OnboardingStep(
            id="welcome",
            title="Welcome to the Team!",
            description="Get introduced to the team, codebase, and onboarding process",
            step_type=StepType.WELCOME,
            difficulty=StepDifficulty.BEGINNER,
            estimated_time_minutes=30,
            learning_objectives=[
                "Understand team structure and roles",
                "Learn about the product and business context",
                "Get familiar with onboarding timeline"
            ],
            resources=[
                {"type": "essential", "title": "Team Overview", "url": "/docs/team"},
                {"type": "video", "title": "Product Demo", "url": "/videos/product-demo"},
                {"type": "documentation", "title": "Company Culture Guide", "url": "/docs/culture"}
            ]
        )
        
        templates["skills_assessment"] = OnboardingStep(
            id="skills_assessment",
            title="Skills Assessment",
            description="Quick assessment to personalize your learning path",
            step_type=StepType.ASSESSMENT,
            difficulty=StepDifficulty.BEGINNER,
            estimated_time_minutes=20,
            learning_objectives=[
                "Identify current skill level",
                "Highlight areas for growth",
                "Customize remaining onboarding steps"
            ],
            completion_criteria={"type": "survey_completed"},
            validation_method="automated"
        )
        
        # Environment Setup
        templates["dev_environment"] = OnboardingStep(
            id="dev_environment",
            title="Development Environment Setup",
            description="Set up your local development environment",
            step_type=StepType.ENVIRONMENT_SETUP,
            difficulty=StepDifficulty.INTERMEDIATE,
            estimated_time_minutes=90,
            learning_objectives=[
                "Install required development tools",
                "Configure IDE and extensions",
                "Run the application locally",
                "Verify development environment"
            ],
            resources=[
                {"type": "essential", "title": "Setup Guide", "url": "/docs/setup"},
                {"type": "video", "title": "Environment Walkthrough", "url": "/videos/setup"},
                {"type": "troubleshooting", "title": "Common Issues", "url": "/docs/setup-troubleshooting"}
            ],
            completion_criteria={
                "type": "command_success",
                "commands": ["npm test", "npm run dev"],
                "expected_output": "All tests passing"
            }
        )
        
        # Code Exploration
        templates["codebase_tour"] = OnboardingStep(
            id="codebase_tour",
            title="Interactive Codebase Tour",
            description="Guided exploration of the codebase architecture",
            step_type=StepType.CODEBASE_EXPLORATION,
            difficulty=StepDifficulty.INTERMEDIATE,
            estimated_time_minutes=120,
            learning_objectives=[
                "Understand overall architecture",
                "Identify key components and modules",
                "Learn coding conventions and patterns",
                "Navigate the codebase confidently"
            ],
            resources=[
                {"type": "essential", "title": "Architecture Overview", "url": "/docs/architecture"},
                {"type": "diagram", "title": "System Diagram", "url": "/docs/architecture-diagram.png"},
                {"type": "exercise", "title": "Code Navigation Exercise", "url": "/exercises/navigation"}
            ]
        )
        
        # Learning Activities
        templates["first_feature"] = OnboardingStep(
            id="first_feature",
            title="Implement Your First Feature",
            description="Complete a small, self-contained feature implementation",
            step_type=StepType.FIRST_TASK,
            difficulty=StepDifficulty.INTERMEDIATE,
            estimated_time_minutes=180,
            learning_objectives=[
                "Apply coding standards and patterns",
                "Use the development workflow",
                "Create and submit a pull request",
                "Receive and incorporate code review feedback"
            ],
            prerequisites=["dev_environment", "codebase_tour"],
            completion_criteria={
                "type": "pull_request",
                "requirements": ["tests_written", "documentation_updated", "review_approved"]
            },
            validation_method="mentor_review"
        )
        
        # Team Integration
        templates["team_standup"] = OnboardingStep(
            id="team_standup",
            title="Participate in Team Standup",
            description="Join daily standup and introduce yourself to the team",
            step_type=StepType.TEAM_INTEGRATION,
            difficulty=StepDifficulty.BEGINNER,
            estimated_time_minutes=30,
            learning_objectives=[
                "Understand standup format and purpose",
                "Practice communicating progress and blockers",
                "Meet team members and their roles"
            ]
        )
        
        return templates
    
    def _initialize_workflow_templates(self) -> Dict[str, List[str]]:
        """Initialize workflow templates for different profiles"""
        return {
            "junior_frontend": [
                "welcome", "skills_assessment", "dev_environment", 
                "frontend_basics", "codebase_tour", "ui_components_exercise",
                "first_feature", "team_standup", "checkpoint_week1"
            ],
            "junior_backend": [
                "welcome", "skills_assessment", "dev_environment",
                "backend_basics", "codebase_tour", "api_exercise", 
                "first_feature", "team_standup", "checkpoint_week1"
            ],
            "mid_fullstack": [
                "welcome", "skills_assessment", "dev_environment",
                "codebase_tour", "architecture_deep_dive", "first_feature",
                "team_integration", "checkpoint_week1"
            ],
            "senior_any": [
                "welcome", "dev_environment", "architecture_review",
                "codebase_tour", "technical_debt_analysis", "first_feature",
                "mentorship_setup", "checkpoint_week1"
            ]
        }
    
    def _get_base_template(self, profile: DeveloperProfile) -> List[str]:
        """Get base workflow template for profile"""
        template_key = f"{profile.experience_level.value}_{profile.role.value}"
        
        # Try exact match first
        if template_key in self.workflow_templates:
            return self.workflow_templates[template_key]
        
        # Fall back to experience level match
        experience_templates = {
            key: template for key, template in self.workflow_templates.items()
            if key.startswith(profile.experience_level.value)
        }
        
        if experience_templates:
            return list(experience_templates.values())[0]
        
        # Default fallback
        return self.workflow_templates["mid_fullstack"]
    
    async def _generate_personalized_steps(
        self, 
        profile: DeveloperProfile, 
        template: List[str]
    ) -> List[OnboardingStep]:
        """Generate personalized steps from template"""
        steps = []
        
        for step_id in template:
            if step_id in self.step_templates:
                # Clone the template step
                template_step = self.step_templates[step_id]
                personalized_step = self._personalize_step(template_step, profile)
                steps.append(personalized_step)
            else:
                # Generate dynamic step if template doesn't exist
                dynamic_step = await self._generate_dynamic_step(step_id, profile)
                if dynamic_step:
                    steps.append(dynamic_step)
        
        return steps
    
    def _personalize_step(self, template_step: OnboardingStep, profile: DeveloperProfile) -> OnboardingStep:
        """Personalize a step template for the specific profile"""
        # Create a copy of the template
        personalized = OnboardingStep(
            id=template_step.id,
            title=template_step.title,
            description=template_step.description,
            step_type=template_step.step_type,
            difficulty=template_step.difficulty,
            estimated_time_minutes=template_step.estimated_time_minutes,
            required=template_step.required,
            learning_objectives=template_step.learning_objectives.copy(),
            prerequisites=template_step.prerequisites.copy(),
            resources=template_step.resources.copy(),
            instructions=template_step.instructions,
            hints=template_step.hints.copy(),
            completion_criteria=template_step.completion_criteria.copy(),
            validation_method=template_step.validation_method
        )
        
        # Adjust time estimate
        personalized.estimated_time_minutes = template_step.get_adjusted_time(profile)
        
        # Personalize resources
        personalized.resources = template_step.get_personalized_resources(profile)
        
        # Add profile-specific hints
        if profile.experience_level == ExperienceLevel.JUNIOR:
            personalized.hints.extend([
                "Take your time to understand each concept thoroughly",
                "Don't hesitate to ask questions if something is unclear",
                "It's normal to feel overwhelmed - focus on one step at a time"
            ])
        elif profile.experience_level == ExperienceLevel.SENIOR:
            personalized.hints.extend([
                "Focus on understanding the architectural decisions",
                "Consider how patterns here compare to your previous experience",
                "Look for opportunities to contribute improvements"
            ])
        
        # Adjust difficulty if needed
        if profile.experience_level == ExperienceLevel.JUNIOR and personalized.difficulty == StepDifficulty.INTERMEDIATE:
            personalized.difficulty = StepDifficulty.BEGINNER
        elif profile.experience_level == ExperienceLevel.SENIOR and personalized.difficulty == StepDifficulty.INTERMEDIATE:
            personalized.difficulty = StepDifficulty.ADVANCED
        
        return personalized
    
    async def _generate_dynamic_step(self, step_id: str, profile: DeveloperProfile) -> Optional[OnboardingStep]:
        """Generate a dynamic step based on step_id and profile"""
        # This would use AI to generate custom steps based on:
        # - Codebase analysis
        # - Profile requirements
        # - Learning objectives
        
        # Placeholder implementation
        logger.info(f"Would generate dynamic step: {step_id} for {profile.role.value}")
        return None
    
    def _optimize_step_order(self, steps: List[OnboardingStep], profile: DeveloperProfile) -> List[OnboardingStep]:
        """Optimize the order of steps based on dependencies and profile"""
        # Simple topological sort based on prerequisites
        ordered_steps = []
        remaining_steps = steps.copy()
        
        while remaining_steps:
            # Find steps with no unmet prerequisites
            ready_steps = []
            for step in remaining_steps:
                prerequisites_met = all(
                    any(completed_step.id == prereq for completed_step in ordered_steps)
                    for prereq in step.prerequisites
                )
                if prerequisites_met:
                    ready_steps.append(step)
            
            if not ready_steps:
                # If no steps are ready, add remaining steps anyway (broken dependencies)
                ready_steps = remaining_steps
            
            # Sort ready steps by difficulty and role relevance
            ready_steps.sort(key=lambda s: (
                s.difficulty.value,
                -s.role_relevance.get(profile.role, 0.5)
            ))
            
            # Add the first ready step
            next_step = ready_steps[0]
            ordered_steps.append(next_step)
            remaining_steps.remove(next_step)
        
        return ordered_steps
    
    def _analyze_feedback_for_adaptations(self, feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze feedback to determine needed adaptations"""
        adaptations = []
        
        # Time-based adaptations
        if feedback.get("average_time_over_estimate", 0) > 1.5:
            adaptations.append({
                "type": "increase_time_estimates",
                "multiplier": 1.3,
                "reason": "Steps taking longer than expected"
            })
        
        # Difficulty adaptations
        if feedback.get("difficulty_too_high_count", 0) >= 2:
            adaptations.append({
                "type": "reduce_difficulty",
                "reason": "Multiple steps reported as too difficult"
            })
        
        # Learning style adaptations
        if feedback.get("resource_type_preferences"):
            adaptations.append({
                "type": "adjust_resource_types",
                "preferences": feedback["resource_type_preferences"],
                "reason": "Resource type preferences identified"
            })
        
        return adaptations
    
    async def _apply_adaptation(self, workflow: OnboardingWorkflow, adaptation: Dict[str, Any]):
        """Apply a specific adaptation to the workflow"""
        adaptation_type = adaptation["type"]
        
        if adaptation_type == "increase_time_estimates":
            multiplier = adaptation["multiplier"]
            for step in workflow.steps:
                step.estimated_time_minutes = int(step.estimated_time_minutes * multiplier)
        
        elif adaptation_type == "reduce_difficulty":
            for step in workflow.steps:
                if step.difficulty == StepDifficulty.ADVANCED:
                    step.difficulty = StepDifficulty.INTERMEDIATE
                elif step.difficulty == StepDifficulty.INTERMEDIATE:
                    step.difficulty = StepDifficulty.BEGINNER
        
        elif adaptation_type == "adjust_resource_types":
            preferences = adaptation["preferences"]
            for step in workflow.steps:
                # Filter resources based on preferences
                step.resources = [
                    resource for resource in step.resources
                    if resource.get("type") in preferences or resource.get("type") == "essential"
                ]
        
        logger.info(f"Applied adaptation: {adaptation_type} - {adaptation['reason']}")
    
    def create_checkpoint_step(self, phase: str, completed_steps: List[str]) -> OnboardingStep:
        """Create a checkpoint step for progress review"""
        return OnboardingStep(
            id=f"checkpoint_{phase}",
            title=f"{phase.title()} Checkpoint",
            description=f"Review progress and plan next steps for {phase}",
            step_type=StepType.CHECKPOINT,
            difficulty=StepDifficulty.BEGINNER,
            estimated_time_minutes=30,
            learning_objectives=[
                "Review progress and achievements",
                "Identify areas for improvement",
                "Plan upcoming learning goals",
                "Get feedback from mentor"
            ],
            completion_criteria={
                "type": "mentor_review",
                "completed_steps": completed_steps
            },
            validation_method="mentor_review"
        )