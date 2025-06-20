"""
Developer Profile Management for Onboarding

Handles developer profiling, experience assessment, and personalization preferences.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime

class ExperienceLevel(Enum):
    """Developer experience levels"""
    JUNIOR = "junior"
    MID = "mid" 
    SENIOR = "senior"
    LEAD = "lead"

class LearningStyle(Enum):
    """Preferred learning approaches"""
    VISUAL = "visual"           # Diagrams, charts, visual examples
    HANDS_ON = "hands_on"       # Interactive coding, trial and error
    READING = "reading"         # Documentation, written tutorials
    AUDITORY = "auditory"       # Video explanations, verbal instructions
    MIXED = "mixed"             # Combination of approaches

class Role(Enum):
    """Developer role focus"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    DEVOPS = "devops"
    DATA = "data"
    QA = "qa"

@dataclass
class DeveloperProfile:
    """
    Comprehensive developer profile for personalized onboarding
    """
    # Basic information
    user_id: str = ""
    name: str = ""
    email: str = ""
    
    # Experience and skills
    experience_level: ExperienceLevel = ExperienceLevel.MID
    years_of_experience: int = 0
    role: Role = Role.FULLSTACK
    programming_languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    
    # Learning preferences
    learning_style: LearningStyle = LearningStyle.MIXED
    preferred_pace: str = "normal"  # slow, normal, fast
    prefers_examples: bool = True
    prefers_theory: bool = False
    
    # Context and goals
    goals: List[str] = field(default_factory=list)
    timezone: str = "UTC"
    availability_hours: int = 8  # hours per day
    previous_companies: List[str] = field(default_factory=list)
    domain_knowledge: List[str] = field(default_factory=list)
    
    # GitHub/External profiles
    github_username: Optional[str] = None
    linkedin_profile: Optional[str] = None
    portfolio_url: Optional[str] = None
    
    # Assessment results
    technical_assessment_score: Optional[float] = None
    communication_style: str = "collaborative"  # direct, collaborative, formal
    problem_solving_approach: str = "methodical"  # exploratory, methodical, intuitive
    
    # Onboarding preferences
    prefers_mentorship: bool = True
    comfortable_with_ambiguity: bool = True
    prefers_structured_learning: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    assessment_completed: bool = False
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if not self.programming_languages:
            self.programming_languages = ["python"]  # Default assumption
        
        if not self.goals:
            self.goals = ["understand_codebase", "contribute_effectively"]
    
    @classmethod
    def from_survey(cls, survey_data: Dict[str, Any]) -> 'DeveloperProfile':
        """Create profile from initial survey responses"""
        profile = cls()
        
        # Map survey responses to profile fields
        if "experience_level" in survey_data:
            profile.experience_level = ExperienceLevel(survey_data["experience_level"])
        
        if "years_experience" in survey_data:
            profile.years_of_experience = int(survey_data["years_experience"])
        
        if "role" in survey_data:
            profile.role = Role(survey_data["role"])
        
        if "learning_style" in survey_data:
            profile.learning_style = LearningStyle(survey_data["learning_style"])
        
        if "programming_languages" in survey_data:
            profile.programming_languages = survey_data["programming_languages"]
        
        if "frameworks" in survey_data:
            profile.frameworks = survey_data["frameworks"]
        
        if "goals" in survey_data:
            profile.goals = survey_data["goals"]
        
        if "github_username" in survey_data:
            profile.github_username = survey_data["github_username"]
        
        if "preferred_pace" in survey_data:
            profile.preferred_pace = survey_data["preferred_pace"]
        
        if "timezone" in survey_data:
            profile.timezone = survey_data["timezone"]
        
        profile.assessment_completed = True
        return profile
    
    @classmethod
    async def from_github_analysis(cls, github_username: str) -> 'DeveloperProfile':
        """Create profile by analyzing GitHub activity"""
        # This would integrate with GitHub API to analyze:
        # - Repository languages
        # - Contribution patterns
        # - Project complexity
        # - Collaboration style
        
        profile = cls()
        profile.github_username = github_username
        
        # Placeholder for GitHub analysis
        # In real implementation, would call GitHub API
        github_data = await cls._analyze_github_profile(github_username)
        
        profile.programming_languages = github_data.get("languages", ["python"])
        profile.frameworks = github_data.get("frameworks", [])
        profile.experience_level = cls._infer_experience_from_github(github_data)
        
        return profile
    
    def get_personalization_context(self) -> Dict[str, Any]:
        """Get context for personalizing onboarding experience"""
        return {
            "experience_level": self.experience_level.value,
            "learning_style": self.learning_style.value,
            "role": self.role.value,
            "languages": self.programming_languages,
            "frameworks": self.frameworks,
            "goals": self.goals,
            "pace": self.preferred_pace,
            "mentorship": self.prefers_mentorship,
            "structured": self.prefers_structured_learning,
            "examples": self.prefers_examples,
            "theory": self.prefers_theory
        }
    
    def get_learning_adjustments(self) -> Dict[str, Any]:
        """Get recommended adjustments for learning content"""
        adjustments = {
            "time_multiplier": 1.0,
            "detail_level": "medium",
            "example_count": 2,
            "theory_depth": "basic"
        }
        
        # Adjust based on experience level
        if self.experience_level == ExperienceLevel.JUNIOR:
            adjustments["time_multiplier"] = 1.5
            adjustments["detail_level"] = "high"
            adjustments["example_count"] = 3
            adjustments["theory_depth"] = "detailed"
        elif self.experience_level == ExperienceLevel.SENIOR:
            adjustments["time_multiplier"] = 0.7
            adjustments["detail_level"] = "low"
            adjustments["example_count"] = 1
            adjustments["theory_depth"] = "minimal"
        
        # Adjust based on learning style
        if self.learning_style == LearningStyle.VISUAL:
            adjustments["visual_aids"] = True
            adjustments["diagrams"] = True
        elif self.learning_style == LearningStyle.HANDS_ON:
            adjustments["interactive_examples"] = True
            adjustments["practice_exercises"] = True
        elif self.learning_style == LearningStyle.READING:
            adjustments["documentation_links"] = True
            adjustments["detailed_explanations"] = True
        
        return adjustments
    
    def should_skip_basic_concepts(self) -> bool:
        """Determine if basic programming concepts can be skipped"""
        return (
            self.experience_level in [ExperienceLevel.SENIOR, ExperienceLevel.LEAD]
            or self.years_of_experience >= 5
        )
    
    def get_recommended_first_tasks(self) -> List[str]:
        """Get task types recommended for this developer profile"""
        if self.experience_level == ExperienceLevel.JUNIOR:
            return ["documentation", "small_bug_fix", "code_review", "unit_test"]
        elif self.experience_level == ExperienceLevel.MID:
            return ["feature_implementation", "refactoring", "integration", "debugging"]
        else:  # Senior/Lead
            return ["architecture_review", "performance_optimization", "design_patterns", "mentoring"]
    
    def get_complexity_preference(self) -> str:
        """Get preferred complexity level for tasks and explanations"""
        complexity_map = {
            ExperienceLevel.JUNIOR: "beginner",
            ExperienceLevel.MID: "intermediate", 
            ExperienceLevel.SENIOR: "advanced",
            ExperienceLevel.LEAD: "expert"
        }
        return complexity_map.get(self.experience_level, "intermediate")
    
    def update_from_feedback(self, feedback: Dict[str, Any]):
        """Update profile based on onboarding feedback"""
        if "pace_too_fast" in feedback and feedback["pace_too_fast"]:
            if self.preferred_pace == "fast":
                self.preferred_pace = "normal"
            elif self.preferred_pace == "normal":
                self.preferred_pace = "slow"
        
        if "pace_too_slow" in feedback and feedback["pace_too_slow"]:
            if self.preferred_pace == "slow":
                self.preferred_pace = "normal"
            elif self.preferred_pace == "normal":
                self.preferred_pace = "fast"
        
        if "needs_more_examples" in feedback and feedback["needs_more_examples"]:
            self.prefers_examples = True
        
        if "needs_more_theory" in feedback and feedback["needs_more_theory"]:
            self.prefers_theory = True
        
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage/transmission"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "experience_level": self.experience_level.value,
            "years_of_experience": self.years_of_experience,
            "role": self.role.value,
            "programming_languages": self.programming_languages,
            "frameworks": self.frameworks,
            "tools": self.tools,
            "learning_style": self.learning_style.value,
            "preferred_pace": self.preferred_pace,
            "prefers_examples": self.prefers_examples,
            "prefers_theory": self.prefers_theory,
            "goals": self.goals,
            "timezone": self.timezone,
            "availability_hours": self.availability_hours,
            "github_username": self.github_username,
            "technical_assessment_score": self.technical_assessment_score,
            "communication_style": self.communication_style,
            "problem_solving_approach": self.problem_solving_approach,
            "prefers_mentorship": self.prefers_mentorship,
            "comfortable_with_ambiguity": self.comfortable_with_ambiguity,
            "prefers_structured_learning": self.prefers_structured_learning,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assessment_completed": self.assessment_completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeveloperProfile':
        """Create profile from dictionary"""
        profile = cls()
        
        for field_name, value in data.items():
            if hasattr(profile, field_name):
                if field_name == "experience_level":
                    profile.experience_level = ExperienceLevel(value)
                elif field_name == "learning_style":
                    profile.learning_style = LearningStyle(value)
                elif field_name == "role":
                    profile.role = Role(value)
                elif field_name in ["created_at", "updated_at"]:
                    setattr(profile, field_name, datetime.fromisoformat(value))
                else:
                    setattr(profile, field_name, value)
        
        return profile
    
    # Static helper methods
    
    @staticmethod
    async def _analyze_github_profile(username: str) -> Dict[str, Any]:
        """Analyze GitHub profile to extract developer information"""
        # Placeholder for GitHub API integration
        # Would analyze:
        # - Repository languages and their usage
        # - Contribution frequency and patterns
        # - Project types and complexity
        # - Collaboration indicators (PRs, issues, reviews)
        
        return {
            "languages": ["python", "javascript"],
            "frameworks": ["fastapi", "react"],
            "experience_indicators": {
                "total_repos": 15,
                "contribution_frequency": "regular",
                "project_complexity": "intermediate"
            }
        }
    
    @staticmethod
    def _infer_experience_from_github(github_data: Dict[str, Any]) -> ExperienceLevel:
        """Infer experience level from GitHub analysis"""
        # Simple heuristic - would be more sophisticated in real implementation
        indicators = github_data.get("experience_indicators", {})
        
        repo_count = indicators.get("total_repos", 0)
        complexity = indicators.get("project_complexity", "basic")
        
        if repo_count >= 20 and complexity == "advanced":
            return ExperienceLevel.SENIOR
        elif repo_count >= 10 and complexity in ["intermediate", "advanced"]:
            return ExperienceLevel.MID
        else:
            return ExperienceLevel.JUNIOR