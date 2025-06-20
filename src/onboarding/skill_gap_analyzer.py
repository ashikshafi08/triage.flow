"""
Skill Gap Analysis Engine using LlamaIndex AgentWorkflow

Provides personalized learning paths by analyzing developer skills vs codebase requirements.
Uses AgentWorkflow for intelligent skill gap detection and learning path generation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from .developer_profile import DeveloperProfile, ExperienceLevel, Role
from ..agent_tools.llm_config import get_llm_instance

logger = logging.getLogger(__name__)

@dataclass
class SkillGap:
    """Represents a skill gap identified for a developer"""
    skill_name: str
    current_level: str  # "none", "basic", "intermediate", "advanced"
    required_level: str
    importance: str  # "critical", "high", "medium", "low"
    learning_resources: List[Dict] = field(default_factory=list)
    estimated_learning_time: int = 0  # hours
    prerequisites: List[str] = field(default_factory=list)
    practice_opportunities: List[str] = field(default_factory=list)

@dataclass 
class PersonalizedLearningPath:
    """A personalized learning path to address skill gaps"""
    developer_id: str
    skill_gaps: List[SkillGap]
    recommended_order: List[str]  # skill names in recommended learning order
    total_estimated_time: int  # hours
    milestones: List[Dict] = field(default_factory=list)
    adaptive_checkpoints: List[Dict] = field(default_factory=list)

class SkillGapAnalyzer:
    """
    LlamaIndex AgentWorkflow-based skill gap analyzer
    
    Uses multiple specialized agents to:
    1. Analyze codebase skill requirements
    2. Assess developer current skills  
    3. Identify gaps and create learning paths
    4. Generate personalized exercises
    """
    
    def __init__(self, workspace_id: str, repo_path: str):
        self.workspace_id = workspace_id
        self.repo_path = repo_path
        self.llm = get_llm_instance()
        
        # Initialize specialized agents
        self.codebase_analyzer = self._create_codebase_analyzer()
        self.skill_assessor = self._create_skill_assessor()
        self.learning_path_generator = self._create_learning_path_generator()
        
        # Create multi-agent workflow
        self.workflow = AgentWorkflow(
            agents=[
                self.codebase_analyzer,
                self.skill_assessor, 
                self.learning_path_generator
            ],
            root_agent="codebase_analyzer",
            initial_state={
                "codebase_analysis": {},
                "skill_assessment": {},
                "learning_path": {},
                "skill_gaps": [],
                "exercises": []
            }
        )
        
        logger.info(f"SkillGapAnalyzer initialized for workspace {workspace_id}")
    
    def _create_codebase_analyzer(self) -> FunctionAgent:
        """Create agent specialized in analyzing codebase skill requirements"""
        
        async def analyze_codebase_skills(ctx: Context, repo_path: str) -> str:
            """
            Analyze codebase to identify required skills and technologies
            
            Args:
                repo_path: Path to the repository to analyze
                
            Returns:
                Analysis of required skills, technologies, and complexity levels
            """
            # Get current state
            state = await ctx.get("state")
            
            # TODO: Integrate with existing AgenticCodebaseExplorer for deep analysis
            from ..agent_tools.core import AgenticCodebaseExplorer
            
            explorer = AgenticCodebaseExplorer(
                session_id=f"skill_analysis_{ctx.workflow_id}",
                repo_path=repo_path
            )
            
            analysis_query = """
            Analyze this codebase and identify all required technical skills:
            
            1. **Programming Languages**: List all languages used with complexity levels
            2. **Frameworks & Libraries**: Identify key frameworks and their usage patterns
            3. **Architectural Patterns**: What design patterns and architectures are used?
            4. **DevOps & Tools**: Build systems, deployment, testing frameworks
            5. **Database Technologies**: Any database usage, ORMs, query complexity
            6. **API Technologies**: REST, GraphQL, authentication patterns
            7. **Frontend Technologies**: If applicable, UI frameworks, state management
            8. **Skill Levels Required**: For each skill, estimate required proficiency level
            
            For each skill, categorize as:
            - Critical: Essential for basic contribution
            - High: Important for effective contribution  
            - Medium: Helpful for advanced work
            - Low: Nice to have
            
            Format as structured analysis with specific examples from the codebase.
            """
            
            analysis_result = await explorer.query(analysis_query)
            
            # Parse and structure the analysis
            codebase_analysis = {
                "repo_path": repo_path,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "raw_analysis": analysis_result,
                "required_skills": self._parse_required_skills(analysis_result),
                "complexity_assessment": self._assess_codebase_complexity(analysis_result)
            }
            
            # Update state
            state["codebase_analysis"] = codebase_analysis
            await ctx.set("state", state)
            
            return f"Codebase analysis completed. Found {len(codebase_analysis['required_skills'])} required skills."
        
        return FunctionAgent(
            name="codebase_analyzer",
            description="Analyzes codebases to identify required technical skills and proficiency levels",
            system_prompt="""
            You are a senior technical architect who specializes in analyzing codebases to understand 
            skill requirements. You have deep experience across multiple technologies and can assess
            the complexity and skill levels needed for different codebases.
            
            Your role is to thoroughly analyze repositories and provide detailed skill requirements
            that will be used to help developers identify learning gaps and create personalized
            learning paths.
            
            Be specific about skill levels needed and provide concrete examples from the codebase.
            """,
            llm=self.llm,
            tools=[
                FunctionTool.from_defaults(
                    fn=analyze_codebase_skills,
                    name="analyze_codebase_skills",
                    description="Analyze a codebase to identify required technical skills"
                )
            ],
            can_handoff_to=["skill_assessor"]
        )
    
    def _create_skill_assessor(self) -> FunctionAgent:
        """Create agent specialized in assessing developer current skills"""
        
        async def assess_developer_skills(ctx: Context, developer_profile: Dict) -> str:
            """
            Assess developer's current skills against codebase requirements
            
            Args:
                developer_profile: Developer profile with experience, languages, etc.
                
            Returns:
                Assessment of current skill levels and identified gaps
            """
            state = await ctx.get("state")
            codebase_analysis = state.get("codebase_analysis", {})
            
            if not codebase_analysis:
                return "ERROR: Codebase analysis not completed. Cannot assess skills without requirements."
            
            # Analyze developer skills vs requirements
            skill_assessment = {
                "developer_id": developer_profile.get("user_id", "unknown"),
                "assessment_timestamp": datetime.utcnow().isoformat(),
                "current_skills": self._map_developer_skills(developer_profile),
                "skill_gaps": self._identify_skill_gaps(
                    developer_profile,
                    codebase_analysis.get("required_skills", [])
                ),
                "strengths": self._identify_skill_strengths(developer_profile, codebase_analysis),
                "learning_preferences": self._extract_learning_preferences(developer_profile)
            }
            
            # Update state
            state["skill_assessment"] = skill_assessment
            state["skill_gaps"] = skill_assessment["skill_gaps"]
            await ctx.set("state", state)
            
            gap_count = len(skill_assessment["skill_gaps"])
            return f"Skill assessment completed. Identified {gap_count} skill gaps to address."
        
        return FunctionAgent(
            name="skill_assessor", 
            description="Assesses developer current skills and identifies gaps against requirements",
            system_prompt="""
            You are an experienced technical mentor who specializes in assessing developer skills
            and identifying learning opportunities. You understand how to evaluate current capabilities
            and map them against project requirements.
            
            Your role is to create accurate, actionable skill assessments that identify specific
            areas for improvement while building on existing strengths. Be encouraging but realistic
            about skill gaps and learning timelines.
            """,
            llm=self.llm,
            tools=[
                FunctionTool.from_defaults(
                    fn=assess_developer_skills,
                    name="assess_developer_skills", 
                    description="Assess developer skills against codebase requirements"
                )
            ],
            can_handoff_to=["learning_path_generator"]
        )
    
    def _create_learning_path_generator(self) -> FunctionAgent:
        """Create agent specialized in generating personalized learning paths"""
        
        async def generate_learning_path(ctx: Context, learning_preferences: Dict) -> str:
            """
            Generate personalized learning path based on skill gaps and preferences
            
            Args:
                learning_preferences: Developer learning style and preferences
                
            Returns:
                Structured learning path with exercises and milestones
            """
            state = await ctx.get("state")
            skill_gaps = state.get("skill_gaps", [])
            skill_assessment = state.get("skill_assessment", {})
            
            if not skill_gaps:
                return "No skill gaps identified. Developer appears well-prepared for this codebase."
            
            # Generate personalized learning path
            learning_path = {
                "path_id": f"path_{datetime.utcnow().timestamp()}",
                "created_timestamp": datetime.utcnow().isoformat(),
                "developer_id": skill_assessment.get("developer_id"),
                "skill_gaps_addressed": skill_gaps,
                "recommended_order": self._optimize_learning_order(skill_gaps),
                "milestones": self._create_learning_milestones(skill_gaps),
                "exercises": await self._generate_practice_exercises(skill_gaps),
                "estimated_timeline": self._calculate_learning_timeline(skill_gaps),
                "adaptive_checkpoints": self._create_adaptive_checkpoints(skill_gaps)
            }
            
            # Update state
            state["learning_path"] = learning_path
            await ctx.set("state", state)
            
            return f"Learning path generated with {len(learning_path['exercises'])} exercises and {len(learning_path['milestones'])} milestones."
        
        async def create_practice_exercise(ctx: Context, skill_name: str, difficulty: str) -> str:
            """
            Create a specific practice exercise for a skill gap
            
            Args:
                skill_name: The specific skill to create exercise for
                difficulty: Target difficulty level (basic/intermediate/advanced)
                
            Returns:
                Detailed practice exercise with instructions and success criteria
            """
            state = await ctx.get("state")
            codebase_analysis = state.get("codebase_analysis", {})
            
            # Create exercise based on actual codebase patterns
            exercise = {
                "skill": skill_name,
                "difficulty": difficulty,
                "title": f"Practice: {skill_name} in {self.repo_path.split('/')[-1]}",
                "description": f"Hands-on exercise to build {skill_name} skills",
                "instructions": self._generate_exercise_instructions(skill_name, difficulty, codebase_analysis),
                "success_criteria": self._generate_success_criteria(skill_name, difficulty),
                "hints": self._generate_exercise_hints(skill_name),
                "estimated_time": self._estimate_exercise_time(skill_name, difficulty),
                "created_timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to exercises in state
            current_exercises = state.get("exercises", [])
            current_exercises.append(exercise)
            state["exercises"] = current_exercises
            await ctx.set("state", state)
            
            return f"Practice exercise created for {skill_name} at {difficulty} level."
        
        return FunctionAgent(
            name="learning_path_generator",
            description="Generates personalized learning paths with exercises and milestones",
            system_prompt="""
            You are an expert learning designer who creates personalized educational experiences
            for software developers. You understand how to structure learning paths that build
            skills progressively and provide practical, hands-on experience.
            
            Your role is to create engaging, achievable learning paths that help developers
            bridge skill gaps effectively. Focus on practical application, real-world examples,
            and incremental progress with clear milestones.
            """,
            llm=self.llm,
            tools=[
                FunctionTool.from_defaults(
                    fn=generate_learning_path,
                    name="generate_learning_path",
                    description="Generate personalized learning path with exercises and milestones"
                ),
                FunctionTool.from_defaults(
                    fn=create_practice_exercise,
                    name="create_practice_exercise", 
                    description="Create specific practice exercise for a skill gap"
                )
            ]
        )
    
    async def analyze_skill_gaps(self, developer_profile: DeveloperProfile) -> PersonalizedLearningPath:
        """
        Main method to analyze skill gaps and generate learning path
        
        Args:
            developer_profile: The developer's profile with current skills
            
        Returns:
            Personalized learning path addressing identified skill gaps
        """
        logger.info(f"Starting skill gap analysis for developer {developer_profile.user_id}")
        
        # Create context for the workflow
        ctx = Context(self.workflow)
        
        # Convert profile to dict for workflow
        profile_dict = developer_profile.to_dict()
        
        # Run the multi-agent workflow
        try:
            # Start with codebase analysis
            response = await self.workflow.run(
                user_msg=f"Analyze skill gaps for developer working on repository: {self.repo_path}",
                ctx=ctx
            )
            
            # Get final state with all analysis results
            final_state = await ctx.get("state")
            
            # Convert to PersonalizedLearningPath object
            learning_path = self._create_learning_path_object(final_state, developer_profile.user_id)
            
            logger.info(f"Skill gap analysis completed. Found {len(learning_path.skill_gaps)} gaps.")
            return learning_path
            
        except Exception as e:
            logger.error(f"Error in skill gap analysis: {e}")
            # Return empty learning path on error
            return PersonalizedLearningPath(
                developer_id=developer_profile.user_id,
                skill_gaps=[],
                recommended_order=[],
                total_estimated_time=0
            )
    
    def _parse_required_skills(self, analysis_result: str) -> List[Dict]:
        """Parse AI analysis result into structured skill requirements"""
        # TODO: Implement parsing logic
        # For now, return example structure
        return [
            {
                "skill_name": "Python",
                "required_level": "intermediate",
                "importance": "critical",
                "examples": ["FastAPI backend", "Async programming", "Type hints"]
            },
            {
                "skill_name": "React",
                "required_level": "intermediate", 
                "importance": "high",
                "examples": ["TypeScript components", "Hooks", "State management"]
            }
        ]
    
    def _assess_codebase_complexity(self, analysis_result: str) -> Dict:
        """Assess overall codebase complexity"""
        return {
            "overall_complexity": "intermediate",
            "architectural_complexity": "high",
            "technology_diversity": "medium",
            "learning_curve": "steep"
        }
    
    def _map_developer_skills(self, profile: Dict) -> Dict:
        """Map developer profile to current skill levels"""
        return {
            "languages": profile.get("programming_languages", []),
            "frameworks": profile.get("frameworks", []),
            "experience_level": profile.get("experience_level", "mid"),
            "years_experience": profile.get("years_of_experience", 0)
        }
    
    def _identify_skill_gaps(self, profile: Dict, required_skills: List[Dict]) -> List[SkillGap]:
        """Identify gaps between current and required skills"""
        gaps = []
        
        for required_skill in required_skills:
            skill_name = required_skill["skill_name"]
            required_level = required_skill["required_level"]
            importance = required_skill["importance"]
            
            # Determine current level (simplified logic)
            current_level = "none"
            if skill_name.lower() in [lang.lower() for lang in profile.get("programming_languages", [])]:
                current_level = "basic"  # Assume basic if listed
            
            # Create gap if current < required
            if self._compare_skill_levels(current_level, required_level) < 0:
                gap = SkillGap(
                    skill_name=skill_name,
                    current_level=current_level,
                    required_level=required_level,
                    importance=importance,
                    estimated_learning_time=self._estimate_learning_time(current_level, required_level)
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_skill_strengths(self, profile: Dict, codebase_analysis: Dict) -> List[str]:
        """Identify areas where developer already has strong skills"""
        strengths = []
        # TODO: Implement strength identification
        return strengths
    
    def _extract_learning_preferences(self, profile: Dict) -> Dict:
        """Extract learning style preferences from profile"""
        return {
            "learning_style": profile.get("learning_style", "mixed"),
            "preferred_pace": profile.get("preferred_pace", "normal"),
            "prefers_examples": profile.get("prefers_examples", True),
            "prefers_hands_on": profile.get("learning_style") == "hands_on"
        }
    
    def _optimize_learning_order(self, skill_gaps: List[SkillGap]) -> List[str]:
        """Optimize the order of learning skills based on dependencies and importance"""
        # Sort by importance and prerequisites
        sorted_gaps = sorted(skill_gaps, key=lambda g: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[g.importance],
            len(g.prerequisites)
        ))
        return [gap.skill_name for gap in sorted_gaps]
    
    def _create_learning_milestones(self, skill_gaps: List[SkillGap]) -> List[Dict]:
        """Create learning milestones for the learning path"""
        milestones = []
        total_time = 0
        
        for i, gap in enumerate(skill_gaps):
            total_time += gap.estimated_learning_time
            milestone = {
                "milestone_id": f"milestone_{i+1}",
                "title": f"Master {gap.skill_name}",
                "description": f"Achieve {gap.required_level} level in {gap.skill_name}",
                "estimated_completion_time": total_time,
                "skills_addressed": [gap.skill_name],
                "success_criteria": [
                    f"Complete practice exercises for {gap.skill_name}",
                    f"Demonstrate {gap.required_level} proficiency",
                    "Apply skill in real codebase context"
                ]
            }
            milestones.append(milestone)
        
        return milestones
    
    async def _generate_practice_exercises(self, skill_gaps: List[SkillGap]) -> List[Dict]:
        """Generate practice exercises for each skill gap"""
        exercises = []
        
        for gap in skill_gaps:
            exercise = {
                "skill": gap.skill_name,
                "title": f"Practice: {gap.skill_name} Fundamentals",
                "difficulty": gap.required_level,
                "estimated_time": gap.estimated_learning_time // 4,  # 25% of total time per exercise
                "type": "hands_on",
                "description": f"Build practical {gap.skill_name} skills through real examples"
            }
            exercises.append(exercise)
        
        return exercises
    
    def _calculate_learning_timeline(self, skill_gaps: List[SkillGap]) -> Dict:
        """Calculate realistic timeline for completing all learning"""
        total_hours = sum(gap.estimated_learning_time for gap in skill_gaps)
        
        return {
            "total_hours": total_hours,
            "estimated_days": total_hours // 6,  # 6 hours per day
            "estimated_weeks": total_hours // 30,  # 30 hours per week
            "intensive_timeline": total_hours // 8,  # 8 hours per day intensive
            "part_time_timeline": total_hours // 10  # 10 hours per week part-time
        }
    
    def _create_adaptive_checkpoints(self, skill_gaps: List[SkillGap]) -> List[Dict]:
        """Create adaptive checkpoints for progress monitoring"""
        checkpoints = []
        
        for i, gap in enumerate(skill_gaps):
            checkpoint = {
                "checkpoint_id": f"checkpoint_{i+1}",
                "skill": gap.skill_name,
                "assessment_type": "practical_application",
                "triggers": [
                    f"After {gap.estimated_learning_time // 2} hours of study",
                    "Before moving to next skill",
                    "When developer requests assessment"
                ],
                "adaptation_rules": {
                    "struggling": "Provide additional resources and extend timeline",
                    "ahead_of_pace": "Introduce advanced concepts early",
                    "different_learning_style": "Adapt delivery method"
                }
            }
            checkpoints.append(checkpoint)
        
        return checkpoints
    
    def _create_learning_path_object(self, final_state: Dict, developer_id: str) -> PersonalizedLearningPath:
        """Convert workflow state to PersonalizedLearningPath object"""
        learning_path_data = final_state.get("learning_path", {})
        skill_gaps_data = final_state.get("skill_gaps", [])
        
        # Convert skill gap dicts to SkillGap objects
        skill_gaps = []
        for gap_data in skill_gaps_data:
            if isinstance(gap_data, dict):
                skill_gap = SkillGap(
                    skill_name=gap_data.get("skill_name", ""),
                    current_level=gap_data.get("current_level", "none"),
                    required_level=gap_data.get("required_level", "basic"),
                    importance=gap_data.get("importance", "medium"),
                    estimated_learning_time=gap_data.get("estimated_learning_time", 10)
                )
                skill_gaps.append(skill_gap)
        
        return PersonalizedLearningPath(
            developer_id=developer_id,
            skill_gaps=skill_gaps,
            recommended_order=learning_path_data.get("recommended_order", []),
            total_estimated_time=learning_path_data.get("estimated_timeline", {}).get("total_hours", 0),
            milestones=learning_path_data.get("milestones", []),
            adaptive_checkpoints=learning_path_data.get("adaptive_checkpoints", [])
        )
    
    # Helper methods
    
    def _compare_skill_levels(self, current: str, required: str) -> int:
        """Compare skill levels. Returns -1 if current < required, 0 if equal, 1 if current > required"""
        levels = {"none": 0, "basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        return levels.get(current, 0) - levels.get(required, 1)
    
    def _estimate_learning_time(self, current_level: str, required_level: str) -> int:
        """Estimate learning time in hours to go from current to required level"""
        level_diff = abs(self._compare_skill_levels(current_level, required_level))
        base_hours = {"basic": 10, "intermediate": 20, "advanced": 40, "expert": 80}
        return base_hours.get(required_level, 20) * level_diff
    
    def _generate_exercise_instructions(self, skill_name: str, difficulty: str, codebase_analysis: Dict) -> str:
        """Generate specific exercise instructions"""
        return f"Practice {skill_name} at {difficulty} level using patterns from the codebase."
    
    def _generate_success_criteria(self, skill_name: str, difficulty: str) -> List[str]:
        """Generate success criteria for an exercise"""
        return [
            f"Demonstrate understanding of {skill_name} concepts",
            f"Complete exercise at {difficulty} level",
            "Apply learning to real codebase examples"
        ]
    
    def _generate_exercise_hints(self, skill_name: str) -> List[str]:
        """Generate helpful hints for an exercise"""
        return [
            f"Start with basic {skill_name} documentation",
            "Look for examples in the existing codebase",
            "Ask for help if stuck for more than 30 minutes"
        ]
    
    def _estimate_exercise_time(self, skill_name: str, difficulty: str) -> int:
        """Estimate time needed for an exercise in minutes"""
        base_times = {"basic": 30, "intermediate": 60, "advanced": 120}
        return base_times.get(difficulty, 60)