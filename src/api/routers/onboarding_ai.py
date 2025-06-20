"""
OnboardAI API Router

Provides API endpoints for the OnboardAI intelligent developer onboarding system.
Integrates all onboarding components: AI core, workflow engine, progress tracking, and personalization.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...onboarding.onboarding_ai_core import OnboardingAICore, OnboardingPhase
from ...onboarding.onboarding_agentic_explorer import OnboardingAgenticExplorer
from ...onboarding.developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle, Role
from ...onboarding.workflow_engine import OnboardingWorkflowEngine
from ...config import settings

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses

class DeveloperSurveyRequest(BaseModel):
    """Initial developer survey for profile creation"""
    name: Optional[str] = None
    email: Optional[str] = None
    experience_level: str = Field(..., description="junior/mid/senior/lead")
    role: str = Field(..., description="frontend/backend/fullstack/mobile/devops/data/qa")
    years_of_experience: int = Field(0, ge=0, le=50)
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    learning_style: str = Field("mixed", description="visual/hands_on/reading/auditory/mixed")
    preferred_pace: str = Field("normal", description="slow/normal/fast")
    goals: List[str] = Field(default_factory=list)
    github_username: Optional[str] = None
    timezone: str = "UTC"
    prefers_mentorship: bool = True
    comfortable_with_ambiguity: bool = True
    prefers_structured_learning: bool = True

class OnboardingQuestionRequest(BaseModel):
    """Request for asking onboarding questions"""
    question: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = None
    current_step_id: Optional[str] = None
    difficulty_preference: Optional[str] = None

class ProgressUpdateRequest(BaseModel):
    """Request for updating learning progress"""
    step_id: str
    time_spent_minutes: int = Field(0, ge=0)
    difficulty_rating: Optional[str] = None  # too_easy/just_right/too_hard
    understanding_level: Optional[str] = None  # beginner/intermediate/advanced/mastered
    feedback: Optional[str] = None
    questions: Optional[str] = None

class ConceptExplanationRequest(BaseModel):
    """Request for concept explanations"""
    concept: str = Field(..., min_length=1, max_length=200)
    file_context: Optional[str] = None
    difficulty_level: str = "auto"

class OnboardingResponse(BaseModel):
    """Standard onboarding response format"""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

# Router setup
router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

# In-memory storage for demo (in production, use proper database)
_active_sessions: Dict[str, Dict[str, Any]] = {}
_developer_profiles: Dict[str, DeveloperProfile] = {}
_repo_path_cache: Dict[str, str] = {}  # Cache GitHub URL -> local path mappings

def _get_session_with_repo_handling(user_id: str, workspace_id: str, repo_path: str) -> Dict[str, Any]:
    """Helper function to handle both GitHub URLs and local paths consistently"""
    from ...local_repo_loader import clone_repo_to_temp_persistent
    
    # Handle GitHub URL caching to avoid re-cloning
    actual_repo_path = repo_path
    if repo_path.startswith("http"):
        if repo_path not in _repo_path_cache:
            try:
                cloned_path = clone_repo_to_temp_persistent(repo_path)
                _repo_path_cache[repo_path] = cloned_path
                logger.info(f"Cached GitHub repo {repo_path} -> {cloned_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to clone GitHub repository: {str(e)}"
                )
        actual_repo_path = _repo_path_cache[repo_path]
    
    return get_or_create_session(user_id, workspace_id, actual_repo_path)

def get_or_create_session(
    user_id: str, 
    workspace_id: str, 
    repo_path: str,
    developer_profile: Optional[DeveloperProfile] = None
) -> Dict[str, Any]:
    """Get or create an onboarding session"""
    session_key = f"{workspace_id}_{user_id}"
    
    if session_key not in _active_sessions:
        # Create new session
        profile = developer_profile or _developer_profiles.get(user_id, DeveloperProfile())
        
        session = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "repo_path": repo_path,
            "developer_profile": profile,
            "ai_core": OnboardingAICore(workspace_id, user_id, repo_path, profile),
            "agentic_explorer": OnboardingAgenticExplorer(
                session_id=f"onboarding_{workspace_id}_{user_id}",
                repo_path=repo_path,
                developer_profile=profile
            ),
            "workflow_engine": OnboardingWorkflowEngine(workspace_id, user_id),
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        _active_sessions[session_key] = session
        logger.info(f"Created new onboarding session for user {user_id}")
    
    # Update last activity
    _active_sessions[session_key]["last_activity"] = datetime.utcnow()
    return _active_sessions[session_key]

@router.post("/profile/survey", response_model=OnboardingResponse)
async def submit_developer_survey(survey: DeveloperSurveyRequest):
    """
    Submit initial developer survey to create personalized profile
    """
    try:
        # Create developer profile from survey
        profile = DeveloperProfile()
        profile.name = survey.name or ""
        profile.email = survey.email or ""
        profile.experience_level = ExperienceLevel(survey.experience_level)
        profile.role = Role(survey.role)
        profile.years_of_experience = survey.years_of_experience
        profile.programming_languages = survey.programming_languages
        profile.frameworks = survey.frameworks
        profile.learning_style = LearningStyle(survey.learning_style)
        profile.preferred_pace = survey.preferred_pace
        profile.goals = survey.goals
        profile.github_username = survey.github_username
        profile.timezone = survey.timezone
        profile.prefers_mentorship = survey.prefers_mentorship
        profile.comfortable_with_ambiguity = survey.comfortable_with_ambiguity
        profile.prefers_structured_learning = survey.prefers_structured_learning
        profile.assessment_completed = True
        
        # Store profile (in production, save to database)
        user_id = survey.email or f"user_{datetime.utcnow().timestamp()}"
        _developer_profiles[user_id] = profile
        
        return OnboardingResponse(
            success=True,
            data={
                "user_id": user_id,
                "profile": profile.to_dict(),
                "personalization_preview": profile.get_personalization_context()
            },
            message="Developer profile created successfully!",
            suggestions=[
                "Start with the codebase overview",
                "Take the interactive tour",
                "Ask questions about the architecture"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error creating developer profile: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create profile: {str(e)}")

@router.post("/session/start", response_model=OnboardingResponse)
async def start_onboarding_session(
    user_id: str,
    workspace_id: str, 
    repo_path: str,  # Can be local path OR GitHub URL
    background_tasks: BackgroundTasks
):
    """
    Start a personalized onboarding session
    Supports both local repository paths and GitHub URLs
    """
    try:
        from ...local_repo_loader import clone_repo_to_temp_persistent
        
        # Determine if it's a GitHub URL or local path
        if repo_path.startswith("http"):
            # It's a GitHub URL - clone it temporarily
            try:
                actual_repo_path = clone_repo_to_temp_persistent(repo_path)
                logger.info(f"Cloned GitHub repo {repo_path} to {actual_repo_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to clone GitHub repository: {str(e)}"
                )
        else:
            # It's a local path - validate it exists
            if not Path(repo_path).exists():
                raise HTTPException(status_code=400, detail=f"Repository path does not exist: {repo_path}")
            actual_repo_path = repo_path
        
        # Get developer profile
        profile = _developer_profiles.get(user_id)
        if not profile:
            # Create default profile if none exists
            profile = DeveloperProfile()
            profile.user_id = user_id
            _developer_profiles[user_id] = profile
        
        # Create or get session
        session = get_or_create_session(user_id, workspace_id, actual_repo_path, profile)
        
        # Initialize the onboarding session
        initial_survey = {
            "experience_level": profile.experience_level.value,
            "role": profile.role.value,
            "learning_style": profile.learning_style.value,
            "programming_languages": profile.programming_languages,
            "goals": profile.goals
        }
        
        # Initialize AI core asynchronously
        background_tasks.add_task(
            session["ai_core"].start_onboarding_session, 
            initial_survey
        )
        
        return OnboardingResponse(
            success=True,
            data={
                "session_id": f"{workspace_id}_{user_id}",
                "repository_info": {
                    "original_input": repo_path,
                    "working_path": actual_repo_path,
                    "is_github_repo": repo_path.startswith("http")
                },
                "developer_profile": profile.to_dict(),
                "estimated_timeline": session["ai_core"]._calculate_timeline(),
                "personalization_notes": session["ai_core"]._get_personalization_notes()
            },
            message=f"Onboarding session started for {profile.role.value} developer!",
            suggestions=[
                "Ask me about the codebase architecture",
                "Request a guided tour of the repository", 
                "Get suggestions for your first task"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error starting onboarding session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.post("/chat", response_model=OnboardingResponse) 
async def ask_onboarding_question(
    user_id: str,
    workspace_id: str,
    repo_path: str,  # Can be local path OR GitHub URL
    question_request: OnboardingQuestionRequest
):
    """
    Ask a question to the onboarding AI assistant
    Supports both local repository paths and GitHub URLs
    """
    try:
        # Get session (will handle GitHub URL if needed)
        session = _get_session_with_repo_handling(user_id, workspace_id, repo_path)
        
        # Ask question using the AI core
        result = await session["ai_core"].ask_question(
            question_request.question,
            question_request.context
        )
        
        return OnboardingResponse(
            success=True,
            data=result,
            message="Question answered successfully",
            metadata={
                "session_id": f"{workspace_id}_{user_id}",
                "response_time": result.get("response_time", 0),
                "confidence_score": result.get("confidence_score", 0.8)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing onboarding question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@router.post("/explain-concept", response_model=OnboardingResponse)
async def explain_concept(
    user_id: str,
    workspace_id: str, 
    repo_path: str,  # Can be local path OR GitHub URL
    concept_request: ConceptExplanationRequest
):
    """
    Get educational explanation of a programming concept
    Supports both local repository paths and GitHub URLs
    """
    try:
        # Get session
        session = _get_session_with_repo_handling(user_id, workspace_id, repo_path)
        
        # Get concept explanation
        explanation = await session["ai_core"].explain_concept(
            concept_request.concept,
            concept_request.file_context
        )
        
        return OnboardingResponse(
            success=True,
            data=explanation,
            message=f"Concept '{concept_request.concept}' explained successfully"
        )
        
    except Exception as e:
        logger.error(f"Error explaining concept: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to explain concept: {str(e)}")

@router.get("/tour", response_model=OnboardingResponse)
async def get_codebase_tour(
    user_id: str,
    workspace_id: str,
    repo_path: str  # Can be local path OR GitHub URL
):
    """
    Generate personalized codebase tour
    Supports both local repository paths and GitHub URLs
    """
    try:
        # Get session
        session = _get_session_with_repo_handling(user_id, workspace_id, repo_path)
        
        # Generate tour
        tour_steps = await session["ai_core"].generate_codebase_tour()
        
        return OnboardingResponse(
            success=True,
            data={
                "tour_steps": tour_steps,
                "estimated_time": sum(step.get("estimated_time", 10) for step in tour_steps),
                "personalized_for": session["developer_profile"].get_personalization_context()
            },
            message="Personalized codebase tour generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating codebase tour: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tour: {str(e)}")

@router.get("/first-tasks", response_model=OnboardingResponse)
async def get_first_task_suggestions(
    user_id: str,
    workspace_id: str,
    repo_path: str  # Can be local path OR GitHub URL
):
    """
    Get suggestions for appropriate first tasks
    Supports both local repository paths and GitHub URLs
    """
    try:
        # Get session
        session = _get_session_with_repo_handling(user_id, workspace_id, repo_path)
        
        # Get task suggestions
        tasks = await session["ai_core"].suggest_first_tasks()
        
        return OnboardingResponse(
            success=True,
            data={
                "suggested_tasks": tasks,
                "selection_criteria": "Tasks selected based on your experience level and role focus",
                "developer_profile": session["developer_profile"].get_personalization_context()
            },
            message="First task suggestions generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting task suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task suggestions: {str(e)}")

@router.post("/progress/update", response_model=OnboardingResponse)
async def update_learning_progress(
    user_id: str,
    workspace_id: str,
    repo_path: str,
    progress_request: ProgressUpdateRequest
):
    """
    Update learning progress and get adaptive recommendations
    """
    try:
        # Get session
        session = get_or_create_session(user_id, workspace_id, repo_path)
        
        # Update progress
        progress_result = await session["ai_core"].track_progress(
            progress_request.step_id,
            progress_request.time_spent_minutes,
            progress_request.difficulty_rating or "just_right"
        )
        
        return OnboardingResponse(
            success=True,
            data=progress_result,
            message="Learning progress updated successfully",
            suggestions=progress_result.get("recommendations", [])
        )
        
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update progress: {str(e)}")

@router.get("/help/stuck", response_model=OnboardingResponse)
async def get_stuck_assistance(
    user_id: str,
    workspace_id: str,
    repo_path: str,
    current_task: str,
    stuck_reason: str
):
    """
    Get help when developer is stuck
    """
    try:
        # Get session
        session = get_or_create_session(user_id, workspace_id, repo_path)
        
        # Get assistance
        assistance = await session["ai_core"].get_stuck_assistance(
            current_task, 
            stuck_reason
        )
        
        return OnboardingResponse(
            success=True,
            data=assistance,
            message="Assistance provided for your current challenge",
            suggestions=[
                "Try the suggested strategies",
                "Ask for more specific help",
                "Take a short break and come back fresh"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error providing stuck assistance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to provide assistance: {str(e)}")

@router.get("/workflow", response_model=OnboardingResponse)
async def get_personalized_workflow(
    user_id: str,
    workspace_id: str,
    repo_path: str
):
    """
    Get personalized onboarding workflow
    """
    try:
        # Get session
        session = get_or_create_session(user_id, workspace_id, repo_path)
        
        # Generate workflow
        workflow = await session["workflow_engine"].create_personalized_workflow(
            session["developer_profile"]
        )
        
        return OnboardingResponse(
            success=True,
            data={
                "workflow": {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "estimated_total_time": workflow.estimated_total_time,
                    "steps": [step.to_dict() for step in workflow.steps]
                },
                "personalization": session["developer_profile"].get_personalization_context()
            },
            message="Personalized workflow created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@router.get("/summary", response_model=OnboardingResponse)
async def get_learning_summary(
    user_id: str,
    workspace_id: str,
    repo_path: str
):
    """
    Get comprehensive learning progress summary
    """
    try:
        # Get session
        session = get_or_create_session(user_id, workspace_id, repo_path)
        
        # Generate summary using agentic explorer
        summary = await session["agentic_explorer"].generate_learning_summary()
        
        return OnboardingResponse(
            success=True,
            data=summary,
            message="Learning summary generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@router.get("/profile/{user_id}", response_model=OnboardingResponse)
async def get_developer_profile(user_id: str):
    """
    Get developer profile information
    """
    try:
        profile = _developer_profiles.get(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Developer profile not found")
        
        return OnboardingResponse(
            success=True,
            data={
                "profile": profile.to_dict(),
                "personalization_context": profile.get_personalization_context(),
                "learning_adjustments": profile.get_learning_adjustments(),
                "recommended_complexity": profile.get_complexity_preference()
            },
            message="Developer profile retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profile: {str(e)}")

@router.put("/profile/{user_id}/update", response_model=OnboardingResponse)
async def update_developer_profile(
    user_id: str,
    feedback: Dict[str, Any]
):
    """
    Update developer profile based on feedback
    """
    try:
        profile = _developer_profiles.get(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Developer profile not found")
        
        # Update profile based on feedback
        profile.update_from_feedback(feedback)
        
        return OnboardingResponse(
            success=True,
            data={
                "updated_profile": profile.to_dict(),
                "changes_applied": list(feedback.keys())
            },
            message="Developer profile updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "OnboardAI",
        "version": "1.0.0",
        "active_sessions": len(_active_sessions),
        "registered_profiles": len(_developer_profiles)
    }