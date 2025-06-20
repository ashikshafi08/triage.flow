"""
Advanced OnboardAI API Router

Provides API endpoints for advanced AI features:
- Skill Gap Analysis with personalized learning paths
- Autonomous Workflow Generation with self-improvement
- A/B Testing for workflow optimization
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ...onboarding.skill_gap_analyzer import SkillGapAnalyzer, PersonalizedLearningPath
from ...onboarding.autonomous_workflow_generator import (
    AutonomousWorkflowGenerator, 
    WorkflowOptimizationGoal,
    WorkflowPerformanceData
)
from ...onboarding.developer_profile import DeveloperProfile
from .onboarding_ai import get_or_create_session, _developer_profiles

logger = logging.getLogger(__name__)

# Pydantic models for advanced features

class SkillGapAnalysisRequest(BaseModel):
    """Request for skill gap analysis"""
    user_id: str
    workspace_id: str
    repo_path: str  # Can be local path OR GitHub URL
    force_reanalysis: bool = False

class LearningPathResponse(BaseModel):
    """Response with personalized learning path"""
    developer_id: str
    skill_gaps: List[Dict[str, Any]]
    learning_path: Dict[str, Any]
    estimated_timeline: Dict[str, Any]
    practice_exercises: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]

class WorkflowGenerationRequest(BaseModel):
    """Request for autonomous workflow generation"""
    user_id: str
    workspace_id: str
    repo_path: str
    optimization_goal: str = Field("balanced", description="speed/thoroughness/retention/satisfaction/balanced")
    project_context: Dict[str, Any] = Field(default_factory=dict)

class WorkflowPerformanceFeedback(BaseModel):
    """Feedback data for workflow performance"""
    workflow_id: str
    user_id: str
    step_completion_times: Dict[str, int]  # step_id -> minutes
    step_satisfaction_scores: Dict[str, float]  # step_id -> 1-5 rating
    overall_satisfaction: float = Field(..., ge=1, le=5)
    completion_rate: float = Field(..., ge=0, le=1)
    help_requests: List[Dict[str, Any]] = Field(default_factory=list)
    bottlenecks_encountered: List[str] = Field(default_factory=list)
    success_factors: List[str] = Field(default_factory=list)
    additional_feedback: Optional[str] = None

class AdvancedOnboardingResponse(BaseModel):
    """Standard response for advanced onboarding features"""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    analysis_metadata: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)

# Router setup
router = APIRouter(prefix="/api/onboarding/advanced", tags=["advanced-onboarding"])

# In-memory storage for advanced features (replace with DB in production)
_skill_analyzers: Dict[str, SkillGapAnalyzer] = {}
_workflow_generators: Dict[str, AutonomousWorkflowGenerator] = {}
_learning_paths: Dict[str, PersonalizedLearningPath] = {}

@router.post("/skill-analysis", response_model=AdvancedOnboardingResponse)
async def analyze_skill_gaps(
    request: SkillGapAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze developer skill gaps and generate personalized learning path
    
    Uses LlamaIndex AgentWorkflow to:
    1. Analyze codebase skill requirements
    2. Assess developer current skills
    3. Identify gaps and create learning paths
    4. Generate personalized exercises
    """
    try:
        # Get developer profile
        profile = _developer_profiles.get(request.user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Developer profile not found")
        
        # Get or create skill analyzer for this workspace
        analyzer_key = f"{request.workspace_id}_{request.repo_path}"
        if analyzer_key not in _skill_analyzers or request.force_reanalysis:
            _skill_analyzers[analyzer_key] = SkillGapAnalyzer(
                workspace_id=request.workspace_id,
                repo_path=request.repo_path
            )
        
        analyzer = _skill_analyzers[analyzer_key]
        
        # Run skill gap analysis in background if it's a complex analysis
        if request.force_reanalysis:
            background_tasks.add_task(
                _run_skill_analysis_background,
                analyzer,
                profile,
                request.user_id
            )
            
            return AdvancedOnboardingResponse(
                success=True,
                message="Skill gap analysis started. Results will be available shortly.",
                data={
                    "analysis_status": "in_progress",
                    "estimated_completion_time": "2-3 minutes"
                }
            )
        
        # Run analysis synchronously for quick results
        learning_path = await analyzer.analyze_skill_gaps(profile)
        
        # Store learning path
        _learning_paths[request.user_id] = learning_path
        
        # Format response
        response_data = {
            "analysis_id": f"analysis_{datetime.utcnow().timestamp()}",
            "developer_id": learning_path.developer_id,
            "skill_gaps_count": len(learning_path.skill_gaps),
            "skill_gaps": [
                {
                    "skill_name": gap.skill_name,
                    "current_level": gap.current_level,
                    "required_level": gap.required_level,
                    "importance": gap.importance,
                    "estimated_learning_time": gap.estimated_learning_time,
                    "learning_resources": gap.learning_resources,
                    "practice_opportunities": gap.practice_opportunities
                }
                for gap in learning_path.skill_gaps
            ],
            "learning_path": {
                "recommended_order": learning_path.recommended_order,
                "total_estimated_time": learning_path.total_estimated_time,
                "milestones": learning_path.milestones,
                "adaptive_checkpoints": learning_path.adaptive_checkpoints
            },
            "personalization_applied": {
                "experience_level": profile.experience_level.value,
                "learning_style": profile.learning_style.value,
                "role_focus": profile.role.value
            }
        }
        
        recommendations = [
            f"Focus on {learning_path.skill_gaps[0].skill_name} first" if learning_path.skill_gaps else "No skill gaps identified",
            f"Estimated learning time: {learning_path.total_estimated_time} hours",
            "Practice exercises have been generated for each skill gap"
        ]
        
        return AdvancedOnboardingResponse(
            success=True,
            data=response_data,
            message=f"Skill gap analysis completed. Found {len(learning_path.skill_gaps)} areas for improvement.",
            recommendations=recommendations,
            analysis_metadata={
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "codebase_analyzed": request.repo_path,
                "analysis_method": "llama_index_agent_workflow"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Skill analysis failed: {str(e)}")

@router.get("/skill-analysis/{user_id}", response_model=AdvancedOnboardingResponse)
async def get_skill_analysis_results(user_id: str):
    """
    Get previously computed skill analysis results
    """
    try:
        learning_path = _learning_paths.get(user_id)
        if not learning_path:
            raise HTTPException(status_code=404, detail="No skill analysis found for this user")
        
        # Format existing results
        response_data = {
            "developer_id": learning_path.developer_id,
            "skill_gaps": [
                {
                    "skill_name": gap.skill_name,
                    "current_level": gap.current_level,
                    "required_level": gap.required_level,
                    "importance": gap.importance,
                    "estimated_learning_time": gap.estimated_learning_time
                }
                for gap in learning_path.skill_gaps
            ],
            "learning_path": {
                "recommended_order": learning_path.recommended_order,
                "total_estimated_time": learning_path.total_estimated_time,
                "milestones": learning_path.milestones
            }
        }
        
        return AdvancedOnboardingResponse(
            success=True,
            data=response_data,
            message="Skill analysis results retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving skill analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.post("/autonomous-workflow", response_model=AdvancedOnboardingResponse)
async def generate_autonomous_workflow(
    request: WorkflowGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate autonomous workflow using AI optimization
    
    Uses LlamaIndex AgentWorkflow to:
    1. Analyze historical performance data
    2. Generate optimized workflows for context
    3. Create A/B test variations
    4. Predict workflow success metrics
    """
    try:
        # Get developer profile
        profile = _developer_profiles.get(request.user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Developer profile not found")
        
        # Get or create autonomous workflow generator
        if request.workspace_id not in _workflow_generators:
            _workflow_generators[request.workspace_id] = AutonomousWorkflowGenerator(
                workspace_id=request.workspace_id
            )
        
        generator = _workflow_generators[request.workspace_id]
        
        # Parse optimization goal
        try:
            optimization_goal = WorkflowOptimizationGoal(request.optimization_goal)
        except ValueError:
            optimization_goal = WorkflowOptimizationGoal.BALANCED
        
        # Generate autonomous workflow
        workflow = await generator.generate_autonomous_workflow(
            developer_profile=profile,
            project_context=request.project_context,
            optimization_goal=optimization_goal
        )
        
        # Format response
        response_data = {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "description": workflow.description,
            "optimization_goal": optimization_goal.value,
            "steps": [
                {
                    "id": step.id,
                    "title": step.title,
                    "description": step.description,
                    "type": step.step_type.value,
                    "difficulty": step.difficulty.value,
                    "estimated_time": step.estimated_time_minutes,
                    "required": step.required
                }
                for step in workflow.steps
            ],
            "total_estimated_time": workflow.estimated_total_time,
            "personalization_applied": {
                "experience_level": profile.experience_level.value,
                "role_focus": profile.role.value,
                "optimization_goal": optimization_goal.value
            },
            "autonomous_features": {
                "self_optimizing": True,
                "adaptive_timing": True,
                "personalized_content": True,
                "predictive_support": True
            }
        }
        
        recommendations = [
            f"Workflow optimized for {optimization_goal.value}",
            f"Total estimated time: {workflow.estimated_total_time // 60} hours",
            "Workflow will adapt based on your progress",
            "Performance will be tracked for continuous improvement"
        ]
        
        return AdvancedOnboardingResponse(
            success=True,
            data=response_data,
            message=f"Autonomous workflow generated with {len(workflow.steps)} optimized steps",
            recommendations=recommendations,
            analysis_metadata={
                "generation_timestamp": datetime.utcnow().isoformat(),
                "optimization_method": "llama_index_multi_agent",
                "personalization_level": "high"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in autonomous workflow generation: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow generation failed: {str(e)}")

@router.post("/workflow-feedback", response_model=AdvancedOnboardingResponse)
async def submit_workflow_feedback(
    feedback: WorkflowPerformanceFeedback,
    background_tasks: BackgroundTasks
):
    """
    Submit workflow performance feedback for continuous improvement
    
    This data is used to:
    1. Optimize existing workflows
    2. Train the autonomous generation system
    3. Identify patterns for future improvements
    4. A/B test workflow variations
    """
    try:
        # Create performance data object
        performance_data = WorkflowPerformanceData(
            workflow_id=feedback.workflow_id,
            developer_profiles=[_developer_profiles.get(feedback.user_id, DeveloperProfile()).to_dict()],
            completion_rates={"overall": feedback.completion_rate},
            average_time_per_step=feedback.step_completion_times,
            satisfaction_scores={
                **feedback.step_satisfaction_scores,
                "overall": feedback.overall_satisfaction
            },
            common_bottlenecks=feedback.bottlenecks_encountered,
            success_factors=feedback.success_factors,
            developer_feedback=[{
                "user_id": feedback.user_id,
                "feedback": feedback.additional_feedback,
                "timestamp": datetime.utcnow().isoformat()
            }],
            timestamp=datetime.utcnow()
        )
        
        # Process feedback in background for workflow evolution
        background_tasks.add_task(
            _process_workflow_feedback,
            feedback.workflow_id,
            performance_data
        )
        
        # Immediate analysis for user
        analysis = {
            "feedback_received": True,
            "performance_summary": {
                "completion_rate": feedback.completion_rate,
                "average_satisfaction": feedback.overall_satisfaction,
                "total_steps_completed": len(feedback.step_completion_times),
                "bottlenecks_identified": len(feedback.bottlenecks_encountered)
            },
            "improvement_areas": _identify_immediate_improvements(feedback),
            "positive_aspects": feedback.success_factors
        }
        
        recommendations = [
            "Thank you for your feedback - it helps improve the system",
            "Your data will be used to optimize future workflows",
            "Consider the improvement suggestions for better experience"
        ]
        
        if feedback.overall_satisfaction < 3:
            recommendations.append("We'll prioritize improvements based on your feedback")
        elif feedback.overall_satisfaction >= 4:
            recommendations.append("Great to hear you had a positive experience!")
        
        return AdvancedOnboardingResponse(
            success=True,
            data=analysis,
            message="Feedback received and will be used for workflow optimization",
            recommendations=recommendations,
            analysis_metadata={
                "feedback_timestamp": datetime.utcnow().isoformat(),
                "workflow_evolution_queued": True
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing workflow feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@router.get("/workflow-optimization/{workspace_id}", response_model=AdvancedOnboardingResponse)
async def get_workflow_optimization_insights(workspace_id: str):
    """
    Get workflow optimization insights for a workspace
    
    Shows:
    - Performance trends across workflows
    - Common optimization patterns
    - A/B test results
    - Recommendations for improvement
    """
    try:
        # Get workflow generator for workspace
        generator = _workflow_generators.get(workspace_id)
        if not generator:
            return AdvancedOnboardingResponse(
                success=True,
                data={
                    "insights": "No workflow data available yet",
                    "recommendation": "Generate some workflows first to see optimization insights"
                },
                message="No optimization data available for this workspace"
            )
        
        # Generate insights from historical data
        insights = {
            "workspace_id": workspace_id,
            "total_workflows_generated": len(generator.workflow_history),
            "performance_trends": _analyze_performance_trends(generator.workflow_history),
            "optimization_patterns": _identify_optimization_patterns(generator.workflow_history),
            "success_factors": _extract_success_factors(generator.workflow_history),
            "recommendations": _generate_optimization_recommendations(generator.workflow_history)
        }
        
        return AdvancedOnboardingResponse(
            success=True,
            data=insights,
            message="Workflow optimization insights generated",
            recommendations=insights["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Error generating optimization insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@router.post("/learning-exercise", response_model=AdvancedOnboardingResponse)
async def generate_personalized_exercise(
    user_id: str,
    skill_name: str,
    difficulty_level: str = "intermediate",
    exercise_type: str = "hands_on"
):
    """
    Generate a personalized practice exercise for a specific skill
    
    Uses the skill gap analysis to create targeted exercises
    """
    try:
        # Get learning path for user
        learning_path = _learning_paths.get(user_id)
        if not learning_path:
            raise HTTPException(status_code=404, detail="No learning path found. Run skill analysis first.")
        
        # Find the specific skill gap
        target_skill = None
        for gap in learning_path.skill_gaps:
            if gap.skill_name.lower() == skill_name.lower():
                target_skill = gap
                break
        
        if not target_skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found in learning path")
        
        # Generate personalized exercise
        exercise = {
            "exercise_id": f"exercise_{datetime.utcnow().timestamp()}",
            "skill_name": skill_name,
            "difficulty_level": difficulty_level,
            "exercise_type": exercise_type,
            "title": f"Practice: {skill_name} for {target_skill.current_level} to {target_skill.required_level}",
            "description": f"Hands-on exercise to improve {skill_name} skills",
            "estimated_time": target_skill.estimated_learning_time // 4,  # 25% of total learning time
            "instructions": [
                f"Review the current codebase for {skill_name} examples",
                f"Identify patterns and best practices for {skill_name}",
                f"Complete the practice implementation",
                "Test your solution and compare with existing code"
            ],
            "success_criteria": [
                f"Demonstrate understanding of {skill_name} concepts",
                f"Implement solution at {difficulty_level} level",
                "Code follows existing codebase patterns",
                "All tests pass"
            ],
            "hints": [
                f"Look for {skill_name} examples in the existing codebase",
                "Start with simple implementations first",
                "Ask for help if stuck for more than 30 minutes",
                "Focus on understanding rather than perfection"
            ],
            "resources": target_skill.learning_resources,
            "next_exercises": [
                f"Advanced {skill_name} patterns",
                f"Integration with other skills",
                f"Real-world {skill_name} project"
            ]
        }
        
        return AdvancedOnboardingResponse(
            success=True,
            data=exercise,
            message=f"Personalized exercise generated for {skill_name}",
            recommendations=[
                f"Complete this exercise to improve your {skill_name} skills",
                f"Estimated time: {exercise['estimated_time']} minutes",
                "Practice regularly for best results"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating exercise: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate exercise: {str(e)}")

# Background task functions

async def _run_skill_analysis_background(
    analyzer: SkillGapAnalyzer,
    profile: DeveloperProfile,
    user_id: str
):
    """Run skill analysis in background"""
    try:
        learning_path = await analyzer.analyze_skill_gaps(profile)
        _learning_paths[user_id] = learning_path
        logger.info(f"Background skill analysis completed for user {user_id}")
    except Exception as e:
        logger.error(f"Background skill analysis failed for user {user_id}: {e}")

async def _process_workflow_feedback(
    workflow_id: str,
    performance_data: WorkflowPerformanceData
):
    """Process workflow feedback for evolution"""
    try:
        # Find the appropriate workflow generator
        for workspace_id, generator in _workflow_generators.items():
            # Evolve workflow based on feedback
            evolved_workflow = await generator.evolve_workflow_from_feedback(
                workflow_id, performance_data
            )
            logger.info(f"Evolved workflow {workflow_id} based on feedback")
            break
    except Exception as e:
        logger.error(f"Error processing workflow feedback: {e}")

# Helper functions

def _identify_immediate_improvements(feedback: WorkflowPerformanceFeedback) -> List[str]:
    """Identify immediate improvements based on feedback"""
    improvements = []
    
    if feedback.overall_satisfaction < 3:
        improvements.append("Overall experience needs improvement")
    
    if feedback.completion_rate < 0.8:
        improvements.append("Workflow completion rate could be higher")
    
    for bottleneck in feedback.bottlenecks_encountered:
        improvements.append(f"Address bottleneck: {bottleneck}")
    
    return improvements

def _analyze_performance_trends(history: List[WorkflowPerformanceData]) -> Dict:
    """Analyze performance trends from historical data"""
    if not history:
        return {"trend": "No data available"}
    
    # Simple trend analysis
    recent_satisfaction = sum(
        data.satisfaction_scores.get("overall", 3) 
        for data in history[-5:]
    ) / min(len(history), 5)
    
    return {
        "recent_average_satisfaction": recent_satisfaction,
        "total_workflows_analyzed": len(history),
        "trend_direction": "improving" if recent_satisfaction > 3.5 else "needs_attention"
    }

def _identify_optimization_patterns(history: List[WorkflowPerformanceData]) -> List[str]:
    """Identify optimization patterns from historical data"""
    patterns = []
    
    if history:
        # Find common success factors
        all_success_factors = []
        for data in history:
            all_success_factors.extend(data.success_factors)
        
        # Count frequency
        factor_counts = {}
        for factor in all_success_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Get top patterns
        top_patterns = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        patterns = [f"{pattern}: {count} occurrences" for pattern, count in top_patterns]
    
    return patterns or ["No patterns identified yet"]

def _extract_success_factors(history: List[WorkflowPerformanceData]) -> List[str]:
    """Extract common success factors"""
    if not history:
        return ["Generate more workflows to identify success factors"]
    
    all_factors = []
    for data in history:
        all_factors.extend(data.success_factors)
    
    # Return unique factors
    return list(set(all_factors)) or ["No success factors identified yet"]

def _generate_optimization_recommendations(history: List[WorkflowPerformanceData]) -> List[str]:
    """Generate optimization recommendations based on historical data"""
    if not history:
        return [
            "Generate more workflows to get optimization recommendations",
            "Submit feedback on completed workflows",
            "Try different optimization goals"
        ]
    
    recommendations = [
        "Continue collecting performance data",
        "Experiment with different workflow structures",
        "Focus on addressing common bottlenecks"
    ]
    
    # Add data-driven recommendations based on patterns
    all_bottlenecks = []
    for data in history:
        all_bottlenecks.extend(data.common_bottlenecks)
    
    if all_bottlenecks:
        most_common_bottleneck = max(set(all_bottlenecks), key=all_bottlenecks.count)
        recommendations.append(f"Priority: Address '{most_common_bottleneck}' bottleneck")
    
    return recommendations