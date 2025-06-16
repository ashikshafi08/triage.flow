"""
FastAPI router for predictive issue resolution endpoints
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from ...prediction.agents.orchestrator import PredictionOrchestrator
from ...prediction.data_models.prediction_models import (
    PredictionRequest,
    PreventionRequest,
    PredictionResponse,
    PreventionResponse,
    DashboardData
)
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prediction", tags=["prediction"])

# Cache for orchestrators to avoid re-initialization
_orchestrator_cache: Dict[str, PredictionOrchestrator] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_key: str):
        await websocket.accept()
        self.active_connections[session_key] = websocket

    def disconnect(self, session_key: str):
        if session_key in self.active_connections:
            del self.active_connections[session_key]

    async def send_progress_update(self, session_key: str, message: dict):
        if session_key in self.active_connections:
            try:
                await self.active_connections[session_key].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending progress update: {e}")
                self.disconnect(session_key)

manager = ConnectionManager()

async def get_orchestrator(repo_owner: str, repo_name: str, repo_path: str = None) -> PredictionOrchestrator:
    """Get or create a prediction orchestrator for the repository"""
    cache_key = f"{repo_owner}/{repo_name}"
    
    if cache_key not in _orchestrator_cache:
        # Use provided repo_path or try to find from session manager
        if not repo_path:
            # Try to find an existing session with this repository
            from ..dependencies import session_manager
            sessions = await session_manager.list_sessions(session_type="repo_chat")
            
            for session_info in sessions:
                if session_info.get("repo_url"):
                    # Extract owner/repo from session repo_url
                    url_parts = session_info["repo_url"].rstrip('/').split('/')
                    session_owner = url_parts[-2] if len(url_parts) >= 2 else ""
                    session_repo = url_parts[-1].replace('.git', '') if url_parts else ""
                    
                    if session_owner == repo_owner and session_repo == repo_name:
                        # Found matching session, get the actual repo path
                        session = await session_manager.get_session(session_info["id"])
                        if session and session.get("repo_path"):
                            repo_path = session["repo_path"]
                            logger.info(f"Found existing session {session_info['id']} for {cache_key} with repo_path: {repo_path}")
                            break
            
            # If still no repo_path, create a new session
            if not repo_path:
                repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
                logger.info(f"Creating new session for {cache_key} with repo_url: {repo_url}")
                session_id, metadata = await session_manager.create_repo_session(repo_url)
                
                # Initialize the session (this will clone the repo)
                await session_manager.initialize_repo_session(session_id)
                
                # Wait a bit for initialization and get the repo path
                await asyncio.sleep(2)  # Give it time to start cloning
                
                session = await session_manager.get_session(session_id)
                if session and session.get("repo_path"):
                    repo_path = session["repo_path"]
                    logger.info(f"Created new session {session_id} for {cache_key} with repo_path: {repo_path}")
                else:
                    # Fallback to default path if session creation failed
                    repo_path = f"/tmp/{repo_owner}_{repo_name}"
                    logger.warning(f"Session creation failed for {cache_key}, using fallback path: {repo_path}")
        
        _orchestrator_cache[cache_key] = PredictionOrchestrator(
            repo_path=repo_path,
            repo_owner=repo_owner,
            repo_name=repo_name
        )
        logger.info(f"Created new orchestrator for {cache_key} with repo_path: {repo_path}")
    
    return _orchestrator_cache[cache_key]

@router.post("/analyze", response_model=PredictionResponse)
async def analyze_repository(request: PredictionRequest):
    """
    Generate a comprehensive prediction report for a repository
    
    This endpoint performs multi-agent analysis to predict potential issues
    and provides actionable prevention strategies.
    """
    try:
        logger.info(f"Starting prediction analysis for {request.repo_owner}/{request.repo_name}")
        
        # Get orchestrator
        orchestrator = await get_orchestrator(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name,
            repo_path=request.repo_path
        )
        
        # Generate prediction report
        prediction_report = await orchestrator.generate_prediction_report(
            prediction_horizon_days=request.prediction_horizon_days
        )
        
        # Convert to dict for JSON response
        report_dict = {
            "repo_owner": prediction_report.repo_owner,
            "repo_name": prediction_report.repo_name,
            "analysis_timestamp": prediction_report.analysis_timestamp.isoformat(),
            "prediction_horizon_days": prediction_report.prediction_horizon_days,
            "predicted_issues": prediction_report.predicted_issues,
            "risk_factors": [
                {
                    "factor_type": rf.factor_type,
                    "severity": rf.severity,
                    "confidence": rf.confidence,
                    "description": rf.description,
                    "affected_files": rf.affected_files,
                    "mitigation_actions": rf.mitigation_actions
                }
                for rf in prediction_report.risk_factors
            ],
            "confidence_score": prediction_report.confidence_score,
            "detected_patterns": [
                {
                    "pattern_id": bp.pattern_id,
                    "pattern_type": bp.pattern_type,
                    "description": bp.description,
                    "confidence": bp.confidence,
                    "historical_occurrences": bp.historical_occurrences,
                    "files_affected": bp.files_affected,
                    "risk_score": bp.risk_score,
                    "prevention_strategies": bp.prevention_strategies
                }
                for bp in prediction_report.detected_patterns
            ],
            "team_patterns": [
                {
                    "pattern_id": tp.pattern_id,
                    "team_members": tp.team_members,
                    "pattern_type": tp.pattern_type,
                    "time_period": tp.time_period,
                    "correlation_strength": tp.correlation_strength,
                    "issue_count": tp.issue_count,
                    "prevention_recommendations": tp.prevention_recommendations
                }
                for tp in prediction_report.team_patterns
            ],
            "immediate_actions": prediction_report.immediate_actions,
            "long_term_strategies": prediction_report.long_term_strategies,
            "analysis_duration_seconds": prediction_report.analysis_duration_seconds,
            "data_sources_used": prediction_report.data_sources_used
        }
        
        return PredictionResponse(
            status="success",
            prediction_report=report_dict,
            analysis_duration_seconds=prediction_report.analysis_duration_seconds
        )
        
    except Exception as e:
        logger.error(f"Error in prediction analysis: {e}")
        return PredictionResponse(
            status="error",
            error=str(e)
        )

@router.post("/prevention", response_model=PreventionResponse)
async def generate_prevention_strategies(request: PreventionRequest):
    """
    Generate targeted prevention strategies for a repository
    
    This endpoint focuses specifically on prevention strategies based on
    current repository analysis and optional context.
    """
    try:
        logger.info(f"Generating prevention strategies for {request.repo_owner}/{request.repo_name}")
        
        # Get orchestrator
        orchestrator = await get_orchestrator(
            repo_owner=request.repo_owner,
            repo_name=request.repo_name
        )
        
        # Generate prevention strategies
        strategies = await orchestrator.generate_prevention_strategies(
            context_description=request.context_description
        )
        
        return PreventionResponse(
            status="success",
            prevention_strategies=strategies
        )
        
    except Exception as e:
        logger.error(f"Error generating prevention strategies: {e}")
        return PreventionResponse(
            status="error",
            error=str(e)
        )

@router.get("/dashboard/{repo_owner}/{repo_name}")
async def get_dashboard_data(repo_owner: str, repo_name: str) -> JSONResponse:
    """
    Get dashboard data for the prediction system
    
    Returns summary statistics, risk breakdowns, and recent predictions
    for display in the frontend dashboard.
    """
    try:
        logger.info(f"Getting dashboard data for {repo_owner}/{repo_name}")
        
        # Get orchestrator
        orchestrator = await get_orchestrator(
            repo_owner=repo_owner,
            repo_name=repo_name
        )
        
        # Create progress callback
        async def send_progress(update):
            session_key = f"{repo_owner}/{repo_name}"
            await manager.send_progress_update(session_key, {
                **update,
                "timestamp": datetime.now().isoformat()
            })
        
        # Get dashboard data with progress updates
        dashboard_data = await orchestrator.get_dashboard_data(progress_callback=send_progress)
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "summary": {"total_risks": 0, "high_priority_risks": 0, "confidence_score": 0.0},
                "risk_breakdown": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "pattern_insights": {},
                "action_items": {"immediate": [], "long_term": []},
                "recent_predictions": [],
                "trend_data": {}
            }
        )

@router.post("/explain-confidence/{repo_owner}/{repo_name}")
async def explain_confidence_score(repo_owner: str, repo_name: str, request_data: dict) -> JSONResponse:
    """
    Generate an AI explanation for the confidence score
    
    Takes the current confidence score and risk factors and generates
    a detailed explanation of how the confidence score was calculated.
    """
    try:
        logger.info(f"Explaining confidence score for {repo_owner}/{repo_name}")
        
        # Get orchestrator
        orchestrator = await get_orchestrator(
            repo_owner=repo_owner,
            repo_name=repo_name
        )
        
        confidence_score = request_data.get('confidence_score', 0.0)
        risk_factors = request_data.get('risk_factors', {})
        pattern_insights = request_data.get('pattern_insights', {})
        
        # Create explanation prompt
        explanation_prompt = f"""
        Explain the confidence score of {confidence_score:.1%} for the predictive analysis of {repo_owner}/{repo_name}.
        
        Risk Breakdown:
        - Critical: {risk_factors.get('critical', 0)}
        - High: {risk_factors.get('high', 0)}
        - Medium: {risk_factors.get('medium', 0)}
        - Low: {risk_factors.get('low', 0)}
        
        Pattern Analysis:
        - Bug patterns detected: {pattern_insights.get('bug_patterns_detected', 0)}
        - Team patterns detected: {pattern_insights.get('team_patterns_detected', 0)}
        - Pattern confidence: {pattern_insights.get('confidence_score', 0.0):.1%}
        
        Please provide a clear, technical explanation of:
        1. What factors contribute to this confidence level
        2. How the data quality affects the score
        3. What would improve the confidence
        4. Any limitations in the current analysis
        
        Keep the explanation concise but informative (2-3 paragraphs).
        """
        
        # Use the agentic explorer to generate explanation
        explanation = await orchestrator.agentic_explorer.query(explanation_prompt)
        
        return JSONResponse(content={
            "explanation": explanation,
            "confidence_score": confidence_score,
            "factors_analyzed": {
                "risk_distribution": risk_factors,
                "pattern_insights": pattern_insights
            }
        })
        
    except Exception as e:
        logger.error(f"Error explaining confidence score: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "explanation": "Unable to generate confidence explanation at this time. Please try again later.",
                "error": str(e)
            }
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for the prediction system"""
    return {"status": "healthy", "service": "predictive_issue_resolution"}

@router.post("/cache/clear/{repo_owner}/{repo_name}")
async def clear_cache(repo_owner: str, repo_name: str):
    """Clear the orchestrator cache for a specific repository"""
    cache_key = f"{repo_owner}/{repo_name}"
    
    if cache_key in _orchestrator_cache:
        del _orchestrator_cache[cache_key]
        logger.info(f"Cleared cache for {cache_key}")
        return {"status": "success", "message": f"Cache cleared for {cache_key}"}
    else:
        return {"status": "info", "message": f"No cache found for {cache_key}"}

@router.get("/metrics/{repo_owner}/{repo_name}")
async def get_prediction_metrics(repo_owner: str, repo_name: str):
    """
    Get prediction accuracy metrics and performance statistics
    
    This endpoint would be used for monitoring the prediction system's
    performance and accuracy over time.
    """
    try:
        # This would integrate with a metrics storage system
        # For now, return placeholder metrics
        metrics = {
            "accuracy_score": 0.75,
            "false_positive_rate": 0.15,
            "false_negative_rate": 0.10,
            "prediction_count": 42,
            "average_confidence": 0.68,
            "last_updated": "2024-06-14T19:00:00Z"
        }
        
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Error getting prediction metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Background task for periodic analysis
async def schedule_periodic_analysis(repo_owner: str, repo_name: str):
    """Background task to run periodic prediction analysis"""
    try:
        logger.info(f"Running periodic analysis for {repo_owner}/{repo_name}")
        
        orchestrator = await get_orchestrator(repo_owner, repo_name)
        await orchestrator.generate_prediction_report()
        
        logger.info(f"Completed periodic analysis for {repo_owner}/{repo_name}")
        
    except Exception as e:
        logger.error(f"Error in periodic analysis: {e}")

@router.websocket("/ws/progress/{repo_owner}/{repo_name}")
async def websocket_progress(websocket: WebSocket, repo_owner: str, repo_name: str):
    """WebSocket endpoint for real-time progress updates during prediction analysis"""
    session_key = f"{repo_owner}/{repo_name}"
    await manager.connect(websocket, session_key)
    
    try:
        # Send initial connection confirmation
        await manager.send_progress_update(session_key, {
            "type": "connection",
            "status": "connected",
            "message": f"Connected to progress updates for {repo_owner}/{repo_name}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for any message from client (ping/pong)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_progress_update(session_key, {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        manager.disconnect(session_key)
        logger.info(f"WebSocket disconnected for {session_key}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_key}: {e}")
        manager.disconnect(session_key)

@router.post("/schedule/{repo_owner}/{repo_name}")
async def schedule_analysis(repo_owner: str, repo_name: str, background_tasks: BackgroundTasks):
    """Schedule a background prediction analysis"""
    background_tasks.add_task(schedule_periodic_analysis, repo_owner, repo_name)
    
    return {
        "status": "scheduled",
        "message": f"Prediction analysis scheduled for {repo_owner}/{repo_name}"
    } 