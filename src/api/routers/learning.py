"""
Learning System API Endpoints

Provides REST API access to the self-improving agent learning capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from ...models import ChatMessage
from ..dependencies import get_session, session_manager
from ...agentic_rag import AgenticRAGSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["learning"])

@router.post("/{session_id}/start")
async def start_learning(
    session_id: str,
    session: Dict[str, Any] = Depends(get_session)
):
    """Start the learning system for a session"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400, 
                detail="Agentic explorer not initialized"
            )
        
        # Start learning for the agent
        await agentic_rag.agentic_explorer.start_learning()
        
        return {
            "session_id": session_id,
            "learning_started": True,
            "message": "Learning system activated for this session"
        }
        
    except Exception as e:
        logger.error(f"Error starting learning for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/stop")
async def stop_learning(
    session_id: str,
    session: Dict[str, Any] = Depends(get_session)
):
    """Stop the learning system for a session"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Stop learning for the agent
        await agentic_rag.agentic_explorer.stop_learning()
        
        return {
            "session_id": session_id,
            "learning_stopped": True,
            "message": "Learning system deactivated for this session"
        }
        
    except Exception as e:
        logger.error(f"Error stopping learning for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/status")
async def get_learning_status(
    session_id: str,
    session: Dict[str, Any] = Depends(get_session)
):
    """Get learning status and performance insights for a session"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Get learning status
        status = await agentic_rag.agentic_explorer.get_learning_status()
        
        return {
            "session_id": session_id,
            "status": status,
            "timestamp": status.get("last_learning"),
            "metrics": {
                "patterns_learned": status.get("patterns_learned", 0),
                "learning_active": status.get("learning_active", False),
                "performance_trend": status.get("performance_summary", {}).get("improvement_trend", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting learning status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/query-with-learning")
async def query_with_learning(
    session_id: str,
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    session: Dict[str, Any] = Depends(get_session)
):
    """Execute a query using the learning-enabled agent"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Execute query with learning
        response = await agentic_rag.agentic_explorer.query_with_learning(
            query=message.content,
            stream=False
        )
        
        # Add user message to session history
        await session_manager.add_message(session_id, "user", message.content)
        
        # Add assistant response to session history
        await session_manager.add_message(
            session_id,
            "assistant",
            response,
            processingType="learning_enabled"
        )
        
        return {
            "session_id": session_id,
            "response": response,
            "learning_applied": True,
            "processing_type": "learning_enabled"
        }
        
    except Exception as e:
        logger.error(f"Error in learning query for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/patterns")
async def get_learned_patterns(
    session_id: str,
    pattern_type: Optional[str] = None,
    limit: int = 10,
    session: Dict[str, Any] = Depends(get_session)
):
    """Get learned patterns for a session"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Get patterns from the memory system
        memory_system = agentic_rag.agentic_explorer.memory_system
        pattern_storage = memory_system.pattern_storage
        
        if pattern_type:
            patterns = await pattern_storage.find_similar_patterns(
                query="",
                pattern_type=pattern_type,
                limit=limit
            )
        else:
            patterns = await pattern_storage.get_success_patterns(
                min_success_rate=0.6,
                min_usage_count=1
            )
            patterns = patterns[:limit]
        
        # Convert patterns to dict format
        pattern_data = []
        for pattern in patterns:
            pattern_dict = pattern.to_dict()
            pattern_data.append({
                "pattern_id": pattern_dict["pattern_id"],
                "type": pattern_dict["pattern_type"],
                "description": pattern_dict["description"],
                "success_rate": pattern_dict["success_rate"],
                "usage_count": pattern_dict["usage_count"],
                "created_at": pattern_dict["created_at"],
                "last_used": pattern_dict["last_used"]
            })
        
        return {
            "session_id": session_id,
            "patterns": pattern_data,
            "total_patterns": len(pattern_data),
            "pattern_type_filter": pattern_type
        }
        
    except Exception as e:
        logger.error(f"Error getting patterns for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/performance")
async def get_performance_metrics(
    session_id: str,
    days: int = 7,
    session: Dict[str, Any] = Depends(get_session)
):
    """Get performance metrics for a session"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Get performance summary
        performance_tracker = agentic_rag.agentic_explorer.performance_tracker
        summary = await performance_tracker.get_performance_summary(days=days)
        
        # Get recent metrics for trend analysis
        recent_metrics = await performance_tracker.get_recent_metrics(days=days)
        
        metrics_data = []
        for metric in recent_metrics[:20]:  # Last 20 interactions
            metrics_data.append({
                "timestamp": metric.timestamp.isoformat(),
                "query": metric.query[:100] + "..." if len(metric.query) > 100 else metric.query,
                "success": metric.success,
                "response_time": metric.response_time,
                "tokens_used": metric.tokens_used,
                "goal_achievement": metric.goal_achievement,
                "error_count": metric.error_count
            })
        
        return {
            "session_id": session_id,
            "performance_summary": summary,
            "recent_interactions": metrics_data,
            "time_period_days": days
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/insights")
async def get_learning_insights(
    session_id: str,
    session: Dict[str, Any] = Depends(get_session)
):
    """Get learning insights and recommendations for improvement"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Get learning insights
        memory_system = agentic_rag.agentic_explorer.memory_system
        insights = await memory_system.get_learning_insights()
        
        return {
            "session_id": session_id,
            "learning_insights": insights,
            "recommendations": {
                "should_continue_learning": insights.get("success_rate", 0) < 0.8,
                "focus_areas": insights.get("improvement_areas", []),
                "strong_patterns": insights.get("common_patterns", [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting learning insights for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/optimize")
async def apply_optimizations(
    session_id: str,
    query: str,
    session: Dict[str, Any] = Depends(get_session)
):
    """Get optimization recommendations for a specific query"""
    try:
        # Get AgenticRAG using the same pattern as other endpoints
        from ..dependencies import get_agentic_rag
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if not agentic_rag or not isinstance(agentic_rag, AgenticRAGSystem):
            raise HTTPException(
                status_code=400,
                detail="AgenticRAG system not available for this session"
            )
        
        if not agentic_rag.agentic_explorer:
            raise HTTPException(
                status_code=400,
                detail="Agentic explorer not initialized"
            )
        
        # Get optimization recommendations
        optimizations = await agentic_rag.agentic_explorer.apply_learned_optimizations(query)
        
        return {
            "session_id": session_id,
            "query": query,
            "optimizations": optimizations,
            "recommendations_applied": len(optimizations.get("strategy_adjustments", [])) > 0
        }
        
    except Exception as e:
        logger.error(f"Error applying optimizations for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))