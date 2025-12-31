from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from ...models import (
    SessionResponse, RepoRequest, RepoSessionResponse, SessionListResponse
)
from ..dependencies import (
    session_manager, github_client, llm_client,
    get_session, get_agentic_rag, logger, settings
)
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

router = APIRouter(tags=["sessions"])

# WebSocket connection management
active_connections: Dict[str, List[WebSocket]] = {}

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates"""
    await websocket.accept()
    
    # Add connection to active connections
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    
    logger.info(f"WebSocket connected for session {session_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive with ping/pong
        while True:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping messages
                if message == "ping":
                    await websocket.send_text("pong")
                else:
                    # Echo other messages back
                    await websocket.send_json({
                        "type": "message_received",
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Remove connection from active connections
        if session_id in active_connections:
            try:
                active_connections[session_id].remove(websocket)
                if not active_connections[session_id]:
                    del active_connections[session_id]
            except ValueError:
                pass  # Connection already removed

async def broadcast_to_session(session_id: str, message: dict):
    """Send message to all WebSocket connections for a session"""
    if session_id in active_connections:
        disconnected = []
        for websocket in active_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            try:
                active_connections[session_id].remove(ws)
            except ValueError:
                pass
        
        # Clean up empty session
        if not active_connections[session_id]:
            del active_connections[session_id]

@router.post("/assistant/sessions", response_model=RepoSessionResponse)
async def create_assistant_session(request: RepoRequest):
    """Create a new repository-only chat session"""
    try:
        # Validate repository URL
        if not request.repo_url.startswith(('https://github.com/', 'http://github.com/')):
            raise HTTPException(status_code=400, detail="Invalid repository URL. Must be a GitHub repository.")
        
        # Create new repo session
        session_id, metadata = await session_manager.create_repo_session(
            request.repo_url,
            request.initial_file,
            request.session_name
        )
        
        # Initialize repository context in background
        background_task = asyncio.create_task(session_manager.initialize_repo_session(session_id))
        
        # Wait a bit for initial status update
        await asyncio.sleep(0.5)
        
        # Get updated session
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return RepoSessionResponse(
            session_id=session_id,
            repo_metadata=session["metadata"],
            status=session["metadata"]["status"],
            message="Repository session created. Cloning and indexing in progress..."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assistant/sessions", response_model=SessionListResponse)
async def list_assistant_sessions(session_type: Optional[str] = Query(None)):
    """List all assistant sessions"""
    try:
        sessions = await session_manager.list_sessions(session_type)
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assistant/sessions/{session_id}/status")
async def get_session_status(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get the current status of a repository session"""
    metadata = session.get("metadata", {})
    
    return {
        "session_id": session_id,
        "status": metadata.get("status", "unknown"),
        "error": metadata.get("error"),
        "repo_info": session.get("repo_context", {}).get("repo_info") if session.get("repo_context") else None,
        "metadata": metadata
    }

@router.delete("/assistant/sessions/{session_id}")
async def delete_assistant_session(session_id: str):
    """Delete an assistant session and clean up resources"""
    if await session_manager.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/assistant/sessions/{session_id}/metadata")
async def get_session_metadata(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get detailed metadata about a session"""
    return {
        "session_id": session_id,
        "type": session.get("type"),
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "metadata": session.get("metadata", {}),
        "message_count": len(session.get("conversation_history", [])),
        "repo_info": session.get("repo_context", {}).get("repo_info") if session.get("repo_context") else None
    }

@router.get("/assistant/sessions/{session_id}/messages")
async def get_assistant_session_messages(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get conversation history for an assistant session"""
    try:
        conversation_history = session.get("conversation_history", [])
        
        # Format messages for frontend consumption
        formatted_messages = []
        for msg in conversation_history:
            formatted_msg = {
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp")
            }
            # Include any additional fields that might be present
            for key, value in msg.items():
                if key not in ["role", "content", "timestamp"]:
                    formatted_msg[key] = value
            formatted_messages.append(formatted_msg)
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "total_messages": len(formatted_messages)
        }
        
    except Exception as e:
        logger.error(f"Error getting session messages for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Founding Member Sessions
class FounderSessionRequest(BaseModel):
    repo_url: str
    session_name: Optional[str] = None

@router.post("/founder/sessions", response_model=SessionResponse)
async def create_founding_session(request: FounderSessionRequest, background_tasks: BackgroundTasks):
    """Create a new session with FoundingMemberAgent for a given repo (async patch linkage)."""
    try:
        # Validate repository URL format
        if not request.repo_url.startswith(('https://github.com/', 'http://github.com/')):
            raise HTTPException(
                status_code=400,
                detail="Invalid repository URL. Must be a GitHub repository URL starting with https://github.com/ or http://github.com/"
            )

        # Create session and initialize in background
        from ...new_rag import LocalRepoContextExtractor
        from ...issue_rag import IssueAwareRAG
        from ...founding_member_agent import FoundingMemberAgent
        
        session_id, metadata = await session_manager.create_repo_session(request.repo_url, session_name=request.session_name)
        session = await session_manager.get_session(session_id)
        session["metadata"]["status"] = "cloning"
        session["metadata"]["progress"] = 0.1
        session["metadata"]["message"] = "Cloning repository..."
        session["metadata"]["tools_ready"] = []
        
        try:
            # Load the repo (cloning)
            code_rag = LocalRepoContextExtractor()
            await code_rag.load_repository(request.repo_url)
            session["metadata"]["status"] = "indexing"
            session["metadata"]["progress"] = 0.4
            session["metadata"]["message"] = "Indexing codebase..."
            owner = metadata["owner"]
            repo = metadata["repo"]
            
            # Issue RAG (fast)
            issue_rag = IssueAwareRAG(owner, repo)
            await issue_rag.initialize(force_rebuild=False, max_issues_for_patch_linkage=10)
            session["metadata"]["status"] = "patch_linkage_pending"
            session["metadata"]["progress"] = 0.7
            session["metadata"]["message"] = "Patch linkage building in background..."
            session["metadata"]["tools_ready"] = ["code_rag", "issue_rag"]
            
            # Store code_rag and issue_rag for later use
            session["_code_rag"] = code_rag
            session["_issue_rag"] = issue_rag
            
            # Start patch linkage and agent setup in background
            async def finish_patch_linkage_and_agent():
                try:
                    # Re-initialize issue_rag with full patch linkage
                    await issue_rag.initialize(force_rebuild=False)
                    session = await session_manager.get_session(session_id)
                    
                    # Create the agent and store in session
                    agent = FoundingMemberAgent(session_id, code_rag, issue_rag)
                    session["founding_member_agent"] = agent
                    session["has_founding_member"] = True
                    session["metadata"]["session_subtype"] = "founding_member"
                    session["metadata"]["status"] = "ready"
                    session["metadata"]["progress"] = 1.0
                    session["metadata"]["message"] = f"FoundingMemberAgent session for {owner}/{repo} is ready."
                    session["metadata"]["tools_ready"] = ["code_rag", "issue_rag", "patch_linkage", "founding_member_agent"]
                except Exception as e:
                    session = await session_manager.get_session(session_id)
                    session["metadata"]["status"] = "error"
                    session["metadata"]["progress"] = 1.0
                    session["metadata"]["message"] = f"Failed to initialize: {str(e)}"
                    session["metadata"]["error"] = str(e)
            
            background_tasks.add_task(finish_patch_linkage_and_agent)
            
            return {"session_id": session_id, "initial_message": f"FoundingMemberAgent session for {owner}/{repo} is initializing. Patch linkage and advanced tools will be available soon."}
            
        except Exception as e:
            session["metadata"]["status"] = "error"
            session["metadata"]["progress"] = 1.0
            session["metadata"]["message"] = f"Failed to initialize: {str(e)}"
            session["metadata"]["error"] = str(e)
            await session_manager.delete_session(session_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize repository session: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/founder/sessions/{session_id}/status")
async def get_founding_session_status(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get the current status and progress of a founding member session."""
    metadata = session.get("metadata", {})
    return {
        "session_id": session_id,
        "status": metadata.get("status", "unknown"),
        "progress": metadata.get("progress", 0.0),
        "message": metadata.get("message", ""),
        "error": metadata.get("error"),
        "session_subtype": metadata.get("session_subtype"),
        "tools_ready": metadata.get("tools_ready", []),
    }

@router.post("/assistant/sessions/{session_id}/sync-repository")
async def sync_repository_data(
    session_id: str, 
    background_tasks: BackgroundTasks, 
    force_full_sync: bool = Query(False, description="Force full rebuild instead of incremental sync"),
    max_new_issues: int = Query(5, description="Maximum new issues to sync in incremental mode"),
    max_new_prs: int = Query(5, description="Maximum new PRs to sync in incremental mode"),
    session: Dict[str, Any] = Depends(get_session),
    agentic_rag = Depends(get_agentic_rag)
):
    """
    Triggers a sync of the repository's issue and patch data.
    By default, performs incremental sync to only fetch new/changed content.
    Use force_full_sync=true to rebuild everything from scratch.
    """
    if not agentic_rag:
        raise HTTPException(status_code=400, detail="AgenticRAG system not initialized for this session.")
    
    if not agentic_rag.issue_rag:
        logger.warning(f"Attempted to sync repo for session {session_id} but issue_rag is not available. Attempting to initialize.")
        raise HTTPException(status_code=400, detail="Issue RAG system not available for this session. Sync cannot proceed.")

    # Update session status to indicate syncing
    if "metadata" not in session:
        session["metadata"] = {}
    
    sync_type = "full rebuild" if force_full_sync else "incremental sync"
    session["metadata"]["status"] = "syncing_issues"
    session["metadata"]["message"] = f"Starting {sync_type} of repository data..."

    async def _sync_task():
        try:
            logger.info(f"Starting repository data sync ({sync_type}) for session {session_id}...")
            
            if force_full_sync:
                # Full rebuild - re-initialize everything
                await agentic_rag.issue_rag.initialize(
                    force_rebuild=True, 
                    max_issues_for_patch_linkage=settings.MAX_PATCH_LINKAGE_ISSUES,
                    max_prs_for_patch_linkage=settings.MAX_PR_TO_PROCESS
                )
                session["metadata"]["status"] = "ready" 
                session["metadata"]["message"] = "Full repository rebuild complete. All context updated."
                logger.info(f"Full repository rebuild complete for session {session_id}.")
            else:
                # Incremental sync - only fetch new/changed content
                sync_result = await agentic_rag.issue_rag.incremental_sync(
                    max_new_issues=max_new_issues,
                    max_new_prs=max_new_prs
                )
                
                if sync_result["status"] == "completed":
                    new_items = sync_result["total_new_items"]
                    session["metadata"]["status"] = "ready"
                    session["metadata"]["message"] = f"Incremental sync complete. Added {new_items} new items ({sync_result['new_issues']} issues, {sync_result['new_prs']} PRs)."
                    session["metadata"]["last_sync_result"] = sync_result
                elif sync_result["status"] == "skipped":
                    session["metadata"]["status"] = "ready"
                    session["metadata"]["message"] = f"Sync skipped: {sync_result['reason']}. Last sync was {sync_result.get('hours_since_last_sync', 0):.1f} hours ago."
                    session["metadata"]["last_sync_result"] = sync_result
                elif sync_result["status"] == "full_initialization":
                    session["metadata"]["status"] = "ready"
                    session["metadata"]["message"] = "System was not initialized, performed full initialization instead."
                    session["metadata"]["last_sync_result"] = sync_result
                else:
                    # Error case
                    session["metadata"]["status"] = "warning_sync_error"
                    session["metadata"]["message"] = f"Incremental sync encountered an error: {sync_result.get('error', 'Unknown error')}"
                    session["metadata"]["last_sync_result"] = sync_result
                
                logger.info(f"Incremental sync result for session {session_id}: {sync_result}")
                
        except Exception as e:
            logger.error(f"Error during repository data sync for session {session_id}: {e}", exc_info=True)
            session["metadata"]["status"] = "error_syncing"
            session["metadata"]["message"] = f"Error during repository data sync: {str(e)}"
            session["metadata"]["error"] = str(e)

    background_tasks.add_task(_sync_task)
    
    return {
        "message": f"Repository {sync_type} started in the background.",
        "sync_type": sync_type,
        "force_full_sync": force_full_sync,
        "max_new_issues": max_new_issues if not force_full_sync else None,
        "max_new_prs": max_new_prs if not force_full_sync else None
    }

@router.get("/sessions/{session_id}/available-tools")
async def get_available_tools(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get available tools for a session"""
    try:
        # Get tools from session metadata
        tools_ready = session.get("metadata", {}).get("tools_ready", [])
        
        # Convert tool names to tool objects with descriptions
        tool_descriptions = {
            "code_rag": "Code repository analysis and retrieval",
            "issue_rag": "GitHub issues and PR analysis", 
            "patch_linkage": "Patch analysis and code changes",
            "founding_member_agent": "Advanced founding member analysis",
            "enhanced_context_retrieval": "Enhanced context retrieval",
            "semantic_search": "Semantic search capabilities",
            "file_structure_analysis": "File structure analysis",
            "related_files_discovery": "Related files discovery",
            "query_analysis": "Query analysis", 
            "technical_requirements_extraction": "Technical requirements extraction",
            "code_references_detection": "Code references detection",
            "code_example_generation": "Code example generation",
            "issue_context_retrieval": "Issue context retrieval",
            "related_issues_search": "Related issues search",
            "issue_history_analysis": "Issue history analysis",
            "multi_index_retrieval": "Multi-index retrieval"
        }
        
        tools = []
        for tool_name in tools_ready:
            tools.append({
                "name": tool_name,
                "description": tool_descriptions.get(tool_name, f"{tool_name} capability")
            })
        
        return {"tools": tools}
        
    except Exception as e:
        logger.error(f"Error getting available tools for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
