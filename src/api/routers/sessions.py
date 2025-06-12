from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from ...models import (
    PromptRequest, PromptResponse, SessionResponse, 
    RepoRequest, RepoSessionResponse, SessionListResponse
)
from ..dependencies import (
    session_manager, github_client, prompt_generator, 
    llm_client, get_session, logger, settings
)
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

router = APIRouter(tags=["sessions"])

@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: PromptRequest):
    try:
        # Create new session
        session_id = session_manager.create_session(
            request.issue_url, 
            request.prompt_type,
            request.llm_config
        )
        
        # Initialize session context in background
        await session_manager.initialize_session_context(session_id)
        
        # Get initial prompt
        session = session_manager.get_session(session_id)
        if not session or not session.get("issue_data"):
            raise HTTPException(status_code=404, detail="Issue not found")
            
        prompt_response = await prompt_generator.generate_prompt(
            request, 
            session["issue_data"]
        )
        
        if prompt_response.status == "error":
            raise HTTPException(status_code=400, detail=prompt_response.error)
            
        # Process initial prompt with LLM
        llm_response = await llm_client.process_prompt(
            prompt_response.prompt,
            prompt_type=request.prompt_type,
            model=request.llm_config.name,
            context=request.context
        )
        
        # Add initial messages to session
        session_manager.add_message(session_id, "system", prompt_response.prompt)
        session_manager.add_message(session_id, "assistant", llm_response.prompt)
        
        return {
            "session_id": session_id,
            "initial_message": llm_response.prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assistant/sessions", response_model=RepoSessionResponse)
async def create_assistant_session(request: RepoRequest):
    """Create a new repository-only chat session"""
    try:
        # Validate repository URL
        if not request.repo_url.startswith(('https://github.com/', 'http://github.com/')):
            raise HTTPException(status_code=400, detail="Invalid repository URL. Must be a GitHub repository.")
        
        # Create new repo session
        session_id, metadata = session_manager.create_repo_session(
            request.repo_url,
            request.initial_file,
            request.session_name
        )
        
        # Initialize repository context in background
        background_task = asyncio.create_task(session_manager.initialize_repo_session(session_id))
        
        # Wait a bit for initial status update
        await asyncio.sleep(0.5)
        
        # Get updated session
        session = session_manager.get_session(session_id)
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
        sessions = session_manager.list_sessions(session_type)
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
    if session_manager.delete_session(session_id):
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
        
        session_id, metadata = session_manager.create_repo_session(request.repo_url, session_name=request.session_name)
        session = session_manager.get_session(session_id)
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
                    session = session_manager.get_session(session_id)
                    
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
                    session = session_manager.get_session(session_id)
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
            session_manager.delete_session(session_id)
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
async def sync_repository_data(session_id: str, background_tasks: BackgroundTasks, session: Dict[str, Any] = Depends(get_session)):
    """
    Triggers a re-sync of the repository's issue and patch data.
    This involves re-running patch linkage and issue indexing.
    """
    agentic_rag = session.get("agentic_rag")
    if not agentic_rag:
        raise HTTPException(status_code=400, detail="AgenticRAG system not initialized for this session.")
    
    if not agentic_rag.issue_rag:
        logger.warning(f"Attempted to sync repo for session {session_id} but issue_rag is not available. Attempting to initialize.")
        raise HTTPException(status_code=400, detail="Issue RAG system not available for this session. Sync cannot proceed.")

    # Update session status to indicate syncing
    if "metadata" not in session:
        session["metadata"] = {}
    session["metadata"]["status"] = "syncing_issues"
    session["metadata"]["message"] = "Re-syncing repository issues, PRs, and patches..."

    async def _sync_task():
        try:
            logger.info(f"Starting repository data sync for session {session_id}...")
            # Calling initialize with force_rebuild=True will re-trigger
            # patch linkage and issue indexing.
            await agentic_rag.issue_rag.initialize(
                force_rebuild=True, 
                max_issues_for_patch_linkage=settings.MAX_ISSUES_TO_PROCESS,
                max_prs_for_patch_linkage=settings.MAX_PR_TO_PROCESS
            )
            session["metadata"]["status"] = "ready" 
            session["metadata"]["message"] = "Repository data sync complete. Full context updated."
            logger.info(f"Repository data sync complete for session {session_id}.")
        except Exception as e:
            logger.error(f"Error during repository data sync for session {session_id}: {e}", exc_info=True)
            session["metadata"]["status"] = "error_syncing"
            session["metadata"]["message"] = f"Error during repository data sync: {str(e)}"
            session["metadata"]["error"] = str(e)

    background_tasks.add_task(_sync_task)
    
    return {"message": "Repository data sync process started in the background."}
