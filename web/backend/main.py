from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from processing import process_issue
from store import jobs
import uuid
import os
import json
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime

from .processing import process_repository, get_indexing_status
from .store import session_manager
from .analysis_processing import process_issue_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IssueRequest(BaseModel):
    issue_url: str
    prompt_type: str  # "explain", "fix", "test", or "summarize"

class AnalyseIssueRequest(BaseModel):
    issue_url: str

class RepoRequest(BaseModel):
    repo_url: str
    session_name: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

@app.post("/process_issue")
async def handle_issue(request: IssueRequest, background_tasks: BackgroundTasks):
    # Validate prompt type
    valid_types = ["explain", "fix", "test", "summarize", "document", "review", "prioritize"]
    if request.prompt_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt type. Must be one of: {', '.join(valid_types)}"
        )
    
    # Create job ID
    job_id = f"job_{uuid.uuid4().hex}"
    
    # Store initial job status
    jobs[job_id] = {"status": "processing", "result": None, "error": None, "progress_log": []}
    
    # Add background task
    background_tasks.add_task(process_issue, job_id, request.issue_url, request.prompt_type)
    
    return {"job_id": job_id, "status": "processing"}

@app.post("/analyse_issue")
async def analyse_issue_endpoint(request: AnalyseIssueRequest, background_tasks: BackgroundTasks):
    job_id = f"analysis_{uuid.uuid4().hex}"
    jobs[job_id] = {"status": "queued", "progress_log": []}
    background_tasks.add_task(process_issue_analysis, job_id, request.issue_url)
    return {"job_id": job_id, "status": "queued"}

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/file-content")
async def get_file_content(session_id: str = Query(...), file_path: str = Query(...)):
    """Get file content with dynamic content handling"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        # Use the agentic_explorer to read the file (fix here)
        content = agentic_rag.agentic_explorer.read_file(file_path)
        
        # If content is JSON string (from chunked reading), parse it
        try:
            content_data = json.loads(content)
            return content_data
        except json.JSONDecodeError:
            # Regular string response (error or small file)
            return {"content": content}
            
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/file-content/stream")
async def stream_file_content(session_id: str = Query(...), file_path: str = Query(...)):
    """Stream large file content in chunks"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        # Use agentic_explorer for streaming
        return StreamingResponse(
            agentic_rag.agentic_explorer.stream_large_file(file_path),
            media_type="application/x-ndjson"
        )
            
    except Exception as e:
        logger.error(f"Error streaming file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/files")
# async def list_files():
#     file_list = []
#     for root, dirs, files in os.walk("."):
#         # Skip hidden files and directories
#         dirs[:] = [d for d in dirs if not d.startswith('.')]
#         for f in files:
#             if not f.startswith('.'):
#                 rel_path = os.path.relpath(os.path.join(root, f), ".")
#                 file_list.append({"path": rel_path})
#     return file_list

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: RepoRequest, background_tasks: BackgroundTasks):
    """Create a new session for repository analysis"""
    try:
        # Create session
        session_id, metadata = session_manager.create_repo_session(
            request.repo_url,
            session_name=request.session_name
        )
        
        # Start initialization in background
        background_tasks.add_task(session_manager.initialize_repo_session, session_id)
        
        return SessionResponse(
            session_id=session_id,
            status="initializing",
            message="Session created and initialization started",
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the status of a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get indexing status
        indexing_status = await get_indexing_status(session_id)
        
        return {
            "session_id": session_id,
            "status": session["metadata"]["status"],
            "indexing": indexing_status,
            "metadata": session["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/index")
async def index_repository(
    session_id: str,
    force_rebuild: bool = Query(False, description="Force rebuild of index"),
    max_issues: int = Query(1000, description="Maximum number of issues to index"),
    max_prs: int = Query(1000, description="Maximum number of PRs to index")
):
    """Index repository issues and PRs"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Start indexing in background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            process_repository,
            session_id,
            force_rebuild=force_rebuild,
            max_issues=max_issues,
            max_prs=max_prs
        )
        
        return {
            "status": "started",
            "message": "Indexing started in background",
            "session_id": session_id,
            "max_issues": max_issues,
            "max_prs": max_prs
        }
        
    except Exception as e:
        logger.error(f"Error starting indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/index/status")
async def get_repository_index_status(session_id: str):
    """Get the status of repository indexing"""
    try:
        return await get_indexing_status(session_id)
    except Exception as e:
        logger.error(f"Error getting indexing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/prs")
async def get_repository_prs(
    session_id: str,
    state: str = Query("all", description="PR state filter: open, closed, all"),
    limit: int = Query(100, description="Maximum number of PRs to return")
):
    """Get repository PRs"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        issue_rag = session.get("issue_rag")
        if not issue_rag:
            raise HTTPException(status_code=400, detail="Repository not indexed yet")
        
        # Get PRs from the index
        prs = await issue_rag.get_prs(state=state, limit=limit)
        
        return {
            "status": "success",
            "prs": prs,
            "total": len(prs)
        }
        
    except Exception as e:
        logger.error(f"Error getting PRs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/prs/{pr_number}")
async def get_pr_details(session_id: str, pr_number: int):
    """Get details for a specific PR"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        issue_rag = session.get("issue_rag")
        if not issue_rag:
            raise HTTPException(status_code=400, detail="Repository not indexed yet")
        
        # Get PR details from the index
        pr_details = await issue_rag.get_pr_details(pr_number)
        if not pr_details:
            raise HTTPException(status_code=404, detail=f"PR #{pr_number} not found")
        
        return {
            "status": "success",
            "pr": pr_details
        }
        
    except Exception as e:
        logger.error(f"Error getting PR details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/prs/{pr_number}/diff")
async def get_pr_diff(session_id: str, pr_number: int):
    """Get the diff for a specific PR"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        issue_rag = session.get("issue_rag")
        if not issue_rag:
            raise HTTPException(status_code=400, detail="Repository not indexed yet")
        
        # Get PR diff from the index
        pr_diff = await issue_rag.get_pr_diff(pr_number)
        if not pr_diff:
            raise HTTPException(status_code=404, detail=f"Diff for PR #{pr_number} not found")
        
        return {
            "status": "success",
            "diff": pr_diff
        }
        
    except Exception as e:
        logger.error(f"Error getting PR diff: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/search/prs")
async def search_prs(
    session_id: str,
    query: str = Query(..., description="Search query for PRs"),
    limit: int = Query(10, description="Maximum number of results to return")
):
    """Search PRs in the repository"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        issue_rag = session.get("issue_rag")
        if not issue_rag:
            raise HTTPException(status_code=400, detail="Repository not indexed yet")
        
        # Search PRs using the index
        results = await issue_rag.search_prs(query, limit=limit)
        
        return {
            "status": "success",
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching PRs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
