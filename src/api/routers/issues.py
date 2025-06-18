from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from ...models import Issue, IssueContextResponse, PullRequestInfo, IssueContextRequest
from ..dependencies import github_client, get_session, logger
from ...issue_rag import IssueAwareRAG
from ...local_repo_loader import get_repo_info
from ...config import settings
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncio
import re
import time

router = APIRouter(prefix="/api", tags=["issues"])

@router.post("/v1/issue_context", response_model=IssueContextResponse)
async def get_issue_context_api(request: IssueContextRequest):
    """
    Get issue context including related issues and patches for a given query and repository.
    Initializes IssueAwareRAG for the repo on each call for this spike.
    """
    try:
        if not request.repo_url.startswith(('https://github.com/', 'http://github.com/')):
            raise HTTPException(status_code=400, detail="Invalid repository URL. Must be a GitHub repository.")

        owner, repo_name_from_url = get_repo_info(request.repo_url)

        issue_rag_system = IssueAwareRAG(owner, repo_name_from_url)
        
        # For this spike, we use force_rebuild=False to leverage caching if the index exists.
        # max_issues_for_patch_linkage is set to a small number for potentially faster cold starts
        # if the index needs to be built.
        await issue_rag_system.initialize(force_rebuild=False, max_issues_for_patch_linkage=settings.MAX_PATCH_LINKAGE_ISSUES)

        if not issue_rag_system.is_initialized():
            # This case should ideally be handled by initialize() raising an error,
            # but as a safeguard:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG system for the repository.")

        context_response = await issue_rag_system.get_issue_context(
            query=request.query,
            max_issues=request.max_issues,
            include_patches=request.include_patches
        )
        # Ensure the response is a Pydantic model or can be converted to one by FastAPI
        return context_response
    except Exception as e:
        logger.error(f"Error in /api/v1/issue_context: {e}")
        import traceback
        traceback.print_exc() # Log full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@router.get("/issues")
async def list_issues(
    repo_url: str, 
    state: str = "open",
    per_page: int = Query(30, description="Number of issues per page (max 100)", ge=1, le=100),
    max_pages: int = Query(10, description="Maximum number of pages to fetch", ge=1, le=50)
):
    """List issues for a given repository URL and state (open/closed/all)."""
    try:
        issues = await github_client.list_issues(repo_url, state, per_page=per_page, max_pages=max_pages)
        # Convert Issue objects to dicts for JSON serialization
        return [issue.model_dump() for issue in issues]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch issues: {str(e)}")

@router.get("/issues/{issue_number}")
async def get_issue_detail(issue_number: int, repo_url: str):
    """Get details for a single issue by number and repo_url."""
    try:
        # Construct the issue URL
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        issue_url = f"{repo_url}/issues/{issue_number}"
        issue_response = await github_client.get_issue(issue_url)
        if issue_response.status != "success" or not issue_response.data:
            raise HTTPException(status_code=404, detail=issue_response.error or "Issue not found")
        return issue_response.data.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch issue: {str(e)}")

@router.get("/prs", response_model=List[PullRequestInfo])
async def list_pull_requests(
    repo_url: str = Query(..., description="Repository URL to fetch pull requests from"),
    state: str = Query("merged", description="State of the pull requests (e.g., open, closed, merged, all)"),
    session_id: Optional[str] = Query(None, description="Optional session ID for context")
):
    """List pull requests for a given repository URL and state."""
    try:
        logger.info(f"Received request for PRs: repo_url={repo_url}, state={state}, session_id={session_id}")
        # Fetch PRs using the github_client
        pr_list = await github_client.list_pull_requests(repo_url, state)
        # The list_pull_requests method in github_client already returns a list of PullRequestInfo Pydantic models.
        return pr_list
    except Exception as e:
        logger.error(f"Failed to fetch pull requests for {repo_url} with state {state}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch pull requests: {str(e)}")

@router.get("/commits")
async def list_commits(
    session_id: str = Query(..., description="Session ID to identify the repository"),
    limit: int = Query(50, description="Maximum number of commits to return"),
    since_date: Optional[str] = Query(None, description="ISO date string to filter commits since"),
    author: Optional[str] = Query(None, description="Filter by author name or email"),
    session: Dict[str, Any] = Depends(get_session)
):
    """List recent commits from the repository associated with the session"""
    try:
        # Get the agentic explorer for this session
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag or not hasattr(agentic_rag, 'commit_index_manager'):
            # Try to get the repo path and initialize commit index
            repo_path = session.get("repo_path")
            if not repo_path:
                raise HTTPException(status_code=400, detail="Repository not available for this session")
            
            # Import commit indexer
            from ...commit_index import CommitIndexManager
            
            # Initialize commit index manager
            commit_manager = CommitIndexManager(repo_path)
            await commit_manager.initialize(max_commits=min(limit * 2, 1000))
            
            if not commit_manager.is_initialized():
                raise HTTPException(status_code=500, detail="Failed to initialize commit index")
                
            # Get commits from the indexer
            commits = list(commit_manager.indexer.commit_metas.values())
        else:
            # Get commits from existing agentic RAG system
            if hasattr(agentic_rag, 'commit_index_manager') and agentic_rag.commit_index_manager:
                commits = list(agentic_rag.commit_index_manager.indexer.commit_metas.values())
            else:
                commits = []

        # Sort commits by date (newest first)
        commits.sort(key=lambda x: x.commit_date, reverse=True)
        
        # Apply filters
        if since_date:
            from datetime import datetime
            since_dt = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
            commits = [c for c in commits if datetime.fromisoformat(c.commit_date.replace('Z', '+00:00')) >= since_dt]
        
        if author:
            commits = [c for c in commits if author.lower() in c.author_name.lower() or author.lower() in c.author_email.lower()]
        
        # Limit results
        commits = commits[:limit]
        
        # Format response
        formatted_commits = []
        for commit in commits:
            formatted_commits.append({
                "sha": commit.sha,
                "subject": commit.subject,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "commit_date": commit.commit_date,
                "files_changed": commit.files_changed,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "is_merge": commit.is_merge,
                "pr_number": commit.pr_number
            })
        
        return {
            "commits": formatted_commits,
            "total": len(formatted_commits),
            "filtered": bool(since_date or author)
        }
        
    except Exception as e:
        logger.error(f"Error fetching commits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Issue Analysis endpoints
class IssueAnalysisRequest(BaseModel):
    issue_url: str = Field(..., description="Full URL of the GitHub issue, e.g. https://github.com/owner/repo/issues/123")

class IssueAnalysisResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[dict] = None

@router.post("/issue_analysis", response_model=IssueAnalysisResponse)
async def start_issue_analysis(request: IssueAnalysisRequest):
    """Kick off the async issue → plan pipeline. Returns a session_id which can be polled."""
    try:
        if not request.issue_url.startswith(("https://github.com/", "http://github.com/")):
            raise HTTPException(status_code=400, detail="Invalid GitHub issue URL")

        # Create session and launch background task
        from ..dependencies import session_manager
        session_id = session_manager.create_session(request.issue_url, prompt_type="issue_analysis")
        session_manager.launch_issue_analysis(session_id)

        return IssueAnalysisResponse(session_id=session_id, status="pending")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to launch issue analysis")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/issue_analysis/{session_id}", response_model=IssueAnalysisResponse)
async def get_issue_analysis_status(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Poll the status/result of an issue analysis run."""
    if session.get("type") != "issue_analysis":
        raise HTTPException(status_code=404, detail="Session not found or not issue analysis")

    return IssueAnalysisResponse(
        session_id=session_id,
        status=session.get("status", "unknown"),
        result=session.get("result"),
    )

class AnalyzeIssueRequest(BaseModel):
    issue_url: str = Field(..., description="Full URL of the GitHub issue")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

@router.post("/analyze-issue")
async def analyze_issue_endpoint(request: AnalyzeIssueRequest):
    """Run comprehensive issue analysis using the agentic issue_analysis pipeline."""
    try:
        logger.info(f"Starting issue analysis for: {request.issue_url}")
        
        # Check cache first
        from ...cache import issue_analysis_cache
        cache_key = f"analysis:{request.issue_url}"
        cached_result = await issue_analysis_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached analysis for: {request.issue_url}")
            return cached_result
        
        # Try to reuse existing session if provided
        analysis_result = None
        if request.session_id:
            from ..dependencies import session_manager, get_agentic_rag as get_agentic_rag_dependency

            # Fetch the session dictionary first
            session_dict = await session_manager.get_session(request.session_id)

            if session_dict:
                try:
                    # Extract repository info from the issue URL to verify compatibility
                    import re
                    issue_match = re.search(r'github\.com/([^/]+)/([^/]+)/issues/(\d+)', request.issue_url)
                    if issue_match:
                        issue_owner, issue_repo, issue_number = issue_match.groups()
                        issue_repo_key = f"{issue_owner}/{issue_repo}"
                        
                        # Check if session is for the same repository
                        session_repo_info = session_dict.get("repo_context", {}).get("repo_info", {})
                        session_owner = session_repo_info.get("owner") or session_dict.get("metadata", {}).get("owner")
                        session_repo = session_repo_info.get("repo") or session_dict.get("metadata", {}).get("repo")
                        session_repo_key = f"{session_owner}/{session_repo}" if session_owner and session_repo else None
                        
                        if session_repo_key == issue_repo_key:
                            logger.info(f"Session {request.session_id} matches issue repository {issue_repo_key}, attempting reuse")
                            
                            # Call get_agentic_rag properly with both session_id and session parameters
                            reconstructed_agentic_rag = await get_agentic_rag_dependency(request.session_id, session_dict)
                            if reconstructed_agentic_rag:
                                logger.info(f"Successfully reused existing RAG system from session {request.session_id}")
                                from ...issue_analysis import analyse_issue_with_existing_rag
                                analysis_result = await analyse_issue_with_existing_rag(
                                    request.issue_url, 
                                    reconstructed_agentic_rag
                                )
                            else:
                                logger.warning(f"Session {request.session_id} found but AgenticRAG could not be obtained")
                        else:
                            logger.info(f"Session {request.session_id} is for different repository ({session_repo_key} vs {issue_repo_key}), running fresh analysis")
                    else:
                        logger.warning(f"Could not extract repository info from issue URL: {request.issue_url}")
                        
                    # If we didn't get a result from reuse, fall back to fresh analysis
                    if not analysis_result:
                        logger.info("Falling back to fresh analysis")
                        from ...issue_analysis import analyse_issue
                        analysis_result = await analyse_issue(request.issue_url)
                        
                except Exception as e:
                    logger.warning(f"Failed to use existing session {request.session_id}: {e}. Running full analysis.")
                    from ...issue_analysis import analyse_issue
                    analysis_result = await analyse_issue(request.issue_url)
            else:
                logger.info(f"Session {request.session_id} not found. Running full analysis.")
                from ...issue_analysis import analyse_issue
                analysis_result = await analyse_issue(request.issue_url)
        else:
            logger.info("No session ID provided, running full analysis")
            from ...issue_analysis import analyse_issue
            analysis_result = await analyse_issue(request.issue_url)
        
        if not analysis_result:
            raise ValueError("Analysis returned no result")
            
        logger.info(f"Issue analysis completed with status: {analysis_result.get('status')}")
        
        # Transform the result to match the expected frontend structure
        session_id_for_response = request.session_id or f"analysis_{hash(request.issue_url) % 10000}"
        
        # Handle PR Detection results
        pr_detection_result = {"has_existing_prs": False, "message": "No existing PRs found"}
        pr_detection_status = "completed"
        
        if analysis_result.get("status") == "skipped" and analysis_result.get("reason") in ["pr_exists", "related_open_prs"]:
            # Issue has existing PRs or related work
            pr_info = analysis_result.get("pr_info", {})
            enhanced_pr_info = analysis_result.get("enhanced_pr_info", {})
            
            if analysis_result.get("reason") == "pr_exists":
                pr_detection_result = {
                    "has_existing_prs": True,
                    "pr_state": pr_info.get("state"),
                    "pr_number": pr_info.get("pr_number"),
                    "pr_url": pr_info.get("pr_url"),
                    "message": f"Found existing {pr_info.get('state', 'unknown')} PR #{pr_info.get('pr_number', 'unknown')}"
                }
            elif analysis_result.get("reason") == "related_open_prs":
                related_open_prs = []
                if enhanced_pr_info.get("related_open_prs"):
                    for pr in enhanced_pr_info["related_open_prs"]:
                        related_open_prs.append({
                            "pr_number": pr.pr_number,
                            "title": pr.title,
                            "author": pr.author,
                            "url": pr.url,
                            "draft": pr.draft,
                            "review_decision": pr.review_decision
                        })
                
                related_merged_prs = []
                if enhanced_pr_info.get("related_merged_prs"):
                    for pr in enhanced_pr_info["related_merged_prs"]:
                        related_merged_prs.append({
                            "pr_number": pr.pr_number,
                            "pr_url": pr.pr_url,
                            "pr_title": pr.pr_title,
                            "merged_at": pr.merged_at
                        })
                
                pr_detection_result = {
                    "has_existing_prs": True,
                    "related_merged_prs": related_merged_prs,
                    "related_open_prs": related_open_prs,
                    "message": enhanced_pr_info.get("message", "Found related work in progress")
                }

        # Create step-by-step structure for UI
        steps = [
            {"step": "PR Detection", "status": pr_detection_status, "result": pr_detection_result},
            {"step": "Issue Classification", "status": "completed" if analysis_result.get("classification") else "error", "result": analysis_result.get("classification")},
            {"step": "Codebase Analysis", "status": "completed" if analysis_result.get("agentic_analysis") else "error", "result": analysis_result.get("agentic_analysis")},
            {"step": "Solution Planning", "status": "completed" if analysis_result.get("plan_markdown") else "error", "result": {"plan_markdown": analysis_result.get("plan_markdown")}}
        ]
        
        # Structure final result for frontend compatibility
        final_result = {}
        
        if classification_data := analysis_result.get("classification"):
            final_result["classification"] = {
                "category": classification_data.get("label", "unknown"),
                "confidence": classification_data.get("confidence", 0.0),
                "reasoning": classification_data.get("reasoning", "")
            }
        
        agentic_data = analysis_result.get("agentic_analysis", {})
        if key_files_primary := agentic_data.get("key_files", {}).get("primary"):
            final_result["related_files"] = key_files_primary
        elif file_discovery_primary := agentic_data.get("file_discovery", {}).get("primary_files"):
            final_result["related_files"] = file_discovery_primary
        
        if plan_markdown := analysis_result.get("plan_markdown"):
            final_result["remediation_plan"] = plan_markdown
        
        if agentic_data:
            final_result["agentic_insights"] = agentic_data
        
        # Extract issue title and number from URL for better caching
        issue_title = None
        issue_number = None
        try:
            import re
            issue_match = re.search(r'/issues/(\d+)', request.issue_url)
            if issue_match:
                issue_number = int(issue_match.group(1))
                
                # Try to get issue title from GitHub API if we have access
                try:
                    repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', request.issue_url)
                    if repo_match:
                        owner, repo = repo_match.groups()
                        repo = repo.replace('.git', '')
                        
                        # Simple GitHub API call to get issue title
                        import httpx
                        async with httpx.AsyncClient() as client:
                            response = await client.get(
                                f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
                                headers={"Accept": "application/vnd.github.v3+json"}
                            )
                            if response.status_code == 200:
                                issue_data = response.json()
                                issue_title = issue_data.get("title", f"Issue #{issue_number}")
                except Exception as e:
                    logger.warning(f"Failed to fetch issue title from GitHub: {e}")
                    issue_title = f"Issue #{issue_number}"
        except Exception as e:
            logger.warning(f"Failed to extract issue info: {e}")

        response_payload = {
            "session_id": session_id_for_response,
            "steps": steps,
            "final_result": final_result,
            "status": analysis_result.get("status"),
            "error": analysis_result.get("error"),
            "cached_at": time.time(),
            "issue_url": request.issue_url,
            "issue_title": issue_title,
            "issue_number": issue_number
        }
        
        # Cache the result for future use
        await issue_analysis_cache.set(cache_key, response_payload)
        logger.info(f"Cached analysis result for: {request.issue_url}")
        
        return response_payload
        
    except Exception as e:
        logger.error(f"Issue analysis failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/cached-analyses/{session_id}")
async def get_cached_analyses(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get all cached issue analyses for a session/repository."""
    try:
        from ...cache import issue_analysis_cache
        
        # Get repository info from session
        repo_url = session.get("repo_url") or session.get("metadata", {}).get("repo_url")
        if not repo_url:
            return {"cached_analyses": []}
        
        # Extract owner/repo from URL for pattern matching
        import re
        match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            return {"cached_analyses": []}
        
        owner, repo = match.groups()
        repo = repo.replace('.git', '')
        repo_pattern = f"{owner}/{repo}"
        
        # Get all cached analyses (this is a simplified approach)
        # In a production system, you'd want to maintain an index
        cached_analyses = []
        
        # Try to get a list of cached keys from Redis if available
        if hasattr(issue_analysis_cache, 'redis') and issue_analysis_cache.redis.initialized:
            try:
                # Scan for keys matching our pattern
                cursor = 0
                while True:
                    cursor, keys = await issue_analysis_cache.redis.redis_client.scan(
                        cursor, match=f"issue_analysis:analysis:*github.com/{repo_pattern}/issues/*", count=100
                    )
                    
                    for key in keys:
                        try:
                            # Get the cached analysis
                            cached_data = await issue_analysis_cache.redis.redis_client.get(key)
                            if cached_data:
                                import json
                                analysis = json.loads(cached_data)
                                
                                # Extract issue info from URL
                                issue_url = analysis.get("issue_url", "")
                                issue_match = re.search(r'/issues/(\d+)', issue_url)
                                issue_number = int(issue_match.group(1)) if issue_match else None
                                
                                # Get issue title from stored data or fallback
                                issue_title = analysis.get("issue_title")
                                if not issue_title and issue_number:
                                    issue_title = f"Issue #{issue_number}"
                                
                                cached_analyses.append({
                                    "issue_url": issue_url,
                                    "cached_at": analysis.get("cached_at", time.time()),
                                    "status": analysis.get("status", "unknown"),
                                    "issue_title": issue_title,
                                    "issue_number": issue_number
                                })
                        except Exception as e:
                            logger.warning(f"Failed to parse cached analysis {key}: {e}")
                            continue
                    
                    if cursor == 0:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to scan Redis for cached analyses: {e}")
        
        # Sort by cached_at descending (newest first)
        cached_analyses.sort(key=lambda x: x["cached_at"], reverse=True)
        
        return {
            "cached_analyses": cached_analyses,
            "repository": repo_pattern,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get cached analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis-cache/{issue_url:path}")
async def get_cached_analysis(issue_url: str):
    """Get a specific cached analysis by issue URL."""
    try:
        from ...cache import issue_analysis_cache
        
        cache_key = f"analysis:{issue_url}"
        cached_result = await issue_analysis_cache.get(cache_key)
        
        if cached_result:
            return {
                "found": True,
                "analysis": cached_result,
                "cached_at": cached_result.get("cached_at"),
                "issue_url": issue_url
            }
        else:
            return {
                "found": False,
                "issue_url": issue_url
            }
            
    except Exception as e:
        logger.error(f"Failed to get cached analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/analysis-cache/{issue_url:path}")
async def delete_cached_analysis(issue_url: str):
    """Delete a specific cached analysis."""
    try:
        from ...cache import issue_analysis_cache
        
        cache_key = f"analysis:{issue_url}"
        deleted = await issue_analysis_cache.delete(cache_key)
        
        return {
            "deleted": deleted,
            "issue_url": issue_url
        }
        
    except Exception as e:
        logger.error(f"Failed to delete cached analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ApplyPatchRequest(BaseModel):
    patch_content: str = Field(..., description="Unified diff content to apply")
    session_id: str = Field(..., description="Session ID to identify the repository")

@router.post("/apply-patch")
async def apply_patch_endpoint(request: ApplyPatchRequest):
    """Apply a unified diff patch to the repository files."""
    import subprocess
    import tempfile
    import os
    
    try:
        logger.info(f"Creating patch for session: {request.session_id}")
        
        # Get session to access repository path and agentic capabilities
        from ..dependencies import session_manager
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        repo_path = session.get("repo_path")
        if not repo_path:
            raise HTTPException(status_code=400, detail="Repository path not found in session")
        
        # Create temporary patch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
            patch_file.write(request.patch_content)
            patch_file_path = patch_file.name
        
        try:
            # Apply patch using git apply
            result = subprocess.run(
                ["git", "apply", "--check", patch_file_path],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Try to provide helpful error message
                error_msg = result.stderr.strip()
                if "does not exist in index" in error_msg:
                    return {
                        "success": False,
                        "error": "Patch cannot be applied - target file not found or has been modified",
                        "details": error_msg
                    }
                elif "patch does not apply" in error_msg:
                    return {
                        "success": False,
                        "error": "Patch does not apply cleanly - file may have changed since analysis",
                        "details": error_msg
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Patch validation failed: {error_msg}",
                        "details": error_msg
                    }
            
            # Actually apply the patch
            result = subprocess.run(
                ["git", "apply", patch_file_path],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Failed to apply patch: {result.stderr.strip()}",
                    "details": result.stderr.strip()
                }
            
            # Get list of modified files
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            modified_files = []
            if status_result.returncode == 0:
                for line in status_result.stdout.strip().split('\n'):
                    if line.strip():
                        # Parse git status format: "XY filename"
                        status = line[:2]
                        filename = line[3:]
                        modified_files.append({
                            "file": filename,
                            "status": status.strip()
                        })
            
            return {
                "success": True,
                "message": "Patch applied successfully",
                "modified_files": modified_files
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(patch_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply patch: {str(e)}")

class PostToGitHubRequest(BaseModel):
    issue_url: str = Field(..., description="Full URL of the GitHub issue")
    analysis_result: dict = Field(..., description="Analysis result data to post")
    custom_message: Optional[str] = Field(None, description="Optional custom message to prepend")

@router.post("/post-to-github")
async def post_analysis_to_github(request: PostToGitHubRequest):
    """Post analysis results as a comment to the GitHub issue."""
    try:
        logger.info(f"Posting analysis to GitHub issue: {request.issue_url}")
        
        # Import and initialize the triage bot
        from ...triage_bot import TriageBot
        triage_bot = TriageBot()
        
        # Post the analysis to GitHub
        result = await triage_bot.post_analysis_to_issue(
            issue_url=request.issue_url,
            analysis_result=request.analysis_result,
            custom_message=request.custom_message
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "comment_url": result["comment_url"],
                "comment_id": result["comment_id"]
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to post to GitHub: {result['error']}"
            )
            
    except Exception as e:
        logger.error(f"Error posting to GitHub: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to post to GitHub: {str(e)}")
