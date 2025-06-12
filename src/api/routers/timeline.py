from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from ..dependencies import session_manager, get_session, logger, settings
import time
import subprocess
import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter(prefix="/api/timeline", tags=["timeline"])

@router.get("/file")
async def get_file_timeline_api(
    session_id: str = Query(..., description="Session ID to identify the repository"),
    file_path: str = Query(..., description="Path to the file relative to repository root"),
    limit: int = Query(50, description="Maximum number of commits to return"),
    session: Dict[str, Any] = Depends(get_session)
):
    """
    🚀 OPTIMIZED: Get timeline of commits for a specific file.
    Enhanced for sub-300ms response times with aggressive optimization.
    """
    start_time = time.time()
    
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized")
        
        # 🚀 PERFORMANCE: Simple cache key and fast lookup
        cache_key = f"timeline_{session_id}_{file_path}_{limit}"
        
        # Check if we have a cached response (simple in-memory cache)
        if not hasattr(get_file_timeline_api, '_timeline_cache'):
            get_file_timeline_api._timeline_cache = {}
        
        # Quick cache check with timestamp
        cached_entry = get_file_timeline_api._timeline_cache.get(cache_key)
        if cached_entry and time.time() - cached_entry['timestamp'] < 300:  # 5 min cache
            logger.info(f"⚡ Timeline cache hit: 0ms for {file_path}")
            return JSONResponse(
                content=cached_entry['data'],
                headers={"X-Response-Time": "0ms", "X-Cache": "hit"}
            )
        
        logger.info(f"Getting timeline for file: {file_path} (original: {file_path})")
        
        # Get timeline from commit index (should be fast)
        commit_index_manager = agentic_rag.agentic_explorer.commit_index_manager
        
        if not commit_index_manager:
            raise HTTPException(status_code=500, detail="Commit index not available")
        
        # 🚀 PERFORMANCE: Optimized timeline retrieval with reduced data
        try:
            timeline_commits = commit_index_manager.get_file_timeline(file_path, limit=limit)
            
            # Log commit index stats for debugging
            stats = commit_index_manager.get_statistics()
            total_commits = stats.get('total_commits', 0)
            total_files = stats.get('total_files_touched', 0)
            logger.info(f"Commit index stats: {total_commits} commits, {total_files} files")
            
            # 🚀 PERFORMANCE: Lightweight timeline format (minimal data for speed)
            lightweight_timeline = []
            for commit_data in timeline_commits[:limit]:  # Enforce limit for performance
                # Extract data from dictionary format (not CommitMeta object)
                timeline_entry = {
                    "sha": commit_data.get("sha", ""),
                    "ts": commit_data.get("date", ""),
                    "loc_added": min(commit_data.get("insertions", 0), 999),  # Cap large numbers for UI
                    "loc_removed": min(commit_data.get("deletions", 0), 999),
                    "pr_number": commit_data.get("pr_number"),
                    "author": (commit_data.get("author", "Unknown")[:30]) if commit_data.get("author") else "Unknown",  # Truncate long names
                    "message": (commit_data.get("subject", "No message")[:100] + ("..." if len(commit_data.get("subject", "")) > 100 else "")) if commit_data.get("subject") else "No message",  # Truncate long messages
                    "change_type": commit_data.get("change_type", "modified"),  # Use actual change type
                    "churn": min((commit_data.get("insertions", 0) + commit_data.get("deletions", 0)), 999)  # Cap churn metric
                }
                lightweight_timeline.append(timeline_entry)
            
            # 🚀 PERFORMANCE: Streamlined response format
            response_data = {
                "file_path": file_path,
                "timeline": lightweight_timeline,
                "total_commits": len(timeline_commits)
            }
            
            # Cache the response for 5 minutes
            get_file_timeline_api._timeline_cache[cache_key] = {
                'data': response_data,
                'timestamp': time.time()
            }
            
            # Clean cache if it gets too large (simple LRU)
            if len(get_file_timeline_api._timeline_cache) > 100:
                oldest_key = min(get_file_timeline_api._timeline_cache.keys(), 
                               key=lambda k: get_file_timeline_api._timeline_cache[k]['timestamp'])
                del get_file_timeline_api._timeline_cache[oldest_key]
            
            end_time = time.time()
            response_time_ms = round((end_time - start_time) * 1000)
            
            logger.info(f"⚡ Timeline API: {response_time_ms}ms for {len(lightweight_timeline)} commits")
            
            return JSONResponse(
                content=response_data,
                headers={
                    "X-Response-Time": f"{response_time_ms}ms",
                    "Cache-Control": "public, max-age=300",  # 5 min browser cache
                    "X-Cache": "miss",
                    "X-Timeline-Count": str(len(lightweight_timeline))
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting timeline for {file_path}: {e}")
            
            # 🚀 PERFORMANCE: Fast fallback to recent repository commits
            try:
                logger.info(f"Using fallback: recent repository commits for {file_path}")
                
                # Get recent commits from the entire repository as fallback (faster than file-specific)
                recent_commits = []
                if hasattr(commit_index_manager, 'indexer') and hasattr(commit_index_manager.indexer, 'commit_metas'):
                    # Take the most recent commits (already sorted by date)
                    all_commits = list(commit_index_manager.indexer.commit_metas.values())
                    all_commits.sort(key=lambda x: x.commit_date, reverse=True)
                    
                    for commit in all_commits[:limit]:
                        fallback_entry = {
                            "sha": commit.sha,
                            "ts": commit.commit_date,
                            "loc_added": min(commit.insertions, 999),
                            "loc_removed": min(commit.deletions, 999),
                            "pr_number": commit.pr_number,
                            "author": commit.author_name[:30] if commit.author_name else "Unknown",
                            "message": commit.subject[:100] + ("..." if len(commit.subject) > 100 else "") if commit.subject else "No message",
                            "change_type": "modified",
                            "churn": min(commit.insertions + commit.deletions, 999)
                        }
                        recent_commits.append(fallback_entry)
                
                fallback_response = {
                    "file_path": file_path,
                    "timeline": recent_commits,
                    "total_commits": len(recent_commits)
                }
                
                end_time = time.time()
                response_time_ms = round((end_time - start_time) * 1000)
                
                logger.info(f"⚡ Fallback timeline: {response_time_ms}ms")
                
                return JSONResponse(
                    content=fallback_response,
                    headers={
                        "X-Response-Time": f"{response_time_ms}ms",
                        "X-Cache": "fallback",
                        "X-Timeline-Count": str(len(recent_commits))
                    }
                )
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for {file_path}: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Timeline unavailable: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        end_time = time.time()
        response_time_ms = round((end_time - start_time) * 1000)
        logger.error(f"Error in optimized timeline API ({response_time_ms}ms): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file timeline: {str(e)}")

@router.get("/hunk")
async def get_hunk_timeline_api(
    session_id: str = Query(..., description="Session ID to identify the repository"),
    file_path: str = Query(..., description="Path to the file relative to repository root"),
    line_start: int = Query(..., description="Starting line number"),
    line_end: int = Query(..., description="Ending line number"),
    limit: int = Query(20, description="Maximum number of commits to return"),
    session: Dict[str, Any] = Depends(get_session)
):
    """
    Get timeline of commits that touched a specific line range in a file.
    More granular than file-level timeline.
    """
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized")
        
        # Get full file timeline first
        full_timeline = agentic_rag.agentic_explorer.commit_index_manager.get_file_timeline(file_path, limit=limit * 2)
        
        # Filter by commits that likely touched the line range
        # For now, we'll return all commits for the file and let frontend handle filtering
        # In a future iteration, we could use git blame for more precise filtering
        
        formatted_timeline = []
        for item in full_timeline[:limit]:
            formatted_timeline.append({
                "sha": item["sha"],
                "ts": item["date"],
                "loc_added": item.get("insertions", 0),
                "loc_removed": item.get("deletions", 0),
                "pr_number": item.get("pr_number"),
                "author": item["author"],
                "message": item["subject"],
                "change_type": item.get("change_type", "modified"),
                "line_range": f"{line_start}-{line_end}"
            })
        
        return {
            "file_path": file_path,
            "line_range": {"start": line_start, "end": line_end},
            "timeline": formatted_timeline,
            "total_commits": len(formatted_timeline)
        }
        
    except Exception as e:
        logger.error(f"Error in get_hunk_timeline_api: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get hunk timeline: {str(e)}")

@router.get("/preview/{sha}/{file_path:path}")
async def get_timeline_preview_api(
    sha: str,
    file_path: str,
    session_id: str = Query(..., description="Session ID to identify the repository"),
    session: Dict[str, Any] = Depends(get_session)
):
    """
    Lightweight preview for timeline scrubbing - just commit metadata without full diff.
    Much faster for real-time timeline navigation.
    """
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized")
        
        # Get commit details from index (very fast - no git operations)
        commit_details = agentic_rag.agentic_explorer.commit_index_manager.get_commit_by_sha(sha)
        if not commit_details:
            raise HTTPException(status_code=404, detail=f"Commit {sha} not found")
        
        # Quick check if file was changed in this commit
        file_changed = file_path in commit_details.files_changed
        change_type = "unknown"
        
        if file_changed:
            if file_path in commit_details.files_added:
                change_type = "added"
            elif file_path in commit_details.files_modified:
                change_type = "modified"
            elif file_path in commit_details.files_deleted:
                change_type = "deleted"
        
        return {
            "sha": sha,
            "file_path": file_path,
            "file_changed": file_changed,
            "change_type": change_type,
            "commit": {
                "sha": commit_details.sha,
                "author": commit_details.author_name,
                "date": commit_details.commit_date,
                "message": commit_details.subject,
                "insertions": commit_details.insertions,
                "deletions": commit_details.deletions,
                "pr_number": commit_details.pr_number,
                "is_merge": commit_details.is_merge
            },
            "has_full_diff": False,  # Indicates this is preview data
            "preview": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_timeline_preview_api: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline preview: {str(e)}")

class TimelineIssueRequest(BaseModel):
    session_id: str
    commit_sha: str
    file_path: str
    title: Optional[str] = None
    description: Optional[str] = None
    line_range: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    next_steps: Optional[str] = None
    labels: Optional[List[str]] = None

@router.post("/create-issue")
async def create_issue_from_timeline(request: TimelineIssueRequest):
    """
    Create a GitHub issue from timeline investigation results.
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Extract repo info from session
        metadata = session.get("metadata", {})
        repo_url = metadata.get("repo_url", "")
        
        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL not found in session")
        
        # Parse repo URL to get owner/repo
        def parse_repo_url(repo_url: str) -> tuple[str, str]:
            """Extract owner and repository name from a GitHub URL."""
            # Remove .git if present
            repo_url = repo_url.replace(".git", "")
            # Split by / and get last two parts
            parts = repo_url.split("/")
            owner = parts[-2]
            repo = parts[-1]
            return owner, repo
        
        owner, repo_name = parse_repo_url(repo_url)
        
        # Create issue using GitHub client
        from ..dependencies import github_client
        
        # Format issue body with timeline context
        issue_body = f"""## Timeline Investigation

**File:** `{request.file_path}`
**Commit:** `{request.commit_sha}`
**Line Range:** {request.line_range or 'N/A'}

{request.description or ''}

### Investigation Context
- Generated from timeline analysis
- Commit: {request.commit_sha}
- Author: {request.author or 'Unknown'}
- Date: {request.date or 'Unknown'}

### Next Steps
{request.next_steps or 'Please investigate the changes in this commit and their impact.'}

---
*This issue was created automatically from timeline analysis in triage.flow*
"""
        
        new_issue = await github_client.create_issue(
            owner=owner,
            repo=repo_name,
            title=request.title or f'Investigation: {request.file_path} changes in {request.commit_sha[:8]}',
            body=issue_body,
            labels=request.labels or ['investigation', 'timeline-generated']
        )
        
        return {
            "success": True,
            "issue": new_issue,
            "url": new_issue.get("html_url")
        }
        
    except Exception as e:
        logger.error(f"Error creating issue from timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create issue: {str(e)}")

# Diff API endpoints
@router.get("/{sha}/{file_path:path}")
async def get_commit_file_diff_api(
    sha: str,
    file_path: str,
    session_id: str = Query(..., description="Session ID to identify the repository"),
    view_type: str = Query("content", description="Type of view: 'content' for file at commit, 'diff' for changes"),
    session: Dict[str, Any] = Depends(get_session)
):
    """
    🚀 OPTIMIZED: Get file content at a specific commit or the diff for that commit.
    Enhanced with caching, compression, and performance monitoring.
    """
    start_time = time.time()
    
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized")
        
        # 🚀 PERFORMANCE: Session-scoped rate limiting (simpler than global cache)
        current_time = time.time()
        cache_key = f"diff_{session_id}_{sha}_{file_path}_{view_type}"
        
        # Check if we're being hit too frequently (basic rate limiting)
        rate_limit_key = f"rate_{session_id}_{int(current_time)}"  # Per-second rate limit
        if not hasattr(get_commit_file_diff_api, '_rate_limits'):
            get_commit_file_diff_api._rate_limits = {}
        
        if rate_limit_key in get_commit_file_diff_api._rate_limits:
            get_commit_file_diff_api._rate_limits[rate_limit_key] += 1
            if get_commit_file_diff_api._rate_limits[rate_limit_key] > 10:  # Max 10 requests per second
                logger.warning(f"Rate limit exceeded for session {session_id}")
                time.sleep(0.1)  # Small delay to prevent spam
        else:
            get_commit_file_diff_api._rate_limits[rate_limit_key] = 1
        
        # Clean old rate limit entries (keep only last 5 seconds)
        current_time_int = int(current_time)
        get_commit_file_diff_api._rate_limits = {
            k: v for k, v in get_commit_file_diff_api._rate_limits.items() 
            if int(k.split('_')[-1]) > current_time_int - 5
        }
        
        repo_path = agentic_rag.repo_path
        if not os.path.exists(repo_path):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        file_content = None
        content_type = "file_content" if view_type == "content" else "diff"
        file_changed = False
        change_type = "unknown"
        is_truncated = False
        original_length = 0
        
        # 🚀 PERFORMANCE: Optimized git operations with timeout
        try:
            os.chdir(repo_path)
            
            if view_type == "content":
                # Get file content at specific commit - optimized command
                result = subprocess.run(
                    ["git", "show", f"{sha}:{file_path}"],
                    capture_output=True,
                    text=True,
                    timeout=3,  # Reduced timeout
                    check=False
                )
                
                if result.returncode == 0:
                    file_content = result.stdout
                    file_changed = True
                    change_type = "modified"
                else:
                    # File might not exist at this commit
                    file_content = f"File '{file_path}' not found at commit {sha[:8]}"
                    change_type = "deleted"
                    
            else:  # diff view
                # Get diff for this specific file - optimized
                result = subprocess.run(
                    ["git", "show", "--format=", "--", file_path],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                    cwd=repo_path
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    file_content = result.stdout
                    file_changed = True
                    change_type = "modified"
                else:
                    file_content = f"No changes found for '{file_path}' in commit {sha[:8]}"
                    change_type = "unknown"
            
            # 🚀 PERFORMANCE: Content size optimization
            original_length = len(file_content) if file_content else 0
            
            # Truncate very large files for faster transmission
            if original_length > 100000:  # 100KB limit
                file_content = file_content[:100000] + "\n\n... [Content truncated for performance. Download to see full file.]"
                is_truncated = True
            
            # Get commit info efficiently
            commit_info_result = subprocess.run(
                ["git", "show", "--format=%an|%ad|%s|%b", "-s", sha],
                capture_output=True,
                text=True,
                timeout=2,  # Quick timeout for metadata
                check=False
            )
            
            commit_info = {
                "sha": sha,
                "author": "Unknown",
                "date": "Unknown",
                "message": "No message",
                "body": "",
                "insertions": 0,
                "deletions": 0,
                "is_merge": False
            }
            
            if commit_info_result.returncode == 0 and commit_info_result.stdout.strip():
                parts = commit_info_result.stdout.strip().split("|", 3)
                if len(parts) >= 3:
                    commit_info.update({
                        "author": parts[0],
                        "date": parts[1],
                        "message": parts[2],
                        "body": parts[3] if len(parts) > 3 else ""
                    })
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Git command timeout for {sha}:{file_path}")
            file_content = f"⚠️ Content loading timed out for performance. File: {file_path}"
            change_type = "timeout"
        except Exception as e:
            logger.error(f"Git error for {sha}:{file_path}: {e}")
            file_content = f"❌ Error loading content: {str(e)}"
            change_type = "error"
        
        response_data = {
            "sha": sha,
            "file_path": file_path,
            "content": file_content or "",
            "content_type": content_type,
            "view_type": view_type,
            "file_changed": file_changed,
            "change_type": change_type,
            "is_truncated": is_truncated,
            "original_length": original_length,
            "commit": commit_info
        }
        
        # 🚀 PERFORMANCE: Add performance metrics
        end_time = time.time()
        response_time_ms = round((end_time - start_time) * 1000)
        
        logger.info(f"Diff API response: {response_time_ms}ms for {sha[:8]}:{file_path}")
        
        # Add performance header
        headers = {
            "X-Response-Time": f"{response_time_ms}ms",
            "Cache-Control": "public, max-age=300",  # 5 minute cache
            "X-Content-Length": str(len(str(response_data))),
        }
        
        return JSONResponse(content=response_data, headers=headers)
        
    except HTTPException:
        raise
    except Exception as e:
        end_time = time.time()
        response_time_ms = round((end_time - start_time) * 1000)
        logger.error(f"Error in optimized diff API ({response_time_ms}ms): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file content: {str(e)}")
