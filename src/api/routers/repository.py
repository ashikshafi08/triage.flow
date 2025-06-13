from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from ..dependencies import session_manager, get_session, get_agentic_rag, logger, settings
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

router = APIRouter(prefix="/api", tags=["repository"])

@router.get("/files")
async def list_files(session_id: str = Query(...), session: Dict[str, Any] = Depends(get_session)):
    if "repo_path" not in session:
        # Check if session is still initializing
        status = session.get("metadata", {}).get("status", "unknown")
        if status in ["initializing", "cloning", "core_ready"]:
            raise HTTPException(status_code=202, detail=f"Repository still initializing (status: {status})")
        else:
            raise HTTPException(status_code=404, detail="No repo loaded for this session")
    repo_path = session["repo_path"]
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if not f.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                file_list.append({"path": rel_path})
    return file_list

@router.get("/file-content")
async def get_file_content(
    session_id: str = Query(...), 
    file_path: str = Query(...),
    session: Dict[str, Any] = Depends(get_session),
    agentic_rag = Depends(get_agentic_rag)
):
    """Get file content with dynamic content handling"""
    try:
        logger.info(f"Getting file content for session {session_id}, file: {file_path}")
        
        if not hasattr(agentic_rag, 'agentic_explorer') or not agentic_rag.agentic_explorer:
            logger.error(f"agentic_explorer not available for session {session_id}")
            raise HTTPException(status_code=400, detail="Agentic explorer not available")
        
        # Use the agentic_explorer to read the file
        logger.debug(f"Reading file {file_path} using agentic_explorer")
        content_response = agentic_rag.agentic_explorer.read_file(file_path)
        logger.debug(f"Got response from agentic_explorer: {type(content_response)}")
        
        # Handle different response types from agentic_explorer
        try:
            # Try to parse as JSON (normal case)
            content_data = json.loads(content_response)
            logger.debug(f"Successfully parsed JSON response")
            
            # Check if it's an error message (plain string)
            if isinstance(content_data, dict) and "content" in content_data:
                # Normal successful response
                return {
                    "content": content_data["content"],
                    "size": content_data.get("size", 0),
                    "type": "text",
                    "encoding": "utf-8"
                }
            elif isinstance(content_data, dict) and "chunks" in content_data:
                # Large file with chunks - combine them
                if isinstance(content_data["content"], list):
                    combined_content = ''.join(content_data["content"])
                else:
                    combined_content = content_data["content"]
                
                return {
                    "content": combined_content,
                    "size": content_data.get("size", 0),
                    "type": "text",
                    "encoding": "utf-8"
                }
            else:
                # Unexpected JSON structure
                logger.warning(f"Unexpected JSON structure: {content_data}")
                return {"content": str(content_data), "size": len(str(content_data)), "type": "text"}
                
        except json.JSONDecodeError:
            logger.debug(f"Content is not JSON, treating as plain text")
            # Plain string response (likely an error message or simple content)
            if "appears to be binary" in content_response or "cannot be read as text" in content_response:
                return {
                    "content": "",
                    "size": 0,
                    "type": "binary",
                    "error": content_response
                }
            elif "does not exist" in content_response or "Error" in content_response:
                logger.error(f"File error: {content_response}")
                raise HTTPException(status_code=404, detail=content_response)
            else:
                # Plain text content
                return {
                    "content": content_response,
                    "size": len(content_response.encode('utf-8')),
                    "type": "text",
                    "encoding": "utf-8"
                }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file content for session {session_id}, file {file_path}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/file-content/stream")
async def stream_file_content(
    session_id: str = Query(...), 
    file_path: str = Query(...),
    agentic_rag = Depends(get_agentic_rag)
):
    """Stream large file content in chunks"""
    try:
        # Use agentic_explorer for streaming
        return StreamingResponse(
            agentic_rag.agentic_explorer.stream_large_file(file_path),
            media_type="application/x-ndjson"
        )
            
    except Exception as e:
        logger.error(f"Error streaming file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tree")
async def get_tree_structure(session_id: str = Query(...), session: Dict[str, Any] = Depends(get_session)):
    """Get the tree structure of the repository"""
    if "repo_path" not in session:
        # Check if session is still initializing
        status = session.get("metadata", {}).get("status", "unknown")
        if status in ["initializing", "cloning", "core_ready"]:
            raise HTTPException(status_code=202, detail=f"Repository still initializing (status: {status})")
        else:
            raise HTTPException(status_code=404, detail="No repo loaded for this session")
    
    repo_path = session["repo_path"]
    
    def build_tree_recursive(current_path: str, relative_path: str = "") -> dict:
        """Recursively build tree structure"""
        items = []
        
        try:
            # Get all items in current directory
            for item in sorted(os.listdir(current_path)):
                # Skip hidden files and directories
                if item.startswith('.'):
                    continue
                
                item_path = os.path.join(current_path, item)
                item_relative_path = os.path.join(relative_path, item) if relative_path else item
                
                if os.path.isdir(item_path):
                    # Directory
                    dir_node = {
                        "name": item,
                        "path": item_relative_path.replace("\\", "/"),  # Normalize path separators
                        "type": "directory",
                        "children": build_tree_recursive(item_path, item_relative_path)
                    }
                    items.append(dir_node)
                else:
                    # File
                    file_node = {
                        "name": item,
                        "path": item_relative_path.replace("\\", "/"),  # Normalize path separators
                        "type": "file"
                    }
                    items.append(file_node)
                    
        except PermissionError:
            # Skip directories we can't read
            pass
        except Exception as e:
            print(f"Error reading directory {current_path}: {e}")
            pass
        
        return items
    
    try:
        tree = build_tree_recursive(repo_path)
        return tree
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building tree structure: {str(e)}")

@router.get("/file-snippet")
async def get_file_snippet(
    session_id: str = Query(..., description="Session ID to identify the repository"),
    file_path: str = Query(..., description="Path to the file relative to repository root"),
    lines: int = Query(10, description="Number of lines to return (default: 10)"),
    start_line: Optional[int] = Query(None, description="Starting line number (1-indexed)"),
    pr_number: Optional[int] = Query(None, description="PR number to show diff for this file"),
    show_diff: bool = Query(False, description="Show diff instead of file content"),
    session: Dict[str, Any] = Depends(get_session)
):
    """Get a snippet of a file for inline preview, with optional RAG-powered diff support"""
    try:
        repo_path = session.get("repo_path")
        if not repo_path:
            raise HTTPException(status_code=400, detail="No repository loaded in this session")
        
        # If diff is requested and we have a PR number, try RAG-based diff first
        if show_diff and pr_number:
            try:
                # Try to use the RAG-based diff system
                agentic_rag = session.get("agentic_rag")
                if agentic_rag and hasattr(agentic_rag, 'agentic_explorer') and agentic_rag.agentic_explorer:
                    # Use the get_pr_diff method from agentic_tools
                    diff_response = agentic_rag.agentic_explorer.get_pr_diff(pr_number)
                    
                    try:
                        diff_data = json.loads(diff_response)
                        if "error" not in diff_data and "full_diff" in diff_data:
                            # Extract diff content for this specific file
                            full_diff = diff_data["full_diff"]
                            file_diff = _extract_file_diff_from_full_diff(full_diff, file_path)
                            
                            if file_diff:
                                return {
                                    "snippet": file_diff,
                                    "file_path": file_path,
                                    "pr_number": pr_number,
                                    "type": "diff",
                                    "truncated": False,
                                    "source": "rag_cache",
                                    "files_changed": diff_data.get("files_changed", []),
                                    "diff_summary": diff_data.get("diff_summary", "")
                                }
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                # Fallback to git-based diff
                return await _get_file_diff_for_pr(repo_path, file_path, pr_number)
            except Exception as e:
                logger.warning(f"Failed to get diff for PR {pr_number}: {e}")
                # Fall back to regular file content
        
        # Regular file content logic
        repo_root = Path(repo_path)
        target_file = repo_root / file_path
        
        # Security check: ensure the file is within the repo directory
        try:
            target_file = target_file.resolve()
            repo_root = repo_root.resolve()
            if not str(target_file).startswith(str(repo_root)):
                raise HTTPException(status_code=403, detail="Access denied: File outside repository")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")
        
        # Check if file exists
        if not target_file.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's actually a file
        if not target_file.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Read the file content
        try:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        total_lines = len(all_lines)
        
        # Determine which lines to return
        if start_line is not None:
            # Start from specific line
            start_idx = max(0, start_line - 1)  # Convert to 0-indexed
            end_idx = min(total_lines, start_idx + lines)
        else:
            # Return first N lines by default
            start_idx = 0
            end_idx = min(total_lines, lines)
        
        # Extract the snippet
        snippet_lines = all_lines[start_idx:end_idx]
        snippet = ''.join(snippet_lines)
        
        # Remove trailing newline if present
        snippet = snippet.rstrip('\n')
        
        return {
            "snippet": snippet,
            "file_path": file_path,
            "start_line": start_idx + 1,  # Convert back to 1-indexed
            "end_line": end_idx,
            "total_lines": total_lines,
            "truncated": end_idx < total_lines,
            "type": "file_content"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file snippet: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/sync-repository")
async def sync_repository_data(
    session_id: str,
    background_tasks: BackgroundTasks,
    session: Dict[str, Any] = Depends(get_session),
    agentic_rag = Depends(get_agentic_rag)
):
    """Triggers a re-sync of the repository's issue and patch data."""
    
    if not agentic_rag.issue_rag:
        logger.warning(f"Attempted to sync repo for session {session_id} but issue_rag is not available.")
        raise HTTPException(status_code=400, detail="Issue RAG system not available for this session. Sync cannot proceed.")

    # Update session status to indicate syncing
    if "metadata" not in session:
        session["metadata"] = {}
    session["metadata"]["status"] = "syncing_issues"
    session["metadata"]["message"] = "Re-syncing repository issues, PRs, and patches..."

    async def _sync_task():
        try:
            logger.info(f"Starting repository data sync for session {session_id}...")
            await agentic_rag.issue_rag.initialize(
                force_rebuild=True, 
                max_issues_for_patch_linkage=settings.MAX_PATCH_LINKAGE_ISSUES,
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

# Helper functions
def _extract_file_diff_from_full_diff(full_diff: str, target_file_path: str) -> Optional[str]:
    """Extract diff content for a specific file from a full PR diff"""
    import re
    from pathlib import Path
    
    # Normalize the target file path (convert to forward slashes, make relative)
    target_path = Path(target_file_path).as_posix()
    
    # Split diff into individual file sections
    # Each file section starts with "diff --git" or "--- a/"
    sections = re.split(r'^(?=diff --git|--- a/)', full_diff, flags=re.MULTILINE)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Look for the target file in this section
        # Check both old and new file paths in case file was renamed
        file_patterns = [
            rf'^diff --git a/([^\s]+) b/([^\s]+)',
            rf'^--- a/([^\s\n]+)',
            rf'^\+\+\+ b/([^\s\n]+)'
        ]
        
        section_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, section, flags=re.MULTILINE)
            if matches:
                if isinstance(matches[0], tuple):
                    section_files.update(matches[0])
                else:
                    section_files.update(matches)
        
        # Normalize section file paths and check for match
        normalized_section_files = {Path(f).as_posix() for f in section_files}
        
        if target_path in normalized_section_files:
            # Found the target file, clean up the diff section
            lines = section.strip().split('\n')
            
            # Remove any leading empty lines
            while lines and not lines[0].strip():
                lines.pop(0)
            
            # Format the diff for better display
            cleaned_diff = _format_diff_for_display('\n'.join(lines))
            
            # If it's a proper diff with additions/deletions, return it
            if any(line.startswith(('+', '-')) for line in lines):
                return cleaned_diff
    
    return None

def _format_diff_for_display(diff_content: str) -> str:
    """Format diff content for better display in the frontend"""
    lines = diff_content.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.startswith('+++') or line.startswith('---'):
            # File headers - make them more readable
            if line.startswith('+++'):
                line = line.replace('+++ b/', '+++ ')
            elif line.startswith('---'):
                line = line.replace('--- a/', '--- ')
        elif line.startswith('@@'):
            # Hunk headers - keep as is but ensure proper formatting
            pass
        elif line.startswith('+') and not line.startswith('+++'):
            # Addition lines - ensure they stand out
            pass
        elif line.startswith('-') and not line.startswith('---'):
            # Deletion lines - ensure they stand out  
            pass
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

async def _get_file_diff_for_pr(repo_path: str, file_path: str, pr_number: int):
    """Get the diff for a specific file in a PR"""
    import subprocess
    from pathlib import Path
    
    try:
        repo_root = Path(repo_path)
        
        # Get PR information from git
        result = subprocess.run([
            "git", "log", "--oneline", "--grep", f"#{pr_number}", "-1"
        ], cwd=repo_root, capture_output=True, text=True, check=True)
        
        if not result.stdout.strip():
            # Try alternative approach - look for merge commits
            result = subprocess.run([
                "git", "log", "--oneline", "--merges", "--grep", f"#{pr_number}", "-1"
            ], cwd=repo_root, capture_output=True, text=True, check=True)
        
        if not result.stdout.strip():
            raise Exception(f"No commit found for PR #{pr_number}")
        
        commit_hash = result.stdout.strip().split()[0]
        
        # Get the diff for this specific file
        diff_result = subprocess.run([
            "git", "show", f"{commit_hash}", "--", file_path
        ], cwd=repo_root, capture_output=True, text=True, check=True)
        
        diff_content = diff_result.stdout
        
        if not diff_content.strip():
            # Try getting diff from parent commit
            diff_result = subprocess.run([
                "git", "diff", f"{commit_hash}^", commit_hash, "--", file_path
            ], cwd=repo_root, capture_output=True, text=True, check=True)
            diff_content = diff_result.stdout
        
        return {
            "snippet": diff_content,
            "file_path": file_path,
            "pr_number": pr_number,
            "commit_hash": commit_hash,
            "type": "diff",
            "truncated": False
        }
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"Git command failed: {e}")
    except Exception as e:
        raise Exception(f"Failed to get diff: {e}")

@router.get("/diff/{sha}/{file_path:path}")
async def get_commit_file_diff_api(
    sha: str,
    file_path: str,
    session_id: str = Query(..., description="Session ID to identify the repository"),
    view_type: str = Query("content", description="Type of view: 'content' for file at commit, 'diff' for changes"),
    session: Dict[str, Any] = Depends(get_session),
    agentic_rag = Depends(get_agentic_rag)
):
    """
    🚀 OPTIMIZED: Get file content at a specific commit or the diff for that commit.
    Enhanced with caching, compression, and performance monitoring.
    """
    import time
    import subprocess
    start_time = time.time()
    
    try:
        if not agentic_rag or not hasattr(agentic_rag, 'get_repo_path'):
            raise HTTPException(status_code=400, detail="Session not properly initialized")
        
        repo_path = agentic_rag.get_repo_path()
        if not repo_path or not os.path.exists(repo_path):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        file_content = None
        content_type = "file_content" if view_type == "content" else "diff"
        file_changed = False
        change_type = "unknown"
        is_truncated = False
        original_length = 0
        
        # Performance: Optimized git operations with timeout
        try:
            if view_type == "content":
                # Get file content at specific commit - optimized command
                result = subprocess.run(
                    ["git", "show", f"{sha}:{file_path}"],
                    capture_output=True,
                    text=True,
                    timeout=3,  # Reduced timeout
                    check=False,
                    cwd=repo_path
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
                    ["git", "show", "--format=", sha, "--", file_path],
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
            
            # Performance: Content size optimization
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
                check=False,
                cwd=repo_path
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
        
        # Performance: Add performance metrics
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


