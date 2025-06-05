from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
from .models import PromptRequest, PromptResponse, ChatMessage, SessionResponse, RepoRequest, RepoSessionResponse, SessionListResponse, Issue
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .prompt_generator import PromptGenerator
from .session_manager import SessionManager
from .conversation_memory import ConversationContextManager
from .cache_manager import response_cache, cleanup_caches_periodically
from .config import settings
import nest_asyncio
import asyncio
import os
from typing import Optional, Dict, Any
import re
import json
from datetime import datetime
import time
import logging
import git
from git import Repo
import tempfile
import shutil
from pathlib import Path
from .new_rag import LocalRepoContextExtractor
from .agentic_tools import AgenticCodebaseExplorer

# Enable nested event loops for Jupyter notebooks
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.2.0"
)

# Add CORS middleware
# Configure CORS origins - make configurable for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients and services
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()
session_manager = SessionManager()
conversation_memory = ConversationContextManager(max_context_tokens=8000)

# Global storage for agentic explorers (in production, use proper session management)
agentic_explorers: Dict[str, AgenticCodebaseExplorer] = {}

# Background task to clean up old sessions
async def cleanup_sessions_periodically():
    while True:
        session_manager.cleanup_sessions()
        await asyncio.sleep(600)  # Clean up every 10 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_sessions_periodically())
    asyncio.create_task(cleanup_caches_periodically())  # Add cache cleanup

@app.get("/")
async def root():
    return {"message": "GH Issue Prompt API"}

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: PromptRequest):
    try:
        # Create new session
        session_id = session_manager.create_session(
            request.issue_url, 
            request.prompt_type,
            request.llm_config # Pass llm_config to session manager
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

@app.post("/sessions/{session_id}/messages", response_model=ChatMessage)
async def handle_chat_message(
    session_id: str, 
    message: ChatMessage, 
    stream: bool = Query(False),
    agentic: bool = Query(False, description="Use agentic system for deeper analysis")
):
    """Enhanced message handler with automatic agentic capabilities when beneficial"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Auto-enable agentic capabilities for the session if not already enabled
        if not session.get("agentic_enabled", False):
            repo_path = session.get("repo_path")
            if repo_path:
                try:
                    agentic_explorer = AgenticCodebaseExplorer(session_id, repo_path)
                    agentic_explorers[session_id] = agentic_explorer
                    session["agentic_enabled"] = True
                    logger.info(f"Auto-enabled agentic capabilities for session {session_id}")
                except Exception as e:
                    logger.warning(f"Could not enable agentic capabilities: {e}")
        
        # Determine if this query would benefit from agentic capabilities
        should_use_agentic = agentic or _should_use_agentic_approach(message.content)
        
        # If agentic mode is beneficial and available, use it
        if should_use_agentic and session.get("agentic_enabled", False) and session_id in agentic_explorers:
            logger.info(f"Using agentic system for query: {message.content[:100]}...")
            return await agentic_query(session_id, message, stream)
        
        # Otherwise, use the enhanced RAG system
        session_manager.add_message(session_id, "user", message.content)
        history = session.get("conversation_history", [])
        user_query = message.content
        rag_instance = session.get("rag_instance")
        fresh_rag_context = {}
        
        # Extract @file_path mentions from message.content
        user_file_contexts = []
        repo_path = session.get("repo_path")
        
        # Extract both file and folder mentions
        file_mentions = re.findall(r"@([\w\-/\\.]+)", message.content)
        folder_mentions = re.findall(r"@folder/([\w\-/\\.]+)", message.content)
        
        # Process individual file mentions
        for rel_path in file_mentions:
            abs_path = os.path.join(repo_path, rel_path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                user_file_contexts.append({
                    "file": rel_path,
                    "content": content[:5000]  # Limit to 5k chars per file
                })
        
        # Process folder mentions - collect all files in the folder
        restricted_files = []
        for folder_path in folder_mentions:
            abs_folder_path = os.path.join(repo_path, folder_path)
            if os.path.exists(abs_folder_path) and os.path.isdir(abs_folder_path):
                # Get all files in the folder (recursively)
                for root, dirs, files in os.walk(abs_folder_path):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for file in files:
                        if not file.startswith('.'):
                            file_abs_path = os.path.join(root, file)
                            file_rel_path = os.path.relpath(file_abs_path, repo_path)
                            restricted_files.append(file_rel_path)
        
        fresh_rag_context["user_selected_files"] = user_file_contexts
        
        # If we have folder mentions, restrict RAG to those files
        if folder_mentions:
            if rag_instance:
                try:
                    fresh_rag_context = await rag_instance.get_relevant_context(
                        user_query, 
                        restrict_files=restricted_files
                    )
                    fresh_rag_context["user_selected_files"] = user_file_contexts
                except Exception as e:
                    print(f"Error getting folder-restricted RAG context: {e}")
                    fresh_rag_context = session.get("repo_context", {})
        else:
            # Regular RAG without folder restrictions
            if rag_instance:
                try:
                    fresh_rag_context = await rag_instance.get_relevant_context(user_query)
                    fresh_rag_context["user_selected_files"] = user_file_contexts
                except Exception as e:
                    print(f"Error getting fresh RAG context: {e}")
                    fresh_rag_context = session.get("repo_context", {})
        # Use production-grade conversation memory instead of simple sliding window
        conversation_context, memory_stats = await conversation_memory.get_conversation_context(
            history, llm_client, use_compression=True
        )
        
        # Log memory statistics for debugging
        print(f"Memory stats for session {session_id}: {memory_stats}")
        
        # Log RAG context retrieval for monitoring
        if fresh_rag_context and fresh_rag_context.get("sources"):
            print(f"RAG retrieved {len(fresh_rag_context['sources'])} sources for query")
            if folder_mentions:
                print(f"Folder-restricted query for folders: {folder_mentions}")
        else:
            print("No RAG sources retrieved for this query")
        
        session_llm_config = session.get("llm_config", {})
        model_name = session_llm_config.name if session_llm_config and hasattr(session_llm_config, 'name') else settings.default_model
        
        # Query classification and caching
        query_type = _classify_query_type(user_query)
        use_cache = settings.ENABLE_RESPONSE_CACHING and _should_use_cached_response(query_type, user_query)
        response_cache_key = None  # Initialize to prevent NameError
        
        # Generate cache key for response caching
        if use_cache:
            # Get repo name from RAG instance if available
            repo_name = ""
            if rag_instance and hasattr(rag_instance, 'repo_info'):
                repo_name = rag_instance.repo_info.get("repo", "")
            
            response_cache_key = response_cache._generate_cache_key(
                user_query, 
                session["prompt_type"],
                restricted_files,
                repo_name
            )
            
            # Try to get cached response
            cached_response = await response_cache.get(response_cache_key)
            if cached_response:
                print(f"Response cache hit for query type: {query_type}")
                session_manager.add_message(session_id, "assistant", cached_response)
                
                if stream:
                    # Stream the cached response
                    async def stream_cached():
                        words = cached_response.split()
                        for i in range(0, len(words), 3):  # Stream 3 words at a time
                            chunk = ' '.join(words[i:i+3]) + ' '
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                            await asyncio.sleep(0.01)  # Small delay for streaming effect
                        yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(stream_cached(), media_type="text/event-stream")
                else:
                    return ChatMessage(
                        role="assistant",
                        content=cached_response
                    )

        if stream:
            full_response_content = ""
            async def generate_stream():
                nonlocal full_response_content
                async for chunk in llm_client.stream_openrouter_response(
                    prompt=conversation_context,  # Use smart context instead of simple history
                    prompt_type=session["prompt_type"],
                    context=fresh_rag_context,
                    model=model_name
                ):
                    # The chunk is already SSE-formatted, don't modify it
                    # Extract content for session storage
                    if chunk.startswith('data: '):
                        data = chunk[6:].strip()
                        if data and data != '[DONE]':
                            try:
                                json_data = json.loads(data)
                                content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    full_response_content += content
                            except:
                                pass
                    yield chunk
                session_manager.add_message(session_id, "assistant", full_response_content)
                
                # Cache the response if appropriate
                if use_cache and full_response_content and response_cache_key:
                    await response_cache.set(response_cache_key, full_response_content, settings.CACHE_TTL_RESPONSE)
                    print(f"Cached response for query type: {query_type}")

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            llm_response = await llm_client.process_prompt(
                prompt=conversation_context,  # Use smart context instead of simple history
                prompt_type=session["prompt_type"],
                context=fresh_rag_context,
                model=model_name
            )
            session_manager.add_message(session_id, "assistant", llm_response.prompt)
            
            # Cache the response if appropriate
            if use_cache and response_cache_key:
                await response_cache.set(response_cache_key, llm_response.prompt, settings.CACHE_TTL_RESPONSE)
                print(f"Cached response for query type: {query_type}")
            
            return ChatMessage(
                role="assistant",
                content=llm_response.prompt
            )
        
    except Exception as e:
        import traceback
        error_details = f"Error in handle_chat_message: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)
        raise HTTPException(status_code=500, detail=str(e))

def _classify_query_type(query: str) -> str:
    """Classify query type for caching and optimization"""
    query_lower = query.lower()
    
    # File/folder exploration queries
    if any(pattern in query_lower for pattern in [
        "what's in", "what is in", "show me", "list files", "list the files",
        "which files", "what files", "folder contains", "directory contains"
    ]):
        return "exploration"
    
    # Code explanation queries
    if any(pattern in query_lower for pattern in [
        "explain", "what does", "how does", "what is", "describe"
    ]):
        return "explanation"
    
    # Implementation queries
    if any(pattern in query_lower for pattern in [
        "how to", "implement", "create", "build", "add", "modify"
    ]):
        return "implementation"
    
    # Debugging queries
    if any(pattern in query_lower for pattern in [
        "bug", "error", "issue", "problem", "fix", "debug", "wrong"
    ]):
        return "debugging"
    
    # Architecture queries
    if any(pattern in query_lower for pattern in [
        "architecture", "structure", "design", "pattern", "organized"
    ]):
        return "architecture"
    
    return "general"

def _should_use_cached_response(query_type: str, query: str) -> bool:
    """Determine if we should try to use cached response"""
    # Always try cache for exploration queries
    if query_type == "exploration":
        return True
    
    # Cache simple explanations
    if query_type == "explanation" and len(query.split()) < 15:
        return True
    
    # Cache architecture queries (they don't change often)
    if query_type == "architecture":
        return True
    
    # Don't cache debugging or complex implementation queries
    if query_type in ["debugging", "implementation"]:
        return False
    
    return False

def _should_use_agentic_approach(query: str) -> bool:
    """Determine if a query would benefit from agentic capabilities"""
    query_lower = query.lower()
    
    # Agentic approach is beneficial for:
    # 1. Complex exploration queries
    agentic_patterns = [
        "explore", "analyze", "find all", "search for", "what files", "which files",
        "how does", "explain how", "trace", "follow", "investigate", "deep dive",
        "architecture", "structure", "relationship", "dependency", "related to",
        "implement", "build", "create", "example", "show me how",
        "debug", "troubleshoot", "fix", "error", "issue", "problem"
    ]
    
    # 2. Multi-step reasoning queries
    multi_step_indicators = [
        "step by step", "walk through", "process", "flow", "sequence",
        "first", "then", "next", "finally", "overall"
    ]
    
    # 3. Comprehensive analysis queries (but be more selective)
    comprehensive_indicators = [
        "comprehensive", "complete", "full", "entire", "everything",
        "overview", "summary", "breakdown", "detailed"
    ]
    
    # Check if query contains agentic patterns
    if any(pattern in query_lower for pattern in agentic_patterns):
        return True
    
    # Check for multi-step reasoning
    if any(indicator in query_lower for indicator in multi_step_indicators):
        return True
    
    # Only use agentic for comprehensive analysis if it's also specific
    if any(indicator in query_lower for indicator in comprehensive_indicators):
        # If it's comprehensive AND specific (mentions files, directories, etc.), use agentic
        specific_terms = ["file", "directory", "folder", "class", "function", "module", "component"]
        if any(term in query_lower for term in specific_terms):
            return True
    
    # Long queries (>15 words) might benefit from agentic approach, but be more conservative
    if len(query.split()) > 15 and any(pattern in query_lower for pattern in agentic_patterns):
        return True
    
    return False

@app.get("/sessions/{session_id}/memory-stats")
async def get_memory_statistics(session_id: str):
    """Get conversation memory statistics for debugging and monitoring"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = session.get("conversation_history", [])
        
        # Get memory statistics
        _, memory_stats = await conversation_memory.get_conversation_context(
            history, llm_client, use_compression=False  # Don't compress for stats
        )
        
        return {
            "session_id": session_id,
            "memory_statistics": memory_stats,
            "session_info": {
                "created_at": session["created_at"].isoformat(),
                "last_accessed": session["last_accessed"].isoformat(),
                "prompt_type": session["prompt_type"],
                "model_name": session.get("llm_config", {}).get("name", "unknown")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files(session_id: str = Query(...)):
    session = session_manager.get_session(session_id)
    
    if not session or "repo_path" not in session:
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

@app.get("/api/file-content")
async def get_file_content(session_id: str = Query(...), file_path: str = Query(...)):
    """Get the content of a specific file from the repository"""
    session = session_manager.get_session(session_id)
    if not session or "repo_path" not in session:
        raise HTTPException(status_code=404, detail="No repo loaded for this session")
    
    repo_path = session["repo_path"]
    
    # normalize and validate file path to prevent directory traversal
    try:
        # Normalize the path and resolve any .. or . components
        normalized_file_path = os.path.normpath(file_path)
        
        # ensure the path doesn't start with / or contain .. 
        if normalized_file_path.startswith('/') or '..' in normalized_file_path:
            raise HTTPException(status_code=403, detail="Invalid file path")
            
        abs_file_path = os.path.normpath(os.path.join(repo_path, normalized_file_path))
        
        # ensure the resolved path is within the repo directory
        if not abs_file_path.startswith(os.path.normpath(repo_path)):
            raise HTTPException(status_code=403, detail="Access denied: File outside repository")
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not os.path.exists(abs_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.isfile(abs_file_path):
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    try:
        # Get file size
        file_size = os.path.getsize(abs_file_path)
        
        # Check if file is likely binary
        def is_binary_file(file_path):
            """Check if a file is binary by reading the first chunk"""
            try:
                with open(file_path, 'rb') as f:
                    chunk = f.read(1024)
                    # If there are null bytes, it's likely binary
                    return b'\x00' in chunk
            except:
                return True
        
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp'}
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml',
            '.xml', '.html', '.css', '.scss', '.sass', '.less', '.php', '.rb', '.go',
            '.rs', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.swift', '.kt', '.scala',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.sql', '.r', '.dockerfile', '.env',
            '.gitignore', '.gitattributes', '.log', '.conf', '.config', '.ini', '.toml'
        }
        
        if file_ext in image_extensions:
            # Handle image files
            try:
                import base64
                with open(abs_file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
                return {
                    "content": content,
                    "size": file_size,
                    "type": "image",
                    "encoding": "base64"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading image file: {str(e)}")
        
        elif file_ext in text_extensions or not is_binary_file(abs_file_path):
            # Handle text files
            try:
                with open(abs_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Limit content size for very large files (>1MB)
                if len(content) > 1024 * 1024:  # 1MB
                    content = content[:1024 * 1024] + "\n\n... (File truncated due to size)"
                
                return {
                    "content": content,
                    "size": file_size,
                    "type": "text",
                    "encoding": "utf-8"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
        
        else:
            # Handle binary files
            return {
                "content": "",
                "size": file_size,
                "type": "binary",
                "encoding": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")

@app.get("/api/tree")
async def get_tree_structure(session_id: str = Query(...)):
    """Get the tree structure of the repository"""
    session = session_manager.get_session(session_id)
    if not session or "repo_path" not in session:
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

@app.get("/cache-stats")
async def get_cache_statistics():
    """Get cache statistics for monitoring performance"""
    from .cache_manager import rag_cache, response_cache, folder_cache
    
    return {
        "rag_cache": rag_cache.get_stats(),
        "response_cache": response_cache.get_stats(),
        "folder_cache": folder_cache.get_stats(),
        "cache_enabled": settings.CACHE_ENABLED,
        "feature_flags": {
            "rag_caching": settings.ENABLE_RAG_CACHING,
            "response_caching": settings.ENABLE_RESPONSE_CACHING,
            "smart_sizing": settings.ENABLE_SMART_SIZING,
            "repo_summaries": settings.ENABLE_REPO_SUMMARIES,
            "prompt_caching": settings.ENABLE_PROMPT_CACHING
        },
        "prompt_caching": {
            "enabled": settings.ENABLE_PROMPT_CACHING,
            "min_tokens": settings.PROMPT_CACHE_MIN_TOKENS,
            "provider": settings.llm_provider
        }
    }

@app.post("/assistant/sessions", response_model=RepoSessionResponse)
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

@app.get("/assistant/sessions", response_model=SessionListResponse)
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

@app.get("/assistant/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current status of a repository session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session.get("metadata", {}).get("status", "unknown"),
        "error": session.get("metadata", {}).get("error"),
        "repo_info": session.get("repo_context", {}).get("repo_info") if session.get("repo_context") else None
    }

@app.delete("/assistant/sessions/{session_id}")
async def delete_assistant_session(session_id: str):
    """Delete an assistant session and clean up resources"""
    if session_manager.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/assistant/sessions/{session_id}/metadata")
async def get_session_metadata(session_id: str):
    """Get detailed metadata about a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "type": session.get("type"),
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "metadata": session.get("metadata", {}),
        "message_count": len(session.get("conversation_history", [])),
        "repo_info": session.get("repo_context", {}).get("repo_info") if session.get("repo_context") else None
    }

@app.get("/assistant/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        conversation_history = session.get("conversation_history", [])
        
        return {
            "session_id": session_id,
            "messages": conversation_history,
            "total_messages": len(conversation_history)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AGENTIC SYSTEM ENDPOINTS
# ============================================================================

@app.post("/assistant/sessions/{session_id}/enable-agentic")
async def enable_agentic_mode(session_id: str):
    """Enable agentic capabilities for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        repo_path = session.get("repo_path")
        if not repo_path:
            raise HTTPException(status_code=400, detail="No repository path found for session")
        
        # Initialize agentic explorer for this session
        agentic_explorer = AgenticCodebaseExplorer(session_id, repo_path)
        agentic_explorers[session_id] = agentic_explorer
        
        # Mark session as agentic-enabled
        session["agentic_enabled"] = True
        
        return {
            "session_id": session_id,
            "agentic_enabled": True,
            "tools_available": [
                "explore_directory",
                "search_codebase", 
                "read_file",
                "analyze_file_structure",
                "rag_query",
                "find_related_files"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error enabling agentic mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/agentic-status")
async def get_agentic_status(session_id: str):
    """Check if agentic mode is enabled for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        is_agentic = session.get("agentic_enabled", False)
        has_explorer = session_id in agentic_explorers
        
        return {
            "session_id": session_id,
            "agentic_enabled": is_agentic,
            "explorer_initialized": has_explorer,
            "tools_available": [
                "explore_directory",
                "search_codebase", 
                "read_file",
                "analyze_file_structure",
                "rag_query",
                "find_related_files"
            ] if is_agentic else []
        }
        
    except Exception as e:
        logger.error(f"Error checking agentic status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/sessions/{session_id}/agentic-query")
async def agentic_query(
    session_id: str, 
    message: ChatMessage,
    stream: bool = Query(False)
):
    try:
        session_info = session_manager.get_session(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        repo_path = session_info["repo_path"]
        
        # Initialize agentic explorer
        explorer = AgenticCodebaseExplorer(session_id, repo_path)
        
        # Extract context files from the message content
        context_files = []
        if message.context_files:
            context_files = message.context_files
        else:
            # Extract from message content using regex
            import re
            mention_pattern = r'@(?:folder\/)?([^\s@]+)'
            matches = re.findall(mention_pattern, message.content)
            context_files = matches
        
        logger.info(f"Agentic query with context files: {context_files}")
        
        # Shared data structure for streaming and background save
        shared_data = {
            "steps": [],
            "final_answer": None,
            "partial": False,
            "suggestions": [],
            "error": None
        }

        # Add user message to session history
        session_manager.add_message(session_id, "user", message.content)

        if stream:
            async def stream_agentic_steps():
                try:
                    # Pass context files to the explorer
                    query_with_context = message.content
                    if context_files:
                        # Add context files information to the query
                        files_context = f"\n\nContext files mentioned: {', '.join(context_files)}"
                        query_with_context = message.content + files_context
                    
                    async for step_json in explorer.stream_query(query_with_context):
                        # Add proper SSE formatting
                        yield f"data: {step_json}\n\n"
                        # Parse and collect agentic output for background save
                        try:
                            parsed = json.loads(step_json)
                            logger.info(f"[DEBUG] Parsed streaming chunk: {parsed.get('type')} - {len(str(parsed))}")
                            
                            if parsed.get("type") == "step" and parsed.get("step"):
                                shared_data["steps"].append(parsed["step"])
                                logger.info(f"[DEBUG] Added step: {parsed['step']['type']} - Total steps: {len(shared_data['steps'])}")
                            elif parsed.get("type") == "final":
                                shared_data["steps"] = parsed.get("steps", shared_data["steps"])
                                shared_data["final_answer"] = parsed.get("final_answer")
                                shared_data["partial"] = parsed.get("partial", False)
                                shared_data["suggestions"] = parsed.get("suggestions", [])
                                logger.info(f"[DEBUG] Final chunk - Answer: {bool(shared_data['final_answer'])}, Steps: {len(shared_data['steps'])}")
                            elif parsed.get("type") == "error":
                                shared_data["error"] = parsed.get("error")
                                shared_data["partial"] = True
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"[DEBUG] Failed to parse streaming chunk: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"[DEBUG] Error processing streaming chunk: {e}")
                            continue

                    logger.info(f"[DEBUG] Streaming completed - Final answer: {bool(shared_data['final_answer'])}, Steps: {len(shared_data['steps'])}")

                except Exception as e:
                    logger.error(f"Error in agentic streaming: {e}")
                    shared_data["error"] = str(e)
                    shared_data["partial"] = True
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            response = StreamingResponse(
                stream_agentic_steps(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable proxy buffering
                }
            )
            
            # After streaming, save the agentic output to history
            async def save_agentic_message():
                # Wait longer to ensure streaming is completely done
                await asyncio.sleep(1.0)  # Increased wait time
                
                logger.info(f"[DEBUG] Saving agentic message - Steps: {len(shared_data['steps'])}, Final answer: {bool(shared_data['final_answer'])}")
                
                # Extract meaningful content from the shared data
                content = None
                
                # First priority: use final_answer if available
                if shared_data["final_answer"] and shared_data["final_answer"].strip():
                    content = shared_data["final_answer"]
                    logger.info(f"[DEBUG] Using final_answer as content")
                
                # Second priority: extract from the last answer step
                elif shared_data["steps"]:
                    for step in reversed(shared_data["steps"]):
                        if step.get('type') == 'answer' and step.get('content') and step.get('content').strip():
                            content = step['content']
                            logger.info(f"[DEBUG] Using answer step as content")
                            break
                    
                    # Third priority: combine all answer steps
                    if not content:
                        answer_contents = []
                        for step in shared_data["steps"]:
                            if step.get('type') == 'answer' and step.get('content'):
                                answer_contents.append(step['content'])
                        if answer_contents:
                            content = '\n\n'.join(answer_contents)
                            logger.info(f"[DEBUG] Using combined answer steps as content")
                
                # Fourth priority: create summary from steps
                if not content and shared_data["steps"]:
                    step_summaries = []
                    for step in shared_data["steps"]:
                        if step.get('type') and step.get('content'):
                            step_type = step['type'].title()
                            content_preview = step['content'][:100] + "..." if len(step['content']) > 100 else step['content']
                            step_summaries.append(f"**{step_type}**: {content_preview}")
                    
                    if step_summaries:
                        content = "## Analysis Complete\n\n" + '\n\n'.join(step_summaries)
                        logger.info(f"[DEBUG] Using step summaries as content")

                # Fallback content
                if not content:
                    if shared_data["error"]:
                        content = f"Analysis encountered an error: {shared_data['error']}"
                        logger.info(f"[DEBUG] Using error as content")
                    else:
                        content = "Agentic analysis completed, but no detailed response was captured."
                        logger.info(f"[DEBUG] Using fallback content - no meaningful data collected")

                logger.info(f"[DEBUG] Final content length: {len(content) if content else 0}")

                try:
                    session_manager.add_message(
                        session_id,
                        role="assistant",
                        content=content,
                        agenticSteps=shared_data["steps"],
                        suggestions=shared_data.get("suggestions", [])
                    )
                    logger.info(f"[DEBUG] Successfully saved agentic message to session {session_id}")
                except Exception as e:
                    logger.error(f"[DEBUG] Failed to save agentic message: {e}")

            # Create background task
            background_tasks = BackgroundTasks()
            background_tasks.add_task(save_agentic_message)
            response.background = background_tasks

            return response
        else:
            # Non-streaming response
            result = await explorer.query(message.content)
            
            try:
                parsed_result = json.loads(result)
                content = parsed_result.get("final_answer", "Analysis completed")
                agentic_steps = parsed_result.get("steps", [])
                suggestions = parsed_result.get("suggestions", [])
            except json.JSONDecodeError:
                content = result
                agentic_steps = []
                suggestions = []
            
            # Add user message to session history
            session_manager.add_message(session_id, "user", message.content)
            
            # Add assistant response to session history
            session_manager.add_message(
                session_id,
                role="assistant", 
                content=content,
                agenticSteps=agentic_steps,
                suggestions=suggestions
            )
            
            return ChatMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error processing agentic query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/sessions/{session_id}/reset-agentic-memory")
async def reset_agentic_memory(session_id: str):
    """Reset the agentic system's memory for a session"""
    try:
        if session_id not in agentic_explorers:
            raise HTTPException(status_code=404, detail="Agentic explorer not found")
        
        explorer = agentic_explorers[session_id]
        explorer.reset_memory()
        
        return {
            "session_id": session_id,
            "memory_reset": True
        }
        
    except Exception as e:
        logger.error(f"Error resetting agentic memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/issues")
async def list_issues(repo_url: str, state: str = "open"):
    """List issues for a given repository URL and state (open/closed/all)."""
    try:
        issues = await github_client.list_issues(repo_url, state)
        # Convert Issue objects to dicts for JSON serialization
        return [issue.model_dump() for issue in issues]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch issues: {str(e)}")

@app.get("/api/issues/{issue_number}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
