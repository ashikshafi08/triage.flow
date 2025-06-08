from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel # Added for IssueContextRequest
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
from .models import PromptRequest, PromptResponse, ChatMessage, SessionResponse, RepoRequest, RepoSessionResponse, SessionListResponse, Issue, IssueContextResponse, PullRequestInfo # Added IssueContextResponse and PullRequestInfo
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .prompt_generator import PromptGenerator
from .session_manager import SessionManager
from .conversation_memory import ConversationContextManager
from .cache_manager import response_cache, cleanup_caches_periodically
from .config import settings
import nest_asyncio
from typing import List
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
from .issue_rag import IssueAwareRAG # Added for the new endpoint
from .local_repo_loader import get_repo_info # Added for the new endpoint
from .chunk_store import ChunkStoreFactory, RedisChunkStore
from .founding_member_agent import FoundingMemberAgent

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
    # Initialize chunk store
    ChunkStoreFactory.get_instance()

# Pydantic model for the new endpoint's request body
class IssueContextRequest(BaseModel):
    query: str
    repo_url: str
    max_issues: Optional[int] = 5
    include_patches: Optional[bool] = True

@app.post("/api/v1/issue_context", response_model=IssueContextResponse)
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
        await issue_rag_system.initialize(force_rebuild=False, max_issues_for_patch_linkage=50)

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
    """Enhanced message handler with AgenticRAG integration"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the AgenticRAG instance from session
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized with AgenticRAG")
        
        # Add user message to session
        session_manager.add_message(session_id, "user", message.content)
        
        # Extract @file_path mentions and folder mentions from message.content
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
                    "content": content[:settings.MAX_USER_FILE_CONTENT_CHARS]  # Use configurable limit
                })
        
        # Extract issue references and add to context
        issue_contexts = []
        repo_url = session.get("metadata", {}).get("repo_url", "")
        if repo_url:
            try:
                issue_contexts = _extract_issue_references(message.content, repo_url)
            except Exception as e:
                print(f"Error extracting issue references: {e}")
        
        # Add current issue context from session metadata
        current_issue_context = session.get("metadata", {}).get("currentIssueContext")
        if current_issue_context:
            issue_contexts.append({
                "number": current_issue_context["number"],
                "title": current_issue_context["title"],
                "body": current_issue_context["body"][:1000],
                "state": current_issue_context["state"],
                "labels": current_issue_context["labels"],
                "comments_count": len(current_issue_context.get("comments", [])),
                "is_current_context": True
            })
            print(f"Added current issue context: #{current_issue_context['number']} - {current_issue_context['title']}")
        
        # Process folder mentions - collect all files in the folder
        restricted_files = []
        for folder_path in folder_mentions:
            abs_folder_path = os.path.join(repo_path, folder_path)
            if os.path.exists(abs_folder_path) and os.path.isdir(abs_folder_path):
                for root, dirs, files in os.walk(abs_folder_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for file in files:
                        if not file.startswith('.'):
                            file_abs_path = os.path.join(root, file)
                            file_rel_path = os.path.relpath(file_abs_path, repo_path)
                            restricted_files.append(file_rel_path)
        
        # Use AgenticRAG for enhanced context retrieval
        user_query = message.content
        
        # Determine if we should force agentic enhancement
        force_agentic = agentic or _should_use_agentic_approach(message.content)
        
        try:
            # Get enhanced context using AgenticRAG
            enhanced_context = await agentic_rag.get_enhanced_context(
                user_query, 
                restrict_files=restricted_files if folder_mentions else None,
                use_agentic_tools=force_agentic
            )
            
            # Add user-selected files and issue contexts
            enhanced_context["user_selected_files"] = user_file_contexts
            enhanced_context["issue_references"] = issue_contexts
            
        except Exception as e:
            print(f"Error getting enhanced context: {e}")
            # Fallback to basic context
            enhanced_context = {
                "user_selected_files": user_file_contexts,
                "issue_references": issue_contexts,
                "error": f"Enhanced context retrieval failed: {str(e)}"
            }
        
        # Use production-grade conversation memory
        history = session.get("conversation_history", [])
        conversation_context, memory_stats = await conversation_memory.get_conversation_context(
            history, llm_client, use_compression=True
        )
        
        # Log context retrieval for monitoring
        if enhanced_context and enhanced_context.get("sources"):
            sources_count = len(enhanced_context["sources"])
            query_analysis = enhanced_context.get("query_analysis", {})
            processing_strategy = query_analysis.get("processing_strategy", "unknown")
            print(f"AgenticRAG retrieved {sources_count} sources using strategy: {processing_strategy}")
            if folder_mentions:
                print(f"Folder-restricted query for folders: {folder_mentions}")
        else:
            print("No enhanced context sources retrieved")
        
        # Get model configuration
        session_llm_config = session.get("llm_config", {})
        model_name = session_llm_config.name if session_llm_config and hasattr(session_llm_config, 'name') else settings.default_model
        
        # Query classification and caching
        query_type = _classify_query_type(user_query)
        use_cache = settings.ENABLE_RESPONSE_CACHING and _should_use_cached_response(query_type, user_query)
        response_cache_key = None
        
        # Generate cache key for response caching
        if use_cache:
            repo_name = agentic_rag.get_repo_info().get("repo", "") if agentic_rag.get_repo_info() else ""
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
                        for i in range(0, len(words), 3):
                            chunk = ' '.join(words[i:i+3]) + ' '
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                            await asyncio.sleep(0.01)
                        yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(stream_cached(), media_type="text/event-stream")
                else:
                    return ChatMessage(role="assistant", content=cached_response)

        if stream:
            full_response_content = ""
            async def generate_stream():
                nonlocal full_response_content
                buffer = ""
                async for chunk in llm_client.stream_openrouter_response(
                    prompt=conversation_context,
                    prompt_type=session["prompt_type"],
                    context=enhanced_context,
                    model=model_name
                ):
                    if chunk.startswith('data: '):
                        data = chunk[6:].strip()
                        if data and data != '[DONE]':
                            try:
                                json_data = json.loads(data)
                                content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    buffer += content
                                    full_response_content += content
                                    
                                    # Send chunks of complete words
                                    if len(buffer) >= 50 or ' ' in buffer:
                                        words = buffer.split(' ')
                                        if len(words) > 1:
                                            chunk_to_send = ' '.join(words[:-1]) + ' '
                                            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk_to_send}}]})}\n\n"
                                            buffer = words[-1]
                            except json.JSONDecodeError:
                                yield f"data: {data}\n\n"
                    yield chunk
                
                # Send any remaining content in buffer
                if buffer:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': buffer}}]})}\n\n"
                
                session_manager.add_message(session_id, "assistant", full_response_content)
                
                if use_cache and full_response_content and response_cache_key:
                    await response_cache.set(response_cache_key, full_response_content, settings.CACHE_TTL_RESPONSE)
                    print(f"Cached response for query type: {query_type}")

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            llm_response = await llm_client.process_prompt(
                prompt=conversation_context,
                prompt_type=session["prompt_type"],
                context=enhanced_context,  # Use enhanced context
                model=model_name
            )
            session_manager.add_message(session_id, "assistant", llm_response.prompt)
            
            # Cache the response if appropriate
            if use_cache and response_cache_key:
                await response_cache.set(response_cache_key, llm_response.prompt, settings.CACHE_TTL_RESPONSE)
                print(f"Cached response for query type: {query_type}")
            
            return ChatMessage(role="assistant", content=llm_response.prompt)
        
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
    
    # Simple queries that DON'T need agentic approach
    simple_patterns = [
        "list files in", "show files in", "what files are in", "files in the",
        "list all files", "show me files", "what's in", "contents of"
    ]
    
    # If it's a simple file listing query, use RAG instead
    if any(pattern in query_lower for pattern in simple_patterns):
        # Unless it's asking for complex analysis along with listing
        complex_indicators = ["analyze", "explain", "how", "why", "implement", "debug"]
        if not any(indicator in query_lower for indicator in complex_indicators):
            return False
    
    # Agentic approach is beneficial for:
    # 1. Complex exploration queries
    agentic_patterns = [
        "explore", "analyze", "find all", "search for", "trace", "follow", 
        "investigate", "deep dive", "architecture", "structure", "relationship", 
        "dependency", "related to", "implement", "build", "create", "example", 
        "show me how", "debug", "troubleshoot", "fix", "error", "issue", "problem"
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
    
    # Long queries (>20 words) might benefit from agentic approach, but be more conservative
    if len(query.split()) > 20 and any(pattern in query_lower for pattern in agentic_patterns):
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
    """Get file content with dynamic content handling"""
    try:
        logger.info(f"Getting file content for session {session_id}, file: {file_path}")
        
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            logger.error(f"AgenticRAG not initialized for session {session_id}")
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
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
    
    metadata = session.get("metadata", {})
    
    return {
        "session_id": session_id,
        "status": metadata.get("status", "unknown"),
        "error": metadata.get("error"),
        "repo_info": session.get("repo_context", {}).get("repo_info") if session.get("repo_context") else None,
        "metadata": metadata  # progress fields
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
    """Check agentic capabilities for a session (now integrated by default)"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="Session not properly initialized with AgenticRAG")
        
        return {
            "session_id": session_id,
            "agentic_enabled": True,
            "integrated_system": "AgenticRAG",
            "tools_available": [
                "enhanced_context_retrieval",
                "query_analysis",
                "technical_requirements_extraction",
                "code_references_detection",
                "semantic_search",
                "file_structure_analysis",
                "related_files_discovery",
                "code_example_generation"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error checking agentic mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/agentic-status")
async def get_agentic_status(session_id: str):
    """Check if AgenticRAG is enabled and initialized for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        is_initialized = agentic_rag is not None
        
        return {
            "session_id": session_id,
            "agentic_enabled": session.get("agentic_enabled", False),
            "agentic_rag_initialized": is_initialized,
            "system_type": "AgenticRAG" if is_initialized else "None",
            "capabilities": [
                "Smart query analysis",
                "Enhanced context retrieval", 
                "Multi-strategy processing",
                "Technical requirements extraction",
                "Code references detection",
                "Semantic enhancement",
                "File relationship analysis"
            ] if is_initialized else []
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
    """Advanced query processing using AgenticRAG's agentic tools"""
    try:
        session_info = session_manager.get_session(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session_info.get("agentic_rag")
        if not agentic_rag or not agentic_rag.agentic_explorer:
            raise HTTPException(status_code=400, detail="AgenticRAG not properly initialized")
        
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
        
        logger.info(f"Advanced agentic query with context files: {context_files}")
        
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

        # Prepare query with enhanced context
        query_with_context = message.content
        if context_files:
            files_context = f"\n\nContext files mentioned: {', '.join(context_files)}"
            query_with_context = message.content + files_context
        
        # Add current issue context if available
        current_issue_context = session_info.get("metadata", {}).get("currentIssueContext")
        if current_issue_context:
            issue_context_text = f"""

Current Issue Context:
- Issue #{current_issue_context['number']}: {current_issue_context['title']}
- State: {current_issue_context['state']}
- Labels: {', '.join(current_issue_context.get('labels', []))}
- Description: {current_issue_context['body'][:500]}{'...' if len(current_issue_context.get('body', '')) > 500 else ''}

This issue is currently in the conversation context. Please consider it when analyzing the codebase or answering questions."""
            query_with_context = message.content + issue_context_text
            print(f"Added issue context to agentic query: #{current_issue_context['number']}")

        if stream:
            async def stream_agentic_steps():
                try:
                    # Use the agentic explorer from AgenticRAG for deep analysis
                    async for step_json in agentic_rag.agentic_explorer.stream_query(query_with_context):
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
                    "X-Accel-Buffering": "no"
                }
            )
            
            # After streaming, save the agentic output to history
            async def save_agentic_message():
                await asyncio.sleep(1.0)
                
                logger.info(f"[DEBUG] Saving agentic message - Steps: {len(shared_data['steps'])}, Final answer: {bool(shared_data['final_answer'])}")
                
                # Extract meaningful content from the shared data
                content = None
                
                if shared_data["final_answer"] and shared_data["final_answer"].strip():
                    content = shared_data["final_answer"]
                    logger.info(f"[DEBUG] Using final_answer as content")
                elif shared_data["steps"]:
                    for step in reversed(shared_data["steps"]):
                        if step.get('type') == 'answer' and step.get('content') and step.get('content').strip():
                            content = step['content']
                            logger.info(f"[DEBUG] Using answer step as content")
                            break
                    
                    if not content:
                        answer_contents = []
                        for step in shared_data["steps"]:
                            if step.get('type') == 'answer' and step.get('content'):
                                answer_contents.append(step['content'])
                        if answer_contents:
                            content = '\n\n'.join(answer_contents)
                            logger.info(f"[DEBUG] Using combined answer steps as content")
                
                if not content and shared_data["steps"]:
                    step_summaries = []
                    for step in shared_data["steps"]:
                        if step.get('type') and step.get('content'):
                            step_type = step['type'].title()
                            content_preview = step['content'][:100] + "..." if len(step['content']) > 100 else step['content']
                            step_summaries.append(f"**{step_type}**: {content_preview}")
                    
                    if step_summaries:
                        content = "## Advanced Analysis Complete\n\n" + '\n\n'.join(step_summaries)
                        logger.info(f"[DEBUG] Using step summaries as content")

                if not content:
                    if shared_data["error"]:
                        content = f"Advanced analysis encountered an error: {shared_data['error']}"
                        logger.info(f"[DEBUG] Using error as content")
                    else:
                        content = "Advanced agentic analysis completed."
                        logger.info(f"[DEBUG] Using fallback content")

                logger.info(f"[DEBUG] Final content length: {len(content) if content else 0}")

                try:
                    session_manager.add_message(
                        session_id,
                        role="assistant",
                        content=content,
                        agenticSteps=shared_data["steps"],
                        suggestions=shared_data.get("suggestions", []),
                        processingType="advanced_agentic"
                    )
                    logger.info(f"[DEBUG] Successfully saved advanced agentic message to session {session_id}")
                except Exception as e:
                    logger.error(f"[DEBUG] Failed to save agentic message: {e}")

            background_tasks = BackgroundTasks()
            background_tasks.add_task(save_agentic_message)
            response.background = background_tasks

            return response
        else:
            # Non-streaming response using agentic explorer
            result = await agentic_rag.agentic_explorer.query(query_with_context)
            
            try:
                parsed_result = json.loads(result)
                content = parsed_result.get("final_answer", "Advanced analysis completed")
                agentic_steps = parsed_result.get("steps", [])
                suggestions = parsed_result.get("suggestions", [])
            except json.JSONDecodeError:
                content = result
                agentic_steps = []
                suggestions = []
            
            # Add assistant response to session history
            session_manager.add_message(
                session_id,
                role="assistant", 
                content=content,
                agenticSteps=agentic_steps,
                suggestions=suggestions,
                processingType="advanced_agentic"
            )
            
            return ChatMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error processing advanced agentic query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/sessions/{session_id}/reset-agentic-memory")
async def reset_agentic_memory(session_id: str):
    """Reset the agentic system's memory for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag or not agentic_rag.agentic_explorer:
            raise HTTPException(status_code=400, detail="AgenticRAG not properly initialized")
        
        agentic_rag.agentic_explorer.reset_memory()
        
        return {
            "session_id": session_id,
            "memory_reset": True,
            "system": "AgenticRAG"
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

@app.get("/api/prs", response_model=List[PullRequestInfo])
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

def _extract_issue_references(message_content: str, repo_url: str) -> list:
    """Extract issue references from message content and return issue data"""
    import re
    
    issues_data = []
    
    # Pattern to match issue numbers (#1234) or issue URLs
    issue_patterns = [
        r'#(\d+)',  # #1234
        r'github\.com/[^/]+/[^/]+/issues/(\d+)',  # GitHub issue URLs
    ]
    
    issue_numbers = set()
    for pattern in issue_patterns:
        matches = re.findall(pattern, message_content)
        issue_numbers.update([int(num) for num in matches])
    
    # Fetch issue details for each referenced issue
    for issue_num in issue_numbers:
        try:
            issue_url = f"{repo_url}/issues/{issue_num}"
            issue_response = asyncio.run(github_client.get_issue(issue_url))
            if issue_response.status == "success" and issue_response.data:
                issues_data.append({
                    "number": issue_response.data.number,
                    "title": issue_response.data.title,
                    "body": issue_response.data.body[:1000],  # Limit body length
                    "state": issue_response.data.state,
                    "labels": issue_response.data.labels,
                    "comments_count": len(issue_response.data.comments)
                })
        except Exception as e:
            print(f"Error fetching issue #{issue_num}: {e}")
            continue
    
    return issues_data

@app.post("/assistant/sessions/{session_id}/add-issue-context")
async def add_issue_context_to_session(session_id: str, issue_data: Dict[str, Any]):
    """Add issue context to session metadata for AI memory"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session metadata with issue context
        if "metadata" not in session:
            session["metadata"] = {}
        
        session["metadata"]["currentIssueContext"] = {
            "number": issue_data["number"],
            "title": issue_data["title"],
            "body": issue_data["body"],
            "state": issue_data["state"],
            "labels": issue_data["labels"],
            "assignees": issue_data.get("assignees", []),
            "comments": issue_data.get("comments", []),
            "created_at": issue_data["created_at"],
            "url": issue_data["url"]
        }
        
        print(f"Added issue #{issue_data['number']} to session {session_id} metadata")
        
        return {
            "session_id": session_id,
            "issue_context_added": True,
            "issue_number": issue_data["number"],
            "issue_title": issue_data["title"]
        }
        
    except Exception as e:
        logger.error(f"Error adding issue context to session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/agentic-rag-info")
async def get_agentic_rag_info(session_id: str):
    """Get detailed information about the AgenticRAG system for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            return {
                "session_id": session_id,
                "agentic_rag_enabled": False,
                "message": "AgenticRAG not initialized for this session"
            }
        
        repo_info = agentic_rag.get_repo_info()
        
        return {
            "session_id": session_id,
            "agentic_rag_enabled": True,
            "repo_info": repo_info,
            "capabilities": {
                "query_analysis": True,
                "enhanced_context_retrieval": True,
                "multi_strategy_processing": True,
                "technical_requirements_extraction": True,
                "code_references_detection": True,
                "semantic_enhancement": True,
                "file_relationship_analysis": True,
                "agentic_tool_integration": True
            },
            "processing_strategies": [
                "agentic_deep - for complex exploration and architecture queries",
                "agentic_focused - for implementation and analysis queries", 
                "agentic_light - for simple queries with minimal enhancement",
                "rag_only - for basic queries without agentic enhancement"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting AgenticRAG info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/sessions/{session_id}/analyze-query")
async def analyze_query_endpoint(session_id: str, query: dict):
    """Analyze a query to understand how AgenticRAG would process it"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        user_query = query.get("text", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Analyze the query
        analysis = await agentic_rag._analyze_query(user_query)
        
        return {
            "session_id": session_id,
            "query": user_query,
            "analysis": analysis,
            "recommendations": {
                "suggested_processing": analysis.get("processing_strategy", "unknown"),
                "expected_enhancement": "Yes" if analysis.get("should_use_agentic", False) else "No",
                "complexity_level": "High" if analysis.get("complexity_score", 0) > 6 else "Medium" if analysis.get("complexity_score", 0) > 3 else "Low"
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/context-preview")
async def get_context_preview(session_id: str, query: str = Query(...), max_sources: int = Query(5)):
    """Get a preview of what context AgenticRAG would retrieve for a query"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        # Get enhanced context but limit sources for preview
        enhanced_context = await agentic_rag.get_enhanced_context(
            query, 
            restrict_files=None,
            use_agentic_tools=True
        )
        
        # Limit sources for preview
        if enhanced_context.get("sources"):
            enhanced_context["sources"] = enhanced_context["sources"][:max_sources]
        
        # Remove large content for preview
        preview_context = enhanced_context.copy()
        for source in preview_context.get("sources", []):
            if len(source.get("content", "")) > 500:
                source["content"] = source["content"][:500] + "... [truncated for preview]"
        
        return {
            "session_id": session_id,
            "query": query,
            "context_preview": preview_context,
            "total_sources_available": len(enhanced_context.get("sources", [])),
            "sources_in_preview": len(preview_context.get("sources", []))
        }
        
    except Exception as e:
        logger.error(f"Error getting context preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/related-issues")
async def get_related_issues(
    session_id: str, 
    query: str = Query(..., description="Query to find similar issues for"),
    max_issues: int = Query(5, description="Maximum number of issues to return"),
    state: str = Query("all", description="Issue state filter: open, closed, all"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity threshold")
):
    """Get related GitHub issues for a query"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        # Check if issue RAG is available
        if not agentic_rag.issue_rag or not agentic_rag.issue_rag.is_initialized():
            return {
                "session_id": session_id,
                "query": query,
                "related_issues": [],
                "total_found": 0,
                "message": "Issue history not available or still indexing",
                "processing_time": 0.0
            }
        
        # Get issue context
        issue_context = await agentic_rag.issue_rag.get_issue_context(
            query, 
            max_issues=max_issues
        )
        
        # Format response
        formatted_issues = []
        for result in issue_context.related_issues:
            if result.similarity >= similarity_threshold:
                issue = result.issue
                formatted_issue = {
                    "number": issue.id,
                    "title": issue.title,
                    "state": issue.state,
                    "url": f"https://github.com/{agentic_rag.repo_info.get('owner')}/{agentic_rag.repo_info.get('repo')}/issues/{issue.id}",
                    "similarity": round(result.similarity, 3),
                    "labels": issue.labels,
                    "created_at": issue.created_at,
                    "closed_at": issue.closed_at,
                    "body_preview": issue.body[:300] + "..." if len(issue.body) > 300 else issue.body,
                    "match_reasons": result.match_reasons,
                    "comments_count": len(issue.comments)
                }
                
                if issue.patch_url:
                    formatted_issue["patch_url"] = issue.patch_url
                
                formatted_issues.append(formatted_issue)
        
        return {
            "session_id": session_id,
            "query": query,
            "related_issues": formatted_issues,
            "total_found": len(formatted_issues),
            "processing_time": issue_context.processing_time,
            "query_analysis": issue_context.query_analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting related issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/sessions/{session_id}/index-issues")
async def index_repository_issues(
    session_id: str,
    force_rebuild: bool = Query(False, description="Force rebuild of issue index"),
    max_issues: int = Query(1000, description="Maximum number of issues to index"),
    max_prs: int = Query(1000, description="Maximum number of PRs to index")
):
    """Index repository issues and PRs for RAG"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get repo info from session
        repo_url = session.get("repo_url")
        if not repo_url:
            raise HTTPException(status_code=400, detail="No repository URL found in session")
        
        # Parse repo owner/name from URL
        owner, repo = parse_repo_url(repo_url)
        
        # Initialize RAG with new parameters
        rag = IssueAwareRAG(owner, repo)
        await rag.initialize(
            force_rebuild=force_rebuild,
            max_issues_for_patch_linkage=max_issues,
            max_prs_for_patch_linkage=max_prs
        )
        
        return {
            "status": "success",
            "message": f"Indexed issues and PRs for {owner}/{repo}",
            "max_issues": max_issues,
            "max_prs": max_prs
        }
        
    except Exception as e:
        logger.error(f"Error indexing repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/sessions/{session_id}/issue-index-status")
async def get_issue_index_status(session_id: str):
    """Get the status of issue indexing for a session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        status = {
            "session_id": session_id,
            "issue_rag_available": agentic_rag.issue_rag is not None,
            "issue_rag_initialized": False,
            "repository": None,
            "total_issues": 0,
            "last_updated": None
        }
        
        if agentic_rag.repo_info:
            status["repository"] = f"{agentic_rag.repo_info.get('owner')}/{agentic_rag.repo_info.get('repo')}"
        
        if agentic_rag.issue_rag:
            status["issue_rag_initialized"] = agentic_rag.issue_rag.is_initialized()
            
            # Try to get metadata if available
            try:
                metadata_file = agentic_rag.issue_rag.indexer.metadata_file
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        status["total_issues"] = metadata.get("total_issues", 0)
                        status["last_updated"] = metadata.get("last_updated")
            except Exception:
                pass
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting issue index status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agentic-rag-features")
async def get_agentic_rag_features():
    """Get information about AgenticRAG features and capabilities"""
    return {
        "system_name": "AgenticRAG",
        "description": "Enhanced RAG system combining semantic retrieval with agentic tool capabilities",
        "features": {
            "intelligent_query_analysis": {
                "description": "Automatically analyzes queries to determine optimal processing approach",
                "capabilities": ["query classification", "complexity assessment", "technical requirements extraction"]
            },
            "multi_strategy_processing": {
                "description": "Uses different processing strategies based on query type and complexity",
                "strategies": ["agentic_deep", "agentic_focused", "agentic_light", "rag_only"]
            },
            "enhanced_context_retrieval": {
                "description": "Combines traditional RAG with agentic tools for richer context",
                "enhancements": ["semantic search", "file relationship analysis", "code structure analysis"]
            },
            "code_intelligence": {
                "description": "Advanced understanding of code structure and relationships",
                "capabilities": ["code reference detection", "file dependency analysis", "example generation"]
            },
            "adaptive_enhancement": {
                "description": "Automatically determines when agentic tools would improve responses",
                "benefits": ["performance optimization", "quality improvement", "resource efficiency"]
            }
        },
        "integration": {
            "rag_system": "LocalRepoContextExtractor",
            "agentic_tools": "AgenticCodebaseExplorer", 
            "combined_benefits": ["better context quality", "smarter processing", "enhanced user experience"]
        }
    }

# Chunk API endpoints
@app.get("/api/agentic/chunk/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Retrieve a chunk by ID"""
    chunk_store = ChunkStoreFactory.get_instance()
    content = chunk_store.retrieve(chunk_id)
    if not content:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"content": content}

@app.delete("/api/agentic/chunk/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """Delete a chunk by ID"""
    chunk_store = ChunkStoreFactory.get_instance()
    if chunk_store.delete(chunk_id):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chunk not found")

@app.get("/api/agentic/redis-health")
async def check_redis_health():
    """Check Redis connection health"""
    try:
        chunk_store = ChunkStoreFactory.get_instance()
        if isinstance(chunk_store, RedisChunkStore):
            # Test Redis connection
            chunk_store.redis.ping()
            return {
                "status": "healthy",
                "store_type": "redis",
                "message": "Redis connection is working"
            }
        else:
            return {
                "status": "degraded",
                "store_type": "memory",
                "message": "Using in-memory store (Redis not available)"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "store_type": "memory",
            "message": f"Redis error: {str(e)}"
        }

class FounderSessionRequest(BaseModel):
    repo_url: str
    session_name: Optional[str] = None

@app.post("/founder/sessions", response_model=SessionResponse)
async def create_founding_session(request: FounderSessionRequest, background_tasks: BackgroundTasks):
    """Create a new session with FoundingMemberAgent for a given repo (async patch linkage)."""
    try:
        # Validate repository URL format
        if not request.repo_url.startswith(('https://github.com/', 'http://github.com/')):
            raise HTTPException(
                status_code=400,
                detail="Invalid repository URL. Must be a GitHub repository URL starting with https://github.com/ or http://github.com/"
            )

        # 1. Create a repo session (gets session_id, metadata)
        session_id, metadata = session_manager.create_repo_session(request.repo_url, session_name=request.session_name)
        session = session_manager.get_session(session_id)
        session["metadata"]["status"] = "cloning"
        session["metadata"]["progress"] = 0.1
        session["metadata"]["message"] = "Cloning repository..."
        session["metadata"]["tools_ready"] = []
        try:
            # 2. Load the repo (cloning)
            code_rag = LocalRepoContextExtractor()
            await code_rag.load_repository(request.repo_url)
            session["metadata"]["status"] = "indexing"
            session["metadata"]["progress"] = 0.4
            session["metadata"]["message"] = "Indexing codebase..."
            owner = metadata["owner"]
            repo = metadata["repo"]
            # 3. Issue RAG (fast)
            issue_rag = IssueAwareRAG(owner, repo)
            await issue_rag.initialize(force_rebuild=False, max_issues_for_patch_linkage=10)  # Small number for fast init
            session["metadata"]["status"] = "patch_linkage_pending"
            session["metadata"]["progress"] = 0.7
            session["metadata"]["message"] = "Patch linkage building in background..."
            session["metadata"]["tools_ready"] = ["code_rag", "issue_rag"]
            # Store code_rag and issue_rag for later use
            session["_code_rag"] = code_rag
            session["_issue_rag"] = issue_rag
            # 4. Start patch linkage and agent setup in background
            async def finish_patch_linkage_and_agent():
                try:
                    # Reuse code_rag and issue_rag
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

@app.get("/founder/sessions/{session_id}/status")
async def get_founding_session_status(session_id: str):
    """Get the current status and progress of a founding member session."""
    session = session_manager.get_session(session_id)
    if not session or "metadata" not in session:
        raise HTTPException(status_code=404, detail="Session not found")
    metadata = session["metadata"]
    return {
        "session_id": session_id,
        "status": metadata.get("status", "unknown"),
        "progress": metadata.get("progress", 0.0),
        "message": metadata.get("message", ""),
        "error": metadata.get("error"),
        "session_subtype": metadata.get("session_subtype"),
        "tools_ready": metadata.get("tools_ready", []),
    }

@app.post("/assistant/sessions/{session_id}/sync-repository")
async def sync_repository_data(session_id: str, background_tasks: BackgroundTasks):
    """
    Triggers a re-sync of the repository's issue and patch data.
    This involves re-running patch linkage and issue indexing.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    agentic_rag = session.get("agentic_rag")
    if not agentic_rag:
        raise HTTPException(status_code=400, detail="AgenticRAG system not initialized for this session.")
    
    if not agentic_rag.issue_rag:
        # This can happen if issue_rag failed to initialize initially or is not part of this session type
        # For repo_chat sessions, initialize_issue_rag_async is called.
        # We might need to re-trigger that if issue_rag is None.
        # For simplicity now, assume issue_rag should exist if sync is meaningful.
        logger.warning(f"Attempted to sync repo for session {session_id} but issue_rag is not available. Attempting to initialize.")
        # Potentially, we could try to kick off agentic_rag.initialize_issue_rag_async(session) here
        # if it's None, but that might complicate status management.
        # For now, let's assume it should have been initialized.
        # A more robust solution might re-run the full initialize_issue_rag_async if issue_rag is None.
        # However, a "sync" usually implies refreshing existing data.
        # If issue_rag is None due to prior init failure, a sync might also fail.
        # The frontend should ideally guide user if initial setup had issues.
        # For now, let's proceed assuming issue_rag should be there for a sync operation.
        # If it's not, it implies a deeper issue with the session's setup.
        raise HTTPException(status_code=400, detail="Issue RAG system not available for this session. Sync cannot proceed.")


    # Update session status to indicate syncing
    if "metadata" not in session: # Should always exist for repo_chat type
        session["metadata"] = {}
    session["metadata"]["status"] = "syncing_issues"
    session["metadata"]["message"] = "Re-syncing repository issues, PRs, and patches..."
    # TODO: Consider a mechanism to persist this status update immediately if needed by polling clients.
    # For now, it's in-memory until the task completes or next status poll.

    async def _sync_task():
        try:
            logger.info(f"Starting repository data sync for session {session_id}...")
            # Calling initialize with force_rebuild=True will re-trigger
            # patch linkage and issue indexing.
            await agentic_rag.issue_rag.initialize(
                force_rebuild=True, 
                max_issues_for_patch_linkage=settings.MAX_ISSUES_TO_PROCESS, # Use configured defaults
                max_prs_for_patch_linkage=settings.MAX_PR_TO_PROCESS
            )
            session["metadata"]["status"] = "ready" 
            session["metadata"]["message"] = "Repository data sync complete. Full context updated."
            logger.info(f"Repository data sync complete for session {session_id}.")
        except Exception as e:
            logger.error(f"Error during repository data sync for session {session_id}: {e}", exc_info=True)
            session["metadata"]["status"] = "error_syncing"
            session["metadata"]["message"] = f"Error during repository data sync: {str(e)}"
            session["metadata"]["error"] = str(e) # Store the error message
        # TODO: Persist final status / notify client more proactively if needed.

    background_tasks.add_task(_sync_task)
    
    return {"message": "Repository data sync process started in the background."}

def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Extract owner and repository name from a GitHub URL."""
    # Remove .git if present
    repo_url = repo_url.replace(".git", "")
    # Split by / and get last two parts
    parts = repo_url.split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

def extract_file_diff_from_full_diff(full_diff: str, target_file_path: str) -> Optional[str]:
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
            cleaned_diff = format_diff_for_display('\n'.join(lines))
            
            # If it's a proper diff with additions/deletions, return it
            if any(line.startswith(('+', '-')) for line in lines):
                return cleaned_diff
    
    return None

def format_diff_for_display(diff_content: str) -> str:
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

@app.get("/api/file-snippet")
async def get_file_snippet(
    session_id: str = Query(..., description="Session ID to identify the repository"),
    file_path: str = Query(..., description="Path to the file relative to repository root"),
    lines: int = Query(10, description="Number of lines to return (default: 10)"),
    start_line: Optional[int] = Query(None, description="Starting line number (1-indexed)"),
    pr_number: Optional[int] = Query(None, description="PR number to show diff for this file"),
    show_diff: bool = Query(False, description="Show diff instead of file content")
):
    """Get a snippet of a file for inline preview, with optional RAG-powered diff support"""
    try:
        # Get the session to access repository path and RAG system
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        repo_path = session.get("repo_path")
        if not repo_path:
            raise HTTPException(status_code=400, detail="No repository loaded in this session")
        
        # If diff is requested and we have a PR number, try RAG-based diff first
        if show_diff and pr_number:
            try:
                # Try to use the RAG-based diff system (similar to agentic_tools.py)
                agentic_rag = session.get("agentic_rag")
                if agentic_rag and hasattr(agentic_rag, 'agentic_explorer') and agentic_rag.agentic_explorer:
                    # Use the get_pr_diff method from agentic_tools
                    diff_response = agentic_rag.agentic_explorer.get_pr_diff(pr_number)
                    
                    try:
                        diff_data = json.loads(diff_response)
                        if "error" not in diff_data and "full_diff" in diff_data:
                            # Extract diff content for this specific file
                            full_diff = diff_data["full_diff"]
                            file_diff = extract_file_diff_from_full_diff(full_diff, file_path)
                            
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
                return await get_file_diff_for_pr(repo_path, file_path, pr_number)
            except Exception as e:
                logger.warning(f"Failed to get diff for PR {pr_number}: {e}")
                # Fall back to regular file content
        
        # Regular file content logic (existing code)
        from pathlib import Path
        import os
        
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

async def get_file_diff_for_pr(repo_path: str, file_path: str, pr_number: int):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
