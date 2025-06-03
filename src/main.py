from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .models import PromptRequest, PromptResponse, ChatMessage, SessionResponse
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
from typing import Optional
import re
import json

# Enable nested event loops for Jupyter notebooks
nest_asyncio.apply()

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.2.0"
)

# Add CORS middleware
# It's good practice to list specific origins in production.
# For development, ["*"] is often used, but sometimes explicit origins work better.
allowed_origins = [
    "http://localhost:8080", # Your Vite frontend dev port from screenshot
    "http://localhost:5173", # Common Vite default
    "http://localhost:3000", # Common React dev port
    # Add any other origins you might be using
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Use the list here
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers
)

# Initialize clients and services
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()
session_manager = SessionManager()
conversation_memory = ConversationContextManager(max_context_tokens=8000)

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
async def handle_chat_message(session_id: str, message: ChatMessage, stream: bool = Query(False)):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
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
        
        # Debug: Log the actual file paths being retrieved from RAG
        if fresh_rag_context and fresh_rag_context.get("sources"):
            print(f"RAG retrieved {len(fresh_rag_context['sources'])} sources:")
            for i, source in enumerate(fresh_rag_context['sources'][:5], 1):
                file_path = source.get('file', 'UNKNOWN')
                print(f"  {i}. {file_path}")
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
    print(f"[DEBUG] /api/files called with session_id: {session_id}")
    session = session_manager.get_session(session_id)
    print(f"[DEBUG] Session found: {session is not None}")
    if session:
        print(f"[DEBUG] Session keys: {session.keys()}")
        print(f"[DEBUG] repo_path in session: {'repo_path' in session}")
        if "repo_path" in session:
            print(f"[DEBUG] repo_path value: {session['repo_path']}")
    
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
    abs_file_path = os.path.join(repo_path, file_path)
    
    # Security check: ensure the file is within the repo directory
    if not abs_file_path.startswith(repo_path):
        raise HTTPException(status_code=403, detail="Access denied: File outside repository")
    
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
    print(f"[DEBUG] /api/tree called with session_id: {session_id}")
    session = session_manager.get_session(session_id)
    print(f"[DEBUG] Session found: {session is not None}")
    if not session or "repo_path" not in session:
        if session:
            print(f"[DEBUG] Session keys: {session.keys()}")
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
            "repo_summaries": settings.ENABLE_REPO_SUMMARIES
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
