from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from ...models import ChatMessage
from ..dependencies import (
    session_manager, llm_client, conversation_memory, 
    get_session, get_agentic_rag, logger, settings
)
from ...cache_manager import response_cache
import re
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List

router = APIRouter(prefix="/sessions", tags=["chat"])

@router.post("/{session_id}/messages", response_model=ChatMessage)
async def handle_chat_message(
    session_id: str, 
    message: ChatMessage, 
    stream: bool = Query(False),
    agentic: bool = Query(False, description="Use agentic system for deeper analysis"),
    session: Dict[str, Any] = Depends(get_session)
):
    """Enhanced message handler with AgenticRAG integration"""
    try:
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
                    "content": content[:settings.MAX_USER_FILE_CONTENT_CHARS]
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
        conversation_context, _ = await conversation_memory.get_conversation_context(
            history, llm_client
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
                context=enhanced_context,
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

@router.get("/{session_id}/memory-stats")
async def get_memory_statistics(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get conversation memory statistics for debugging and monitoring"""
    try:
        history = session.get("conversation_history", [])
        
        # Get memory statistics
        _, memory_stats = await conversation_memory.get_conversation_context(
            history, llm_client, use_compression=False
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

@router.post("/{session_id}/add-issue-context")
async def add_issue_context_to_session(
    session_id: str, 
    issue_data: Dict[str, Any],
    session: Dict[str, Any] = Depends(get_session)
):
    """Add issue context to session metadata for AI memory"""
    try:
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

@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    try:
        conversation_history = session.get("conversation_history", [])
        
        return {
            "session_id": session_id,
            "messages": conversation_history,
            "total_messages": len(conversation_history)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
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

def _extract_issue_references(message_content: str, repo_url: str) -> list:
    """Extract issue references from message content and return issue data"""
    import re
    from ..dependencies import github_client
    
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