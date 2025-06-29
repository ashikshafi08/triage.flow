from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from ...models import ChatMessage
from ..dependencies import (
    session_manager, get_session, get_agentic_rag, get_chunk_store, logger, settings
)
from ...agentic_rag import AgenticRAGSystem # Import AgenticRAGSystem for isinstance check
from ...chunk_store import RedisChunkStore
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

router = APIRouter(prefix="/assistant/sessions", tags=["agentic"])

@router.post("/{session_id}/enable-agentic")
async def enable_agentic_mode(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Check agentic capabilities for a session (now integrated by default)"""
    try:
        # First check if we have a functioning agentic_rag system
        agentic_rag = session.get("agentic_rag")
        
        # If no agentic_rag in session, try to get/create one through the dependency
        if not agentic_rag:
            try:
                # Use the dependency to get or create an AgenticRAG instance
                from ..dependencies import get_agentic_rag
                agentic_rag = await get_agentic_rag(session_id, session)
                # Store it in the session for future use
                session["agentic_rag"] = agentic_rag
                logger.info(f"Successfully created/retrieved AgenticRAG for session {session_id}")
            except HTTPException as he:
                logger.error(f"Failed to get AgenticRAG for session {session_id}: {he.detail}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"AgenticRAG system not available: {he.detail}"
                )
            except Exception as e:
                logger.error(f"Unexpected error getting AgenticRAG for session {session_id}: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to initialize AgenticRAG system: {str(e)}"
                )
        
        # Check if it's a valid AgenticRAG instance
        if not isinstance(agentic_rag, AgenticRAGSystem):
            logger.error(f"Invalid AgenticRAG instance type for session {session_id}: {type(agentic_rag)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Invalid AgenticRAG system type: {type(agentic_rag)}"
            )
        
        # Check core components
        core_tools = []
        has_rag_extractor = bool(agentic_rag.rag_extractor)
        has_agentic_explorer = bool(agentic_rag.agentic_explorer)
        has_issue_rag = bool(agentic_rag.issue_rag and agentic_rag.issue_rag.is_initialized())
        has_composite_retriever = bool(getattr(agentic_rag, '_use_composite', False))
        
        if has_rag_extractor:
            core_tools.extend([
                "enhanced_context_retrieval",
                "semantic_search",
                "file_structure_analysis",
                "related_files_discovery"
            ])
        
        if has_agentic_explorer:
            core_tools.extend([
                "query_analysis", 
                "technical_requirements_extraction",
                "code_references_detection",
                "code_example_generation"
            ])
        
        if has_issue_rag:
            core_tools.extend([
                "issue_context_retrieval",
                "related_issues_search",
                "issue_history_analysis"
            ])
        
        if has_composite_retriever:
            core_tools.append("multi_index_retrieval")
        
        # Get session metadata for additional context
        session_metadata = session.get("metadata", {})
        repo_status = session_metadata.get("status", "unknown")
        issue_rag_ready = session_metadata.get("issue_rag_ready", False)
        
        # Determine overall system status
        if has_rag_extractor and has_agentic_explorer:
            if has_issue_rag:
                system_status = "fully_operational"
                status_message = "All agentic capabilities available including issue context"
            else:
                system_status = "core_operational"
                if repo_status == "warning_issue_rag_failed":
                    status_message = "Core agentic capabilities available. Issue context unavailable due to GitHub API issues."
                else:
                    status_message = "Core agentic capabilities available. Issue context still loading or unavailable."
        else:
            system_status = "limited"
            status_message = "Limited agentic capabilities available"
        
        return {
            "session_id": session_id,
            "agentic_enabled": True,
            "integrated_system": "AgenticRAG",
            "system_status": system_status,
            "status_message": status_message,
            "components": {
                "rag_extractor": has_rag_extractor,
                "agentic_explorer": has_agentic_explorer, 
                "issue_rag": has_issue_rag,
                "composite_retriever": has_composite_retriever
            },
            "tools_available": core_tools,
            "repository": {
                "status": repo_status,
                "issue_rag_ready": issue_rag_ready,
                "repo_info": agentic_rag.get_repo_info() if agentic_rag else None
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error checking agentic mode for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{session_id}/agentic-status")
async def get_agentic_status(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Check if AgenticRAG is enabled and initialized for a session"""
    try:
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

@router.post("/{session_id}/agentic-query")
async def agentic_query(
    session_id: str, 
    message: ChatMessage,
    stream: bool = Query(False),
    session: Dict[str, Any] = Depends(get_session), # Keep get_session for other session data
    agentic_rag: Any = Depends(get_agentic_rag) # Use get_agentic_rag for the agentic_rag object
):
    """Advanced query processing using AgenticRAG's agentic tools"""
    try:
        logger.info(f"agentic_query for session {session_id}: Received agentic_rag param. Type: {type(agentic_rag)}, Value: {str(agentic_rag)[:100]}")

        if not isinstance(agentic_rag, AgenticRAGSystem):
            logger.error(f"CRITICAL: agentic_rag param in agentic_query is NOT an AgenticRAGSystem instance. Actual Type: {type(agentic_rag)}. Value: {str(agentic_rag)[:200]}")
            # Also log what session.get('agentic_rag') shows at this point for comparison
            session_rag_value = session.get("agentic_rag")
            logger.error(f"For comparison, session.get('agentic_rag') in agentic_query shows Type: {type(session_rag_value)}, Value: {str(session_rag_value)[:100]}")
            raise HTTPException(status_code=500, detail=f"Internal server error: AgenticRAG system received as incorrect type: {type(agentic_rag)}.")

        if not agentic_rag.agentic_explorer:
            logger.error(f"CRITICAL: agentic_rag IS an AgenticRAGSystem instance, but agentic_rag.agentic_explorer is None or Falsy. Explorer: {agentic_rag.agentic_explorer}. Session ID: {session_id}")
            raise HTTPException(status_code=500, detail="Internal server error: AgenticRAG explorer component not initialized.")
        
        logger.info(f"agentic_query for session {session_id}: agentic_rag is type {type(agentic_rag)} and agentic_explorer is type {type(agentic_rag.agentic_explorer)}. Proceeding.")

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
        await session_manager.add_message(session_id, "user", message.content)

        # Prepare query with enhanced context
        query_with_context = message.content
        if context_files:
            files_context = f"\n\nContext files mentioned: {', '.join(context_files)}"
            query_with_context = message.content + files_context
        
        # Add current issue context if available
        current_issue_context = session.get("metadata", {}).get("currentIssueContext")
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
                    await session_manager.add_message(
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
            await session_manager.add_message(
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

@router.post("/{session_id}/reset-agentic-memory")
async def reset_agentic_memory(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Reset the agentic system's memory for a session"""
    try:
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

@router.get("/{session_id}/agentic-rag-info")
async def get_agentic_rag_info(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get detailed information about the AgenticRAG system for a session"""
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            return {
                "session_id": session_id,
                "agentic_rag_enabled": False,
                "message": "AgenticRAG not initialized for this session"
            }
        
        repo_info = agentic_rag.get_repo_info()
        
        # NEW: Get composite retrieval information
        composite_stats = agentic_rag.get_composite_statistics()
        composite_enabled = agentic_rag._use_composite if hasattr(agentic_rag, '_use_composite') else False
        
        response = {
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
                "agentic_tool_integration": True,
                "composite_multi_index_retrieval": composite_enabled  # NEW
            },
            "processing_strategies": [
                "agentic_deep - for complex exploration and architecture queries",
                "agentic_focused - for implementation and analysis queries", 
                "agentic_light - for simple queries with minimal enhancement",
                "rag_only - for basic queries without agentic enhancement"
            ]
        }
        
        # NEW: Add composite retrieval details if available
        if composite_enabled and composite_stats:
            response["composite_retrieval"] = {
                "enabled": True,
                "statistics": composite_stats,
                "description": "LlamaIndex-style multi-index retrieval with intelligent routing"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting AgenticRAG info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/analyze-query")
async def analyze_query_endpoint(session_id: str, query: dict, session: Dict[str, Any] = Depends(get_session)):
    """Analyze a query to understand how AgenticRAG would process it"""
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        user_query = query.get("text", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Analyze the query - simple sync analysis
        word_count = len(user_query.split())
        query_lower = user_query.lower()
        
        # Determine complexity
        if word_count > 20:
            complexity = "complex"
        elif word_count > 10:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        # Check for agentic patterns
        agentic_patterns = [
            "explain", "analyze", "how does", "implement", "create", "find all",
            "comprehensive", "detailed", "step by step"
        ]
        should_use_agentic = any(pattern in query_lower for pattern in agentic_patterns)
        
        analysis = {
            "query_type": "general",
            "complexity": complexity,
            "should_use_agentic": should_use_agentic,
            "confidence": 0.7,
            "processing_time": 0.01,
            "complexity_score": min(word_count / 5, 10)  # Score out of 10
        }
        
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

@router.get("/{session_id}/context-preview")
async def get_context_preview(
    session_id: str, 
    query: str = Query(...), 
    max_sources: int = Query(5),
    session: Dict[str, Any] = Depends(get_session)
):
    """Get a preview of what context AgenticRAG would retrieve for a query"""
    try:
        agentic_rag = session.get("agentic_rag")
        if not agentic_rag:
            raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
        
        # Get enhanced context but limit sources for preview
        enhanced_context_list = await agentic_rag.get_enhanced_context(
            query, 
            restrict_files=None,
            use_agentic_tools=True
        )
        
        # Convert list to dict format for compatibility
        enhanced_context = {"sources": enhanced_context_list or []}
        
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
            "total_sources_available": len(enhanced_context_list or []),
            "sources_in_preview": len(preview_context.get("sources", []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting context preview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context preview: {str(e)}")

@router.get("/{session_id}/related-issues")
async def get_related_issues(
    session_id: str, 
    query: str = Query(..., description="Query to find similar issues for"),
    max_issues: int = Query(5, description="Maximum number of issues to return"),
    state: str = Query("all", description="Issue state filter: open, closed, all"),
    similarity_threshold: float = Query(0.7, description="Minimum similarity threshold"),
    session: Dict[str, Any] = Depends(get_session)
):
    """Get related GitHub issues for a query"""
    try:
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

@router.post("/{session_id}/index-issues")
async def index_repository_issues(
    session_id: str,
    force_rebuild: bool = Query(False, description="Force rebuild of issue index"),
    max_issues: int = Query(1000, description="Maximum number of issues to index"),
    max_prs: int = Query(1000, description="Maximum number of PRs to index"),
    session: Dict[str, Any] = Depends(get_session)
):
    """Index repository issues and PRs for RAG"""
    try:
        # Get repo info from session
        repo_url = session.get("repo_url")
        if not repo_url:
            raise HTTPException(status_code=400, detail="No repository URL found in session")
        
        # Parse repo owner/name from URL
        def parse_repo_url(repo_url: str) -> tuple[str, str]:
            """Extract owner and repository name from a GitHub URL."""
            # Remove .git if present
            repo_url = repo_url.replace(".git", "")
            # Split by / and get last two parts
            parts = repo_url.split("/")
            owner = parts[-2]
            repo = parts[-1]
            return owner, repo
        
        owner, repo = parse_repo_url(repo_url)
        
        # Initialize RAG with new parameters
        from ...issue_rag import IssueAwareRAG
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

@router.get("/{session_id}/issue-index-status")
async def get_issue_index_status(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    """Get the status of issue indexing for a session"""
    try:
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

# Chunk API endpoints
@router.get("/agentic/chunk/{chunk_id}")
async def get_chunk(chunk_id: str, chunk_store = Depends(get_chunk_store)):
    """Retrieve a chunk by ID"""
    content = chunk_store.retrieve(chunk_id)
    if not content:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"content": content}

@router.delete("/agentic/chunk/{chunk_id}")
async def delete_chunk(chunk_id: str, chunk_store = Depends(get_chunk_store)):
    """Delete a chunk by ID"""
    if chunk_store.delete(chunk_id):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chunk not found")

@router.get("/agentic/redis-health")
async def check_redis_health(chunk_store = Depends(get_chunk_store)):
    """Check Redis connection health"""
    try:
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

# General agentic features endpoint
@router.get("/agentic-rag-features")
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
