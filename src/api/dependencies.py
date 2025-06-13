from typing import Dict, Any
from fastapi import HTTPException, Depends
from ..github_client import GitHubIssueClient
from ..llm_client import LLMClient
from ..prompt_generator import PromptGenerator
from ..session_manager import SessionManager
from ..conversation_memory import ConversationContextManager
from ..config import settings
from ..chunk_store import ChunkStoreFactory
import logging

logger = logging.getLogger(__name__)

# Initialize shared clients and services
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()
session_manager = SessionManager()
conversation_memory = ConversationContextManager(max_context_tokens=8000)

# In-memory cache for AgenticRAG instances to avoid recreation on every request
agentic_rag_cache: Dict[str, Any] = {}

# Dependency to get session
async def get_session(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Dependency to get agentic RAG
async def get_agentic_rag(session_id: str, session: Dict[str, Any] = Depends(get_session)): # session is now injected
    current_agentic_rag_value = session.get("agentic_rag")
    logger.critical(f"[GET_AGENTIC_RAG ENTRY] session_id={session_id}, initial agentic_rag type={type(current_agentic_rag_value)}, value='{str(current_agentic_rag_value)[:60]}'")

    # Use repository-based cache key instead of session-based
    repo_info = session.get("repo_context", {}).get("repo_info", {})
    repo_key = f"{repo_info.get('owner', 'unknown')}_{repo_info.get('repo', 'unknown')}"
    
    # Check if we have a cached instance first (use repo-based key for sharing across sessions)
    if repo_key != "unknown_unknown" and repo_key in agentic_rag_cache:
        cached_instance = agentic_rag_cache[repo_key]
        logger.critical(f"[GET_AGENTIC_RAG CACHE_HIT] session_id={session_id}, repo_key={repo_key}, returning cached instance type={type(cached_instance)}")
        return cached_instance

    if not current_agentic_rag_value or isinstance(current_agentic_rag_value, str):
        if session.get("repo_path") and session.get("type") == "repo_chat":
            logger.critical(f"[GET_AGENTIC_RAG RECREATE_START] session_id={session_id}, repo_key={repo_key}, placeholder_value='{current_agentic_rag_value}'")
            try:
                from ..agentic_rag import AgenticRAGSystem # Ensure import is within try if it's heavy
                
                recreated_agentic_rag = AgenticRAGSystem(repo_key)  # Use repo key instead of session_id
                recreated_agentic_rag.repo_path = session["repo_path"]
                recreated_agentic_rag.repo_info = repo_info
                
                from ..agent_tools import AgenticCodebaseExplorer # Ensure import is within try
                recreated_agentic_rag.agentic_explorer = AgenticCodebaseExplorer(
                    repo_key,  # Use repo key instead of session_id
                    session["repo_path"], 
                    issue_rag_system=None 
                )
                
                try:
                    logger.info(f"Initializing commit index for recreated AgenticRAG with repo key {repo_key}")
                    await recreated_agentic_rag.agentic_explorer.initialize_commit_index(force_rebuild=False)
                    logger.info(f"Commit index initialized for repo key {repo_key}")
                except Exception as e_commit:
                    logger.error(f"[GET_AGENTIC_RAG RECREATE_COMMIT_FAIL] repo_key={repo_key}, error: {e_commit}")
                    # Allow continuation, some tools might not work if commit index fails

                # Try to restore issue RAG system if it was previously available
                try:
                    if session.get("metadata", {}).get("issue_rag_ready"):
                        logger.info(f"Attempting to restore issue RAG system for repo key {repo_key}")
                        from ..issue_rag import IssueAwareRAG
                        
                        # Initialize issue RAG system
                        owner = repo_info.get('owner')
                        repo_name = repo_info.get('repo')
                        
                        if owner and repo_name:
                            issue_rag = IssueAwareRAG(owner, repo_name)
                            await issue_rag.initialize(force_rebuild=False)
                            
                            recreated_agentic_rag.issue_rag = issue_rag
                            recreated_agentic_rag.agentic_explorer.issue_rag_system = issue_rag
                            
                            # Update the sub-components that depend on issue_rag_system
                            if hasattr(recreated_agentic_rag.agentic_explorer, 'pr_ops'):
                                recreated_agentic_rag.agentic_explorer.pr_ops.issue_rag_system = issue_rag
                            if hasattr(recreated_agentic_rag.agentic_explorer, 'issue_ops'):
                                recreated_agentic_rag.agentic_explorer.issue_ops.issue_rag_system = issue_rag
                                
                            logger.info(f"Issue RAG system restored successfully for repo key {repo_key}")
                        else:
                            logger.warning(f"Cannot restore issue RAG: missing owner/repo info for {repo_key}")
                    else:
                        logger.info(f"Issue RAG not previously ready for {repo_key}, skipping restoration")
                except Exception as e_issue_rag:
                    logger.warning(f"Failed to restore issue RAG for repo key {repo_key}: {e_issue_rag}")
                    # Continue without issue RAG - basic functionality will still work

                # Cache the recreated instance for future requests (use repo-based key for sharing)
                agentic_rag_cache[repo_key] = recreated_agentic_rag
                
                # This updates the 'session' dictionary that was injected by FastAPI.
                # This modified 'session' dictionary is local to this request's scope.
                # The 'agentic_rag' object itself is what's returned by this dependency.
                session["agentic_rag"] = recreated_agentic_rag 
                
                logger.critical(f"[GET_AGENTIC_RAG RECREATE_SUCCESS] session_id={session_id}, repo_key={repo_key}, RETURNING RECREATED INSTANCE type={type(recreated_agentic_rag)}")
                return recreated_agentic_rag # Return the new instance directly
                
            except Exception as e_recreate:
                logger.error(f"[GET_AGENTIC_RAG RECREATE_FAIL_GENERAL] session_id={session_id}, error: {e_recreate}")
                raise HTTPException(status_code=500, detail=f"AgenticRAG system recreation failed: {str(e_recreate)}")
        else:
            conditions_not_met_msg = f"AgenticRAG is '{str(current_agentic_rag_value)[:60]}' but conditions not met for recreation. Repo path: {session.get('repo_path')}, Type: {session.get('type')}"
            logger.critical(f"[GET_AGENTIC_RAG NO_RECREATE_CONDITIONS] session_id={session_id}, {conditions_not_met_msg}")
            raise HTTPException(status_code=400, detail=f"AgenticRAG not initialized and cannot be recreated: {conditions_not_met_msg}")
    
    # If current_agentic_rag_value was already a valid instance, cache it for future requests
    if repo_key != "unknown_unknown":
        agentic_rag_cache[repo_key] = current_agentic_rag_value
    logger.critical(f"[GET_AGENTIC_RAG RETURN_EXISTING] session_id={session_id}, repo_key={repo_key}, type={type(current_agentic_rag_value)}")
    return current_agentic_rag_value

# Dependency to get chunk store
def get_chunk_store():
    return ChunkStoreFactory.get_instance()
