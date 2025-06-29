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
# Changed to use repository-based keys (e.g., "apache_airflow") instead of session IDs
agentic_rag_cache: Dict[str, Any] = {}

# Dependency to get session
async def get_session(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Dependency to get agentic RAG
async def get_agentic_rag(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    current_agentic_rag_value = session.get("agentic_rag")
    
    # Extract repository info for cache key
    repo_info = session.get("repo_context", {}).get("repo_info", {})
    owner = repo_info.get('owner') or session.get("metadata", {}).get("owner")
    repo = repo_info.get('repo') or session.get("metadata", {}).get("repo")
    
    if not owner or not repo:
        raise HTTPException(status_code=400, detail="Repository information not available in session")
    
    # Use repository-based cache key for sharing AgenticRAG instances across sessions
    repo_key = f"{owner}/{repo}"
    
    # Get current repository path for validation
    current_repo_path = session.get("repo_path")
    if not current_repo_path:
        raise HTTPException(status_code=400, detail="Repository not initialized for this session")
    
    logger.info(f"[GET_AGENTIC_RAG] session_id={session_id}, repo_key={repo_key}, repo_path={current_repo_path}")
    
    # Check repository-based cache first, but validate the repo path matches
    if repo_key in agentic_rag_cache:
        cached_instance = agentic_rag_cache[repo_key]
        cached_repo_path = getattr(cached_instance, 'repo_path', None)
        
        # Validate that the cached instance is for the same repository path
        if cached_repo_path and cached_repo_path == current_repo_path:
            logger.info(f"[CACHE_HIT] Using cached AgenticRAG for {repo_key} with matching repo path")
            return cached_instance
        else:
            logger.warning(f"[CACHE_MISMATCH] Cached instance for {repo_key} has different repo path: {cached_repo_path} vs {current_repo_path}")
            # Remove the invalid cached instance
            del agentic_rag_cache[repo_key]

    # If we have a valid AgenticRAG instance in session, validate and cache it
    if current_agentic_rag_value and not isinstance(current_agentic_rag_value, str):
        session_instance_repo_path = getattr(current_agentic_rag_value, 'repo_path', None)
        
        # Only cache if the session instance matches the current repository path
        if session_instance_repo_path == current_repo_path:
            logger.info(f"[CACHE_STORE] Caching existing AgenticRAG instance for {repo_key}")
            agentic_rag_cache[repo_key] = current_agentic_rag_value
            return current_agentic_rag_value
        else:
            logger.warning(f"[SESSION_MISMATCH] Session AgenticRAG instance has different repo path: {session_instance_repo_path} vs {current_repo_path}")
            # Don't cache the mismatched instance
    
    logger.info(f"[RECREATE] Creating new AgenticRAG instance for {repo_key}")
    
    try:
        from ..agentic_rag import AgenticRAGSystem
        
        # Create new instance using repo_key for consistent caching
        recreated_agentic_rag = AgenticRAGSystem(repo_key)
        recreated_agentic_rag.repo_path = session["repo_path"]
        recreated_agentic_rag.repo_info = {"owner": owner, "repo": repo}
        
        # Initialize the RAG extractor (core RAG component)
        from ..new_rag import LocalRepoContextExtractor
        rag_extractor = LocalRepoContextExtractor()
        rag_extractor.current_repo_path = session["repo_path"]
        recreated_agentic_rag.rag_extractor = rag_extractor
        
        # Initialize code explorer with existing indexes if available
        from ..agent_tools import AgenticCodebaseExplorer
        recreated_agentic_rag.agentic_explorer = AgenticCodebaseExplorer(
            repo_key,
            session["repo_path"], 
            issue_rag_system=None 
        )
        
        # Initialize composite retriever after core systems are ready
        logger.info(f"Initializing composite retriever for {repo_key}")
        await recreated_agentic_rag._initialize_composite_retriever()
        
        # Initialize commit index (should load from existing cache if available)
        try:
            logger.info(f"Loading commit index for {repo_key}")
            await recreated_agentic_rag.agentic_explorer.initialize_commit_index(force_rebuild=False)
            stats = recreated_agentic_rag.agentic_explorer.commit_index_manager.get_statistics()
            logger.info(f"Commit index loaded for {repo_key}: {stats}")
        except Exception as e_commit:
            logger.error(f"Failed to load commit index for {repo_key}: {e_commit}")
            # Continue without commit index

        # Restore issue RAG if it was previously initialized
        try:
            if session.get("metadata", {}).get("issue_rag_ready"):
                logger.info(f"Restoring issue RAG for {repo_key}")
                from ..issue_rag import IssueAwareRAG
                
                # Create a fresh IssueAwareRAG instance to avoid coroutine reuse issues
                issue_rag = IssueAwareRAG(owner, repo)
                # Load existing index without rebuild
                await issue_rag.initialize(force_rebuild=False)
                
                if issue_rag.is_initialized():
                    recreated_agentic_rag.issue_rag = issue_rag
                    recreated_agentic_rag.agentic_explorer.issue_rag_system = issue_rag
                    
                    # Update sub-components
                    if hasattr(recreated_agentic_rag.agentic_explorer, 'pr_ops'):
                        recreated_agentic_rag.agentic_explorer.pr_ops.issue_rag_system = issue_rag
                    if hasattr(recreated_agentic_rag.agentic_explorer, 'issue_ops'):
                        recreated_agentic_rag.agentic_explorer.issue_ops.issue_rag_system = issue_rag
                    
                    # Update composite retriever with issue RAG
                    if recreated_agentic_rag._use_composite and recreated_agentic_rag.composite_retriever:
                        recreated_agentic_rag.composite_retriever.indices["issues"] = issue_rag
                        logger.info(f"Updated composite retriever with issue RAG for {repo_key}")
                        
                    logger.info(f"Issue RAG restored successfully for {repo_key}")
                else:
                    logger.warning(f"Issue RAG failed to initialize for {repo_key}")
        except Exception as e_issue_rag:
            logger.warning(f"Failed to restore issue RAG for {repo_key}: {e_issue_rag}")

        # Cache the new instance using repository key
        agentic_rag_cache[repo_key] = recreated_agentic_rag
        
        logger.info(f"[RECREATE_SUCCESS] AgenticRAG created and cached for {repo_key}")
        return recreated_agentic_rag
        
    except Exception as e:
        logger.error(f"Failed to recreate AgenticRAG for {repo_key}: {e}")
        raise HTTPException(status_code=500, detail=f"AgenticRAG system recreation failed: {str(e)}")

# Dependency to get chunk store
def get_chunk_store():
    return ChunkStoreFactory.get_instance()
