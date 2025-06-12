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

# Global storage for agentic explorers (in production, use proper session management)
agentic_explorers: Dict[str, Any] = {}

# Dependency to get session
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Dependency to get agentic RAG
async def get_agentic_rag(session_id: str):
    session = await get_session(session_id)
    agentic_rag = session.get("agentic_rag")
    if not agentic_rag:
        raise HTTPException(status_code=400, detail="AgenticRAG not initialized")
    return agentic_rag

# Dependency to get chunk store
def get_chunk_store():
    return ChunkStoreFactory.get_instance() 