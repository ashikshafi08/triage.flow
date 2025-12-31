"""Session management with Redis/in-memory fallback - tinygrad-style (537→300 lines)"""
import uuid, os, shutil, json, logging, asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from .github_client import GitHubIssueClient
from .agentic_rag import AgenticRAGSystem
from .config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions with Redis persistence and in-memory fallback."""

    def __init__(self):
        self.github_client = GitHubIssueClient()
        self.session_timeout = timedelta(hours=24)
        self.sessions_cache = None
        self.use_redis = False
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._redis_init_attempted = False

    # ============================================================================
    # Storage Layer (Redis with in-memory fallback)
    # ============================================================================

    async def _ensure_redis_initialized(self):
        """Lazy Redis initialization."""
        if self._redis_init_attempted: return
        self._redis_init_attempted = True

        try:
            from .cache.redis_cache_manager import EnhancedCacheManager
            self.sessions_cache = EnhancedCacheManager(namespace="sessions", default_ttl=86400)
            await self.sessions_cache.redis.initialize()
            self.use_redis = self.sessions_cache.redis.initialized
            logger.info(f"SessionManager: Redis {'enabled' if self.use_redis else 'not ready, using in-memory'}")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.use_redis = False

    def _convert_datetimes(self, data: Dict, to_iso: bool = False) -> Dict:
        """Convert datetime fields to/from ISO strings."""
        result = data.copy()
        for field in ["created_at", "last_accessed"]:
            if field in result:
                if to_iso and isinstance(result[field], datetime):
                    result[field] = result[field].isoformat()
                elif not to_iso and isinstance(result[field], str):
                    result[field] = datetime.fromisoformat(result[field])
                elif not to_iso and not isinstance(result[field], datetime):
                    result[field] = datetime.now()
        return result

    async def _get_session_from_storage(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from storage."""
        await self._ensure_redis_initialized()
        if self.use_redis:
            if (data := await self.sessions_cache.get(session_id)):
                return self._convert_datetimes(data, to_iso=False)
            return None
        return self.sessions.get(session_id)

    async def _store_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Store session to storage."""
        await self._ensure_redis_initialized()
        if self.use_redis:
            storage = self._convert_datetimes(session_data, to_iso=True)
            # Remove non-serializable objects
            for key in ["agentic_rag", "agentic_rag_for_issue_repo", "founding_member_agent", "_code_rag", "_issue_rag"]:
                if key in storage:
                    if key == "agentic_rag" and hasattr(storage[key], 'repo_info'):
                        storage[f"{key}_metadata"] = {"type": "AgenticRAGSystem", "repo_info": storage[key].repo_info, "initialized": True}
                    del storage[key]
            await self.sessions_cache.set(session_id, storage)
        else:
            self.sessions[session_id] = session_data

    async def _list_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions."""
        await self._ensure_redis_initialized()
        if not self.use_redis: return self.sessions

        import redis.asyncio as redis
        client = redis.from_url(settings.redis_url)
        try:
            keys = await client.keys(f"{self.sessions_cache.namespace}:*")
            sessions = {}
            for key in keys:
                sid = key.decode().split(":", 1)[1]
                if (data := await self._get_session_from_storage(sid)): sessions[sid] = data
            return sessions
        finally:
            await client.close()

    async def _delete_from_storage(self, session_id: str) -> bool:
        """Delete session from storage."""
        await self._ensure_redis_initialized()
        if self.use_redis: return await self.sessions_cache.delete(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    # ============================================================================
    # Session Creation
    # ============================================================================

    async def create_repo_session(self, repo_url: str, initial_file: Optional[str] = None,
                                   session_name: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """Create session for repository chat."""
        session_id = str(uuid.uuid4())
        url_parts = repo_url.rstrip('/').split('/')
        owner, repo = (url_parts[-2] if len(url_parts) >= 2 else "unknown"), url_parts[-1].replace('.git', '')

        if not session_name:
            session_name = f"{owner}/{repo}" + (f" - {os.path.basename(initial_file)}" if initial_file else "")

        storage_path = f"/tmp/triage_sessions/{session_id}"
        os.makedirs(storage_path, exist_ok=True)

        metadata = {"repo_url": repo_url, "owner": owner, "repo": repo, "session_name": session_name,
                    "initial_file": initial_file, "storage_path": storage_path, "status": "initializing"}

        await self._store_session(session_id, {
            "id": session_id, "type": "repo_chat", "repo_url": repo_url,
            "created_at": datetime.now(), "last_accessed": datetime.now(),
            "conversation_history": [], "metadata": metadata, "prompt_type": "chat", "llm_config": None
        })
        return session_id, metadata

    # ============================================================================
    # Session Initialization
    # ============================================================================

    async def initialize_repo_session(self, session_id: str) -> None:
        """Initialize repository context for repo-chat session."""
        session = await self._get_session_from_storage(session_id)
        if not session or session["type"] != "repo_chat": return

        try:
            owner, repo = session["metadata"]["owner"], session["metadata"]["repo"]
            repo_key = f"{owner}/{repo}"

            # Check cache first
            from .api.dependencies import agentic_rag_cache
            if repo_key in agentic_rag_cache:
                logger.info(f"Reusing cached AgenticRAG for {repo_key}")
                rag = agentic_rag_cache[repo_key]
                session.update({"agentic_rag": rag, "repo_path": rag.repo_path,
                               "repo_context": {"repo_info": rag.repo_info}, "agentic_enabled": True})
                session["metadata"].update({"status": "ready", "message": "Repository loaded from cache.",
                                           "issue_rag_ready": rag.issue_rag is not None})
                await self._store_session(session_id, session)
                return

            # Fresh initialization
            session["metadata"].update({"status": "cloning", "message": "Cloning repository..."})
            await self._store_session(session_id, session)

            agentic_rag = AgenticRAGSystem(repo_key)
            await agentic_rag.initialize_core_systems(session["repo_url"])

            session.update({"agentic_rag": agentic_rag, "repo_path": agentic_rag.get_repo_path(),
                           "repo_context": {"repo_info": agentic_rag.get_repo_info()}, "agentic_enabled": True})
            session["metadata"].update({"status": "core_ready",
                                       "message": "Core repository indexed. Issue context loading..."})
            agentic_rag_cache[repo_key] = agentic_rag
            await self._store_session(session_id, session)

            # Background issue RAG init
            try:
                asyncio.create_task(agentic_rag.initialize_issue_rag_async(session))
            except Exception as e:
                logger.warning(f"Failed to start issue RAG init: {e}")
                session["metadata"].update({"status": "warning_issue_rag_failed",
                                           "message": "Core ready. Issue context failed.", "issue_rag_ready": False})
                await self._store_session(session_id, session)

            # Save metadata to disk
            try:
                with open(os.path.join(session["metadata"]["storage_path"], "metadata.json"), 'w') as f:
                    json.dump(session["metadata"], f, indent=2)
            except Exception as e: logger.warning(f"Failed to save metadata: {e}")

        except Exception as e:
            logger.error(f"Error initializing repo session {session_id}: {e}")
            session["metadata"].update({"status": "error", "error": str(e), "message": f"Failed: {e}"})
            session.pop("agentic_rag", None)
            session["agentic_enabled"] = False
            await self._store_session(session_id, session)

    # ============================================================================
    # Session Operations
    # ============================================================================

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        if (session := await self._get_session_from_storage(session_id)):
            session["last_accessed"] = datetime.now()
        return session

    async def update_session_last_accessed_and_store(self, session_id: str):
        """Update and persist last_accessed time."""
        if (session := await self._get_session_from_storage(session_id)):
            session["last_accessed"] = datetime.now()
            await self._store_session(session_id, session)

    async def list_sessions(self, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by type."""
        sessions_list = []
        for session_id, data in (await self._list_all_sessions()).items():
            if session_type and data.get("type") != session_type: continue

            info = {"id": session_id, "type": data.get("type", "unknown"),
                    "created_at": data["created_at"].isoformat(), "last_accessed": data["last_accessed"].isoformat(),
                    "metadata": data.get("metadata", {}), "message_count": len(data.get("conversation_history", []))}

            if data["type"] == "repo_chat":
                info.update({"repo_url": data.get("repo_url"), "session_name": data.get("metadata", {}).get("session_name")})
            elif data["type"] == "issue_analysis":
                info.update({"issue_url": data.get("issue_url"), "prompt_type": data.get("prompt_type")})
            sessions_list.append(info)

        return sorted(sessions_list, key=lambda x: x["last_accessed"], reverse=True)

    async def delete_session(self, session_id: str) -> bool:
        """Delete session and clean up resources."""
        session = await self._get_session_from_storage(session_id)
        if not session: return False

        # Cleanup AgenticRAG instances
        for key in ["agentic_rag", "agentic_rag_for_issue_repo"]:
            if (rag := session.get(key)) and isinstance(rag, AgenticRAGSystem):
                try: asyncio.create_task(rag.cleanup())
                except Exception as e: logger.error(f"Error scheduling {key} cleanup: {e}")

        # Cleanup FoundingMemberAgent
        if (agent := session.get("founding_member_agent")) and hasattr(agent, 'explorer'):
            try:
                if hasattr(agent.explorer, 'reset_memory'): agent.explorer.reset_memory()
            except Exception as e: logger.error(f"Error cleaning up FoundingMemberAgent: {e}")

        # Cleanup cache
        try:
            from .api.dependencies import agentic_rag_cache
            if session_id in agentic_rag_cache:
                del agentic_rag_cache[session_id]
        except Exception as e: logger.debug(f"Cache cleanup skipped: {e}")

        # Cleanup storage for repo sessions
        if session.get("type") == "repo_chat":
            if (path := session.get("metadata", {}).get("storage_path")) and os.path.exists(path):
                try: shutil.rmtree(path)
                except Exception as e: logger.error(f"Error cleaning up storage at {path}: {e}")

        await self._delete_from_storage(session_id)
        logger.info(f"Deleted session {session_id}")
        return True

    async def add_message(self, session_id: str, role: str, content: str = "", **kwargs) -> None:
        """Add message to conversation history."""
        if (session := await self._get_session_from_storage(session_id)):
            session["conversation_history"].append({"role": role, "content": content,
                                                    "timestamp": datetime.now().isoformat(), **kwargs})
            session["last_accessed"] = datetime.now()
            await self._store_session(session_id, session)

    async def cleanup_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.now()
        for session_id, data in (await self._list_all_sessions()).items():
            if now - data["last_accessed"] > self.session_timeout:
                logger.info(f"Cleaning up expired session: {session_id}")
                await self.delete_session(session_id)
