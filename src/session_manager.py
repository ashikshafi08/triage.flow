from uuid import UUID
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .github_client import GitHubIssueClient
from .new_rag import LocalRepoContextExtractor
from .agentic_rag import AgenticRAGSystem  # Import the new integrated system
from .repo_summarizer import RepositorySummarizer
from .config import settings
from .models import Issue, IssueComment  # Import the models
import asyncio
import os
import shutil
import json
import logging
import aiofiles
import asyncio
from src.issue_analysis.analyzer import analyse_issue  # new import

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.github_client = GitHubIssueClient()
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        
    def create_session(self, issue_url: str, prompt_type: str, llm_config: Optional[Any] = None) -> str:
        """Create a new session for issue analysis"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "type": "issue_analysis",
            "issue_url": issue_url,
            "prompt_type": prompt_type,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "conversation_history": [],
            "llm_config": llm_config,
            "status": "pending",
            "result": None
        }
        return session_id
    
    def create_repo_session(self, repo_url: str, initial_file: Optional[str] = None, session_name: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """Create a new session for repository-only chat"""
        session_id = str(uuid.uuid4())
        
        # Extract repo info from URL
        url_parts = repo_url.rstrip('/').split('/')
        owner = url_parts[-2] if len(url_parts) >= 2 else "unknown"
        repo = url_parts[-1].replace('.git', '') if url_parts else "unknown"
        
        # Generate session name if not provided
        if not session_name:
            session_name = f"{owner}/{repo}"
            if initial_file:
                session_name += f" - {os.path.basename(initial_file)}"
        
        # Create session storage paths
        session_storage_path = f"/tmp/triage_sessions/{session_id}"
        os.makedirs(session_storage_path, exist_ok=True)
        
        metadata = {
            "repo_url": repo_url,
            "owner": owner,
            "repo": repo,
            "session_name": session_name,
            "initial_file": initial_file,
            "storage_path": session_storage_path,
            "status": "initializing"
        }
        
        self.sessions[session_id] = {
            "id": session_id,
            "type": "repo_chat",
            "repo_url": repo_url,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "conversation_history": [],
            "metadata": metadata,
            "prompt_type": "chat",  # Default for repo chat
            "llm_config": None
        }
        
        return session_id, metadata
    
    async def initialize_repo_session(self, session_id: str) -> None:
        """Initialize repository context for a repo-only session using AgenticRAG"""
        session = self.sessions.get(session_id)
        if not session or session["type"] != "repo_chat":
            logger.error(f"Session {session_id} not found or not a repo_chat session for initialization.")
            return
        
        try:
            session["metadata"]["status"] = "cloning"
            session["metadata"]["message"] = "Cloning repository..."
            # TODO: Persist/notify UI if a mechanism exists to update status progressively

            agentic_rag = AgenticRAGSystem(session_id)
            session["agentic_rag"] = agentic_rag # Store early

            # Initialize core systems (blocking part of this background task)
            await agentic_rag.initialize_core_systems(session["repo_url"])
            
            # Update session with info from core systems
            session["repo_path"] = agentic_rag.get_repo_path()
            session["repo_context"] = { "repo_info": agentic_rag.get_repo_info() }
            session["agentic_enabled"] = True # Core agentic tools for code analysis are ready

            session["metadata"]["status"] = "core_ready"
            session["metadata"]["message"] = "Core repository indexed. Chat ready for code analysis. Issue context loading in background..."
            # TODO: Persist/notify UI

            # Kick off asynchronous initialization of IssueAwareRAG.
            # This task will run independently. AgenticRAGSystem.initialize_issue_rag_async
            # is responsible for updating the session's status upon its completion or failure,
            # as it receives the 'session' dictionary directly.
            asyncio.create_task(agentic_rag.initialize_issue_rag_async(session))
            logger.info(f"Session {session_id}: Kicked off background task for IssueAwareRAG initialization.")
            
            # Save metadata to disk for persistence (reflecting core_ready state)
            # The initialize_issue_rag_async will modify session["metadata"] further.
            # Subsequent status checks or a dedicated save mechanism would pick up those changes.
            metadata_path = os.path.join(session["metadata"]["storage_path"], "metadata.json")
            with open(metadata_path, 'w') as f:
                # Dump the entire session metadata as it stands at this point
                json.dump(session["metadata"], f, indent=2) 
                
        except Exception as e:
            logger.error(f"Error during core repository session initialization for {session_id}: {e}")
            session["metadata"]["status"] = "error"
            session["metadata"]["error"] = str(e)
            session["metadata"]["message"] = f"Failed to initialize core session systems: {str(e)}"
            # TODO: Persist/notify UI of error
            # Errors in this main background task are critical for core functionality.
            # Errors in initialize_issue_rag_async are handled within that task to set specific statuses.
    
    async def initialize_session_context(self, session_id: str) -> None:
        """Initialize session context based on session type"""
        session = self.sessions.get(session_id)
        if not session:
            return
            
        if session["type"] == "repo_chat":
            # This is now handled by the background task created in /assistant/sessions
            # No direct call here needed anymore as it's part of the main session init flow.
            # If called, it might re-trigger, which could be an issue or intended for re-init.
            # For now, let's assume it's mainly for the issue_analysis type.
            logger.info(f"Repo chat session {session_id} initialization is handled by create_assistant_session flow.")
            pass # Or decide if re-initialization logic is needed here.
        else: # issue_analysis type
            try:
                # Get issue data
                issue_response = await self.github_client.get_issue(session["issue_url"])
                if issue_response.status == "success" and issue_response.data:
                    session["issue_data"] = issue_response.data
                    
                    # Extract repo URL from issue URL
                    url_parts = session["issue_url"].split('/')
                    owner = url_parts[3]
                    repo_name_from_url = url_parts[4] # Renamed to avoid conflict
                    repo_url = f"https://github.com/{owner}/{repo_name_from_url}.git"
                    
                    # Initialize AgenticRAG system for the issue's repository
                    # This will set up core RAG and kick off issue_rag_async for this repo
                    agentic_rag = AgenticRAGSystem(session_id + "_issue_repo") # Unique ID for this instance
                    
                    await agentic_rag.initialize_core_systems(repo_url)
                    asyncio.create_task(agentic_rag.initialize_issue_rag_async(session)) # Pass main session for status updates
                                        
                    session["agentic_rag_for_issue_repo"] = agentic_rag # Store it separately
                    # session["repo_context"] will be populated by agentic_rag methods as needed
                    session["repo_path_for_issue_repo"] = agentic_rag.get_repo_path()
                    session["agentic_enabled"] = True
                    
                    logger.info(f"Context initialization for issue {session['issue_url']} started.")
                    
            except Exception as e:
                logger.error(f"Error initializing context for issue session {session_id}: {e}")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID and update last accessed time"""
        session = self.sessions.get(session_id)
        if session:
            session["last_accessed"] = datetime.now()
        return session
    
    def list_sessions(self, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by type"""
        sessions_list = []
        for session_id, session_data in self.sessions.items(): # Renamed session to session_data
            if session_type and session_data.get("type") != session_type:
                continue
                
            session_info = {
                "id": session_id,
                "type": session_data.get("type", "unknown"),
                "created_at": session_data["created_at"].isoformat(),
                "last_accessed": session_data["last_accessed"].isoformat(),
                "metadata": session_data.get("metadata", {}),
                "message_count": len(session_data.get("conversation_history", []))
            }
            
            if session_data["type"] == "repo_chat":
                session_info["repo_url"] = session_data.get("repo_url")
                session_info["session_name"] = session_data.get("metadata", {}).get("session_name")
            elif session_data["type"] == "issue_analysis": # Added elif for clarity
                session_info["issue_url"] = session_data.get("issue_url")
                session_info["prompt_type"] = session_data.get("prompt_type")
                
            sessions_list.append(session_info)
        
        sessions_list.sort(key=lambda x: x["last_accessed"], reverse=True)
        return sessions_list
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and clean up associated resources"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Clean up AgenticRAG resources (main one for repo_chat)
        if "agentic_rag" in session and session["agentic_rag"]:
            try:
                # AgenticRAGSystem.cleanup is async
                asyncio.create_task(session["agentic_rag"].cleanup())
            except Exception as e:
                logger.error(f"Error scheduling AgenticRAG cleanup for session {session_id}: {e}")

        # Clean up AgenticRAG for issue_analysis sessions
        if "agentic_rag_for_issue_repo" in session and session["agentic_rag_for_issue_repo"]:
            try:
                asyncio.create_task(session["agentic_rag_for_issue_repo"].cleanup())
            except Exception as e:
                logger.error(f"Error scheduling AgenticRAG (issue_repo) cleanup for session {session_id}: {e}")

        # Clean up FoundingMemberAgent resources
        if "founding_member_agent" in session and session["founding_member_agent"]:
            try:
                agent = session["founding_member_agent"]
                if hasattr(agent, 'explorer') and hasattr(agent.explorer, 'reset_memory'):
                    agent.explorer.reset_memory() # Assuming this is synchronous
            except Exception as e:
                logger.error(f"Error cleaning up FoundingMemberAgent for session {session_id}: {e}")
        
        # Clean up storage for repo sessions
        if session.get("type") == "repo_chat":
            storage_path = session.get("metadata", {}).get("storage_path")
            if storage_path and os.path.exists(storage_path):
                try:
                    shutil.rmtree(storage_path)
                    logger.info(f"Cleaned up session storage for {session_id} at {storage_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up session storage for {session_id} at {storage_path}: {e}")
        
        del self.sessions[session_id]
        logger.info(f"Deleted session {session_id} from memory.")
        return True
    
    def add_message(self, session_id: str, role: str, content: str = "", **kwargs) -> None:
        """Add a message to the conversation history, supporting extra fields."""
        session = self.sessions.get(session_id)
        if session:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            message.update(kwargs)
            session["conversation_history"].append(message)
            session["last_accessed"] = datetime.now()
    
    def cleanup_sessions(self) -> None:
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data["last_accessed"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            self.delete_session(session_id)

    # ------------------------------------------------------------------
    # Issue analysis orchestration (agentic pipeline)
    # ------------------------------------------------------------------
    async def _run_issue_analysis(self, session_id: str):
        """Internal helper that executes the analyse_issue pipeline and updates session."""
        session = self.sessions.get(session_id)
        if not session or session["type"] != "issue_analysis":
            logger.error("Session %s not found or not issue_analysis", session_id)
            return

        issue_url: str = session["issue_url"]
        try:
            session["status"] = "running"
            result = await analyse_issue(issue_url)
            session["status"] = result.get("status", "completed")
            session["result"] = result
        except Exception as exc:
            logger.exception("Issue analysis failed for session %s", session_id)
            session["status"] = "error"
            session["error"] = str(exc)

    def launch_issue_analysis(self, session_id: str) -> None:
        """Public method to start analysis in background using asyncio.create_task."""
        if session_id not in self.sessions:
            raise ValueError(f"Invalid session_id {session_id}")
        # Ensure not already running
        if self.sessions[session_id].get("status") in {"running", "completed"}:
            return
        # Kick off background task
        asyncio.create_task(self._run_issue_analysis(session_id))
