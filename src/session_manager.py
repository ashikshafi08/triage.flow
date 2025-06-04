from uuid import UUID
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .github_client import GitHubIssueClient
from .new_rag import LocalRepoContextExtractor
from .repo_summarizer import RepositorySummarizer
from .config import settings
from .models import Issue, IssueComment  # Import the models
import asyncio
import os
import shutil
import json

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
            "llm_config": llm_config
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
        """Initialize repository context for a repo-only session"""
        session = self.sessions.get(session_id)
        if not session or session["type"] != "repo_chat":
            return
        
        try:
            # Update status
            session["metadata"]["status"] = "cloning"
            
            # Initialize RAG extractor
            rag_extractor = LocalRepoContextExtractor()
            
            # Load repository
            await rag_extractor.load_repository(session["repo_url"])
            
            # Update session with RAG instance and repo path
            session["rag_instance"] = rag_extractor
            session["repo_path"] = rag_extractor.current_repo_path
            session["repo_context"] = {
                "repo_info": {
                    "owner": session["metadata"]["owner"],
                    "repo": session["metadata"]["repo"],
                    "url": session["repo_url"],
                    "languages": rag_extractor.repo_info.get("languages", {}) if hasattr(rag_extractor, 'repo_info') else {}
                }
            }
            
            # Update metadata
            session["metadata"]["status"] = "ready"
            session["metadata"]["repo_path"] = rag_extractor.current_repo_path
            
            # Save metadata to disk for persistence
            metadata_path = os.path.join(session["metadata"]["storage_path"], "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(session["metadata"], f)
                
        except Exception as e:
            session["metadata"]["status"] = "error"
            session["metadata"]["error"] = str(e)
            raise
    
    async def initialize_session_context(self, session_id: str) -> None:
        """Initialize session context based on session type"""
        session = self.sessions.get(session_id)
        if not session:
            return
            
        if session["type"] == "repo_chat":
            await self.initialize_repo_session(session_id)
        else:
            # Original issue-based initialization
            try:
                # Get issue data
                issue_response = await self.github_client.get_issue(session["issue_url"])
                if issue_response.status == "success":
                    session["issue_data"] = issue_response.data
                    
                    # Extract repo URL from issue URL
                    url_parts = session["issue_url"].split('/')
                    owner = url_parts[3]
                    repo = url_parts[4]
                    repo_url = f"https://github.com/{owner}/{repo}.git"
                    
                    # Initialize RAG extractor
                    rag_extractor = LocalRepoContextExtractor()
                    await rag_extractor.load_repository(repo_url)
                    
                    # Get issue context
                    context = await rag_extractor.get_issue_context(
                        issue_response.data.title,
                        issue_response.data.body
                    )
                    
                    session["rag_instance"] = rag_extractor
                    session["repo_context"] = context
                    session["repo_path"] = rag_extractor.current_repo_path
                    
            except Exception as e:
                print(f"Error initializing session context: {e}")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID and update last accessed time"""
        session = self.sessions.get(session_id)
        if session:
            session["last_accessed"] = datetime.now()
        return session
    
    def list_sessions(self, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by type"""
        sessions_list = []
        for session_id, session in self.sessions.items():
            if session_type and session.get("type") != session_type:
                continue
                
            session_info = {
                "id": session_id,
                "type": session.get("type", "unknown"),
                "created_at": session["created_at"].isoformat(),
                "last_accessed": session["last_accessed"].isoformat(),
                "metadata": session.get("metadata", {}),
                "message_count": len(session.get("conversation_history", []))
            }
            
            # Add type-specific info
            if session["type"] == "repo_chat":
                session_info["repo_url"] = session.get("repo_url")
                session_info["session_name"] = session.get("metadata", {}).get("session_name")
            else:
                session_info["issue_url"] = session.get("issue_url")
                session_info["prompt_type"] = session.get("prompt_type")
                
            sessions_list.append(session_info)
        
        # Sort by last accessed, most recent first
        sessions_list.sort(key=lambda x: x["last_accessed"], reverse=True)
        return sessions_list
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and clean up associated resources"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Clean up storage for repo sessions
        if session.get("type") == "repo_chat":
            storage_path = session.get("metadata", {}).get("storage_path")
            if storage_path and os.path.exists(storage_path):
                try:
                    shutil.rmtree(storage_path)
                except Exception as e:
                    print(f"Error cleaning up session storage: {e}")
        
        # Remove from memory
        del self.sessions[session_id]
        return True
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        session = self.sessions.get(session_id)
        if session:
            session["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            session["last_accessed"] = datetime.now()
    
    def cleanup_sessions(self) -> None:
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_accessed"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
            print(f"Cleaned up expired session: {session_id}")
