from uuid import UUID
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from .github_client import GitHubIssueClient
from .new_rag import LocalRepoContextExtractor
from .repo_summarizer import RepositorySummarizer
from .config import settings
from .models import Issue, IssueComment  # Import the models
import asyncio

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.github_client = GitHubIssueClient()
        
    def create_session(self, issue_url: str, prompt_type: str, llm_config: Optional[Dict] = None) -> str:
        """Create a new session with issue data"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "id": session_id,
            "issue_url": issue_url,
            "prompt_type": prompt_type,
            "llm_config": llm_config,  # Store LLM config in session
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "conversation_history": [],
            "issue_data": None,
            "repo_context": None,
            "repo_overview": None,  # Add repo overview
            "rag_instance": None,
            "repo_path": None  # Store repo path for file access
        }
        
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID and update last accessed time"""
        session = self.sessions.get(session_id)
        if session:
            session["last_accessed"] = datetime.now()
        return session

    def update_session(self, session_id: str, update: Dict):
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id].update(update)
            self.sessions[session_id]["last_accessed"] = datetime.now()
            
    async def initialize_session_context(self, session_id: str):
        """Initialize session with issue data and repository context"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        # Fetch issue data using the async method
        issue_response = await self.github_client.get_issue(session["issue_url"])
        if issue_response.status != "success" or not issue_response.data:
            raise Exception(f"Failed to fetch issue: {issue_response.error}")
            
        # Store the Issue object directly, not a dictionary
        session["issue_data"] = issue_response.data
        
        # Get repository info from the issue data
        issue_info = self.github_client._extract_issue_info(session["issue_url"])
        if not issue_info:
            raise Exception("Invalid GitHub issue URL")
            
        owner, repo, _ = issue_info
        repo_url = f"https://github.com/{owner}/{repo}.git"
        branch = "main"  # Default branch assumption
        
        # Initialize RAG for repository context
        rag = LocalRepoContextExtractor()
        
        # Load repository
        await rag.load_repository(repo_url, branch)
        session["rag_instance"] = rag
        session["repo_path"] = rag.current_repo_path  # Store the repo path
        
        # Generate repository overview if enabled
        if settings.ENABLE_REPO_SUMMARIES:
            try:
                summarizer = RepositorySummarizer(rag.current_repo_path)
                overview = await summarizer.generate_repo_overview()
                session["repo_overview"] = overview
                print(f"Generated repo overview: {overview['total_files']} files, {len(overview['languages'])} languages")
            except Exception as e:
                print(f"Failed to generate repo overview: {e}")
                session["repo_overview"] = None
        
        # Get initial context
        issue_context = await rag.get_issue_context(
            issue_response.data.title,
            issue_response.data.body or ""
        )
        session["repo_context"] = issue_context
        
        self.sessions[session_id] = session

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session conversation history"""
        session = self.sessions.get(session_id)
        if session:
            session["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            session["last_accessed"] = datetime.now()

    def cleanup_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session["last_accessed"] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            print(f"Cleaned up {len(sessions_to_remove)} old sessions")
