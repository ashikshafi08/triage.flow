import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .local_rag import LocalRepoContextExtractor
from .prompt_generator import PromptGenerator
from .github_client import GitHubIssueClient
from .models import IssueResponse, LLMConfig # Added LLMConfig

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        
    def create_session(self, issue_url: str, prompt_type: str, llm_config: LLMConfig) -> str: # Added llm_config parameter
        """Create a new session with initial context"""
        session_id = str(uuid.uuid4())
        
        # Initialize session data
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "issue_url": issue_url,
            "prompt_type": prompt_type,
            "llm_config": llm_config.model_dump(), # Store llm_config as dict
            "conversation_history": [],
            "repo_context": None,
            "issue_data": None,
            "rag_instance": None # Added rag_instance
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data and update last accessed time"""
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
        """Load repo context and issue data for a session"""
        session = self.get_session(session_id)
        if not session or session["repo_context"]:
            return
            
        try:
            # Extract repo info from URL
            url_parts = session["issue_url"].split('/')
            owner = url_parts[3]
            repo = url_parts[4]
            repo_url = f"https://github.com/{owner}/{repo}.git"
            
            # Load repository context
            rag = LocalRepoContextExtractor()
            await rag.load_repository(repo_url)
            
            github_client = GitHubIssueClient()
            issue_data = await github_client.get_issue(session["issue_url"])
            
            if isinstance(issue_data, IssueResponse) and issue_data.status == "success":
                context = await rag.get_issue_context(
                    issue_data.data.title,
                    issue_data.data.body
                )
                self.update_session(session_id, {
                    "repo_context": context,
                    "issue_data": issue_data.data,
                    "rag_instance": rag # Store the RAG instance
                })
        except Exception as e:
            print(f"Error initializing session context: {e}")

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        session = self.get_session(session_id)
        if session:
            session["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            })
            
    def cleanup_sessions(self, max_age_minutes: int = 60):
        """Remove inactive sessions"""
        now = datetime.now()
        expired = [sid for sid, session in self.sessions.items() 
                  if (now - session["last_accessed"]) > timedelta(minutes=max_age_minutes)]
        for sid in expired:
            del self.sessions[sid]
