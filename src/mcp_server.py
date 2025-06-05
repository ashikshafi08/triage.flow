from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import os
from datetime import datetime
import asyncio
from .models import ChatMessage
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .session_manager import SessionManager
from .new_rag import LocalRepoContextExtractor
from .config import settings

class TriageFlowMCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="triage.flow MCP Server",
            description="Managed Cloud Platform for Repository Intelligence",
            version="1.0.0"
        )
        
        # Initialize core components
        self.github_client = GitHubIssueClient()
        self.llm_client = LLMClient()
        self.session_manager = SessionManager()
        self.repo_extractor = LocalRepoContextExtractor()
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS.split(","),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register MCP API routes"""
        
        @self.app.post("/mcp/repositories")
        async def load_repository(repo_url: str, branch: str = "main"):
            """Load a repository into the MCP server"""
            try:
                repo_context = await self.repo_extractor.get_or_load_repository(repo_url, branch)
                return {
                    "status": "success",
                    "repo_info": repo_context
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/repositories")
        async def list_repositories():
            """List all loaded repositories"""
            try:
                repos = self.repo_extractor.list_loaded_repositories()
                return [{"repo_url": repo.url, "owner": repo.owner, "repo": repo.name} for repo in repos]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/query")
        async def query_repository(
            repo_url: str,
            query: str,
            session_id: Optional[str] = None,
            model: Optional[str] = None
        ):
            """Query a repository with natural language"""
            try:
                # Get or create session
                if not session_id:
                    session_id = self.session_manager.create_session("repo_chat", {"repo_url": repo_url})
                
                # Get repository context
                repo_context = await self.repo_extractor.get_or_load_repository(repo_url)
                
                # Process query
                response = await self.llm_client.process_prompt(
                    prompt=query,
                    prompt_type="explore",
                    context=repo_context,
                    model=model
                )
                
                # Add to session history
                self.session_manager.add_message(session_id, "user", query)
                self.session_manager.add_message(session_id, "assistant", response.prompt)
                
                return {
                    "response": response.prompt,
                    "session_id": session_id,
                    "model_used": response.model_used
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/sessions/{session_id}")
        async def get_session_history(session_id: str):
            """Get conversation history for a session"""
            try:
                session = self.session_manager.get_session(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                return session["conversation_history"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

# Create and export the MCP server instance
mcp_server = TriageFlowMCPServer()
app = mcp_server.app 