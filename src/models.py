from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class IssueComment(BaseModel):
    body: str
    user: str
    created_at: datetime

class Issue(BaseModel):
    number: int
    title: str
    body: str
    state: str
    created_at: datetime
    url: str
    labels: List[str] = []
    assignees: List[str] = []
    comments: List[IssueComment] = []

class IssueResponse(BaseModel):
    status: str
    data: Optional[Issue] = None
    error: Optional[str] = None

class LLMConfig(BaseModel):
    provider: str
    name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    additional_params: Dict[str, Any] = {}

class PromptRequest(BaseModel):
    issue_url: str
    prompt_type: str
    llm_config: LLMConfig
    context: Dict[str, Any] = {}

class PromptResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class SessionResponse(BaseModel):
    session_id: str
    initial_message: str

class RepoRequest(BaseModel):
    """Request model for creating repository-only chat sessions"""
    repo_url: str = Field(..., description="GitHub repository URL")
    initial_file: Optional[str] = Field(None, description="Initial file to focus on")
    session_name: Optional[str] = Field(None, description="Custom session name")

class RepoSessionResponse(BaseModel):
    """Response model for repository session creation"""
    session_id: str
    repo_metadata: Dict[str, Any]
    status: str
    message: Optional[str] = None

class SessionListResponse(BaseModel):
    """Response model for listing user sessions"""
    sessions: List[Dict[str, Any]]
    total: int
