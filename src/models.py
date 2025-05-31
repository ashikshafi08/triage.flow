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
