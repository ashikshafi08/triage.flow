from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Issue(BaseModel):
    number: int
    title: str
    body: str
    state: str
    created_at: datetime
    url: str
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)

class IssueResponse(BaseModel):
    status: str
    data: Optional[Issue] = None
    error: Optional[str] = None

class PromptRequest(BaseModel):
    issue_url: str
    prompt_type: str = Field(..., description="Type of prompt to generate (explain, fix, test, summarize)")
    model: str = Field(..., description="LLM model to use (gpt-4, claude, etc.)")
    context: Optional[dict] = Field(default_factory=dict)

class PromptResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    error: Optional[str] = None 