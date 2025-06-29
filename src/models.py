from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from .patch_linkage import DiffDoc
import asyncio

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
    closed_at: Optional[datetime] = None # Add closed_at field
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
    context: Optional[Dict[str, Any]] = None
    llm_config: Optional[Any] = None

class IssueContextRequest(BaseModel):
    query: str
    repo_url: str
    max_issues: Optional[int] = 5
    include_patches: Optional[bool] = True

class PromptResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str
    context_files: Optional[List[str]] = Field(None, description="List of file paths mentioned in the message")

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

class IssueDoc(BaseModel):
    id: int
    state: str
    title: str
    body: str
    comments: List[str]
    labels: List[str]
    created_at: str
    closed_at: Optional[str]
    patch_url: Optional[str]
    repo: str
    # Enhanced fields for issue closing tracking
    closed_by_commit: Optional[str] = None  # SHA of closing commit
    closed_by_pr: Optional[int] = None      # PR number that closed it
    closed_by_author: Optional[str] = None  # Author who closed it
    closed_event_data: Optional[Dict[str, Any]] = None  # Additional closing event data

class IssueSearchResult(BaseModel):
    """Result from issue similarity search"""
    issue: IssueDoc
    similarity: float
    match_reasons: List[str]

@dataclass
class IssueContextResponse:
    related_issues: List[IssueSearchResult]
    total_found: int
    query_analysis: Dict[str, Any]
    processing_time: float
    patches: Optional[List['PatchSearchResult']] = None

@dataclass
class PatchSearchResult:
    """Represents a search result for a diff/patch"""
    patch: DiffDoc
    similarity: float
    match_reasons: List[str] = field(default_factory=list)

class PullRequestUser(BaseModel):
    login: str

class PullRequestInfo(BaseModel):
    number: int
    title: str
    merged_at: Optional[str] = None
    files_changed: List[str] = []
    issue_id: Optional[int] = None
    url: Optional[str] = None
    user: Optional[PullRequestUser] = None
    body: Optional[str] = None

@dataclass
class PullRequestReview:
    """Represents a pull request review"""
    author: str
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED
    submitted_at: str
    body: Optional[str] = None

@dataclass
class PullRequestReviewer:
    """Represents a requested reviewer"""
    login: Optional[str] = None
    name: Optional[str] = None  # For teams
    type: str = "User"  # User or Team

@dataclass
class PullRequestStatusCheck:
    """Represents CI/status check information"""
    state: str  # SUCCESS, FAILURE, PENDING, ERROR
    context: Optional[str] = None
    description: Optional[str] = None

@dataclass
class EnhancedPullRequestInfo:
    """Enhanced PR info with review and status data"""
    number: int
    title: str
    state: str  # open, closed, merged
    merged_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    url: Optional[str] = None
    body: Optional[str] = None
    user: Optional['PullRequestUser'] = None
    files_changed: List[str] = field(default_factory=list)
    issue_id: Optional[int] = None
    
    # Review information
    review_decision: Optional[str] = None  # APPROVED, REVIEW_REQUIRED, CHANGES_REQUESTED
    reviews: List[PullRequestReview] = field(default_factory=list)
    review_requests: List[PullRequestReviewer] = field(default_factory=list)
    
    # Status information
    mergeable: Optional[str] = None  # MERGEABLE, CONFLICTING, UNKNOWN
    status_checks: List[PullRequestStatusCheck] = field(default_factory=list)
    
    # Additional metadata
    draft: bool = False
    commits_count: Optional[int] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None
