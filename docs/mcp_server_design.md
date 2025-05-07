# triage.flow MCP Server: Technical Design Document

## 1. System Architecture

### 1.1 Overview

The triage.flow MCP Server transforms the existing issue analysis tool into a comprehensive repository intelligence system. It maintains persistent repository context and enables natural language conversations about code, issues, and architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                    triage.flow MCP Server                   │
├─────────────┬─────────────────────────┬────────────────────┤
│ Repository  │                         │                    │
│ Management  │  Conversation Engine    │  LLM Orchestration │
│             │                         │                    │
├─────────────┼─────────────────────────┼────────────────────┤
│             │                         │                    │
│  Context    │  Session Management     │  Tool Registry     │
│  Store      │                         │                    │
└─────────────┴─────────────────────────┴────────────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  Repository     │ │  User Sessions  │ │  External Services  │
│  - FAISS Index  │ │  - Chat History │ │  - GitHub API       │
│  - Clone Cache  │ │  - Preferences  │ │  - LLM Providers    │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
```

### 1.2 Core Components

#### 1.2.1 Repository Management
- Handles repository cloning, updating, and indexing
- Maintains FAISS vector indices for each repository
- Implements language-aware parsing and context extraction

#### 1.2.2 Conversation Engine
- Processes natural language queries about repositories
- Maintains conversation context and history
- Generates contextually relevant responses using RAG

#### 1.2.3 LLM Orchestration
- Manages connections to multiple LLM providers
- Handles fallback strategies and model selection
- Optimizes prompt construction for different query types

#### 1.2.4 Session Management
- Tracks user sessions and conversation history
- Manages authentication and authorization
- Persists user preferences and repository access

#### 1.2.5 Tool Registry
- Exposes MCP-compatible tools for clients
- Handles tool versioning and capability discovery
- Provides standardized error handling and response formatting

## 2. Data Models

### 2.1 Repository Context

```python
class RepositoryContext:
    """Persistent context for a repository"""
    repo_url: str
    owner: str
    repo_name: str
    branch: str
    clone_path: str
    index_path: str
    last_updated: datetime
    languages: Dict[str, str]  # language -> display_name
    vector_index: Optional[VectorStoreIndex]
    query_engine: Optional[QueryEngine]
```

### 2.2 Session Model

```python
class ChatSession:
    """User chat session with history"""
    session_id: str
    user_id: str
    repository_url: str
    created_at: datetime
    last_active: datetime
    messages: List[ChatMessage]
    preferences: Dict[str, Any]
```

### 2.3 Chat Message

```python
class ChatMessage:
    """Individual message in a chat session"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    context_sources: Optional[List[ContextSource]]
    metadata: Dict[str, Any]
```

### 2.4 Context Source

```python
class ContextSource:
    """Source of context used in a response"""
    file_path: str
    language: str
    snippet: str
    relevance_score: float
    line_numbers: Optional[Tuple[int, int]]
```

## 3. API Design

### 3.1 MCP Tools

#### 3.1.1 Repository Management Tools

```python
@mcp_tool
async def load_repository(repo_url: str, branch: str = "main") -> Dict[str, Any]:
    """Load a repository into the MCP server"""
    # Implementation details...
    return {"status": "success", "repo_info": {...}}

@mcp_tool
async def list_repositories() -> List[Dict[str, Any]]:
    """List all loaded repositories"""
    # Implementation details...
    return [{"repo_url": "...", "owner": "...", "repo": "...", "languages": [...]}]
```

#### 3.1.2 Conversation Tools

```python
@mcp_tool
async def query_repository(
    repo_url: str,
    query: str,
    session_id: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Query a repository with natural language"""
    # Implementation details...
    return {
        "response": "Detailed answer...",
        "sources": [...],  # Context sources
        "session_id": "..."  # New or existing session
    }

@mcp_tool
async def continue_conversation(
    session_id: str,
    message: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Continue an existing conversation"""
    # Implementation details...
    return {
        "response": "Follow-up answer...",
        "sources": [...],
        "session_id": session_id
    }
```

#### 3.1.3 Issue Analysis Tools

```python
@mcp_tool
async def analyze_issue(
    issue_url: str,
    analysis_type: str = "explain",
    session_id: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a GitHub issue with repository context"""
    # Implementation details...
    return {
        "response": "Issue analysis...",
        "sources": [...],
        "session_id": "..."
    }
```

### 3.2 MCP Resources

```python
@mcp_resource
async def get_repository_info(repo_url: str) -> Dict[str, Any]:
    """Get information about a loaded repository"""
    # Implementation details...
    return {
        "repo_url": "...",
        "owner": "...",
        "repo": "...",
        "languages": [...],
        "file_count": 123,
        "last_updated": "..."
    }

@mcp_resource
async def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """Get conversation history for a session"""
    # Implementation details...
    return [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."},
        # ...
    ]
```

## 4. Implementation Plan

### 4.1 Phase 1: Core Server Implementation

1. Create `TriageFlowMCPServer` class with repository management
   - Adapt existing `LocalRepoContextExtractor` for persistent storage
   - Implement repository caching and update mechanisms
   - Add session management and conversation history

2. Implement MCP tool and resource handlers
   - Create MCP server configuration
   - Implement tool registration and dispatching
   - Add authentication and rate limiting

3. Extend prompt templates for conversational interaction
   - Create new templates for code exploration, implementation guidance
   - Enhance context integration for multi-turn conversations
   - Add source attribution and confidence scoring

### 4.2 Phase 2: Enhanced Capabilities

1. Implement advanced repository analysis
   - Add dependency graph extraction
   - Support architectural pattern recognition
   - Enable cross-repository context (for monorepos or related projects)

2. Add specialized conversation modes
   - Code review mode with diff analysis
   - Implementation planning with step-by-step guidance
   - Debugging mode with error analysis

3. Improve performance and scalability
   - Implement background indexing and cache warming
   - Add distributed vector store support
   - Optimize token usage with context pruning

### 4.3 Phase 3: Client Integrations

1. Create CLI client
   - Interactive chat with repository selection
   - Session management and history browsing
   - Export/import of conversations

2. Develop VS Code extension
   - Sidebar integration with chat interface
   - Context-aware code actions
   - Inline code explanations and suggestions

3. Build web interface
   - Team collaboration features
   - Repository and issue browsing
   - Visualization of code relationships

## 5. File Structure

```
triage_flow_mcp/
├── server/
│   ├── __init__.py
│   ├── server.py             # Main MCP server implementation
│   ├── repository.py         # Repository management
│   ├── conversation.py       # Conversation engine
│   ├── session.py            # Session management
│   ├── tools/                # MCP tool implementations
│   │   ├── __init__.py
│   │   ├── repository_tools.py
│   │   ├── conversation_tools.py
│   │   └── issue_tools.py
│   ├── resources/            # MCP resource implementations
│   │   ├── __init__.py
│   │   ├── repository_resources.py
│   │   └── session_resources.py
│   └── models/               # Data models
│       ├── __init__.py
│       ├── repository.py
│       ├── session.py
│       └── message.py
├── clients/
│   ├── cli/                  # Command-line client
│   ├── vscode/               # VS Code extension
│   └── web/                  # Web interface
└── shared/                   # Shared utilities and types
    ├── __init__.py
    ├── config.py
    └── types.py
```

## 6. Integration with Existing Code

### 6.1 Adapting LocalRepoContextExtractor

The existing `LocalRepoContextExtractor` class will be extended to support:
- Persistent storage of cloned repositories
- Incremental updates to avoid full re-cloning
- Background indexing for improved performance

```python
class ManagedRepositoryExtractor(LocalRepoContextExtractor):
    """Enhanced extractor with persistence and session awareness"""
    
    def __init__(self, storage_path: str):
        super().__init__()
        self.storage_path = storage_path
        self.repo_cache = {}
        
    async def get_or_load_repository(self, repo_url: str, branch: str = "main") -> None:
        """Get cached repository or load if not present"""
        cache_key = f"{repo_url}:{branch}"
        if cache_key in self.repo_cache:
            # Check if update needed
            if self._needs_update(cache_key):
                await self._update_repository(repo_url, branch)
            return self.repo_cache[cache_key]
        
        # Load repository
        await self.load_repository(repo_url, branch)
        self.repo_cache[cache_key] = {
            "query_engine": self.query_engine,
            "repo_info": self.repo_info,
            "last_updated": datetime.now()
        }
        return self.repo_cache[cache_key]
```

### 6.2 Enhancing PromptGenerator

The `PromptGenerator` will be extended with conversational capabilities:

```python
class ConversationalPromptGenerator(PromptGenerator):
    """Enhanced prompt generator for conversations"""
    
    def __init__(self):
        super().__init__()
        # Add conversational templates
        self.prompt_templates.update({
            "explore": """Please explain the following code or concept from the repository:

Query: {query}

{context}

Please provide:
1. A clear explanation of the code/concept
2. How it fits into the overall architecture
3. Any important implementation details
4. Examples of usage if available""",

            "implement": """Please provide guidance on implementing the following feature:

Feature: {query}

{context}

Please provide:
1. Suggested approach for implementation
2. Relevant files that would need to be modified
3. Potential challenges or considerations
4. Testing strategy""",

            # Additional templates...
        })
        
    async def generate_conversation_prompt(
        self, 
        query: str, 
        context: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None
    ) -> PromptResponse:
        """Generate a prompt for conversational interaction"""
        # Implementation details...
```

## 7. Deployment Considerations

### 7.1 Server Requirements

- Python 3.8+ runtime
- 4+ CPU cores for vector operations
- 8+ GB RAM for in-memory indices
- SSD storage for repository cache
- Optional GPU for embedding generation

### 7.2 Scaling Strategy

- Horizontal scaling with repository sharding
- Redis for session state and rate limiting
- Distributed vector store (FAISS or Pinecone)
- Background workers for repository indexing

### 7.3 Security Considerations

- API key authentication for MCP clients
- GitHub token management for private repositories
- Rate limiting to prevent abuse
- Sanitization of user inputs
- Isolation of repository clones

## 8. Testing Strategy

### 8.1 Unit Tests

- Repository management functions
- Conversation processing
- Tool and resource handlers
- Session management

### 8.2 Integration Tests

- End-to-end conversation flows
- Repository loading and querying
- MCP protocol compliance
- Error handling and recovery

### 8.3 Performance Tests

- Repository indexing benchmarks
- Query latency measurements
- Memory usage profiling
- Concurrent request handling

## 9. Monitoring and Telemetry

- Request/response logging
- Performance metrics collection
- Error tracking and alerting
- Usage statistics for optimization

## 10. Future Extensions

- Multi-repository context for complex queries
- Code generation with repository-specific patterns
- Integration with CI/CD for automated code review
- Collaborative sessions for team problem-solving
- Fine-tuning models on repository-specific data
