# Frontend-to-Backend Dependency Analysis
## Critical Path Discovery for Python → Mastra Migration

**Date**: 2026-01-05
**Objective**: Map what the React frontend ACTUALLY uses from the Python backend to identify migration priorities

---

## Executive Summary

### Key Findings
- **Frontend uses only 17 API functions** from `/lib/api.ts`
- **Backend exposes 51 API endpoints** across 5 routers
- **~66% of Python code is NOT in the critical path** from frontend
- **Must migrate**: ~8,000 LOC (session management, agentic query, repository operations)
- **Can skip/defer**: ~14,000 LOC (workflows, timeline, git tools, founding member agent)

### Migration Impact
```
Total Python Backend:    ~22,000 LOC
Frontend Dependencies:    ~8,000 LOC (36%)
Optional/Unused:         ~14,000 LOC (64%)
```

---

## Part 1: Frontend API Consumption

### Frontend API Functions (`/lib/api.ts`)
The React frontend uses exactly **17 API functions**:

#### Session Management (7 functions)
1. `createRepoSession` → `POST /assistant/sessions`
2. `listAssistantSessions` → `GET /assistant/sessions`
3. `getSessionStatus` → `GET /assistant/sessions/{session_id}/status`
4. `deleteAssistantSession` → `DELETE /assistant/sessions/{session_id}`
5. `getSessionMetadata` → `GET /assistant/sessions/{session_id}/metadata`
6. `getSessionMessages` → `GET /assistant/sessions/{session_id}/messages`
7. `enableAgenticMode` → `POST /assistant/sessions/{session_id}/enable-agentic`

#### Chat & Agentic Query (2 functions)
8. `sendMessage` → `POST /assistant/sessions/{session_id}/agentic-query?stream=true` (SSE streaming)
9. `resetAgenticMemory` → `POST /assistant/sessions/{session_id}/reset-agentic-memory`

#### Repository Operations (2 functions)
10. `getRepositoryTree` → `GET /api/tree?session_id={sessionId}`
11. `getFileContent` → `GET /api/file-content?session_id={sessionId}&file_path={filePath}`

#### Issue Analysis (5 functions)
12. `analyzeIssue` → `POST /api/analyze-issue`
13. `getCachedAnalysis` → `GET /api/analysis-cache/{issueUrl}`
14. `postAnalysisToGitHub` → `POST /api/post-to-github`
15. `getCachedAnalyses` → `GET /api/cached-analyses/{sessionId}`
16. `deleteCachedAnalysis` → `DELETE /api/analysis-cache/{issueUrl}`

#### Pull Requests (1 function)
17. `listPullRequests` → `GET /api/prs?repo_url={repoUrl}&state={state}`

### Frontend Request/Response Patterns

#### Critical: snake_case Responses Expected
The frontend expects **snake_case** in ALL responses. From the code analysis:
- Session responses: `session_id`, `repo_metadata`, `repo_url`, `initial_file`, `session_name`
- Message responses: `agenticSteps`, `processingType`, `context_files`
- Issue analysis: `final_result`, `related_files`, `remediation_plan`, `agentic_insights`

**Implication**: Python's Pydantic models handle this via `alias_generator`. Mastra migration must preserve snake_case in JSON responses.

#### Critical: SSE Streaming for Chat
The `sendMessage` function uses **Server-Sent Events (SSE)** streaming:
```typescript
// Frontend expects SSE stream with JSON chunks
async function* sendMessage(...): AsyncGenerator<StreamedAgenticResponse> {
  // Parses "data: {json}\n\n" chunks
  // Handles types: 'step', 'final', 'error', 'status'
}
```

**Python Implementation** (`agentic.py:236`):
```python
async def stream_agentic_steps():
    async for step_json in agentic_rag.agentic_explorer.stream_query(query):
        yield f"data: {step_json}\n\n"
```

**Implication**: Mastra migration must implement SSE streaming with identical JSON structure.

---

## Part 2: Backend API Endpoints (51 total)

### Router 1: Sessions (`/assistant/sessions`) - **CRITICAL PATH**
Used by frontend: ✅ 7/10 endpoints

| Method | Endpoint | Frontend Usage | Python Module Dependencies |
|--------|----------|----------------|---------------------------|
| POST | `/assistant/sessions` | ✅ `createRepoSession` | `session_manager`, `github_client` |
| GET | `/assistant/sessions` | ✅ `listAssistantSessions` | `session_manager` |
| GET | `/{session_id}/status` | ✅ `getSessionStatus` | `session_manager` |
| DELETE | `/{session_id}` | ✅ `deleteAssistantSession` | `session_manager` |
| GET | `/{session_id}/metadata` | ✅ `getSessionMetadata` | `session_manager` |
| GET | `/{session_id}/messages` | ✅ `getSessionMessages` | `session_manager` |
| POST | `/{session_id}/sync-repository` | ❌ Not used | `agentic_rag.issue_rag` |
| GET | `/sessions/{session_id}/available-tools` | ❌ Not used | `session_manager` |
| POST | `/founder/sessions` | ❌ Not used | `founding_member_agent` |
| GET | `/founder/sessions/{session_id}/status` | ❌ Not used | `founding_member_agent` |

**WebSocket endpoint** (unused by frontend):
- `WS /ws/{session_id}` - Real-time updates (not used, frontend uses SSE instead)

### Router 2: Agentic (`/assistant/sessions`) - **CRITICAL PATH**
Used by frontend: ✅ 2/15 endpoints

| Method | Endpoint | Frontend Usage | Python Module Dependencies |
|--------|----------|----------------|---------------------------|
| POST | `/{session_id}/agentic-query` | ✅ `sendMessage` (SSE) | `agentic_rag.agentic_explorer` |
| POST | `/{session_id}/reset-agentic-memory` | ✅ `resetAgenticMemory` | `agentic_rag.agentic_explorer` |
| POST | `/{session_id}/enable-agentic` | ✅ `enableAgenticMode` | `agentic_rag` |
| GET | `/{session_id}/agentic-status` | ❌ Not used | `agentic_rag` |
| GET | `/{session_id}/agentic-rag-info` | ❌ Not used | `agentic_rag` |
| POST | `/{session_id}/analyze-query` | ❌ Not used | `agentic_rag` |
| GET | `/{session_id}/context-preview` | ❌ Not used | `agentic_rag.get_enhanced_context` |
| GET | `/{session_id}/related-issues` | ❌ Not used | `issue_rag.get_issue_context` |
| POST | `/{session_id}/index-issues` | ❌ Not used | `issue_rag` |
| GET | `/{session_id}/issue-index-status` | ❌ Not used | `issue_rag` |
| GET | `/agentic/chunk/{chunk_id}` | ❌ Not used | `chunk_store` |
| DELETE | `/agentic/chunk/{chunk_id}` | ❌ Not used | `chunk_store` |
| GET | `/agentic/redis-health` | ❌ Not used | `chunk_store` |
| GET | `/agentic-rag-features` | ❌ Not used | Static info |

### Router 3: Repository (`/api`) - **CRITICAL PATH**
Used by frontend: ✅ 2/7 endpoints

| Method | Endpoint | Frontend Usage | Python Module Dependencies |
|--------|----------|----------------|---------------------------|
| GET | `/tree` | ✅ `getRepositoryTree` | `session_manager` (repo_path) |
| GET | `/file-content` | ✅ `getFileContent` | `agentic_rag.agentic_explorer.read_file` |
| GET | `/files` | ❌ Not used | `session_manager` (repo_path) |
| GET | `/file-content/stream` | ❌ Not used | `agentic_explorer.stream_large_file` |
| GET | `/file-snippet` | ❌ Not used | `session_manager` (repo_path), git |
| POST | `/sync-repository` | ❌ Not used | `issue_rag` |
| GET | `/diff/{sha}/{file_path:path}` | ❌ Not used | git subprocess |

### Router 4: Issues (`/api`) - **CRITICAL PATH**
Used by frontend: ✅ 6/12 endpoints

| Method | Endpoint | Frontend Usage | Python Module Dependencies |
|--------|----------|----------------|---------------------------|
| POST | `/analyze-issue` | ✅ `analyzeIssue` | `issue_analysis.analyse_issue` |
| GET | `/analysis-cache/{issue_url:path}` | ✅ `getCachedAnalysis` | `cache.issue_analysis_cache` |
| POST | `/post-to-github` | ✅ `postAnalysisToGitHub` | `triage_bot.TriageBot` |
| GET | `/cached-analyses/{session_id}` | ✅ `getCachedAnalyses` | `cache.issue_analysis_cache` |
| DELETE | `/analysis-cache/{issue_url:path}` | ✅ `deleteCachedAnalysis` | `cache.issue_analysis_cache` |
| GET | `/prs` | ✅ `listPullRequests` | `github_client` |
| POST | `/v1/issue_context` | ❌ Not used | `issue_rag.IssueAwareRAG` |
| GET | `/issues` | ❌ Not used | `github_client` |
| GET | `/issues/{issue_number}` | ❌ Not used | `github_client` |
| GET | `/commits` | ❌ Not used | `commit_index.CommitIndexManager` |
| POST | `/issue_analysis` | ❌ Not used | Legacy endpoint |
| GET | `/issue_analysis/{session_id}` | ❌ Not used | Legacy endpoint |
| POST | `/apply-patch` | ❌ Not used | git subprocess |

### Router 5: Workflows (`/api`) - **NOT IN CRITICAL PATH**
Used by frontend: ❌ 0/6 endpoints

| Method | Endpoint | Frontend Usage | Python Module Dependencies |
|--------|----------|----------------|---------------------------|
| POST | `/{session_id}/workflows/create` | ❌ Not used | `llamaindex_workflows` |
| POST | `/{session_id}/workflows/{workflow_id}/execute` | ❌ Not used | `llamaindex_workflows` |
| GET | `/{session_id}/workflows/{workflow_id}/status` | ❌ Not used | `llamaindex_workflows` |
| POST | `/{session_id}/workflows/{workflow_id}/pause` | ❌ Not used | `llamaindex_workflows` |
| POST | `/{session_id}/workflows/{workflow_id}/resume` | ❌ Not used | `llamaindex_workflows` |
| GET | `/{session_id}/workflows` | ❌ Not used | `llamaindex_workflows` |

### Router 6: Timeline (`/api`) - **NOT IN CRITICAL PATH**
Used by frontend: ❌ 0/11 endpoints (entire router unused)

All timeline endpoints (file history, blame, git operations) are **NOT called by frontend**.

---

## Part 3: Python Module Call Graph

### Critical Path: Frontend → Python Modules

```
Frontend API Call
    ↓
FastAPI Router
    ↓
Dependencies (dependencies.py)
    ├── session_manager.SessionManager
    ├── github_client.GitHubIssueClient
    ├── agentic_rag.AgenticRAGSystem
    └── chunk_store.ChunkStoreFactory
    ↓
Core Business Logic Modules
    ├── session_manager.py (300 LOC) ← CRITICAL
    ├── agentic_rag.py (800 LOC) ← CRITICAL
    ├── agent_tools/core.py (AgenticCodebaseExplorer, 350 LOC) ← CRITICAL
    ├── issue_rag.py (700 LOC) ← CRITICAL for issue analysis
    ├── github_client.py (400 LOC) ← CRITICAL
    ├── new_rag.py (LocalRepoContextExtractor, 600 LOC) ← CRITICAL
    ├── cache/redis_cache_manager.py (200 LOC) ← CRITICAL
    ├── triage_bot.py (500 LOC) ← CRITICAL for GitHub posting
    └── models.py (150 LOC) ← CRITICAL
```

### Module Dependency Breakdown

#### Tier 1: MUST MIGRATE (Frontend dependencies)
These modules are directly called by frontend-facing endpoints:

| Module | LOC | Called By | Purpose |
|--------|-----|-----------|---------|
| `session_manager.py` | 300 | Sessions router | Session CRUD, initialization |
| `agentic_rag.py` | 800 | Agentic router | AgenticRAG orchestration |
| `agent_tools/core.py` | 350 | Agentic router | AgenticCodebaseExplorer |
| `new_rag.py` | 600 | Session init | LocalRepoContextExtractor |
| `issue_rag.py` | 700 | Issue analysis | IssueAwareRAG system |
| `github_client.py` | 400 | Multiple | GitHub API wrapper |
| `triage_bot.py` | 500 | Issues router | Post analysis to GitHub |
| `cache/redis_cache_manager.py` | 200 | Multiple | Redis caching |
| `models.py` | 150 | All routers | Pydantic models |
| `config.py` | 100 | Multiple | Settings management |
| `local_repo_loader.py` | 200 | Issue analysis | Repo cloning/loading |
| `patch_linkage.py` | 300 | Issue RAG | Patch analysis |
| `llm_client.py` | 550 | Issue analysis | LLM interactions |
| **TOTAL** | **~5,150** | | |

#### Tier 2: CRITICAL SUPPORT (Not directly called, but required by Tier 1)
These modules are imported by Tier 1 modules:

| Module | LOC | Required By | Purpose |
|--------|-----|-------------|---------|
| `commit_index.py` | 400 | agentic_rag | Commit indexing |
| `agent_tools/utilities.py` | 200 | agent_tools | Helper functions |
| `agent_tools/file_operations.py` | 400 | AgenticCodebaseExplorer | File reading/parsing |
| `agent_tools/search_operations.py` | 400 | AgenticCodebaseExplorer | Code search |
| `agent_tools/query_processor.py` | 550 | AgenticCodebaseExplorer | Query processing |
| `agent_tools/prompts.py` | 400 | AgenticCodebaseExplorer | Prompt templates |
| `agent_tools/response_handling.py` | 550 | AgenticCodebaseExplorer | Response formatting |
| **TOTAL** | **~2,900** | | |

#### Tier 3: OPTIONAL/SKIP (Not in frontend path)
These modules are NOT called by any frontend-facing endpoint:

| Module | LOC | Unused Feature |
|--------|-----|----------------|
| `agent_tools/llamaindex_workflows.py` | 1,000 | Workflow system (unused endpoints) |
| `agent_tools/llamaindex_comprehensive_workflow.py` | 1,000 | Comprehensive workflow (unused) |
| `agent_tools/workflow_state.py` | 400 | Workflow state management |
| `agent_tools/workflow_agents.py` | 500 | Workflow agents |
| `agent_tools/context_manager.py` | 600 | Context management (not in use) |
| `agent_tools/tool_registry.py` | 550 | Tool registry (not in use) |
| `agent_tools/agent_pool.py` | 100 | Agent pooling (not in use) |
| `founding_member_agent.py` | 600 | Founding member features |
| `agent_tools/git_operations.py` | 600 | Git operations (timeline router) |
| `agent_tools/issue_operations.py` | 1,100 | Advanced issue ops (unused) |
| `agent_tools/pr_operations.py` | 600 | PR operations (not exposed) |
| `agent_tools/context_aware_tools.py` | 1,000 | Context-aware tools (not used) |
| `agent_tools/code_generation.py` | 700 | Code generation (not used) |
| `api/routers/timeline.py` | 550 | Timeline router (unused) |
| `enhanced_persistence.py` | 300 | Enhanced persistence (not used) |
| `conversation_memory.py` | 200 | Conversation memory (not used) |
| `response_formatter.py` | 600 | Response formatting (not used) |
| `analyzer.py` | 200 | Analysis utilities (not used) |
| `classifier.py` | 150 | Classification (not used) |
| `plan_generator.py` | 400 | Plan generation (not used) |
| `pr_checker.py` | 300 | PR checking (not used) |
| **TOTAL** | **~12,000+** | |

---

## Part 4: snake_case Transformation Analysis

### Where Does snake_case Conversion Happen?

**Answer**: Pydantic models with `ConfigDict(alias_generator=to_camel)`

However, examining the codebase:
```python
# models.py - Uses snake_case natively
class RepoSessionResponse(BaseModel):
    session_id: str  # Already snake_case
    repo_metadata: Dict[str, Any]
    status: str
```

**No explicit camelCase ↔ snake_case transformation found** in the Python backend.

**Implication**: The Python backend already returns snake_case by default. Mastra migration can preserve this pattern without additional transformation layers.

### Response Examples from Code

#### Session Creation Response
```json
{
  "session_id": "abc123",
  "repo_metadata": {
    "owner": "apache",
    "repo": "airflow",
    "status": "cloning"
  },
  "status": "cloning",
  "message": "Repository session created. Cloning and indexing in progress..."
}
```

#### Agentic Query SSE Stream
```
data: {"type": "step", "step": {"type": "thought", "content": "...", "step": 1}}

data: {"type": "step", "step": {"type": "action", "content": "...", "step": 2}}

data: {"type": "final", "final_answer": "...", "steps": [...], "suggestions": [...]}
```

---

## Part 5: WebSocket/SSE Analysis

### Frontend Uses SSE, NOT WebSocket

**Finding**: The frontend uses **Server-Sent Events (SSE)** for streaming, not WebSocket.

#### SSE Implementation (`sendMessage` in api.ts)
```typescript
async function* sendMessage(...): AsyncGenerator<StreamedAgenticResponse> {
  const response = await fetch(url, {
    headers: { 'Accept': 'text/event-stream' }
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  // Parse "data: {json}\n\n" chunks
  while ((eolIndex = buffer.indexOf('\n\n')) >= 0) {
    if (line.startsWith('data: ')) {
      const jsonData = line.substring(6);
      yield JSON.parse(jsonData);
    }
  }
}
```

#### Python SSE Implementation (`agentic.py`)
```python
async def stream_agentic_steps():
    async for step_json in agentic_rag.agentic_explorer.stream_query(query):
        yield f"data: {step_json}\n\n"

return StreamingResponse(
    stream_agentic_steps(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache"}
)
```

#### WebSocket Endpoint (Unused)
```python
# sessions.py - WebSocket endpoint NOT used by frontend
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(...):
    # Real-time updates - but frontend uses SSE instead
```

**Implication**:
- ✅ Mastra migration must implement SSE streaming for chat
- ❌ Can skip WebSocket endpoint (not used)

---

## Part 6: Session Management Architecture

### How Sessions Work

```
1. Frontend calls createRepoSession(repo_url)
   ↓
2. POST /assistant/sessions
   ↓
3. session_manager.create_repo_session(repo_url)
   ↓
4. Background initialization:
   - Clone repository
   - Initialize AgenticRAG
   - Initialize IssueRAG (optional)
   - Update session.metadata.status
   ↓
5. Frontend polls getSessionStatus(session_id)
   ↓
6. When status === "ready":
   - Frontend calls sendMessage(session_id, query)
   - Backend streams agentic response via SSE
```

### Session Data Structure
```python
session = {
    "id": "uuid",
    "type": "repo_chat",
    "created_at": datetime,
    "last_accessed": datetime,
    "repo_path": "/tmp/repos/owner_repo",
    "repo_url": "https://github.com/owner/repo",
    "conversation_history": [],
    "metadata": {
        "owner": "owner",
        "repo": "repo",
        "status": "ready|cloning|error",
        "issue_rag_ready": bool,
        "agentic_enabled": bool
    },
    "agentic_rag": AgenticRAGSystem instance  # Not serialized to Redis
}
```

### Session Storage
- **Redis** (primary): Sessions serialized to Redis with 24h TTL
- **In-memory** (fallback): Python dict if Redis unavailable
- **Non-serializable objects** removed before Redis storage: `agentic_rag`, `founding_member_agent`, etc.
- **Re-creation logic**: `get_agentic_rag` dependency reconstructs AgenticRAG from session metadata

**Implication**: Mastra migration needs:
- Redis-backed session storage
- Background task system for repo initialization
- Lazy reconstruction of RAG systems from session metadata

---

## Part 7: Critical Paths Summary

### Path 1: Session Creation → Chat (MOST CRITICAL)
```
Frontend:
  createRepoSession(repo_url)
    ↓
Python:
  session_manager.create_repo_session()
    ├── Clone repo (local_repo_loader.py)
    ├── Initialize AgenticRAG (agentic_rag.py)
    │   ├── LocalRepoContextExtractor (new_rag.py)
    │   ├── AgenticCodebaseExplorer (agent_tools/core.py)
    │   └── IssueAwareRAG (issue_rag.py) [optional]
    └── Store session (redis_cache_manager.py)
    ↓
Frontend:
  sendMessage(session_id, query)
    ↓
Python:
  agentic_rag.agentic_explorer.stream_query()
    ├── Query processing (agent_tools/query_processor.py)
    ├── File search (agent_tools/search_operations.py)
    ├── File reading (agent_tools/file_operations.py)
    ├── LLM interaction (llm_client.py)
    └── Response streaming (SSE)
```

**Modules in path**: ~8,000 LOC (Tier 1 + Tier 2)

### Path 2: Issue Analysis (IMPORTANT)
```
Frontend:
  analyzeIssue(issue_url)
    ↓
Python:
  issue_analysis.analyse_issue()
    ├── GitHub API (github_client.py)
    ├── IssueAwareRAG (issue_rag.py)
    ├── LLM classification (llm_client.py)
    ├── Agentic analysis (agentic_rag.py)
    └── Plan generation (llm_client.py)
    ↓
  Cache result (cache/redis_cache_manager.py)
    ↓
Frontend:
  postAnalysisToGitHub(issue_url, result)
    ↓
Python:
  triage_bot.TriageBot.post_analysis_to_issue()
```

**Modules in path**: ~3,500 LOC

### Path 3: Repository Browser (MINOR)
```
Frontend:
  getRepositoryTree(session_id)
    ↓
Python:
  Build tree from session.repo_path
    ↓
Frontend:
  getFileContent(session_id, file_path)
    ↓
Python:
  agentic_explorer.read_file(file_path)
```

**Modules in path**: ~500 LOC

---

## Part 8: Missing Features (Python-Only)

### Features NOT Exposed to Frontend

#### 1. Workflow System (~3,000 LOC)
- `llamaindex_workflows.py`, `llamaindex_comprehensive_workflow.py`
- Router: `/api/{session_id}/workflows/*` (6 unused endpoints)
- **Status**: Backend code exists, frontend never calls it
- **Decision**: Can be skipped in initial migration

#### 2. Timeline/Git History (~2,000 LOC)
- `api/routers/timeline.py` (11 unused endpoints)
- File history, git blame, commit visualization
- **Status**: Entire router unused
- **Decision**: Skip migration

#### 3. Founding Member Agent (~600 LOC)
- `founding_member_agent.py`
- Router: `/founder/sessions` (2 unused endpoints)
- **Status**: Session type exists but frontend doesn't create/use
- **Decision**: Skip migration

#### 4. Context-Aware Tools (~1,000 LOC)
- `agent_tools/context_aware_tools.py`
- Advanced context management features
- **Status**: Not called by AgenticCodebaseExplorer
- **Decision**: Skip migration

#### 5. Enhanced Persistence (~300 LOC)
- `enhanced_persistence.py`
- Advanced session persistence features
- **Status**: Not used by session_manager
- **Decision**: Skip migration

---

## Part 9: Migration Priority Matrix

### Priority 1: MUST MIGRATE (Weeks 1-3)
**LOC**: ~8,000
**Risk**: HIGH - Breaks frontend if not migrated

| Module | LOC | Migration Complexity | Notes |
|--------|-----|---------------------|-------|
| `session_manager.py` | 300 | Medium | Redis + background tasks |
| `agentic_rag.py` | 800 | High | Core orchestration |
| `agent_tools/core.py` | 350 | High | SSE streaming |
| `new_rag.py` | 600 | High | RAG implementation |
| `issue_rag.py` | 700 | High | Issue indexing |
| `github_client.py` | 400 | Medium | GitHub API wrapper |
| `triage_bot.py` | 500 | Medium | GitHub posting |
| `llm_client.py` | 550 | Medium | LLM provider abstraction |
| `models.py` | 150 | Low | Pydantic → TypeScript types |
| `cache/*` | 200 | Medium | Redis caching |
| `local_repo_loader.py` | 200 | Medium | Repo cloning |
| `patch_linkage.py` | 300 | Medium | Patch analysis |
| Support modules | 2,900 | Medium-High | File ops, search, prompts |

### Priority 2: OPTIONAL MIGRATE (Weeks 4-6)
**LOC**: ~5,000
**Risk**: LOW - Not used by frontend

| Module | LOC | Decision |
|--------|-----|----------|
| `commit_index.py` | 400 | Migrate if time permits |
| `agent_tools/issue_operations.py` | 1,100 | Skip (unused endpoints) |
| `agent_tools/pr_operations.py` | 600 | Skip (unused endpoints) |
| `agent_tools/code_generation.py` | 700 | Skip (not exposed) |
| `agent_tools/context_manager.py` | 600 | Skip (not in use) |
| `agent_tools/tool_registry.py` | 550 | Skip (not in use) |
| Other unused modules | 1,050 | Skip |

### Priority 3: SKIP MIGRATION
**LOC**: ~9,000
**Risk**: NONE - Dead code

| Module | LOC | Reason |
|--------|-----|--------|
| `agent_tools/llamaindex_workflows.py` | 1,000 | Unused endpoints |
| `agent_tools/llamaindex_comprehensive_workflow.py` | 1,000 | Unused endpoints |
| `agent_tools/workflow_*` | 900 | Workflow system unused |
| `founding_member_agent.py` | 600 | Feature not exposed |
| `api/routers/timeline.py` | 550 | Entire router unused |
| `agent_tools/git_operations.py` | 600 | Timeline features |
| `enhanced_persistence.py` | 300 | Not in use |
| `conversation_memory.py` | 200 | Not in use |
| `response_formatter.py` | 600 | Not in use |
| Other unused | 3,250 | Various unused features |

---

## Part 10: Estimated LOC Breakdown

### Original Backend
```
Total Python Backend:        ~22,000 LOC
  ├── API Routers:            ~3,500 LOC
  ├── Core Business Logic:    ~8,000 LOC
  ├── Agent Tools:           ~10,000 LOC
  └── Git Tools:                ~500 LOC
```

### Migration Requirements
```
MUST Migrate (Priority 1):   ~8,000 LOC (36%)
  ├── Tier 1 (Direct deps):   ~5,150 LOC
  └── Tier 2 (Support):        ~2,900 LOC

CAN Skip (Priority 3):       ~14,000 LOC (64%)
  ├── Unused endpoints:        ~4,000 LOC
  ├── Workflow system:         ~3,000 LOC
  ├── Timeline features:       ~2,000 LOC
  ├── Founding member:           ~600 LOC
  ├── Unused agent tools:      ~4,000 LOC
  └── Other dead code:           ~400 LOC
```

### Migration Efficiency Gain
- **Before**: 22,000 LOC Python + 5,000 LOC TypeScript frontend = 27,000 LOC total
- **After**: 8,000 LOC TypeScript (Mastra) + 5,000 LOC frontend = 13,000 LOC total
- **Reduction**: 52% total codebase reduction

---

## Part 11: Migration Roadmap

### Phase 0: Foundation (Week 1)
- ✅ Install Mastra dependencies
- ✅ Set up LibSQL database
- ✅ Create basic Mastra server structure
- ✅ Implement snake_case response serialization

### Phase 1: Session Management (Week 1-2)
**Target**: Replace `session_manager.py` + `agentic_rag.py` initialization

1. Create Mastra agent for session management
2. Implement Redis-backed session storage
3. Background task: Repository cloning
4. Background task: RAG initialization
5. Endpoints:
   - `POST /assistant/sessions`
   - `GET /assistant/sessions`
   - `GET /assistant/sessions/{id}/status`
   - `DELETE /assistant/sessions/{id}`
   - `GET /assistant/sessions/{id}/metadata`
   - `GET /assistant/sessions/{id}/messages`

**Verification**: Frontend session creation works, status polling works

### Phase 2: Agentic Query (Week 2-3)
**Target**: Replace `agent_tools/core.py` (AgenticCodebaseExplorer)

1. Implement SSE streaming in Mastra
2. Create query processing agent
3. Implement file search + file reading tools
4. Create prompt system
5. Endpoints:
   - `POST /assistant/sessions/{id}/agentic-query` (SSE)
   - `POST /assistant/sessions/{id}/enable-agentic`
   - `POST /assistant/sessions/{id}/reset-agentic-memory`

**Verification**: Chat streaming works, same quality as Python

### Phase 3: Repository Operations (Week 3)
**Target**: Replace `new_rag.py` + file operations

1. Implement repository tree generation
2. Implement file content reading
3. Endpoints:
   - `GET /api/tree`
   - `GET /api/file-content`

**Verification**: File browser works

### Phase 4: Issue Analysis (Week 3-4)
**Target**: Replace `issue_rag.py` + `triage_bot.py`

1. Implement issue indexing (RAG)
2. Create issue analysis agent
3. Implement GitHub posting
4. Implement caching layer
5. Endpoints:
   - `POST /api/analyze-issue`
   - `GET /api/analysis-cache/{url}`
   - `POST /api/post-to-github`
   - `GET /api/cached-analyses/{id}`
   - `DELETE /api/analysis-cache/{url}`
   - `GET /api/prs`

**Verification**: Issue analysis produces equivalent results

### Phase 5: Cleanup (Week 4)
1. Remove Python backend
2. Update frontend to point to Mastra :4111
3. Performance testing
4. Documentation

---

## Part 12: API Contract Preservation

### Critical: Frontend Expects Exact JSON Structure

#### Example: Session Creation Response
**Python** (current):
```json
{
  "session_id": "abc123",
  "repo_metadata": {
    "owner": "apache",
    "repo": "airflow"
  },
  "status": "cloning",
  "message": "Repository session created..."
}
```

**Mastra** (must match):
```typescript
// Mastra endpoint must return identical structure
app.post('/assistant/sessions', async (req, res) => {
  return {
    session_id: sessionId,  // snake_case
    repo_metadata: { ... },
    status: "cloning",
    message: "..."
  };
});
```

#### Example: SSE Stream Format
**Python** (current):
```
data: {"type": "step", "step": {"type": "thought", "content": "...", "step": 1}}

data: {"type": "final", "final_answer": "...", "steps": [...]}
```

**Mastra** (must match):
```typescript
// Exact SSE format required
async function* streamAgenticSteps() {
  yield `data: ${JSON.stringify({type: "step", step: {...}})}\n\n`;
  yield `data: ${JSON.stringify({type: "final", final_answer: "...", steps: [...]})}\n\n`;
}
```

---

## Conclusion

### Key Takeaways

1. **66% of Python code is not in the critical path**
   - Only 8,000 LOC must be migrated
   - 14,000 LOC can be skipped (unused features)

2. **Frontend uses 17 API functions across 4 routers**
   - Sessions (7 functions)
   - Agentic query (2 functions)
   - Repository (2 functions)
   - Issue analysis (6 functions)

3. **SSE streaming is critical**
   - Frontend expects Server-Sent Events, not WebSocket
   - Exact JSON structure must be preserved

4. **snake_case is already the standard**
   - No transformation layer needed
   - Mastra should preserve snake_case in responses

5. **Migration can focus on 4 core systems**
   - Session management
   - AgenticRAG orchestration
   - Repository operations
   - Issue analysis

### Migration Impact: 36% of Codebase

```
┌─────────────────────────────────────────────┐
│  Python Backend LOC Distribution            │
├─────────────────────────────────────────────┤
│  ████████ MUST MIGRATE      36% (8,000)     │
│  █████████████████ SKIP     64% (14,000)    │
└─────────────────────────────────────────────┘
```

### Recommendation

**Migrate in phases, focus on critical path first:**

1. **Week 1-2**: Session management + basic chat (no agentic)
2. **Week 2-3**: Agentic query with SSE streaming
3. **Week 3**: Repository operations (tree + file content)
4. **Week 3-4**: Issue analysis system
5. **Week 4**: Cleanup + performance testing

**Total Effort**: 4 weeks with parallel run validation

**LOC Reduction**: 52% (from 27,000 → 13,000 LOC)

**Risk Mitigation**: Keep Python backend running in parallel during migration, frontend can route to either based on feature flag.
