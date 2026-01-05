# Migration Priority Matrix
## Python → Mastra: What to Migrate First (and What to Skip)

**Date**: 2026-01-05
**Analysis Basis**: Frontend API consumption analysis + Backend call graph tracing

---

## Executive Summary

### The 40% Rule Achievement

**Finding**: By analyzing what the React frontend actually calls, we discovered that **64% of the Python backend is NOT in the critical path**.

```
┌────────────────────────────────────────────┐
│  Python Backend Distribution               │
├────────────────────────────────────────────┤
│  CRITICAL (must migrate):     8,000 LOC    │
│  OPTIONAL (can defer):        5,000 LOC    │
│  DEAD CODE (skip entirely): 9,000 LOC      │
│  ────────────────────────────────────────  │
│  TOTAL:                      22,000 LOC    │
└────────────────────────────────────────────┘

Migration target: 36% of codebase (8,000 LOC)
Skip entirely:   64% of codebase (14,000 LOC)
```

---

## Priority Matrix

### Priority 1: MUST MIGRATE (Weeks 1-3)
**Impact**: Breaks frontend if skipped
**Complexity**: High
**Total LOC**: ~8,000

| Module | LOC | Week | Risk | Verification |
|--------|-----|------|------|--------------|
| `session_manager.py` | 300 | 1 | HIGH | Session creation works |
| `cache/redis_cache_manager.py` | 200 | 1 | MEDIUM | Sessions persist |
| `github_client.py` | 400 | 1 | LOW | Issue fetch works |
| `models.py` → TypeScript types | 150 | 1 | LOW | Type safety |
| `config.py` → Mastra config | 100 | 1 | LOW | Settings load |
| `agentic_rag.py` | 800 | 2 | HIGH | AgenticRAG init |
| `new_rag.py` | 600 | 2 | HIGH | Code indexing |
| `agent_tools/core.py` | 350 | 2 | HIGH | SSE streaming |
| `agent_tools/file_operations.py` | 400 | 2 | MEDIUM | File reading |
| `agent_tools/search_operations.py` | 400 | 2 | MEDIUM | Code search |
| `agent_tools/query_processor.py` | 550 | 2 | HIGH | Query analysis |
| `agent_tools/prompts.py` | 400 | 2 | MEDIUM | Prompt templates |
| `agent_tools/response_handling.py` | 550 | 2-3 | MEDIUM | SSE formatting |
| `llm_client.py` | 550 | 2 | MEDIUM | LLM providers |
| `issue_rag.py` | 700 | 3 | HIGH | Issue indexing |
| `local_repo_loader.py` | 200 | 3 | MEDIUM | Repo cloning |
| `patch_linkage.py` | 300 | 3 | MEDIUM | Patch analysis |
| `triage_bot.py` | 500 | 3 | MEDIUM | GitHub posting |
| `commit_index.py` | 400 | 3 | MEDIUM | Commit history |

**Total**: 7,850 LOC

---

### Priority 2: OPTIONAL DEFER (Weeks 4-6+)
**Impact**: Nice to have, not frontend-critical
**Complexity**: Medium
**Total LOC**: ~5,000

| Module | LOC | Reason to Defer | Migration Later? |
|--------|-----|-----------------|------------------|
| `agent_tools/issue_operations.py` | 1,100 | Advanced features not exposed | Maybe |
| `agent_tools/pr_operations.py` | 600 | PR features not in frontend | Maybe |
| `agent_tools/code_generation.py` | 700 | Code gen not used | No |
| `agent_tools/context_manager.py` | 600 | Context management unused | No |
| `agent_tools/tool_registry.py` | 550 | Tool registry not active | No |
| `agent_tools/agent_pool.py` | 100 | Agent pooling unused | No |
| `response_formatter.py` | 600 | Formatter not in use | No |
| `conversation_memory.py` | 200 | Memory system unused | No |
| `enhanced_persistence.py` | 300 | Enhanced features unused | No |
| `analyzer.py` | 200 | Analysis utils unused | No |

**Total**: 4,950 LOC

---

### Priority 3: SKIP ENTIRELY (Never Migrate)
**Impact**: Zero - not called by frontend
**Complexity**: N/A
**Total LOC**: ~9,000

| Module | LOC | Unused Feature |
|--------|-----|----------------|
| `agent_tools/llamaindex_workflows.py` | 1,000 | Workflow system (0/6 endpoints used) |
| `agent_tools/llamaindex_comprehensive_workflow.py` | 1,000 | Comprehensive workflow |
| `agent_tools/workflow_state.py` | 400 | Workflow state |
| `agent_tools/workflow_agents.py` | 500 | Workflow agents |
| `api/routers/timeline.py` | 550 | Timeline router (0/11 endpoints used) |
| `agent_tools/git_operations.py` | 600 | Git operations (timeline) |
| `agent_tools/context_aware_tools.py` | 1,000 | Context-aware tools |
| `founding_member_agent.py` | 600 | Founding member features |
| `classifier.py` | 150 | Classification utils |
| `plan_generator.py` | 400 | Plan generation (inline instead) |
| `pr_checker.py` | 300 | PR checking |
| Various git tools | 2,500 | Git blame, history, etc. |

**Total**: 9,000 LOC

---

## Week-by-Week Migration Plan

### Week 1: Foundation + Sessions
**Goal**: Create Mastra project, implement session management

**Tasks**:
1. Set up Mastra project structure
   ```bash
   pnpm create mastra@latest
   cd mastra
   pnpm install @mastra/core @mastra/rag @mastra/libsql @mastra/memory
   ```

2. Implement session storage
   - LibSQL schema for sessions
   - Redis caching (optional)
   - Session CRUD operations

3. Implement repo initialization
   - Background task: clone repo
   - Background task: initialize RAG
   - Status polling endpoint

4. Migrate endpoints:
   - ✅ `POST /assistant/sessions`
   - ✅ `GET /assistant/sessions`
   - ✅ `GET /assistant/sessions/{id}/status`
   - ✅ `DELETE /assistant/sessions/{id}`
   - ✅ `GET /assistant/sessions/{id}/metadata`
   - ✅ `GET /assistant/sessions/{id}/messages`

**Verification**:
```typescript
// Frontend test
const session = await createRepoSession('https://github.com/apache/airflow');
// Should return: { session_id, repo_metadata, status: "cloning" }

const status = await getSessionStatus(session.session_id);
// Should poll until: { status: "ready" }
```

**Parallel Run**: Compare session creation time Python vs. Mastra

---

### Week 2: Agentic Query (SSE Streaming)
**Goal**: Implement chat with SSE streaming

**Tasks**:
1. Create AgenticCodebaseExplorer agent
   - Query processing
   - File search tools
   - File reading tools

2. Implement SSE streaming
   ```typescript
   async function* streamAgenticSteps(query: string) {
     // Yield chunks: data: {...}\n\n
     for await (const step of agent.execute(query)) {
       yield `data: ${JSON.stringify({
         type: 'step',
         step: { type: step.type, content: step.content, step: step.number }
       })}\n\n`;
     }
   }
   ```

3. Create tools:
   - `searchCodebase()` - Semantic + AST search
   - `readFile()` - Parse and return file content
   - `analyzeQuery()` - Extract intent

4. Migrate endpoints:
   - ✅ `POST /assistant/sessions/{id}/agentic-query?stream=true`
   - ✅ `POST /assistant/sessions/{id}/enable-agentic`
   - ✅ `POST /assistant/sessions/{id}/reset-agentic-memory`

**Verification**:
```typescript
// Frontend test
for await (const chunk of sendMessage(sessionId, "Where is the DAG parsing logic?")) {
  console.log(chunk.type, chunk.step?.content);
}
// Should stream: thought → action → observation → answer
```

**Parallel Run**:
- Compare response quality (manual review)
- Compare streaming latency
- Verify chunk format matches

---

### Week 3: Repository + Issue Analysis
**Goal**: File browser + issue analysis

**Tasks**:
1. Repository operations
   - Tree generation
   - File content reading
   - Migrate endpoints:
     - ✅ `GET /api/tree`
     - ✅ `GET /api/file-content`

2. Issue analysis system
   - Migrate IssueAwareRAG
   - Implement analysis pipeline
   - Implement caching
   - Migrate endpoints:
     - ✅ `POST /api/analyze-issue`
     - ✅ `GET /api/analysis-cache/{url}`
     - ✅ `POST /api/post-to-github`
     - ✅ `GET /api/cached-analyses/{id}`
     - ✅ `DELETE /api/analysis-cache/{url}`
     - ✅ `GET /api/prs`

**Verification**:
```typescript
// Test file browser
const tree = await getRepositoryTree(sessionId);
const file = await getFileContent(sessionId, 'src/main.py');

// Test issue analysis
const result = await analyzeIssue('https://github.com/apache/airflow/issues/1234');
// Should return: { steps, final_result, status: "completed" }
```

**Parallel Run**:
- Compare issue analysis quality
- Verify cache hit/miss behavior
- Check GitHub API rate limits

---

### Week 4: Polish + Cutover
**Goal**: Production-ready migration

**Tasks**:
1. Performance optimization
   - Benchmark all endpoints
   - Optimize slow queries
   - Add connection pooling

2. Error handling
   - Match Python error messages
   - Handle edge cases

3. Documentation
   - API documentation
   - Deployment guide

4. Cutover plan
   - Feature flag rollout
   - Monitoring setup
   - Rollback procedure

**Verification**:
- Load testing (100 concurrent users)
- Error rate < 0.1%
- P95 latency within 20% of Python

---

## Module-by-Module Migration Guide

### Module 1: session_manager.py → Mastra Session Service

**Python** (300 LOC):
```python
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.use_redis = False

    async def create_repo_session(self, repo_url, ...):
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "type": "repo_chat",
            "repo_url": repo_url,
            "created_at": datetime.now(),
            "metadata": {"status": "initializing"}
        }
        await self._store_session(session_id, session)
        return session_id, metadata
```

**Mastra** (TypeScript):
```typescript
// mastra/src/services/session-manager.ts
export class SessionManager {
  constructor(
    private db: LibSQLDatabase,
    private redis?: RedisCache
  ) {}

  async createRepoSession(repoUrl: string) {
    const sessionId = randomUUID();
    const session = {
      id: sessionId,
      type: 'repo_chat',
      repo_url: repoUrl,
      created_at: new Date(),
      metadata: { status: 'initializing' }
    };

    // Store in LibSQL
    await this.db.insert('sessions', session);

    // Cache in Redis
    if (this.redis) {
      await this.redis.set(`session:${sessionId}`, session, 86400);
    }

    return { session_id: sessionId, ...session.metadata };
  }
}
```

**LOC**: 300 → ~150 (50% reduction)

---

### Module 2: agentic_rag.py → Mastra Agent

**Python** (800 LOC):
```python
class AgenticRAGSystem:
    def __init__(self, repo_key: str):
        self.rag_extractor = LocalRepoContextExtractor()
        self.agentic_explorer = AgenticCodebaseExplorer(...)
        self.issue_rag = IssueAwareRAG(...)

    async def get_enhanced_context(self, query, ...):
        # Complex RAG logic
        return enhanced_context
```

**Mastra** (TypeScript):
```typescript
// mastra/src/agents/agentic-rag.ts
import { Agent } from '@mastra/core';
import { ragTool } from '@mastra/rag';

export const agenticRagAgent = new Agent({
  name: 'AgenticRAG',
  instructions: 'You are a code analysis expert...',
  model: {
    provider: 'ANTHROPIC',
    name: 'claude-sonnet-4.5',
  },
  tools: {
    searchCodebase: ragTool({ /* config */ }),
    readFile: createFileTool(),
    analyzeQuery: createQueryTool(),
  },
});
```

**LOC**: 800 → ~200 (75% reduction with Mastra abstractions)

---

### Module 3: agent_tools/core.py → Mastra Tools

**Python** (350 LOC):
```python
class AgenticCodebaseExplorer:
    async def stream_query(self, query: str):
        # Process query
        steps = await self.query_processor.process(query)

        # Search codebase
        results = await self.search_operations.search(query)

        # Stream results
        for step in steps:
            yield json.dumps({"type": "step", "step": step})
```

**Mastra** (TypeScript):
```typescript
// mastra/src/tools/codebase-search.ts
import { createTool } from '@mastra/core';

export const searchCodebaseTool = createTool({
  id: 'search-codebase',
  description: 'Search codebase semantically',
  inputSchema: z.object({
    query: z.string(),
  }),
  execute: async ({ context, input }) => {
    // Semantic search using Mastra RAG
    const results = await context.rag.search(input.query);
    return { files: results };
  },
});
```

**LOC**: 350 → ~100 per tool × 3 tools = 300 LOC (15% reduction)

---

## snake_case Response Serialization

### Requirement
Frontend expects **snake_case** in all JSON responses.

### Python Implementation
Pydantic models already use snake_case natively:
```python
class RepoSessionResponse(BaseModel):
    session_id: str  # Already snake_case
    repo_metadata: Dict[str, Any]
```

### Mastra Implementation
Use custom serialization:
```typescript
// mastra/src/utils/serialization.ts
export function toSnakeCase(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(toSnakeCase);
  } else if (obj !== null && typeof obj === 'object') {
    return Object.keys(obj).reduce((acc, key) => {
      const snakeKey = key.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
      acc[snakeKey] = toSnakeCase(obj[key]);
      return acc;
    }, {} as any);
  }
  return obj;
}

// Apply to all responses
app.use((req, res, next) => {
  const originalJson = res.json.bind(res);
  res.json = (data) => originalJson(toSnakeCase(data));
  next();
});
```

---

## SSE Streaming Implementation

### Python Reference
```python
async def stream_agentic_steps():
    async for step_json in agentic_rag.agentic_explorer.stream_query(query):
        yield f"data: {step_json}\n\n"

return StreamingResponse(
    stream_agentic_steps(),
    media_type="text/event-stream"
)
```

### Mastra Implementation
```typescript
// mastra/src/routes/agentic.ts
app.post('/assistant/sessions/:id/agentic-query', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const stream = agenticRagAgent.generateStream({
    messages: [{ role: 'user', content: req.body.content }],
    onStepStart: (step) => {
      const chunk = {
        type: 'step',
        step: {
          type: step.type,
          content: step.content,
          step: step.number
        }
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    },
  });

  for await (const chunk of stream) {
    // Handle final chunk
    if (chunk.done) {
      res.write(`data: ${JSON.stringify({
        type: 'final',
        final_answer: chunk.content,
        steps: chunk.steps
      })}\n\n`);
      res.end();
    }
  }
});
```

---

## Testing Strategy

### Unit Tests
```typescript
// mastra/tests/session-manager.test.ts
describe('SessionManager', () => {
  it('should create session with snake_case response', async () => {
    const manager = new SessionManager(db);
    const response = await manager.createRepoSession('https://github.com/apache/airflow');

    expect(response).toHaveProperty('session_id');
    expect(response).toHaveProperty('repo_metadata');
    expect(response.repo_metadata).toHaveProperty('owner', 'apache');
    expect(response.repo_metadata).toHaveProperty('repo', 'airflow');
  });
});
```

### Integration Tests
```typescript
// mastra/tests/integration/chat.test.ts
describe('Agentic Query SSE', () => {
  it('should stream chunks in correct format', async () => {
    const chunks: any[] = [];

    const stream = await fetch('http://localhost:4111/assistant/sessions/abc/agentic-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: 'test query' })
    });

    const reader = stream.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split('\n\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));
          chunks.push(data);
        }
      }
    }

    expect(chunks[0].type).toBe('step');
    expect(chunks[chunks.length - 1].type).toBe('final');
  });
});
```

### Parallel Run Tests
```typescript
// Compare Python vs. Mastra
describe('Parallel Run Comparison', () => {
  it('should return equivalent session creation response', async () => {
    const pythonResponse = await fetch('http://localhost:8000/assistant/sessions', {
      method: 'POST',
      body: JSON.stringify({ repo_url: 'https://github.com/apache/airflow' })
    }).then(r => r.json());

    const mastraResponse = await fetch('http://localhost:4111/assistant/sessions', {
      method: 'POST',
      body: JSON.stringify({ repo_url: 'https://github.com/apache/airflow' })
    }).then(r => r.json());

    expect(mastraResponse).toMatchObject({
      session_id: expect.any(String),
      repo_metadata: expect.objectContaining({
        owner: 'apache',
        repo: 'airflow'
      }),
      status: 'cloning'
    });

    expect(mastraResponse).toHaveProperty('session_id');
    expect(mastraResponse).toHaveProperty('repo_metadata');
    // Structure matches, IDs will differ
  });
});
```

---

## Migration Checklist

### Pre-Migration
- [ ] Analyze frontend API usage (DONE ✅)
- [ ] Map Python modules to critical paths (DONE ✅)
- [ ] Identify skip-able modules (DONE ✅)
- [ ] Create migration priority matrix (DONE ✅)

### Week 1: Foundation
- [ ] Install Mastra dependencies
- [ ] Set up LibSQL database
- [ ] Create session storage service
- [ ] Implement Redis caching (optional)
- [ ] Migrate session endpoints (7 endpoints)
- [ ] Test: Session creation works

### Week 2: Agentic Query
- [ ] Create AgenticRAG agent
- [ ] Implement SSE streaming
- [ ] Create codebase search tool
- [ ] Create file reading tool
- [ ] Migrate agentic endpoints (3 endpoints)
- [ ] Test: Chat streaming works

### Week 3: Repository + Issues
- [ ] Implement repository tree generation
- [ ] Implement file content reading
- [ ] Create IssueRAG system
- [ ] Implement issue analysis pipeline
- [ ] Migrate repository endpoints (2 endpoints)
- [ ] Migrate issue endpoints (6 endpoints)
- [ ] Test: Issue analysis works

### Week 4: Cutover
- [ ] Performance benchmarking
- [ ] Error handling polish
- [ ] Documentation
- [ ] Feature flag setup
- [ ] Monitoring setup
- [ ] Production deployment
- [ ] Rollback plan documented

---

## Success Criteria

### Functional Requirements
- ✅ All 17 frontend API functions work
- ✅ SSE streaming format matches Python
- ✅ snake_case responses preserved
- ✅ Session persistence works (LibSQL/Redis)
- ✅ Issue analysis quality equivalent

### Performance Requirements
- ✅ Response time within 20% of Python
- ✅ P95 latency < 2 seconds for chat
- ✅ Session creation < 5 seconds
- ✅ Issue analysis < 30 seconds

### Quality Requirements
- ✅ Content similarity > 95% (parallel run)
- ✅ Error rate < 0.1%
- ✅ Unit test coverage > 80%
- ✅ Integration tests pass

### Operational Requirements
- ✅ Monitoring dashboards
- ✅ Logging/tracing
- ✅ Rollback procedure
- ✅ Documentation complete

---

## Estimated Effort

### LOC Migration
- **Must Migrate**: 8,000 LOC Python → ~4,000 LOC TypeScript (50% reduction)
- **Skip Entirely**: 14,000 LOC (64% of backend)

### Time Estimates
- **Week 1**: Foundation + Sessions (2 engineers)
- **Week 2**: Agentic Query (2 engineers)
- **Week 3**: Repository + Issues (2 engineers)
- **Week 4**: Polish + Cutover (1 engineer)

**Total**: 4 weeks, 2 engineers = 8 engineer-weeks

### Cost Savings
- **Development time**: 50% reduction (Mastra abstractions)
- **Maintenance burden**: 52% LOC reduction
- **Single language**: No Python↔TypeScript context switching

---

## Risk Mitigation

### High Risk: SSE Streaming Format
**Mitigation**: Unit tests for chunk serialization, parallel run validation

### High Risk: RAG Quality Degradation
**Mitigation**: A/B testing, manual review of 100 sample queries

### Medium Risk: Performance Regression
**Mitigation**: Load testing, benchmarking, caching strategy

### Low Risk: Migration Bugs
**Mitigation**: Feature flag, gradual rollout, rollback plan

---

## Conclusion

By focusing on the **36% of code that the frontend actually uses**, we can:

1. ✅ Migrate in 4 weeks instead of 12 weeks
2. ✅ Skip 14,000 LOC of unused Python code
3. ✅ Reduce total codebase by 52%
4. ✅ Maintain 100% feature parity with frontend

**Next Step**: Begin Week 1 migration (Foundation + Sessions)
