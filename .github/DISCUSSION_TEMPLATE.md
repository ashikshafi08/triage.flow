# [RFC] LlamaIndex → Mastra Migration Plan

## 📋 **Summary**

Migrate triage.flow's agentic system from Python (LlamaIndex) to TypeScript (Mastra), reducing codebase by 82% while maintaining full feature parity.

**Current State:** 22,297 lines Python + scattered architecture
**Target State:** 4,000 lines TypeScript + unified Mastra architecture
**Timeline:** 5 weeks (with 3-day pre-migration cleanup)
**Risk Level:** Medium (phased approach with parallel validation)

---

## 🎯 **Problem Statement**

### What We're Solving

1. **Language Fragmentation**: Python backend + TypeScript frontend creates context-switching overhead
2. **Architectural Debt**:
   - 3 duplicate RAG implementations (1,355 lines doing same thing)
   - 1,480 lines of duplicated code across 47 files
   - 21 scattered files in src/ root
3. **Maintenance Burden**:
   - Complex LlamaIndex event-driven workflows (~2,000 lines)
   - Manual orchestration code that Mastra handles natively
4. **Scale Issues**:
   - 64% of Python code unused by frontend
   - 15% can be deleted as dead code

### Why Now?

- Mastra v0.24.9 is stable with production-ready features
- We have comprehensive SPEC.md documenting migration path
- 5 agent audits completed showing clear consolidation opportunities
- Frontend API contracts well-documented

---

## 📊 **Audit Findings**

### Code Analysis (5 Specialized Agents)

| Finding | Details | Impact |
|---------|---------|--------|
| **Dead Code** | 1,952 lines unused (tests, demos, utilities) | Delete immediately |
| **RAG Duplication** | 3 files (1,355 lines) → can be 1 file (600 lines) | 55% reduction |
| **Code Duplication** | 1,480 lines duplicated patterns | 810-1,120 line savings |
| **Frontend Usage** | Only 17/51 API functions used (33%) | 64% of code skippable |
| **Feature Inventory** | 45 tools, only 22 frequently used | Focus on 23% must-haves |

### What Frontend Actually Uses

```
17 API Functions:
├─ Session Management (7): create, list, status, delete, metadata, messages, enable
├─ Chat (2): sendMessage (SSE), resetMemory
├─ Repository (2): getTree, getFileContent
└─ Issues (6): analyze, getCached, postToGitHub, list, delete, listPRs
```

**Critical Finding:** Workflow router (835 lines) has 0 frontend calls → DELETE

---

## 🏗️ **Proposed Architecture**

### Before (Python/LlamaIndex)
```
React Frontend :5173
    ↓
Python FastAPI :8000
    ├─ LlamaIndex Workflows (2,000 LOC)
    ├─ AgenticRAG (3 files, 1,355 LOC)
    ├─ Agent Tools (21 files, 8,736 LOC)
    ├─ Session Manager + Memory
    └─ GitHub Client
```

### After (TypeScript/Mastra)
```
React Frontend :5173
    ↓
Mastra Server :4111
    ├─ Workflows (Mastra native, 300 LOC)
    ├─ RAG (Unified, 600 LOC → @mastra/rag)
    ├─ Agents (Mastra, 800 LOC)
    ├─ Tools (Native, 1,200 LOC)
    └─ Memory (Mastra LibSQL)
```

---

## 📅 **Proposed Timeline**

### Pre-Migration (3 Days)
**Goal:** Clean up Python codebase for clarity

- **Day 1**: Delete dead code (1,952 lines)
  - Empty test files
  - Unused utilities (triage_bot, conversation_memory, response_formatter)
  - Python cache cleanup

- **Day 2-3**: Consolidate RAG systems
  - Create `unified_rag.py` combining code + issue indexing
  - Delete `new_rag.py`, `issue_rag.py`, `agentic_rag.py`
  - Apply existing decorators to reduce boilerplate

**Outcome:** 22,297 → 19,390 LOC (-13%), clearer interfaces

---

### Phase 0: Foundation (Week 1)
**Goal:** Core infrastructure working

```typescript
packages/mastra/src/
├── rag/
│   ├── codebaseRag.ts      (replaces new_rag.py)
│   ├── issueRag.ts         (replaces issue_rag.py)
│   └── compositeRag.ts     (replaces agentic_rag.py)
├── context/
│   └── manager.ts          (replaces context_manager.py)
└── db/
    └── storage.ts          (LibSQL setup)
```

**Deliverables:**
- [ ] Mastra monorepo setup
- [ ] LibSQL + vector storage configured
- [ ] CodebaseRag can index repository
- [ ] Semantic search returns relevant results
- [ ] Unit tests pass

**Success Criteria:**
```bash
curl -X POST localhost:4111/test/index \
  -d '{"repo_url":"https://github.com/facebook/react"}'
# Returns: {"indexed": true, "files": 1234, "chunks": 5678}
```

---

### Phase 1: Session Management (Week 2)
**Goal:** Frontend can create sessions

```typescript
packages/mastra/src/
├── server/routes/
│   └── sessions.ts         (7 endpoints)
├── workflows/
│   └── repositoryIndexing.ts
└── tools/
    ├── createSession.ts    (native, not HTTP proxy)
    └── searchCodebase.ts   (native)
```

**Deliverables:**
- [ ] POST `/assistant/sessions` creates session + indexes repo
- [ ] GET `/assistant/sessions` lists sessions
- [ ] GET `/assistant/sessions/:id/status` returns progress
- [ ] DELETE `/assistant/sessions/:id` cleanup
- [ ] snake_case response middleware
- [ ] Parallel validation: Mastra vs Python API

**Success Criteria:**
- Frontend session creation works without Python
- Indexing completes with status tracking
- <20% latency difference from Python

---

### Phase 2: Agentic Chat (Week 3)
**Goal:** SSE streaming chat works

```typescript
packages/mastra/src/
├── agents/
│   ├── orchestrator.ts     (main coordinator)
│   ├── codeAnalysis.ts     (code expert)
│   └── issueResolution.ts  (issue expert)
├── workflows/
│   └── agenticQuery.ts
└── server/routes/
    └── chat.ts             (SSE streaming)
```

**Deliverables:**
- [ ] POST `/assistant/sessions/:id/agentic-query` with SSE
- [ ] Streaming format matches Python exactly:
  ```
  data: {"type": "step", "step": {...}}
  data: {"type": "final", "final_answer": "...", "steps": [...]}
  ```
- [ ] Agents use RAG from Phase 0
- [ ] Memory persists across requests
- [ ] POST `/assistant/sessions/:id/reset-memory`

**Success Criteria:**
- Chat responses stream correctly
- Quality matches Python (95%+ similarity)
- Frontend works without code changes

---

### Phase 3: Repository & Issues (Week 4)
**Goal:** Full feature parity

```typescript
packages/mastra/src/
├── tools/
│   ├── github/
│   │   ├── issues.ts
│   │   └── pullRequests.ts
│   ├── git/
│   │   ├── blame.ts
│   │   └── history.ts
│   └── codebase/
│       ├── fileTree.ts
│       └── fileRead.ts
└── server/routes/
    ├── repository.ts       (2 endpoints)
    └── issues.ts           (6 endpoints)
```

**Deliverables:**
- [ ] GET `/api/tree` returns file tree
- [ ] GET `/api/file-content` returns file content
- [ ] POST `/api/analyze-issue` analyzes GitHub issues
- [ ] GET `/api/cached-analyses/:id` retrieves cached results
- [ ] POST `/api/post-to-github` posts analysis comments
- [ ] All 22 frequently-used tools implemented natively

**Success Criteria:**
- All frontend features work
- No HTTP calls to Python API
- Issue analysis quality equivalent

---

### Phase 4: Cutover & Cleanup (Week 5)
**Goal:** Production ready, Python deleted

**Deliverables:**
- [ ] Performance testing (load test with 100 concurrent users)
- [ ] Feature flag deployment (gradual rollout: 10% → 50% → 100%)
- [ ] Monitoring dashboards (Mastra metrics)
- [ ] DELETE entire `src/` Python directory (19,390 lines)
- [ ] DELETE `requirements.txt`
- [ ] Update deployment configs (single Node.js container)
- [ ] Documentation updates

**Success Criteria:**
- All frontend operations work
- p95 latency <20% difference from Python
- Zero Python code in production
- Rollback plan tested

---

## 🎯 **Success Metrics**

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total LOC | 27,000 | 13,000 | -52% |
| Backend LOC | 22,000 | 4,000 | -82% |
| Languages | 2 | 1 | Unified |
| RAG Files | 3 | 1 | Consolidated |
| API Latency (p95) | Baseline | <1.2x | <20% increase |
| Feature Coverage | 100% | 100% | No regressions |
| Test Coverage | 40% | 80% | Improved |

---

## ⚠️ **Risks & Mitigations**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Mastra RAG quality < LlamaIndex | High | Medium | Keep Python RAG as fallback in Phase 1-2 |
| Performance regression | Medium | Low | Parallel validation every phase |
| Feature gaps discovered | Medium | Medium | Comprehensive testing checklist |
| Team unfamiliarity with Mastra | Low | High | Pair programming, docs review |
| Breaking frontend | High | Low | API contracts preserved, snake_case enforced |

### Rollback Strategy

**Phase 1-3 Rollback:**
- Keep Python API running on :8000
- Frontend switches via env var `VITE_API_URL`
- Zero downtime

**Phase 4+ Rollback:**
```bash
git checkout main -- src/
docker-compose up python-api
# Update frontend: VITE_API_URL=http://localhost:8000
```

---

## 📝 **Open Questions**

1. **LibSQL vs PostgreSQL for vectors?**
   - SPEC.md recommends LibSQL (simpler), but pgvector more mature
   - **Decision needed:** Test LibSQL vector search quality first week

2. **Parallel validation threshold?**
   - How similar must outputs be? (Proposed: 95% content similarity)
   - **Decision needed:** Define comparison methodology

3. **Feature flag granularity?**
   - Per-endpoint or all-or-nothing?
   - **Decision needed:** Gradual rollout strategy

4. **Deployment approach?**
   - Blue-green deployment or canary?
   - **Decision needed:** Infrastructure requirements

---

## 💬 **Discussion Points**

**For Team Feedback:**

1. **Timeline:** Is 5 weeks realistic? Should we add buffer?
2. **Pre-migration cleanup:** Worth 3 days or start migration immediately?
3. **RAG strategy:** Hybrid (keep Python RAG) or pure Mastra?
4. **Testing:** Automated parallel validation or manual QA?
5. **Rollout:** Gradual (10% → 50% → 100%) or big bang?

**Comment below with:**
- ✅ Agree with plan
- 🤔 Concerns about...
- 💡 Suggestion:...
- ❓ Question:...

---

## 📚 **References**

- [SPEC.md](../SPEC.md) - Detailed migration specification
- [FRONTEND_BACKEND_DEPENDENCY_MAP.md](../FRONTEND_BACKEND_DEPENDENCY_MAP.md) - API usage analysis
- [FEATURE_INVENTORY.md](../FEATURE_INVENTORY.md) - Complete feature list
- [Mastra Documentation](https://mastra.ai/docs)
- [Agent Audit Reports](../.github/AGENT_AUDITS.md) - 5 agent analyses

---

**Status:** 🟡 Awaiting Team Feedback
**Created:** 2026-01-05
**Author:** Migration Team
**Next Steps:** Gather feedback → Create Epic Issue → Start pre-migration cleanup
