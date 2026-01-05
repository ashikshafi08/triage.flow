# [EPIC] LlamaIndex → Mastra Migration

**Related Discussion:** #[DISCUSSION_NUMBER]
**Project Board:** [Migration Tracking](https://github.com/ashikshafi08/triage.flow/projects/[PROJECT_NUMBER])
**Timeline:** 5 weeks + 3 days pre-migration
**Status:** 🔴 Not Started

---

## 📊 **Progress Overview**

```
[▱▱▱▱▱▱▱▱▱▱] 0% Complete (0/29 tasks)

Pre-Migration: ▱▱▱ (0/3 tasks)
Phase 0:       ▱▱▱▱▱ (0/5 tasks)
Phase 1:       ▱▱▱▱▱▱ (0/6 tasks)
Phase 2:       ▱▱▱▱▱ (0/5 tasks)
Phase 3:       ▱▱▱▱▱▱ (0/6 tasks)
Phase 4:       ▱▱▱▱ (0/4 tasks)
```

---

## 🎯 **High-Level Goals**

- [ ] Reduce codebase from 27,000 → 13,000 LOC (52% reduction)
- [ ] Migrate from Python/LlamaIndex to TypeScript/Mastra
- [ ] Maintain 100% feature parity (17 frontend API functions)
- [ ] Achieve <20% latency increase
- [ ] Zero Python code in production

---

## 📋 **Task Checklist**

### Pre-Migration Cleanup (3 days)

- [ ] #[ISSUE_1]: Delete dead code (1,952 lines)
- [ ] #[ISSUE_2]: Consolidate 3 RAG systems into 1 unified RAG
- [ ] #[ISSUE_3]: Apply existing decorators to reduce boilerplate

**Success Criteria:**
- Python LOC reduced from 22,297 → 19,390
- Unified RAG interface ready for Mastra migration
- All existing tests still passing

---

### Phase 0: Foundation (Week 1)

- [ ] #[ISSUE_4]: Set up Mastra monorepo structure
- [ ] #[ISSUE_5]: Implement CodebaseRag (replaces new_rag.py)
- [ ] #[ISSUE_6]: Implement ContextManager (replaces context_manager.py)
- [ ] #[ISSUE_7]: Set up LibSQL storage + vector search
- [ ] #[ISSUE_8]: Create unit tests for RAG indexing

**Success Criteria:**
```bash
✅ pnpm --filter mastra tsc --noEmit  # Types compile
✅ pnpm --filter mastra test          # Unit tests pass
✅ Sample repo indexes successfully
✅ Semantic search returns relevant results
```

---

### Phase 1: Session Management (Week 2)

- [ ] #[ISSUE_9]: Implement createSession tool (native, not proxy)
- [ ] #[ISSUE_10]: Implement searchCodebase tool (native)
- [ ] #[ISSUE_11]: Create repositoryIndexing workflow
- [ ] #[ISSUE_12]: Add session API routes with snake_case middleware
- [ ] #[ISSUE_13]: Set up parallel validation script
- [ ] #[ISSUE_14]: Frontend integration testing

**Success Criteria:**
```bash
✅ POST /assistant/sessions works without Python
✅ Frontend session creation succeeds
✅ Indexing progress tracking works
✅ <20% latency difference from Python
```

---

### Phase 2: Agentic Chat (Week 3)

- [ ] #[ISSUE_15]: Create orchestrator agent
- [ ] #[ISSUE_16]: Create codeAnalysis agent
- [ ] #[ISSUE_17]: Create issueResolution agent
- [ ] #[ISSUE_18]: Implement SSE streaming for agentic query
- [ ] #[ISSUE_19]: Set up thread-scoped memory

**Success Criteria:**
```bash
✅ SSE streaming format matches Python exactly
✅ Chat responses quality ≥95% similarity to Python
✅ Memory persists across requests
✅ Frontend chat works without code changes
```

---

### Phase 3: Repository & Issues (Week 4)

- [ ] #[ISSUE_20]: Implement GitHub tools (issues, PRs)
- [ ] #[ISSUE_21]: Implement Git tools (blame, history, diff)
- [ ] #[ISSUE_22]: Implement file operations (tree, read, search)
- [ ] #[ISSUE_23]: Add repository API routes
- [ ] #[ISSUE_24]: Add issue analysis API routes
- [ ] #[ISSUE_25]: Migrate 22 frequently-used tools

**Success Criteria:**
```bash
✅ All 17 frontend API functions work
✅ No HTTP calls to Python API remain
✅ Issue analysis quality equivalent to Python
✅ File browsing works correctly
```

---

### Phase 4: Cutover & Cleanup (Week 5)

- [ ] #[ISSUE_26]: Performance testing + optimization
- [ ] #[ISSUE_27]: Feature flag deployment (10% → 50% → 100%)
- [ ] #[ISSUE_28]: Delete Python codebase (19,390 lines)
- [ ] #[ISSUE_29]: Update deployment configs + documentation

**Success Criteria:**
```bash
✅ Load test: 100 concurrent users
✅ p95 latency <1.2x Python baseline
✅ Zero Python code in production
✅ Rollback plan tested
✅ Monitoring dashboards operational
```

---

## 📈 **Metrics Dashboard**

| Metric | Baseline | Current | Target | Status |
|--------|----------|---------|--------|--------|
| Total LOC | 27,000 | 27,000 | 13,000 | 🔴 |
| Backend LOC | 22,000 | 22,000 | 4,000 | 🔴 |
| Languages | 2 | 2 | 1 | 🔴 |
| Test Coverage | 40% | 40% | 80% | 🔴 |
| API Latency (p95) | 450ms | N/A | <540ms | 🔴 |
| Feature Coverage | 100% | 100% | 100% | 🟢 |

---

## ⚠️ **Active Risks**

| Risk | Impact | Mitigation | Owner | Status |
|------|--------|------------|-------|--------|
| Mastra RAG quality | High | Parallel validation, Python fallback | TBD | 🟡 Monitoring |
| Performance regression | Medium | Profiling, optimization | TBD | 🟡 Monitoring |
| Feature gaps | Medium | Comprehensive testing | TBD | 🟡 Monitoring |

---

## 📚 **Related Documents**

- [RFC Discussion](#[DISCUSSION_NUMBER]) - Team feedback & decisions
- [SPEC.md](../SPEC.md) - Detailed technical specification
- [Agent Audit Reports](../docs/AGENT_AUDITS.md) - Analysis findings
- [Migration Guide](../docs/MIGRATION_GUIDE.md) - Implementation details

---

## 🔄 **Status Updates**

### Week 0 (Pre-Migration)
- TBD

### Week 1 (Phase 0)
- TBD

### Week 2 (Phase 1)
- TBD

### Week 3 (Phase 2)
- TBD

### Week 4 (Phase 3)
- TBD

### Week 5 (Phase 4)
- TBD

---

## 💬 **Quick Links**

- **Start Issue:** #[ISSUE_1] (Delete dead code)
- **Current Sprint:** N/A
- **Blockers:** None
- **Last Updated:** 2026-01-05

---

**Epic Owner:** @ashikshafi08
**Reviewers:** @team
**Labels:** `epic` `migration` `mastra` `refactor` `high-priority`
