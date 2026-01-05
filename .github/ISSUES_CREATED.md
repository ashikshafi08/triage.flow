# GitHub Issues Created for Migration

**Created:** 2026-01-05
**Total Issues:** 30 (1 Epic + 29 Sub-issues)
**Repository:** ashikshafi08/triage.flow
**Branch:** feat/migrate-agents-to-mastra

---

## Epic Issue

- **#4**: [EPIC] LlamaIndex → Mastra Migration
  - URL: https://github.com/ashikshafi08/triage.flow/issues/4
  - Central tracking dashboard with 29-task checklist

---

## Pre-Migration (3 days)

- **#5**: [Pre-Migration] Delete dead code (1,952 lines)
- **#6**: [Pre-Migration] Consolidate 3 RAG systems → 1 unified RAG
- **#7**: [Pre-Migration] Apply decorators (200-300 lines)

**Goal:** Clean up codebase before migration begins

---

## Phase 0: Foundation (Week 1)

- **#8**: [Phase 0] Set up Mastra monorepo structure
- **#9**: [Phase 0] Implement CodebaseRag (replaces new_rag.py)
- **#10**: [Phase 0] Implement ContextManager (replaces context_manager.py)
- **#11**: [Phase 0] Set up LibSQL storage + vector search
- **#12**: [Phase 0] Create unit tests for RAG indexing

**Goal:** Build TypeScript foundation for migration

---

## Phase 1: Session Management (Week 2)

- **#13**: [Phase 1] Implement createSession tool (native, not proxy)
- **#14**: [Phase 1] Implement searchCodebase tool (native)
- **#15**: [Phase 1] Create repositoryIndexing workflow
- **#16**: [Phase 1] Add session API routes with snake_case middleware
- **#17**: [Phase 1] Set up parallel validation script
- **#18**: [Phase 1] Frontend integration testing

**Goal:** Session creation and indexing work without Python

---

## Phase 2: Agentic Chat (Week 3)

- **#19**: [Phase 2] Create orchestrator agent
- **#20**: [Phase 2] Create codeAnalysis agent
- **#21**: [Phase 2] Create issueResolution agent
- **#22**: [Phase 2] Implement SSE streaming for agentic query
- **#23**: [Phase 2] Set up thread-scoped memory

**Goal:** Chat functionality fully migrated to Mastra

---

## Phase 3: Repository & Issues (Week 4)

- **#24**: [Phase 3] Implement GitHub tools (issues, PRs)
- **#25**: [Phase 3] Implement Git tools (blame, history, diff)
- **#26**: [Phase 3] Implement file operations (tree, read, search)
- **#27**: [Phase 3] Add repository API routes
- **#28**: [Phase 3] Add issue analysis API routes
- **#29**: [Phase 3] Migrate 22 frequently-used tools

**Goal:** All 17 frontend API functions work with Mastra

---

## Phase 4: Cutover & Cleanup (Week 5)

- **#30**: [Phase 4] Performance testing + optimization
- **#31**: [Phase 4] Feature flag deployment (10% → 50% → 100%)
- **#32**: [Phase 4] Delete Python codebase (19,390 lines)
- **#33**: [Phase 4] Update deployment configs + documentation

**Goal:** Full production cutover and Python deletion

---

## Labels Created

### Phase Labels
- `pre-migration` - Cleanup (Days 1-3)
- `phase-0` - Foundation (Week 1)
- `phase-1` - Sessions (Week 2)
- `phase-2` - Chat (Week 3)
- `phase-3` - Features (Week 4)
- `phase-4` - Cutover (Week 5)

### Priority Labels
- `p0` - Critical, blocking
- `p1` - High priority
- `p2` - Medium priority
- `p3` - Nice to have

### Category Labels
- `epic` - Epic tracking issue
- `migration` - Migration related
- `mastra` - Mastra-specific
- `cleanup` - Deleting code
- `refactor` - Restructuring
- `rag` - RAG-related
- `agents` - Agent work
- `testing` - Test improvements
- `performance` - Performance optimization
- `deployment` - Deployment related
- `documentation` - Docs updates

---

## Next Steps

1. **Start with Issue #5**: Delete dead code (1 day)
   ```bash
   gh issue develop 5 --checkout
   # OR
   git checkout -b pre-migration/delete-dead-code
   ```

2. **Track Progress**: Update Epic Issue #4 after each completion

3. **Create PRs**: Use "Closes #X" in PR description to auto-link

4. **Weekly Review**: Check Epic Issue dashboard every Friday

---

## GitHub Workflow Commands

```bash
# View all migration issues
gh issue list --label "migration"

# View issues for current phase
gh issue list --label "pre-migration" --state open

# Start working on an issue
gh issue develop 5 --checkout

# Create PR when done
gh pr create --fill
# Automatically links to issue via "Closes #X"

# View Epic progress
gh issue view 4
```

---

## Success Metrics

Upon completion:
- ✅ 27,000 → 13,000 LOC (52% reduction)
- ✅ Python + TypeScript → TypeScript only
- ✅ 100% feature parity (17 frontend functions)
- ✅ <20% latency increase
- ✅ Zero Python code in production

---

**Migration Duration:** 5 weeks + 3 days
**Created By:** AI-assisted migration planning
**Last Updated:** 2026-01-05
