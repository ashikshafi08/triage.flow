# Delete Dead Code (Pre-Migration Day 1)

**Epic:** #[EPIC_NUMBER]
**Phase:** Pre-Migration
**Effort:** 1 day
**Priority:** P0 (Blocking)

---

## 📋 **Description**

Remove 1,952 lines of unused/dead code to clean up the codebase before Mastra migration.

**Context:**
- Agent audit identified 290 lines of empty/demo test files
- 1,662 lines of utilities used by ≤1 endpoint
- Removing these simplifies migration planning

---

## ✅ **Acceptance Criteria**

- [ ] All empty test files deleted (290 lines)
- [ ] Unused utilities verified and removed (1,662 lines)
- [ ] Python `__pycache__` directories cleaned
- [ ] Remaining tests still pass
- [ ] No breaking changes to existing functionality

---

## 📝 **Tasks**

### Immediate Deletions (100% Safe)
```bash
# Empty/demo test files
rm tests/test_new_rag.py
rm tests/test_vulnerable_temp.py
rm tests/test_safety_api_demo.py
rm tests/test_codesplitter.py
rm tests/test_token.py

# Python cache cleanup
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
```

### Verify Before Deleting
```bash
# Check if triage_bot.py is only used in 1 endpoint
grep -r "triage_bot" src/api/routers/
# Expected: Only issues.py line 695

# Check conversation_memory.py usage
grep -r "conversation_memory" src/
# Expected: Only chat.py uses it (verify chat works without it)

# Check response_formatter.py usage
grep -r "response_formatter" src/
# Expected: Only response_handling.py with fallback logic
```

### Delete After Verification
```bash
rm src/triage_bot.py                # 487 lines
rm src/conversation_memory.py       # 326 lines
rm src/response_formatter.py        # 583 lines
rm src/language_config.py           # 266 lines (if new_rag.py is low usage)
```

---

## 🧪 **Testing Plan**

1. **Before deletion:**
   ```bash
   pytest tests/ -v --tb=short
   # Record: X tests passed
   ```

2. **After each deletion:**
   ```bash
   pytest tests/ -v --tb=short
   # Verify: X tests still pass (no reduction except deleted tests)
   ```

3. **API endpoint verification:**
   ```bash
   # Start Python server
   uvicorn src.main:app --reload

   # Test critical endpoints
   curl -X POST localhost:8000/assistant/sessions -d '{"repo_url":"..."}'
   curl -X POST localhost:8000/assistant/sessions/{id}/agentic-query -d '{"content":"..."}'
   ```

---

## 📊 **Success Metrics**

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Python LOC | 22,297 | ? | 20,345 (-1,952) |
| Test files | 10 | 5 | 5 |
| Python cache | ~5-10 MB | 0 MB | Clean |
| Passing tests | X | X | Same |

---

## ⚠️ **Risks**

| Risk | Mitigation |
|------|------------|
| Breaking undocumented dependencies | Grep for usage before deleting |
| Tests fail after deletion | Run pytest after each step |
| Frontend breaks | Test critical user flows |

---

## 📚 **References**

- [Dead Code Analysis Report](../../docs/DEAD_CODE_ANALYSIS.md)
- Agent 2 audit findings

---

## 🔗 **Related Issues**

- Blocks: #[ISSUE_2] (RAG consolidation)
- Epic: #[EPIC_NUMBER]

---

**Assignee:** TBD
**Labels:** `pre-migration` `cleanup` `technical-debt` `p0`
**Estimate:** 1 day
