# Migration Workflow Guide

This document explains how we're managing the LlamaIndex → Mastra migration using industry-standard GitHub workflows.

---

## 🏗️ **Structure Overview**

```
GitHub Discussion (RFC)
    ↓
Epic Issue (Tracking)
    ↓
GitHub Project (Kanban)
    ↓
Sub-Issues (Tasks) → Pull Requests → Merge
```

---

## 📚 **Components**

### 1. **GitHub Discussion** - RFC (Request for Comments)

**Purpose:** Gather team feedback on migration approach

**What it contains:**
- Problem statement & motivation
- Proposed architecture
- Timeline & phases
- Risks & mitigations
- Open questions for team discussion

**When to use:**
- Before starting major refactors
- When making architectural decisions
- To gather consensus

**Example:**
```
Title: [RFC] LlamaIndex → Mastra Migration Plan
URL: https://github.com/ashikshafi08/triage.flow/discussions/X

Comments:
- @alice: "Agree with hybrid RAG approach"
- @bob: "Concern: Timeline might be tight"
- @charlie: "Suggestion: Add performance benchmarks"
```

---

### 2. **Epic Issue** - Central Tracking

**Purpose:** Single source of truth for overall migration progress

**What it contains:**
- Progress checklist (29 tasks)
- Links to all sub-issues
- Metrics dashboard
- Status updates
- Active risks

**Why it's useful:**
- See entire migration at a glance
- Track progress week by week
- Identify blockers quickly

**Example:**
```
Title: [EPIC] LlamaIndex → Mastra Migration
Progress: [███▱▱▱▱▱▱▱] 30% (9/29 tasks)

Current Phase: Phase 1 (Session Management)
Blockers: None
```

---

### 3. **GitHub Project** - Visual Kanban

**Purpose:** Track task status visually

**Columns:**
1. **Backlog** - Future work
2. **To Do** - Ready to start
3. **In Progress** - Actively working
4. **Review** - PR submitted
5. **Done** - Merged & deployed

**Custom Fields:**
- **Phase**: Pre-Migration | Phase 0-4
- **Priority**: P0 (Critical) | P1 (High) | P2 (Medium) | P3 (Low)
- **LOC Impact**: Lines of code added/removed
- **Estimate**: Days of effort

**Example View:**
```
┌─────────┬──────────┬────────────┬─────────┬──────┐
│ Backlog │ To Do    │ In Progress│ Review  │ Done │
├─────────┼──────────┼────────────┼─────────┼──────┤
│ #5      │ #4       │ #1         │ #3      │ #2   │
│ #6      │          │            │         │      │
│ #7      │          │            │         │      │
└─────────┴──────────┴────────────┴─────────┴──────┘
```

---

### 4. **Sub-Issues** - Granular Tasks

**Purpose:** Bite-sized work items (1-3 days each)

**Anatomy of a good issue:**
```markdown
# Title: Delete dead code (1,952 lines)

Epic: #[EPIC_NUMBER]
Phase: Pre-Migration
Effort: 1 day
Priority: P0

## Description
Remove unused code to simplify codebase before migration.

## Acceptance Criteria
- [ ] Empty test files deleted
- [ ] Unused utilities removed
- [ ] Tests still pass

## Tasks
1. Delete empty test files
2. Verify utility usage with grep
3. Run pytest to confirm no breaks

## Testing Plan
...

## Success Metrics
- LOC: 22,297 → 20,345 (-1,952)

## Risks
...
```

**Labels:**
- `pre-migration`, `phase-0`, `phase-1`, etc. - Which phase
- `p0`, `p1`, `p2`, `p3` - Priority
- `cleanup`, `refactor`, `rag`, `agents` - Category
- `blocked` - Can't start yet

---

### 5. **Pull Requests** - Implementation

**Purpose:** Code review & merge

**Best Practices:**
```markdown
# Title: [Pre-Migration] Delete dead code

Closes #1
Epic: #[EPIC_NUMBER]

## Changes
- Deleted 5 empty test files (290 lines)
- Removed triage_bot.py (487 lines)
- Cleaned Python cache

## Testing
- ✅ pytest: 45/45 tests pass
- ✅ API endpoints: All working
- ✅ Frontend: No regressions

## Metrics
- LOC: 22,297 → 20,345 (-1,952)

## Screenshots
[Before/After git diff stat]
```

**Review checklist:**
- [ ] Tests pass
- [ ] No breaking changes
- [ ] Documentation updated
- [ ] Epic issue checklist updated

---

## 🔄 **Typical Workflow**

### Day-to-Day Process

```bash
# 1. Pick next issue from "To Do" column
gh project item-move [ISSUE_ID] --column "In Progress"

# 2. Create feature branch
git checkout -b pre-migration/delete-dead-code

# 3. Do the work
rm tests/test_*.py
pytest

# 4. Commit with issue reference
git add -A
git commit -m "chore: delete dead code

- Removed 5 empty test files
- Cleaned Python cache
- All tests still passing

Closes #1"

# 5. Create PR
gh pr create --fill
# Automatically links to issue via "Closes #1"

# 6. Move issue to Review
# (Automatically happens when PR created)

# 7. After review + approval
gh pr merge --squash

# 8. Issue automatically moves to "Done"

# 9. Update Epic issue
# (Manually check the box)
```

---

## 🎯 **Sprint Planning**

### Weekly Cycle

**Monday (Planning):**
```bash
# 1. Review Epic progress
open https://github.com/ashikshafi08/triage.flow/issues/[EPIC_NUMBER]

# 2. Move next week's issues to "To Do"
gh project item-move [ISSUE_4] --column "To Do"
gh project item-move [ISSUE_5] --column "To Do"
gh project item-move [ISSUE_6] --column "To Do"

# 3. Team assigns issues
gh issue edit [ISSUE_4] --add-assignee @alice
gh issue edit [ISSUE_5] --add-assignee @bob
```

**Daily (Standup):**
- What did I complete yesterday?
- What am I working on today?
- Any blockers?

**Friday (Retrospective):**
- Update Epic issue with status
- Identify risks
- Adjust next week's plan

---

## 📊 **Tracking Progress**

### Epic Issue Checklist
Update manually after each PR merge:

```markdown
### Phase 1: Session Management

- [x] #9: Implement createSession tool
- [x] #10: Implement searchCodebase tool
- [ ] #11: Create repositoryIndexing workflow
- [ ] #12: Add session API routes
- [ ] #13: Set up parallel validation
- [ ] #14: Frontend integration testing

Progress: [███▱▱▱] 33% (2/6 tasks)
```

### GitHub Project Automation

Issues automatically move:
- **Backlog → To Do**: Manual (during planning)
- **To Do → In Progress**: When you start working
- **In Progress → Review**: When PR created (via "Closes #X")
- **Review → Done**: When PR merged

---

## 🏷️ **Labeling Strategy**

### Phase Labels
- `pre-migration` - Cleanup work (Days 1-3)
- `phase-0` - Foundation (Week 1)
- `phase-1` - Sessions (Week 2)
- `phase-2` - Chat (Week 3)
- `phase-3` - Full features (Week 4)
- `phase-4` - Cutover (Week 5)

### Priority Labels
- `p0` - Blocking, must do now
- `p1` - High priority
- `p2` - Medium priority
- `p3` - Nice to have

### Category Labels
- `cleanup` - Deleting code
- `refactor` - Restructuring
- `rag` - RAG-related
- `agents` - Agent work
- `testing` - Test improvements
- `documentation` - Docs updates

### Status Labels
- `blocked` - Can't start (dependency)
- `in-review` - PR submitted
- `needs-feedback` - Waiting for input

---

## 💡 **Example Scenarios**

### Scenario 1: Starting Issue #1

```bash
# 1. Check Epic to understand context
gh issue view [EPIC_NUMBER]

# 2. Read issue details
gh issue view 1

# 3. Move to "In Progress"
gh project item-move 1 --column "In Progress"

# 4. Create branch
git checkout -b pre-migration/delete-dead-code

# 5. Do the work...

# 6. Create PR
gh pr create \
  --title "[Pre-Migration] Delete dead code" \
  --body "Closes #1" \
  --label "pre-migration,p0"

# 7. PR automatically links to issue and moves it to Review
```

---

### Scenario 2: Issue is Blocked

```bash
# 1. Issue #5 depends on #4 finishing
gh issue edit 5 --add-label "blocked"

# 2. Add comment explaining
gh issue comment 5 --body "Blocked by #4 (monorepo setup)"

# 3. Work on something else
gh project item-move 6 --column "In Progress"

# 4. When #4 merges, unblock
gh issue edit 5 --remove-label "blocked"
gh project item-move 5 --column "To Do"
```

---

### Scenario 3: Risk Identified

```bash
# 1. Update Epic issue
gh issue comment [EPIC_NUMBER] --body "
## ⚠️ New Risk

**Risk**: Mastra RAG quality lower than expected
**Impact**: High
**Mitigation**: Keep Python RAG as fallback for Phase 1-2
**Owner**: @alice
**Status**: 🟡 Monitoring
"

# 2. Create new issue if needed
gh issue create \
  --title "Investigate Mastra RAG quality" \
  --body "Epic: #[EPIC_NUMBER]

Discovered during Phase 0 testing that Mastra RAG returns
lower quality results than LlamaIndex for code search.

**Action Items:**
- [ ] Run benchmark comparison
- [ ] Document quality gaps
- [ ] Decide: Improve Mastra or keep Python fallback
" \
  --label "phase-0,rag,p1,investigation"
```

---

## 🚀 **Quick Start Commands**

### Create Everything
```bash
# Run the automated script
./.github/scripts/create_migration_issues.sh

# Or manually:
gh discussion create --title "[RFC] Migration" --body-file .github/DISCUSSION_TEMPLATE.md
gh project create --title "Mastra Migration"
gh issue create --title "[EPIC] Migration" --body-file .github/EPIC_ISSUE.md
```

### Daily Workflow
```bash
# View your assigned issues
gh issue list --assignee @me

# Start working on issue
gh issue develop 1 --checkout

# Create PR when done
gh pr create --fill

# Check project status
gh project view [PROJECT_NUMBER]
```

---

## 📚 **References**

- [GitHub Projects Documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [GitHub Discussions Guide](https://docs.github.com/en/discussions)
- [GitHub CLI Manual](https://cli.github.com/manual/)

---

## ✅ **Checklist for Starting Migration**

- [ ] Read RFC Discussion
- [ ] Understand Epic Issue structure
- [ ] Familiarize with Project board
- [ ] Pick first issue from "To Do"
- [ ] Follow daily workflow
- [ ] Keep Epic updated weekly

---

**This is how professional teams manage large-scale refactors!** 🎯
