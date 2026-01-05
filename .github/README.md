# Mastra Migration - GitHub Workflow

This directory contains all templates and scripts for managing the LlamaIndex → Mastra migration using professional GitHub workflows.

---

## 📁 **What's Here**

```
.github/
├── README.md                           ← You are here
├── DISCUSSION_TEMPLATE.md              ← RFC for team feedback
├── EPIC_ISSUE.md                       ← Central tracking issue
├── ISSUE_TEMPLATES/
│   └── 01_delete_dead_code.md          ← Example sub-issue template
└── scripts/
    └── create_migration_issues.sh      ← Automated setup script
```

---

## 🚀 **Quick Start (2 Options)**

### Option 1: Automated Setup (Recommended)
```bash
# Run the script to create everything automatically
./.github/scripts/create_migration_issues.sh

# This creates:
# - GitHub Discussion (RFC)
# - GitHub Project (Kanban board)
# - Epic Issue (tracking)
# - All 29 sub-issues
```

### Option 2: Manual Setup
```bash
# 1. Create Discussion
gh discussion create \
  --title "[RFC] LlamaIndex → Mastra Migration Plan" \
  --body-file .github/DISCUSSION_TEMPLATE.md \
  --category "General"

# 2. Create Project
gh project create \
  --owner ashikshafi08 \
  --title "Mastra Migration"

# 3. Create Epic Issue
gh issue create \
  --title "[EPIC] LlamaIndex → Mastra Migration" \
  --body-file .github/EPIC_ISSUE.md \
  --label "epic,migration,mastra"

# 4. Create individual sub-issues
# (Use ISSUE_TEMPLATES/ as reference)
```

---

## 📊 **Structure Hierarchy**

```
┌─────────────────────────────────────┐
│   GitHub Discussion (RFC)           │  ← Team discussion & feedback
│   "Should we migrate? How?"         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Epic Issue (Tracking)             │  ← Progress dashboard
│   [███▱▱▱▱▱▱▱] 30% (9/29 tasks)     │
└──────────────┬──────────────────────┘
               │
               ├──────────────────────┐
               │                      │
               ▼                      ▼
┌─────────────────────┐   ┌─────────────────────┐
│  GitHub Project     │   │  Sub-Issues         │
│  (Kanban Board)     │   │  (Granular Tasks)   │
│                     │   │                     │
│  Backlog │ Todo     │   │  #1: Delete code    │
│  In Prog │ Review   │   │  #2: Consolidate    │
│  Done    │          │   │  #3: Decorators     │
└─────────────────────┘   └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Pull Requests      │
                          │  (Implementation)   │
                          │                     │
                          │  PR #X: Closes #1   │
                          │  PR #Y: Closes #2   │
                          └─────────────────────┘
```

---

## 🎯 **Typical Day-to-Day Flow**

### 1. **Monday (Planning)**
- Review Epic Issue progress
- Move next issues to "To Do" column
- Assign issues to team members

### 2. **During the Week**
```bash
# Pick next issue
gh issue list --label "phase-0" --state open

# Start working
git checkout -b phase-0/monorepo-setup

# Do the work...
# Create PR when done
gh pr create --fill

# PR automatically links to issue and updates project
```

### 3. **Friday (Review)**
- Update Epic Issue with weekly status
- Identify any blockers or risks
- Plan next week's work

---

## 📝 **Creating New Sub-Issues**

Use this template structure:

```markdown
# [Title: Action-oriented, specific]

**Epic:** #[EPIC_NUMBER]
**Phase:** [Pre-Migration | Phase 0-4]
**Effort:** [1-3 days]
**Priority:** [P0-P3]

## Description
[What needs to be done and why]

## Acceptance Criteria
- [ ] Specific, measurable outcome
- [ ] Another outcome
- [ ] Tests pass

## Tasks
1. Concrete step 1
2. Concrete step 2

## Testing Plan
[How to verify it works]

## Success Metrics
- LOC: X → Y
- Performance: <20% increase

## Risks
- [What could go wrong]
```

---

## 🏷️ **Label Convention**

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
- `cleanup` - Deleting code
- `refactor` - Restructuring
- `rag` - RAG-related
- `agents` - Agent work
- `testing` - Tests

---

## 📚 **Related Documentation**

- [MIGRATION_WORKFLOW_GUIDE.md](../docs/MIGRATION_WORKFLOW_GUIDE.md) - Detailed process guide
- [SPEC.md](../SPEC.md) - Technical migration spec
- [FEATURE_INVENTORY.md](../FEATURE_INVENTORY.md) - What we're migrating
- [FRONTEND_BACKEND_DEPENDENCY_MAP.md](../FRONTEND_BACKEND_DEPENDENCY_MAP.md) - API dependencies

---

## 🔗 **Useful Commands**

### View Epic Progress
```bash
gh issue view [EPIC_NUMBER]
```

### View Project Board
```bash
gh project view [PROJECT_NUMBER]
```

### View Your Assigned Issues
```bash
gh issue list --assignee @me
```

### Create PR from Branch
```bash
gh pr create \
  --title "[Phase 0] Your change" \
  --body "Closes #X" \
  --label "phase-0,p0"
```

### Update Epic Checklist
```bash
# Manually edit the issue
gh issue edit [EPIC_NUMBER]
```

---

## ✅ **Pre-Flight Checklist**

Before running the setup script:

- [ ] GitHub CLI installed (`gh --version`)
- [ ] Authenticated (`gh auth status`)
- [ ] On correct repository (`ashikshafi08/triage.flow`)
- [ ] "General" discussion category exists
- [ ] Have write access to repository

---

## 🤝 **Team Collaboration**

### For Solo Work
- Use Discussion to document decisions
- Update Epic Issue weekly
- Keep Project board current

### For Team Work
- Assign issues during sprint planning
- Use PR reviews for code quality
- Comment on Epic Issue for status updates
- Tag teammates with @mentions in issues

---

## 📈 **Success Indicators**

You're doing it right if:

✅ Epic Issue shows clear progress (30%, 60%, 90%)
✅ Project board reflects reality
✅ Issues are small (1-3 days each)
✅ PRs link to issues (`Closes #X`)
✅ Discussion has team feedback
✅ No orphaned issues (all have context)

---

## 🆘 **Need Help?**

- **GitHub Docs**: https://docs.github.com/en/issues
- **GitHub CLI Manual**: https://cli.github.com/manual/
- **Migration Guide**: [../docs/MIGRATION_WORKFLOW_GUIDE.md](../docs/MIGRATION_WORKFLOW_GUIDE.md)

---

**Ready to start?** Run `./.github/scripts/create_migration_issues.sh` 🚀
