# Comprehensive Feature Inventory: Python Codebase

**Total Python LOC:** ~28,836 lines
**Repository:** triage.flow
**Purpose:** GitHub issue triage and repository analysis with AI agents

---

## Executive Summary

This inventory categorizes every feature in the Python codebase by:
1. **User-facing** (exposed via API endpoints)
2. **Backend-only** (internal implementation)
3. **Migration Priority** (must-have, should-have, can-skip)
4. **Mastra Compatibility** (easy, moderate, complex, LlamaIndex-specific)

---

## 1. USER-FACING FEATURES (Frontend-Exposed API Endpoints)

### 1.1 Session Management
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| Create repo session | `POST /assistant/sessions` | `sessions.py:100-134` | ~150 | **MUST** | Easy |
| List sessions | `GET /assistant/sessions` | `sessions.py:136-146` | ~50 | **MUST** | Easy |
| Get session status | `GET /assistant/sessions/{id}/status` | `sessions.py:148-159` | ~50 | **MUST** | Easy |
| Delete session | `DELETE /assistant/sessions/{id}` | `sessions.py:161-167` | ~30 | **MUST** | Easy |
| Get session metadata | `GET /assistant/sessions/{id}/metadata` | `sessions.py:169-180` | ~50 | Should | Easy |
| WebSocket session updates | `WS /ws/{id}` | `sessions.py:20-99` | ~150 | Should | Moderate |
| Sync repository data | `POST /assistant/sessions/{id}/sync-repository` | `sessions.py:320-406` | ~200 | Should | Moderate |

**Category Total:** ~680 LOC
**Migration Assessment:** 100% must migrate - core session functionality

---

### 1.2 Chat & Messaging
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| Send chat message | `POST /sessions/{id}/messages` | `chat.py:18-250` | ~450 | **MUST** | Moderate |
| Get session messages | `GET /sessions/{id}/messages` | `chat.py:314-327` | ~40 | **MUST** | Easy |
| Memory statistics | `GET /sessions/{id}/memory-stats` | `chat.py:252-275` | ~50 | Should | Easy |
| Add issue context | `POST /sessions/{id}/add-issue-context` | `chat.py:277-312` | ~80 | Should | Easy |
| Streaming responses | `stream=true` param | `chat.py:187-229` | ~150 | Should | Moderate |

**Category Total:** ~770 LOC
**Migration Assessment:** 90% must migrate - replace with Mastra agent streaming

---

### 1.3 Repository Browsing
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| List files | `GET /api/files` | `repository.py:11-28` | ~40 | **MUST** | Easy |
| Get file content | `GET /api/file-content` | `repository.py:30-111` | ~180 | **MUST** | Easy |
| Stream large files | `GET /api/file-content/stream` | `repository.py:113-129` | ~50 | Should | Moderate |
| Get tree structure | `GET /api/tree` | `repository.py:131-189` | ~120 | **MUST** | Easy |
| Get file snippet | `GET /api/file-snippet` | `repository.py:191-304` | ~250 | Should | Moderate |
| Get commit file diff | `GET /api/diff/{sha}/{path}` | `repository.py:478-633` | ~300 | Should | Moderate |

**Category Total:** ~940 LOC
**Migration Assessment:** 60% must migrate - file browsing is core, diff is nice-to-have

---

### 1.4 Issue Management
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| Get issue context | `POST /api/v1/issue_context` | `issues.py:16-51` | ~80 | **MUST** | Complex (RAG) |
| List issues | `GET /api/issues` | `issues.py:53-66` | ~40 | **MUST** | Easy |
| Get issue detail | `GET /api/issues/{num}` | `issues.py:68-81` | ~40 | **MUST** | Easy |
| Analyze issue | `POST /api/analyze-issue` | `issues.py:227-442` | ~500 | **MUST** | Complex (Agentic) |
| Get cached analyses | `GET /api/cached-analyses/{id}` | `issues.py:444-525` | ~180 | Should | Easy |
| Apply patch | `POST /api/apply-patch` | `issues.py:575-681` | ~250 | Should | Moderate |
| Post to GitHub | `POST /api/post-to-github` | `issues.py:688-720` | ~80 | Should | Easy |

**Category Total:** ~1,170 LOC
**Migration Assessment:** 80% must migrate - issue analysis is core value prop

---

### 1.5 Pull Request Management
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| List PRs | `GET /api/prs` | `issues.py:83-98` | ~40 | **MUST** | Easy |
| Get PR diff | Tool method | `pr_operations.py` | ~200 | Should | Easy |
| Get PR summary | Tool method | `pr_operations.py` | ~150 | Should | Moderate |
| Find open PRs | Tool method | `pr_operations.py` | ~180 | Should | Moderate |
| Check PR readiness | Tool method | `pr_operations.py` | ~150 | Should | Moderate |

**Category Total:** ~720 LOC
**Migration Assessment:** 50% must migrate - PR listing is core, analysis is nice-to-have

---

### 1.6 Commit History & Timeline
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| List commits | `GET /api/commits` | `issues.py:100-181` | ~180 | Should | Moderate |
| File timeline | `GET /api/timeline/file` | `timeline.py:14-176` | ~320 | **MUST** | Complex (Git+Index) |
| Hunk timeline | `GET /api/timeline/hunk` | `timeline.py:178-227` | ~120 | Should | Complex |
| Timeline preview | `GET /api/timeline/preview/{sha}/{path}` | `timeline.py:229-286` | ~150 | Should | Moderate |
| Create issue from timeline | `POST /api/timeline/create-issue` | `timeline.py:300-371` | ~180 | Should | Easy |
| Commit diff | `GET /api/timeline/{sha}/{path}` | `timeline.py:374-551` | ~350 | Should | Moderate |

**Category Total:** ~1,300 LOC
**Migration Assessment:** 40% must migrate - timeline is unique feature but complex

---

### 1.7 Agentic Query Processing
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| Enable agentic mode | `POST /assistant/sessions/{id}/enable-agentic` | `agentic.py:17-132` | ~250 | **MUST** | Easy (config) |
| Get agentic status | `GET /assistant/sessions/{id}/agentic-status` | `agentic.py:134-159` | ~60 | Should | Easy |
| Agentic query (streaming) | `POST /assistant/sessions/{id}/agentic-query` | `agentic.py:161-381` | ~450 | **MUST** | Complex (Mastra) |
| Reset agentic memory | `POST /assistant/sessions/{id}/reset-agentic-memory` | `agentic.py:383-401` | ~50 | Should | Easy |
| Get AgenticRAG info | `GET /assistant/sessions/{id}/agentic-rag-info` | `agentic.py:403-456` | ~120 | Should | Easy |
| Analyze query | `POST /assistant/sessions/{id}/analyze-query` | `agentic.py:458-511` | ~120 | Should | Easy |
| Context preview | `GET /assistant/sessions/{id}/context-preview` | `agentic.py:513-558` | ~100 | Should | Moderate |
| Get related issues | `GET /assistant/sessions/{id}/related-issues` | `agentic.py:560-627` | ~150 | Should | Moderate |
| Index issues | `POST /assistant/sessions/{id}/index-issues` | `agentic.py:629-675` | ~100 | Should | Moderate |
| Issue index status | `GET /assistant/sessions/{id}/issue-index-status` | `agentic.py:677-715` | ~90 | Should | Easy |

**Category Total:** ~1,490 LOC
**Migration Assessment:** 60% must migrate - agentic query is core, rest is nice-to-have

---

### 1.8 Workflow Management (LlamaIndex Workflows)
| Feature | Endpoint | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------|----------------|-----|----------|-------------------|
| Create workflow | `POST /assistant/sessions/{id}/workflows/create` | `workflows.py:73-127` | ~150 | Should | **LlamaIndex-specific** |
| Execute workflow | `POST /assistant/sessions/{id}/workflows/{wf}/execute` | `workflows.py:130-194` | ~180 | Should | **LlamaIndex-specific** |
| Get workflow status | `GET /assistant/sessions/{id}/workflows/{wf}/status` | `workflows.py:197-232` | ~100 | Should | **LlamaIndex-specific** |
| Pause workflow | `POST /assistant/sessions/{id}/workflows/{wf}/pause` | `workflows.py:235-269` | ~80 | Can skip | **LlamaIndex-specific** |
| Resume workflow | `POST /assistant/sessions/{id}/workflows/{wf}/resume` | `workflows.py:272-323` | ~120 | Can skip | **LlamaIndex-specific** |
| List workflows | `GET /assistant/sessions/{id}/workflows` | `workflows.py:326-359` | ~80 | Should | **LlamaIndex-specific** |
| Get workflow details | `GET /assistant/sessions/{id}/workflows/{wf}` | `workflows.py:362-409` | ~120 | Should | **LlamaIndex-specific** |
| Workflow WebSocket | `WS /assistant/sessions/{id}/workflows/{wf}/ws` | `workflows.py:412-477` | ~150 | Can skip | **LlamaIndex-specific** |

**Category Total:** ~980 LOC
**Migration Assessment:** **0% must migrate** - Replace entirely with Mastra Workflows

---

## 2. BACKEND-ONLY FEATURES (Not Exposed to Frontend)

### 2.1 RAG & Indexing Systems
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| LocalRepoContextExtractor | `new_rag.py` | ~800 | AgenticRAG | **MUST** | Complex (RAG) |
| IssueAwareRAG | `issue_rag.py` | ~600 (compressed) | Multiple | **MUST** | Complex (RAG) |
| CompositeAgenticRetriever | `agentic_rag.py:36-139` | ~240 | AgenticRAG | Should | Complex |
| Patch linkage builder | `patch_linkage.py` | ~400 | IssueRAG | Should | Moderate |
| Commit indexing | `commit_index.py` | ~500 | Timeline/Git | **MUST** | Moderate |

**Category Total:** ~2,540 LOC
**Migration Assessment:** 70% must migrate - core RAG features

---

### 2.2 Git Operations
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| Git blame tools | `git_tools/git_blame_tools.py` | ~300 | Agents | Should | Easy |
| Git history tools | `git_tools/git_history_tools.py` | ~400 | Agents | Should | Easy |
| Issue closing detection | `git_tools/issue_closing_tools.py` | ~250 | IssueOps | Should | Moderate |
| Commit metadata extraction | `commit_index.py` | ~500 | Timeline | **MUST** | Moderate |

**Category Total:** ~1,450 LOC
**Migration Assessment:** 50% must migrate - git blame/history can be simplified

---

### 2.3 Agent Tools & Operations
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| Tool registry (all 45 tools) | `agent_tools/tool_registry.py` | ~422 | Agents | **MUST** | Easy (map to Mastra) |
| File operations | `agent_tools/file_operations.py` | ~200 | Tools | **MUST** | Easy |
| Search operations | `agent_tools/search_operations.py` | ~300 | Tools | **MUST** | Easy |
| Code generation | `agent_tools/code_generation.py` | ~250 | Tools | Should | Moderate |
| Git operations wrapper | `agent_tools/git_operations.py` | ~400 | Tools | Should | Easy |
| Issue operations | `agent_tools/issue_operations.py` | ~350 | Tools | **MUST** | Moderate |
| PR operations | `agent_tools/pr_operations.py` | ~350 | Tools | Should | Moderate |
| Context manager | `agent_tools/context_manager.py` | ~200 | All | Should | Easy |
| Query processor | `agent_tools/query_processor.py` | ~150 | Agentic | Should | Easy |

**Category Total:** ~2,622 LOC
**Migration Assessment:** 70% must migrate - most tools map directly to Mastra

---

### 2.4 Agent & Workflow Engines
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| AgenticCodebaseExplorer | `agent_tools/core.py` | ~400 (compressed) | Multiple | **MUST** | **Mastra replaces** |
| LlamaIndex ReAct agent | Various | ~500 | Explorer | **SKIP** | **Use Mastra agents** |
| Workflow engine | `agent_tools/llamaindex_workflows.py` | ~987 | Workflows | **SKIP** | **Use Mastra workflows** |
| Agent pool | `agent_tools/agent_pool.py` | ~150 | Explorer | **SKIP** | **Use Mastra** |
| Comprehensive analysis workflow | `agent_tools/llamaindex_comprehensive_workflow.py` | ~400 | Workflows | Should | **Mastra replaces** |

**Category Total:** ~2,437 LOC
**Migration Assessment:** **10% must migrate** - Most is replaced by Mastra's agent system

---

### 2.5 Caching & Storage
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| Redis cache manager | `cache/redis_cache_manager.py` | ~250 | All | **MUST** | Easy |
| Multi-tier caching | `cache/__init__.py` | ~150 | All | Should | Easy |
| Chunk store (Redis/Memory) | `chunk_store.py` | ~200 | Large outputs | Should | Easy |

**Category Total:** ~600 LOC
**Migration Assessment:** 80% must migrate - caching is critical for performance

---

### 2.6 LLM & Conversation
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| LLM client (OpenRouter) | `llm_client.py` | ~300 | All | **MUST** | Easy (use Mastra) |
| Conversation memory | `conversation_memory.py` | ~200 | Chat | **MUST** | Easy (use Mastra) |
| Prompt engineering | `agent_tools/prompts.py` | ~150 | Agents | **MUST** | Easy |

**Category Total:** ~650 LOC
**Migration Assessment:** 100% must migrate - but simplified with Mastra

---

### 2.7 GitHub Integration
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| GitHub client | `github_client.py` | ~400 | Multiple | **MUST** | Easy |
| Triage bot | `triage_bot.py` | ~200 | Posting | Should | Easy |

**Category Total:** ~600 LOC
**Migration Assessment:** 100% must migrate - core integration

---

### 2.8 Utilities & Support
| Feature | Implementation | LOC | Used By | Priority | Mastra Complexity |
|---------|----------------|-----|---------|----------|-------------------|
| Models & schemas | `models.py` | ~300 | All | **MUST** | Easy |
| Config management | `config.py` | ~150 | All | **MUST** | Easy |
| Language detection | `language_config.py` | ~100 | Tree-sitter | Should | Easy |
| Async helpers | `utils/async_helpers.py` | ~80 | All | Should | Easy |
| Decorators | `utils/decorators.py` | ~100 | All | Should | Easy |

**Category Total:** ~730 LOC
**Migration Assessment:** 80% must migrate - supporting infrastructure

---

## 3. AGENT CAPABILITIES (45 Tools Available)

### 3.1 File Operations (4 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `explore_directory` | List directory contents | ✅ | **MUST** | Easy to map |
| `read_file` | Read complete file | ✅ | **MUST** | Easy to map |
| `analyze_file_structure` | Analyze file/directory | ✅ | Should | Moderate |
| `stream_large_file` | Stream large files | ❌ | Can skip | Not needed |

---

### 3.2 Search Operations (3 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `search_codebase` | Pattern/concept search | ✅ | **MUST** | Easy to map |
| `find_related_files` | Find related files | ✅ | Should | Moderate |
| `semantic_content_search` | Semantic search | ✅ | **MUST** | Easy (use RAG) |

---

### 3.3 Code Generation (2 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `generate_code_example` | Generate examples | ✅ | Should | Easy to map |
| `write_complete_code` | Write full files | ❌ | Can skip | Not needed |

---

### 3.4 Git Operations (15 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `git_blame_function` | Blame for function | ✅ | Should | Moderate |
| `who_last_edited_line` | Line-level blame | ❌ | Can skip | Not needed |
| `git_blame_at_commit` | Historical blame | ❌ | Can skip | Not needed |
| `find_commits_touching_function` | Function history | ✅ | Should | Moderate |
| `get_function_evolution` | Function evolution | ❌ | Can skip | Complex |
| `find_pr_closing_commit` | PR merge commit | ✅ | Should | Easy |
| `get_issue_closing_info` | Issue closing info | ✅ | **MUST** | Easy |
| `get_open_issues_related_to_commit` | Related issues | ❌ | Can skip | Not needed |
| `find_when_feature_was_added` | Feature origin | ❌ | Can skip | Complex |
| `search_commits` | Semantic commit search | ✅ | Should | Moderate |
| `get_file_timeline` | File commit history | ✅ | **MUST** | Moderate |
| `get_file_commit_statistics` | File stats | ❌ | Can skip | Not needed |
| `get_commit_details` | Commit metadata | ✅ | Should | Easy |
| `analyze_commit_patterns` | Pattern analysis | ❌ | Can skip | Complex |
| `who_implemented_this` | Original author | ❌ | Can skip | Complex |

---

### 3.5 Issue Operations (8 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `analyze_github_issue` | Analyze issue | ✅ | **MUST** | Moderate |
| `find_issue_related_files` | Issue→files | ✅ | **MUST** | Complex (RAG) |
| `related_issues` | Similar issues | ✅ | **MUST** | Complex (RAG) |
| `get_issue_closing_info` | How closed | ✅ | Should | Easy |
| `find_issues_related_to_file` | File→issues | ✅ | Should | Moderate |
| `get_issue_resolution_summary` | Resolution summary | ❌ | Can skip | Not needed |
| `check_issue_status_and_linked_pr` | Issue+PR status | ✅ | Should | Easy |
| `regression_detector` | Detect regressions | ❌ | Can skip | Complex |

---

### 3.6 PR Operations (13 tools)
| Tool | Function | Used? | Priority | Mastra Tool |
|------|----------|-------|----------|-------------|
| `get_pr_for_issue` | Issue→PR | ✅ | Should | Easy |
| `get_pr_diff` | PR diff | ✅ | Should | Easy |
| `get_files_changed_in_pr` | PR files | ✅ | Should | Easy |
| `get_pr_summary` | PR summary | ✅ | Should | Moderate |
| `find_open_prs_for_issue` | Open PRs for issue | ✅ | Should | Easy |
| `get_open_pr_status` | PR status | ✅ | Should | Easy |
| `find_open_prs_by_files` | File→PRs | ❌ | Can skip | Not needed |
| `search_open_prs` | Search PRs | ❌ | Can skip | Not needed |
| `check_pr_readiness` | Check merge-ready | ❌ | Can skip | Not needed |
| `find_feature_introducing_pr` | Feature→PR | ❌ | Can skip | Complex |
| `get_pr_details_from_github` | PR metadata | ✅ | Should | Easy |
| `get_pr_analysis` | Full PR analysis | ✅ | Should | Moderate |

---

## 4. WORKFLOW CAPABILITIES

### 4.1 LlamaIndex Workflows (To Be Replaced)
| Workflow | Purpose | LOC | Priority | Mastra Replacement |
|----------|---------|-----|----------|-------------------|
| TriageFlowAgentWorkflow | Base workflow | ~400 | **SKIP** | Mastra Workflow |
| LinearSwarmWorkflow | Sequential agents | ~150 | **SKIP** | Mastra Workflow |
| OrchestratorWorkflow | Central coordinator | ~150 | **SKIP** | Mastra Workflow |
| ComprehensiveAnalysisWorkflow | Full repo analysis | ~400 | Should port | Mastra Workflow |

**Total:** ~1,100 LOC
**Migration Assessment:** **0% must migrate** - Completely replaced by Mastra

---

## 5. RAG CAPABILITIES

### 5.1 Code Indexing
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| Tree-sitter parsing | `new_rag.py` | ~200 | **MUST** | Complex |
| Chunk creation | `new_rag.py` | ~150 | **MUST** | Moderate |
| FAISS vector index | `new_rag.py` | ~100 | **MUST** | Easy (use Mastra) |
| BM25 hybrid search | `new_rag.py` | ~100 | Should | Easy (use Mastra) |

---

### 5.2 Issue Indexing
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| GitHub API crawling | `issue_rag.py` | ~150 | **MUST** | Easy |
| Issue embedding | `issue_rag.py` | ~100 | **MUST** | Easy (use Mastra) |
| FAISS issue index | `issue_rag.py` | ~100 | **MUST** | Easy (use Mastra) |
| BM25 issue search | `issue_rag.py` | ~80 | Should | Easy (use Mastra) |

---

### 5.3 Hybrid Search & Reranking
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| Dense+sparse fusion | `agentic_rag.py` | ~100 | Should | Easy (Mastra has this) |
| Query routing | `agentic_rag.py` | ~80 | Should | Easy |
| Result reranking | `agentic_rag.py` | ~50 | Should | Easy (Mastra has this) |

---

## 6. MEMORY & CONTEXT CAPABILITIES

| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| Chat memory buffer | `conversation_memory.py` | ~200 | **MUST** | Easy (use Mastra) |
| Session state | `session_manager.py` | ~300 | **MUST** | Easy |
| Agent memory | Integrated in agents | ~150 | **MUST** | Easy (use Mastra) |
| Context-aware tools | `context_aware_tools.py` | ~200 | Should | Moderate |
| Execution context | `context_manager.py` | ~200 | Should | Moderate |

**Total:** ~1,050 LOC
**Migration Assessment:** 80% must migrate - session/chat memory critical

---

## 7. INTEGRATION CAPABILITIES

### 7.1 GitHub API
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| List issues | `github_client.py` | ~50 | **MUST** | Easy |
| Get issue details | `github_client.py` | ~50 | **MUST** | Easy |
| List PRs | `github_client.py` | ~50 | **MUST** | Easy |
| Create issue | `github_client.py` | ~50 | Should | Easy |
| Post comment | `github_client.py` | ~50 | Should | Easy |
| GraphQL queries | `github_client.py` | ~150 | Should | Moderate |

**Total:** ~400 LOC
**Migration Assessment:** 100% must migrate

---

### 7.2 Git Operations
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| Clone repository | `local_repo_loader.py` | ~100 | **MUST** | Easy |
| Git blame | `git_tools/` | ~300 | Should | Easy |
| Git log parsing | `commit_index.py` | ~200 | **MUST** | Moderate |
| Diff extraction | Various | ~150 | Should | Easy |

**Total:** ~750 LOC
**Migration Assessment:** 60% must migrate

---

### 7.3 Redis Caching
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| Multi-tier cache | `cache/` | ~400 | **MUST** | Easy |
| RAG caching | Integrated | ~150 | **MUST** | Easy |
| Response caching | Integrated | ~100 | Should | Easy |

**Total:** ~650 LOC
**Migration Assessment:** 100% must migrate

---

### 7.4 Database
| Feature | Implementation | LOC | Priority | Mastra Complexity |
|---------|----------------|-----|----------|-------------------|
| No database used | N/A | 0 | N/A | N/A |

**Note:** System uses file-based storage (FAISS indexes, JSONL) and Redis cache. No SQL database.

---

## 8. MIGRATION PRIORITY SUMMARY

### MUST-HAVE Features (Cannot skip)
| Category | Features | LOC | % of Total |
|----------|----------|-----|------------|
| Session Management | 5 endpoints | ~300 | 1% |
| Chat & Messaging | 3 endpoints | ~540 | 2% |
| Repository Browsing | 4 endpoints | ~560 | 2% |
| Issue Management | 4 endpoints | ~620 | 2% |
| RAG Systems | 3 systems | ~1,800 | 6% |
| Agent Tools | 15 tools | ~1,400 | 5% |
| Git Integration | Core features | ~450 | 2% |
| GitHub Integration | All features | ~400 | 1% |
| Caching | Redis system | ~650 | 2% |
| **TOTAL MUST-HAVE** | | **~6,720** | **23%** |

---

### SHOULD-HAVE Features (High value, can defer)
| Category | Features | LOC | % of Total |
|----------|----------|-----|------------|
| Timeline & History | File timeline | ~320 | 1% |
| Agentic Query | Advanced features | ~800 | 3% |
| Agent Tools | 15 additional tools | ~1,200 | 4% |
| Git Operations | Blame/history | ~700 | 2% |
| Context Management | Execution context | ~400 | 1% |
| **TOTAL SHOULD-HAVE** | | **~3,420** | **12%** |

---

### CAN-SKIP Features (Low ROI or replaceable)
| Category | Features | LOC | % of Total |
|----------|----------|-----|------------|
| LlamaIndex Workflows | All workflow system | ~2,100 | 7% |
| LlamaIndex Agents | ReAct agent system | ~500 | 2% |
| Advanced Git Tools | 8 complex tools | ~800 | 3% |
| Advanced PR Tools | 5 complex tools | ~400 | 1% |
| Comprehensive Workflow | Full repo analysis | ~400 | 1% |
| **TOTAL CAN-SKIP** | | **~4,200** | **15%** |

---

## 9. MASTRA IMPLEMENTATION COMPLEXITY

### Easy (Direct mapping, <1 day each)
- Session CRUD operations
- File browsing
- GitHub API wrappers
- Basic agent tools (15 tools)
- Caching infrastructure
- **Total:** ~4,500 LOC

---

### Moderate (Some adaptation needed, 1-3 days each)
- Chat with streaming
- Issue analysis pipeline
- Commit indexing
- Context-aware tools
- Git blame/history
- **Total:** ~3,800 LOC

---

### Complex (Significant work, 3-7 days each)
- RAG systems (code + issue indexing)
- Agentic query with streaming steps
- Timeline with commit index
- Patch linkage builder
- **Total:** ~3,200 LOC

---

### LlamaIndex-Specific (Replace entirely)
- ReAct agent system
- Workflow engine
- Agent pool
- **Total:** ~3,100 LOC (don't port, use Mastra equivalents)

---

## 10. MIGRATION STRATEGY RECOMMENDATIONS

### Phase 1: Core Foundation (Week 1-2)
**Must implement:** ~6,720 LOC
- Session management
- Repository browsing
- GitHub integration
- Basic agent tools
- Caching layer

**Mastra components:**
- Mastra agents (replace ReAct)
- Mastra workflows (replace LlamaIndex workflows)
- Mastra RAG (replace LocalRepoContextExtractor)
- Mastra tools (map 15 core tools)

---

### Phase 2: RAG & Intelligence (Week 3-4)
**Must implement:** ~3,000 LOC
- Issue-aware RAG
- Code indexing with tree-sitter
- Commit indexing
- Hybrid search

**Mastra components:**
- Mastra vector stores
- Mastra RAG with reranking
- Custom tools for git operations

---

### Phase 3: Advanced Features (Week 5-6)
**Should implement:** ~3,420 LOC
- Timeline feature
- Advanced agentic queries
- Context management
- Git history tools

**Mastra components:**
- Mastra memory
- Mastra workflow orchestration
- Custom git tools

---

### Phase 4: Polish & Testing (Week 7-8)
- Integration testing
- Performance optimization
- Documentation
- Frontend integration

---

## 11. KEY METRICS

| Metric | Value |
|--------|-------|
| **Total Python LOC** | ~28,836 |
| **Must-migrate LOC** | ~6,720 (23%) |
| **Should-migrate LOC** | ~3,420 (12%) |
| **Can-skip LOC** | ~4,200 (15%) |
| **LlamaIndex-specific (replace)** | ~3,100 (11%) |
| **Supporting/util LOC** | ~11,396 (39%) |

---

## 12. CONCLUSION

**What we're actually building:**

1. **Core Value Props (Must-have):**
   - Intelligent GitHub issue triage with RAG
   - Agentic code exploration with 15 core tools
   - Real-time chat interface with streaming
   - File/PR/commit browsing

2. **Unique Differentiators (Should-have):**
   - File timeline visualization
   - Patch linkage (issue→PR→commit)
   - Advanced git blame/history

3. **What we can skip:**
   - LlamaIndex-specific workflows (use Mastra)
   - Complex git archaeology tools
   - Comprehensive repo analysis workflow (defer)

**Estimated Mastra Implementation:**
- **Core features:** 6,720 LOC → ~2,000 LOC Mastra (70% reduction via framework)
- **Advanced features:** 3,420 LOC → ~1,500 LOC Mastra (56% reduction)
- **Total new code:** ~3,500 LOC Mastra + integration
- **Time estimate:** 6-8 weeks for complete migration

**ROI:**
- Migrate 35% of codebase (23% must + 12% should)
- Skip 15% entirely (replaced by Mastra)
- 50% reduction in code to maintain
- Gain: Mastra's workflow engine, better agents, cleaner architecture
