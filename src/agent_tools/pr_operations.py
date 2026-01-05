"""PR operations for finding and analyzing pull requests - tinygrad-style (1017→400 lines)"""
import json, logging, asyncio, re, subprocess, time
from typing import List, Dict, Any, Optional, Annotated, TYPE_CHECKING
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from ..unified_rag import IssueAwareRAG
    from ..git_tools import GitHistoryTools
    from llama_index.core.llms import LLM

try:
    from ..config import settings
    from ..cache.redis_cache_manager import RedisCacheManager
    from ..utils.decorators import log_errors, safe_op
except ImportError:
    class MockSettings: openrouter_api_key = summarization_model = None
    settings = MockSettings()
    RedisCacheManager = None
    # Fallback decorators if import fails
    def log_errors(fn): return fn
    def safe_op(*args, **kwargs):
        def decorator(fn): return fn
        return decorator

logger = logging.getLogger(__name__)

def _run_sync(coro):
    """Run async coroutine from sync context, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor() as ex: return ex.submit(asyncio.run, coro).result(timeout=30)
    except RuntimeError: return asyncio.run(coro)

def _json_error(msg: str, **kwargs) -> str:
    return json.dumps({"error": msg, **kwargs})


class PROperations:
    """PR operations with caching and multi-source fallback search."""

    def __init__(self, repo_path: Path, issue_rag_system: Optional['IssueAwareRAG'],
                 git_history_tools: 'GitHistoryTools', llm_instance: 'LLM',
                 chunk_large_output_func: callable, extract_repo_info_func: callable):
        self.repo_path, self.issue_rag_system = repo_path, issue_rag_system
        self.git_history_tools, self.llm = git_history_tools, llm_instance
        self._chunk_large_output, self._extract_repo_info = chunk_large_output_func, extract_repo_info_func
        self.pr_cache = RedisCacheManager(namespace="pr_ops", default_ttl=1800, max_memory_items=500) if RedisCacheManager else None
        self._repo_context = None

    def _cache_key(self, key_type: str, *args) -> str:
        """Create repository-aware cache key."""
        if not self._repo_context:
            owner, name = self._extract_repo_info(self.repo_path) if self._extract_repo_info else (None, None)
            self._repo_context = f"{owner}/{name}" if owner and name else str(self.repo_path)
        return f"{key_type}:{self._repo_context}:{':'.join(map(str, args))}" if args else f"{key_type}:{self._repo_context}"

    def _cached(self, key: str, fetch_fn):
        """Get cached result or fetch and cache."""
        if not self.pr_cache: return fetch_fn()
        async def _get():
            if (cached := await self.pr_cache.get(key)): return cached
            result = await fetch_fn() if asyncio.iscoroutinefunction(fetch_fn) else fetch_fn()
            await self.pr_cache.set(key, result)
            return result
        return _run_sync(_get())

    def _git(self, *args) -> subprocess.CompletedProcess:
        """Run git command, return result."""
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=self.repo_path)

    def _parse_pr_refs(self, output: str, limit: int = 100) -> List[Dict]:
        """Parse git log output for PR references."""
        prs = []
        for line in (output.strip().split('\n') if output.strip() else [])[:limit]:
            if '|' not in line: continue
            parts = line.split('|', 2)
            sha, subject = parts[0], parts[1] if len(parts) > 1 else ""
            date = parts[2] if len(parts) > 2 else None
            for pr_num in re.findall(r'(?:Merge pull request #|PR|merge.*#|#)(\d+)', subject, re.I):
                if pr_num not in [str(p.get('pr_number')) for p in prs]:
                    prs.append({"pr_number": int(pr_num), "commit_sha": sha[:8], "subject": subject, **({"date": date} if date else {})})
        return prs

    # ============================================================================
    # Core PR Lookup Methods
    # ============================================================================

    @log_errors
    def get_pr_for_issue(self, issue_identifier: Annotated[str, "Issue identifier (number or #number)"]) -> str:
        """Find PRs that reference or fix a specific issue."""
        issue_num = int(issue_identifier.lstrip('#'))

        def _fetch():
            # Try Issue RAG first
            if self.issue_rag_system and hasattr(self.issue_rag_system, 'indexer'):
                try:
                    if (pb := getattr(self.issue_rag_system.indexer, 'patch_builder', None)):
                        if (links := pb.load_patch_links().get(issue_num)):
                            return json.dumps({"issue_number": issue_num, "found_prs": [
                                {"issue_id": l.issue_id, "pr_number": l.pr_number, "merged_at": l.merged_at,
                                 "pr_title": l.pr_title, "pr_url": l.pr_url, "files_changed": l.files_changed}
                                for l in links
                            ]})
                except Exception as e: logger.warning(f"Issue RAG patch_builder failed: {e}")

            # Fallback: git log
            result = self._git("log", "--grep", f"#{issue_num}", "--pretty=format:%H|%s", "--all", "-100")
            if result.returncode == 0 and (prs := self._parse_pr_refs(result.stdout)):
                return json.dumps({"issue_number": issue_num, "found_prs": prs, "source": "git_log"})
            return json.dumps({"issue_number": issue_num, "found_prs": [], "message": "No PRs found"})

        return self._cached(self._cache_key("pr_for_issue", issue_num), _fetch)

    @log_errors
    def get_pr_diff(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get the diff content for a merged PR."""
        def _fetch():
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                return _json_error("Issue RAG or diff_docs not available.")
            if not (diff_doc := self.issue_rag_system.indexer.diff_docs.get(pr_number)):
                return _json_error(f"No cached diff for PR #{pr_number}.")
            if not (diff_path := Path(diff_doc.diff_path)).exists():
                return _json_error(f"Diff file not found at {diff_doc.diff_path}.")
            try:
                return json.dumps({"pr_number": pr_number, "diff_summary": diff_doc.diff_summary,
                                   "full_diff": diff_path.read_text(encoding='utf-8', errors='ignore')})
            except Exception as e: return _json_error(f"Error reading diff: {e}")
        return self._cached(self._cache_key("pr_diff", pr_number), _fetch)

    @log_errors
    def get_files_changed_in_pr(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get list of files changed in a PR."""
        def _fetch():
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                return _json_error("Issue RAG or diff_docs not available.")
            if not (diff_doc := self.issue_rag_system.indexer.diff_docs.get(pr_number)):
                return _json_error(f"No cached diff for PR #{pr_number}.")
            return json.dumps({"pr_number": pr_number, "files_changed": diff_doc.files_changed})
        return self._cached(self._cache_key("pr_files", pr_number), _fetch)

    @log_errors
    def get_pr_summary(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get summary of a PR."""
        def _fetch():
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                return _json_error("Issue RAG or diff_docs not available.")
            if not (diff_doc := self.issue_rag_system.indexer.diff_docs.get(pr_number)):
                return _json_error(f"No cached diff for PR #{pr_number}.")
            return json.dumps({"pr_number": pr_number, "summary": diff_doc.diff_summary or "Summary not available.",
                              "files_changed": diff_doc.files_changed})
        return self._cached(self._cache_key("pr_summary", pr_number), _fetch)

    # ============================================================================
    # Open PR Search (unified fallback pattern)
    # ============================================================================

    def _search_open_prs_multi(self, search_type: str, issue_num: int = None,
                                file_paths: List[str] = None, query: str = None, limit: int = 5) -> Dict:
        """Unified multi-source open PR search with fallback."""
        result = {"search_type": search_type, "open_prs": [], "search_methods": [], "found_via": None}
        if issue_num: result["issue_number"] = issue_num
        if file_paths: result["file_paths"] = file_paths
        if query: result["query"] = query

        # Method 1: Issue RAG index
        rag = self.issue_rag_system
        if rag and (hasattr(rag, 'is_initialized') and rag.is_initialized() if hasattr(rag, 'is_initialized') else True):
            try:
                if (open_pr_docs := getattr(getattr(rag, 'indexer', None), 'open_pr_docs', None)):
                    for pr_num, doc in open_pr_docs.items():
                        match = False
                        if issue_num:  # Search by issue reference
                            pr_text = f"{doc.title} {doc.body}".lower()
                            match = any(ref in pr_text for ref in [f"#{issue_num}", f"fixes #{issue_num}", f"closes #{issue_num}"])
                        elif file_paths:  # Search by files
                            match = any(any(f in pf or pf in f for pf in doc.files_changed) for f in file_paths)
                        elif query:  # Keyword search
                            pr_text = f"{doc.title} {doc.body}".lower()
                            words = query.lower().split()
                            match = sum(1 for w in words if w in pr_text) / max(len(words), 1) > 0.3

                        if match:
                            result["open_prs"].append({
                                "pr_number": pr_num, "title": doc.title, "author": doc.author, "url": doc.url,
                                "created_at": doc.created_at, "review_decision": doc.review_decision,
                                "draft": doc.draft, "files_changed": doc.files_changed[:5], "source": "issue_rag_index"
                            })

                    if result["open_prs"]:
                        result.update({"found_via": "issue_rag_index", "status": "found"})
                        result["search_methods"].append("issue_rag_index")
                        return result
            except Exception as e:
                logger.warning(f"Issue RAG search failed: {e}")
                result["search_methods"].append(f"issue_rag_failed: {e}")

        # Method 2: Git log search
        try:
            grep_term = f"#{issue_num}" if issue_num else (query or "")
            if grep_term or file_paths:
                cmd = ["log", "--pretty=format:%H|%s|%ad", "--date=short", "--since=90.days.ago", "--all"]
                if grep_term: cmd.insert(1, "--grep"); cmd.insert(2, grep_term)
                if file_paths: cmd.extend(["--"] + file_paths)

                proc = self._git(*cmd)
                if proc.returncode == 0 and (commits := self._parse_pr_refs(proc.stdout, limit * 2)):
                    result["recent_commits"] = commits[:limit]
                    result["search_methods"].append("git_log_search")
        except Exception as e:
            result["search_methods"].append(f"git_log_failed: {e}")

        # Method 3: Branch search (for issue number or file-related)
        try:
            proc = self._git("branch", "-r", "--format=%(refname:short)")
            if proc.returncode == 0:
                branches = [b.strip() for b in proc.stdout.strip().split('\n') if b.strip()]
                related = []
                if issue_num:
                    related = [b for b in branches if str(issue_num) in b]
                elif file_paths:
                    for fp in file_paths:
                        fname = fp.split('/')[-1].split('.')[0]
                        if len(fname) > 3:
                            related.extend(b for b in branches if fname.lower() in b.lower())
                if related:
                    result["related_branches"] = list(set(related))[:10]
                    result["search_methods"].append("branch_search")
        except Exception as e:
            result["search_methods"].append(f"branch_search_failed: {e}")

        # Final status
        result["status"] = "found" if result["open_prs"] else "not_found"
        result["message"] = f"Found {len(result['open_prs'])} open PR(s)" if result["open_prs"] else \
                           f"No open PRs found using: {', '.join(result['search_methods'])}"
        return result

    def find_open_prs_for_issue(self, issue_number: Annotated[int, "Issue number"]) -> str:
        """Find open PRs related to an issue."""
        def _fetch():
            return json.dumps(self._search_open_prs_multi("by_issue", issue_num=issue_number), indent=2)
        return self._cached(f"open_prs_for_issue:{issue_number}", _fetch)

    def find_open_prs_by_files(self, file_paths: Annotated[List[str], "List of file paths"]) -> str:
        """Find open PRs that touch specified files."""
        def _fetch():
            return json.dumps(self._search_open_prs_multi("by_files", file_paths=file_paths), indent=2)
        return self._cached(f"prs_by_files:{':'.join(sorted(file_paths))}", _fetch)

    def search_open_prs(self, query: Annotated[str, "Search query"], limit: Annotated[int, "Limit"] = 5) -> str:
        """Search open PRs by keywords."""
        def _fetch():
            return json.dumps(self._search_open_prs_multi("by_query", query=query, limit=limit), indent=2)
        return self._cached(f"search_prs:{query}:{limit}", _fetch)

    # ============================================================================
    # GitHub API Integration
    # ============================================================================

    @log_errors
    def get_pr_details_from_github(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get PR details from GitHub API."""
        def _fetch():
            try:
                from ..github_client import GitHubIssueClient
                from .utilities import get_repo_url_from_path

                if not (repo_url := get_repo_url_from_path(self.repo_path)):
                    return _json_error("Cannot determine repository URL")

                async def _get_pr():
                    client = GitHubIssueClient()
                    if not (pr := await client.get_pr_detailed_info(repo_url, pr_number)):
                        return {"error": f"PR #{pr_number} not found", "repo_url": repo_url}
                    return {
                        "number": pr.number, "title": pr.title, "state": pr.state, "url": pr.url,
                        "body": pr.body, "author": pr.user.login if pr.user else None,
                        "created_at": pr.created_at, "updated_at": pr.updated_at, "merged_at": pr.merged_at,
                        "files_changed": pr.files_changed, "review_decision": pr.review_decision,
                        "mergeable": pr.mergeable, "draft": pr.draft,
                        "additions": pr.additions, "deletions": pr.deletions,
                        "reviews": [{"author": r.author, "state": r.state, "submitted_at": r.submitted_at} for r in pr.reviews],
                        "status_checks": [{"state": c.state, "context": c.context} for c in pr.status_checks]
                    }

                return json.dumps(_run_sync(_get_pr()), indent=2)
            except Exception as e:
                logger.error(f"Error fetching PR from GitHub: {e}")
                return _json_error(str(e), pr_number=pr_number)

        return self._cached(f"github_pr:{pr_number}", _fetch)

    def get_pr_analysis(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get comprehensive PR analysis combining local and GitHub data."""
        def _fetch():
            local = json.loads(self.get_pr_diff(pr_number))
            github = json.loads(self.get_pr_details_from_github(pr_number))

            # Generate summary
            summary_parts = []
            if github and not github.get("error"):
                summary_parts.extend([
                    f"**PR #{github['number']}: {github['title']}**",
                    f"State: {github['state']} | Author: {github.get('author', 'Unknown')}",
                ])
                if github.get('files_changed'):
                    summary_parts.append(f"Files: {len(github['files_changed'])} ({', '.join(github['files_changed'][:3])}...)")
                if github.get('additions') or github.get('deletions'):
                    summary_parts.append(f"Changes: +{github.get('additions', 0)} -{github.get('deletions', 0)}")
            elif local and not local.get("error"):
                summary_parts.append(f"**PR #{pr_number}**: {local.get('diff_summary', 'No summary')}")

            return json.dumps({
                "pr_number": pr_number, "timestamp": time.time(),
                "local_data": local, "github_data": github,
                "summary": "\n".join(summary_parts) or f"PR #{pr_number} - No details available"
            }, indent=2)

        return self._cached(f"pr_analysis:{pr_number}", _fetch)

    # ============================================================================
    # Simple Status Methods
    # ============================================================================

    def get_open_pr_status(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Get status of an open PR from local index."""
        def _fetch():
            if not self.issue_rag_system or not hasattr(self.issue_rag_system.indexer, 'open_pr_docs'):
                return _json_error("Issue RAG or open_pr_docs not available.")
            if not (doc := self.issue_rag_system.indexer.open_pr_docs.get(pr_number)):
                return _json_error(f"Open PR #{pr_number} not found in index.")
            return json.dumps(doc.to_dict() if hasattr(doc, 'to_dict') else {"pr_number": pr_number, "title": doc.title}, indent=2)
        return self._cached(f"open_pr_status:{pr_number}", _fetch)

    def check_pr_readiness(self, pr_number: Annotated[int, "PR number"]) -> str:
        """Check if a PR is ready to merge."""
        status = json.loads(self.get_open_pr_status(pr_number))
        if status.get("error"): return json.dumps(status)
        return json.dumps({
            "pr_number": pr_number,
            "ready": status.get("review_decision") == "APPROVED" and status.get("mergeable", False),
            "review_decision": status.get("review_decision"), "mergeable": status.get("mergeable"),
            "draft": status.get("draft", False)
        })

    def find_feature_introducing_pr(self, feature_name: Annotated[str, "Feature name"]) -> str:
        """Find the PR that introduced a specific feature."""
        def _fetch():
            result = self._git("log", "--grep", feature_name, "--pretty=format:%H|%s", "-10", "--all")
            if result.returncode == 0 and (prs := self._parse_pr_refs(result.stdout)):
                return json.dumps({"feature_name": feature_name, "introducing_pr": prs[0], "related_prs": prs[:3]}, indent=2)
            return json.dumps({"feature_name": feature_name, "message": f"No PR found for '{feature_name}'"})
        return self._cached(f"feature_pr:{feature_name}", _fetch)
