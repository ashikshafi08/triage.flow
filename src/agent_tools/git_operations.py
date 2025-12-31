"""Git operations for agentic codebase explorer - tinygrad-style rewrite (868→350 lines)"""
import json, logging, subprocess, asyncio, re, os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Annotated, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..git_tools import GitBlameTools, GitHistoryTools
    from ..commit_index import CommitIndexManager

logger = logging.getLogger(__name__)

def _run_sync(coro):
    """Run async coroutine from sync context, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor() as ex:
            return ex.submit(asyncio.run, coro).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)


class GitOperations:
    def __init__(self, git_blame_tools: 'GitBlameTools', git_history_tools: 'GitHistoryTools',
                 commit_index_manager: 'CommitIndexManager', get_current_head_sha_func: callable,
                 chunk_large_output_func: callable):
        self.git_blame_tools, self.git_history_tools = git_blame_tools, git_history_tools
        self.commit_index_manager = commit_index_manager
        self._get_current_head_sha, self._chunk_large_output = get_current_head_sha_func, chunk_large_output_func
        self.repo_path = self.git_blame_tools.repo_path

    def _git(self, *args, check=False) -> subprocess.CompletedProcess:
        """Run git command, return result."""
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=self.repo_path, check=check)

    def _parse_commit_line(self, line: str, sep='|', n=4) -> Optional[Dict]:
        """Parse commit line into dict."""
        parts = line.split(sep, n)
        if len(parts) < n: return None
        keys = ['sha', 'author', 'date', 'subject'][:len(parts)]
        if len(parts) > 4: keys = ['sha', 'author', 'email', 'date', 'subject'][:len(parts)]
        return dict(zip(keys, parts))

    def _json_error(self, e, **context) -> str:
        return json.dumps({**context, "error": str(e)})

    # ============================================================================
    # Blame Operations
    # ============================================================================

    def git_blame_function(self, function_name: Annotated[str, "Function/class name"],
                           file_path: Annotated[Optional[str], "File path (optional)"] = None) -> str:
        """Get git blame information for a specific function or class."""
        try:
            if not (sha := self._get_current_head_sha()): return self._json_error("Could not get HEAD SHA")
            result = self.git_blame_tools.git_blame_function_at_commit(file_path, function_name, sha)
            return json.dumps(result, indent=2)
        except Exception as e:
            return self._json_error(e, function_name=function_name, file_path=file_path)

    def who_last_edited_line(self, file_path: Annotated[str, "File path"],
                              line_number: Annotated[int, "Line number"]) -> str:
        """Get information about who last edited a specific line."""
        try:
            if not (sha := self._get_current_head_sha()): return self._json_error("Could not get HEAD SHA")
            result = self.git_blame_tools.git_blame_at_commit(file_path, sha, line_start=line_number, line_end=line_number)
            return json.dumps(result, indent=2)
        except Exception as e:
            return self._json_error(e, file_path=file_path, line_number=line_number)

    def git_blame_at_commit(self, file_path: Annotated[str, "File path"], commit_sha: Annotated[str, "Commit SHA"],
                            line_start: Annotated[Optional[int], "Start line"] = None,
                            line_end: Annotated[Optional[int], "End line"] = None) -> str:
        try:
            return json.dumps(self.git_blame_tools.git_blame_at_commit(file_path, commit_sha, line_start, line_end), indent=2)
        except Exception as e:
            return self._json_error(e)

    # ============================================================================
    # History Operations
    # ============================================================================

    def find_commits_touching_function(self, function_name: Annotated[str, "Function name"],
                                        file_path: Annotated[str, "File path"],
                                        limit: Annotated[int, "Max commits"] = 10) -> str:
        try:
            result = self.git_history_tools.find_commits_touching_function(function_name, file_path, limit)
            return self._chunk_large_output(json.dumps(result, indent=2))
        except Exception as e:
            return self._json_error(e)

    def get_function_evolution(self, function_name: Annotated[str, "Function name"],
                               file_path: Annotated[str, "File path"],
                               max_versions: Annotated[int, "Max versions"] = 5) -> str:
        try:
            result = self.git_history_tools.get_function_evolution(function_name, file_path, max_versions)
            return self._chunk_large_output(json.dumps(result, indent=2))
        except Exception as e:
            return self._json_error(e)

    def get_code_lifespan(self, file_path: Annotated[str, "File path"],
                          line_range: Annotated[Optional[str], "Line range e.g. '10-20'"] = None) -> str:
        try:
            start, end = (map(int, line_range.split('-')) if line_range else (None, None)) if line_range else (None, None)
            result = self.git_history_tools.get_code_lifespan(file_path, start, end)
            return self._chunk_large_output(json.dumps(result, indent=2))
        except Exception as e:
            return self._json_error(e)

    def find_open_issues_for_commit(self, commit_sha: Annotated[str, "Commit SHA"]) -> str:
        try:
            result = self.git_history_tools.find_open_issues_for_commit(commit_sha)
            return json.dumps(result, indent=2)
        except Exception as e:
            return self._json_error(e)

    # ============================================================================
    # Feature Analysis
    # ============================================================================

    def find_when_feature_was_added(self, feature_search_term: Annotated[str, "Feature/pattern to find"]) -> str:
        """Find when a feature was first added using commit index or git log."""
        try:
            # Try commit index first
            if self.commit_index_manager.is_initialized():
                try:
                    results = _run_sync(self.commit_index_manager.search_commits(
                        query=f"add {feature_search_term}", k=10, sort_by_date=True))
                    if results and (c := results[0].commit):
                        return json.dumps({"search_term": feature_search_term, "method": "commit_index",
                            "introducing_commit": {"sha": c.sha, "date": c.commit_date,
                                                   "subject": c.subject, "author": c.author_name}}, indent=2)
                except Exception as e:
                    logger.warning(f"Commit index failed, using git log: {e}")

            # Fallback to git log
            result = self._git("log", "-S", feature_search_term, "--pretty=format:%H|%an|%ae|%ad|%s",
                               "--date=short", "--reverse")
            if not result.stdout.strip():
                return json.dumps({"search_term": feature_search_term, "message": "No commits found"})

            if (commit := self._parse_commit_line(result.stdout.strip().split('\n')[0], n=5)):
                show = self._git("show", "--name-only", "--pretty=format:", commit['sha'])
                commit['files_changed'] = [f for f in show.stdout.strip().split('\n') if f.strip()]
                return json.dumps({"search_term": feature_search_term, "introducing_commit": commit,
                                   "method": "git_log"}, indent=2)
            return self._json_error("Parse failed", search_term=feature_search_term)
        except Exception as e:
            return self._json_error(e, search_term=feature_search_term)

    def get_file_history(self, file_path: Annotated[str, "File path"]) -> str:
        """Get timeline of commits that touched a file."""
        try:
            result = self._git("log", "--pretty=format:%H|%an|%ad|%s", "--date=iso", "--", file_path, check=True)
            if not result.stdout.strip():
                return json.dumps({"file_path": file_path, "message": "No history found"})

            timeline = []
            for line in result.stdout.strip().split('\n'):
                if (c := self._parse_commit_line(line)):
                    c['issue_refs'] = re.findall(r'#(\d+)', c.get('subject', ''))
                    timeline.append(c)

            return self._chunk_large_output(json.dumps({
                "file_path": file_path, "timeline": timeline[:50], "total_commits": len(timeline)}, indent=2))
        except Exception as e:
            return self._json_error(e, file_path=file_path)

    def summarize_feature_evolution(self, feature_query: Annotated[str, "Feature keyword"]) -> str:
        """Summarize how a feature evolved over time."""
        try:
            timeline = []
            # Search commit messages
            for cmd in [["--grep=" + feature_query], ["-S", feature_query]]:
                result = self._git("log", *cmd, "--oneline", "--date=short", "--pretty=format:%H|%ad|%s")
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line and (c := self._parse_commit_line(line, n=3)):
                            c['type'] = 'commit'
                            if c not in timeline: timeline.append(c)

            # Sort and dedupe
            seen = set()
            unique = [t for t in timeline if t['sha'] not in seen and not seen.add(t['sha'])]
            unique.sort(key=lambda x: x.get('date', ''), reverse=True)

            return self._chunk_large_output(json.dumps({
                "feature_query": feature_query, "timeline": unique[:30],
                "total_events": len(unique)}, indent=2))
        except Exception as e:
            return self._json_error(e, feature_query=feature_query)

    def who_implemented_this(self, feature_name: Annotated[str, "Feature/function name"],
                             file_path: Annotated[Optional[str], "File path"] = None) -> str:
        """Find who originally implemented a feature."""
        try:
            # Validate path
            if file_path:
                abs_path = os.path.join(self.repo_path, file_path)
                if not abs_path.startswith(os.path.realpath(self.repo_path)):
                    return self._json_error("Path outside repository")

            cmd = ["log", "--reverse", "--pretty=format:%H|%an|%ae|%ad|%s", "-S", feature_name]
            if file_path: cmd.extend(["--", file_path])
            result = self._git(*cmd)

            if not result.stdout.strip():
                return json.dumps({"feature_name": feature_name, "message": "No commits found"})

            if not (commit := self._parse_commit_line(result.stdout.strip().split('\n')[0], n=5)):
                return self._json_error("Parse failed", feature_name=feature_name)

            # Get files changed
            show = self._git("show", "--name-only", "--pretty=format:", commit['sha'])
            commit['files_changed'] = [f for f in show.stdout.strip().split('\n') if f.strip()]
            commit['issue_refs'] = re.findall(r'#(\d+)', commit.get('subject', ''))

            # Get contributors
            contrib_cmd = ["log", "--pretty=format:%an", "-S", feature_name]
            if file_path: contrib_cmd.extend(["--", file_path])
            contrib = self._git(*contrib_cmd)
            contributors = list(set(contrib.stdout.strip().split('\n'))) if contrib.returncode == 0 else []

            return json.dumps({
                "feature_name": feature_name, "file_path": file_path,
                "initial_implementation": commit,
                "contributors": {"implementer": commit['author'], "all": contributors[:5],
                                 "total": len(contributors)}}, indent=2)
        except Exception as e:
            return self._json_error(e, feature_name=feature_name, file_path=file_path)

    # ============================================================================
    # Commit Index Operations
    # ============================================================================

    def search_commits(self, query: Annotated[str, "Search query"],
                       k: Annotated[int, "Number of results"] = 10,
                       author_filter: Annotated[Optional[str], "Author filter"] = None,
                       file_filter: Annotated[Optional[str], "File filter"] = None,
                       path: Annotated[Optional[str], "File path (alt)"] = None) -> str:
        """Search commits using index or git log fallback."""
        effective_filter = path or file_filter
        try:
            if self.commit_index_manager.is_initialized():
                results = _run_sync(self.commit_index_manager.search_commits(
                    query, k=k, author_filter=author_filter,
                    file_filter=[effective_filter] if effective_filter else None))
                return json.dumps([r.to_dict() for r in results], indent=2)
        except Exception as e:
            logger.warning(f"Index search failed: {e}")

        return self._search_commits_fallback(query, k, author_filter, effective_filter)

    def _search_commits_fallback(self, query: str, k: int, author: Optional[str], file: Optional[str]) -> str:
        cmd = ["log", "--pretty=format:%H|%an|%ad|%s", "--date=short", f"-{k}"]
        if author: cmd.extend(["--author", author])
        if file: cmd.extend(["--", file])
        result = self._git(*cmd)
        commits = [c for line in result.stdout.strip().split('\n')
                   if (c := self._parse_commit_line(line)) and query.lower() in c.get('subject', '').lower()]
        return json.dumps(commits[:k], indent=2)

    def get_file_timeline(self, file_path: Annotated[str, "File path"],
                          limit: Annotated[int, "Max commits"] = 20) -> str:
        try:
            if self.commit_index_manager.is_initialized():
                results = _run_sync(self.commit_index_manager.get_file_timeline(file_path, limit))
                return json.dumps([r.to_dict() for r in results], indent=2)
        except Exception as e:
            logger.warning(f"Index timeline failed: {e}")

        result = self._git("log", "--pretty=format:%H|%an|%ad|%s", "--date=short", f"-{limit}", "--", file_path)
        timeline = [c for line in result.stdout.strip().split('\n') if (c := self._parse_commit_line(line))]
        return json.dumps({"file_path": file_path, "timeline": timeline}, indent=2)

    def get_file_commit_statistics(self, file_path: Annotated[str, "File path"]) -> str:
        try:
            if self.commit_index_manager.is_initialized():
                stats = _run_sync(self.commit_index_manager.get_file_statistics(file_path))
                return json.dumps(stats, indent=2)
        except Exception as e:
            logger.warning(f"Index stats failed: {e}")

        # Fallback
        result = self._git("log", "--pretty=format:%an", "--", file_path)
        authors = result.stdout.strip().split('\n') if result.returncode == 0 else []
        author_counts = {}
        for a in authors: author_counts[a] = author_counts.get(a, 0) + 1
        return json.dumps({"file_path": file_path, "total_commits": len(authors),
                           "unique_authors": len(set(authors)),
                           "top_authors": sorted(author_counts.items(), key=lambda x: -x[1])[:5]}, indent=2)

    def get_commit_details(self, commit_sha: Annotated[Optional[str], "Commit SHA"] = None,
                           commit_message: Annotated[Optional[str], "Search message"] = None) -> str:
        try:
            if commit_sha:
                if self.commit_index_manager.is_initialized():
                    result = _run_sync(self.commit_index_manager.get_commit(commit_sha))
                    if result: return json.dumps(result.to_dict(), indent=2)
                return self._get_commit_details_fallback(commit_sha)

            if commit_message:
                return self._find_commit_by_message(commit_message)

            return self._json_error("Provide commit_sha or commit_message")
        except Exception as e:
            return self._json_error(e)

    def _find_commit_by_message(self, msg: str) -> str:
        result = self._git("log", "--pretty=format:%H|%an|%ae|%ad|%s", "--grep", msg, "-i", "--max-count=10")
        commits = [c for line in result.stdout.strip().split('\n')
                   if (c := self._parse_commit_line(line, n=5)) and msg.lower() in c.get('subject', '').lower()]

        if len(commits) == 1:
            return self._get_commit_details_fallback(commits[0]['sha'])
        elif commits:
            return json.dumps({"search_message": msg, "matches": commits,
                               "message": f"Found {len(commits)} matches"}, indent=2)
        return json.dumps({"search_message": msg, "error": "No commits found"})

    def _get_commit_details_fallback(self, sha: str) -> str:
        result = self._git("show", "--pretty=format:%H|%an|%ae|%ad|%s|%b", "--name-status", sha)
        if result.returncode != 0:
            return self._json_error(f"Commit {sha} not found")

        lines = result.stdout.strip().split('\n')
        if not lines: return self._json_error("No output")

        parts = lines[0].split('|', 5)
        if len(parts) < 6: return self._json_error("Parse failed")

        files = [{"status": s, "file": f} for line in lines[1:]
                 if '\t' in line for s, f in [line.split('\t', 1)]]

        return json.dumps({"sha": parts[0], "author": parts[1], "email": parts[2],
                           "date": parts[3], "subject": parts[4], "body": parts[5],
                           "files_changed": files}, indent=2)

    def analyze_commit_patterns(self, analysis_type: Annotated[str, "'authors', 'files', 'messages', or 'general'"] = "general") -> str:
        try:
            if analysis_type == "authors":
                result = self._git("shortlog", "-sn", "--all")
                authors = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        authors.append({"name": parts[1], "commits": int(parts[0])})
                return json.dumps({"analysis_type": "authors", "top_contributors": authors[:10]}, indent=2)

            elif analysis_type == "files":
                result = self._git("log", "--pretty=format:", "--name-only")
                counts = {}
                for f in result.stdout.strip().split('\n'):
                    if f: counts[f] = counts.get(f, 0) + 1
                top = sorted(counts.items(), key=lambda x: -x[1])[:10]
                return json.dumps({"analysis_type": "files",
                                   "most_changed": [{"file": f, "changes": c} for f, c in top]}, indent=2)

            elif analysis_type == "messages":
                result = self._git("log", "--pretty=format:%s", "-100")
                words = {}
                for msg in result.stdout.strip().split('\n'):
                    for w in msg.lower().split():
                        if len(w) > 3: words[w] = words.get(w, 0) + 1
                top = sorted(words.items(), key=lambda x: -x[1])[:10]
                return json.dumps({"analysis_type": "messages",
                                   "common_words": [{"word": w, "count": c} for w, c in top]}, indent=2)

            else:  # general
                total = self._git("rev-list", "--all", "--count")
                contrib = self._git("shortlog", "-sn", "--all")
                result = {"analysis_type": "general"}
                if total.returncode == 0: result["total_commits"] = int(total.stdout.strip())
                if contrib.returncode == 0:
                    lines = contrib.stdout.strip().split('\n')
                    result["total_contributors"] = len(lines)
                    if lines and '\t' in lines[0]:
                        parts = lines[0].split('\t')
                        result["top_contributor"] = {"name": parts[1], "commits": int(parts[0])}
                return json.dumps(result, indent=2)
        except Exception as e:
            return self._json_error(e)
