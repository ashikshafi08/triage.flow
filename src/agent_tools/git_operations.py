# src/agent_tools/git_operations.py

import json
import logging
import subprocess
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Optional, Annotated, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..git_tools import GitBlameTools, GitHistoryTools
    from ..commit_index import CommitIndexManager
    from .utilities import get_current_head_sha_func, chunk_large_output_func # Renaming to avoid conflict

# For standalone utilities if needed, or they can be passed via core_explorer
from .utilities import get_current_head_sha as get_current_head_sha_util
from .utilities import chunk_large_output as chunk_large_output_util

logger = logging.getLogger(__name__)

class GitOperations:
    def __init__(self, 
                 git_blame_tools: 'GitBlameTools', 
                 git_history_tools: 'GitHistoryTools',
                 commit_index_manager: 'CommitIndexManager',
                 # Pass utility functions if they need instance data like repo_path
                 # or if we want to make dependencies explicit
                 get_current_head_sha_func: callable, # e.g., core_explorer._get_current_head_sha
                 chunk_large_output_func: callable # e.g., core_explorer._chunk_large_output
                ):
        self.git_blame_tools = git_blame_tools
        self.git_history_tools = git_history_tools
        self.commit_index_manager = commit_index_manager
        self._get_current_head_sha = get_current_head_sha_func
        self._chunk_large_output = chunk_large_output_func
        # Get repo_path from one of the tools
        self.repo_path = self.git_blame_tools.repo_path

    def git_blame_function(
        self,
        function_name: Annotated[str, "Name of the function or class to get blame information for"],
        file_path: Annotated[Optional[str], "File path to search in. If not provided, will search the entire codebase"] = None
    ) -> str:
        """Get git blame information for a specific function or class"""
        try:
            head_sha = self._get_current_head_sha() # Uses the passed function
            if not head_sha: return json.dumps({"error": "Could not get current HEAD SHA."})
            result = self.git_blame_tools.git_blame_function_at_commit(file_path, function_name, head_sha)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in git_blame_function: {e}")
            return json.dumps({"function_name": function_name, "file_path": file_path, "error": str(e)})

    def who_last_edited_line(
        self,
        file_path: Annotated[str, "Path to the file to check"],
        line_number: Annotated[int, "Line number to get blame information for"]
    ) -> str:
        """Get information about who last edited a specific line in a file"""
        try:
            head_sha = self._get_current_head_sha()
            if not head_sha: return json.dumps({"error": "Could not get current HEAD SHA."})
            result = self.git_blame_tools.git_blame_at_commit(file_path, head_sha, line_start=line_number, line_end=line_number)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in who_last_edited_line: {e}")
            return json.dumps({"file_path": file_path, "line_number": line_number, "error": str(e)})

    def git_blame_at_commit(
        self,
        file_path: Annotated[str, "Path to the file to blame"],
        commit_sha: Annotated[str, "Commit SHA to run blame at"],
        line_start: Annotated[Optional[int], "Starting line number (optional)"] = None,
        line_end: Annotated[Optional[int], "Ending line number (optional)"] = None
    ) -> str:
        try:
            result = self.git_blame_tools.git_blame_at_commit(file_path, commit_sha, line_start, line_end)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in git_blame_at_commit: {e}")
            return json.dumps({"error": str(e)})

    def find_commits_touching_function(
        self,
        function_name: Annotated[str, "Name of the function to track"],
        file_path: Annotated[str, "File containing the function"],
        limit: Annotated[int, "Maximum number of commits to return"] = 10
    ) -> str:
        try:
            result = self.git_history_tools.find_commits_touching_function(function_name, file_path, limit)
            return self._chunk_large_output(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Error finding commits for function: {e}")
            return json.dumps({"error": str(e)})

    def get_function_evolution(
        self,
        function_name: Annotated[str, "Name of the function to track evolution for"],
        file_path: Annotated[str, "File containing the function"],
        limit: Annotated[int, "Maximum number of commits to analyze"] = 5
    ) -> str:
        try:
            result = self.git_history_tools.get_function_evolution(function_name, file_path, limit)
            return self._chunk_large_output(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Error getting function evolution: {e}")
            return json.dumps({"error": str(e)})

    def find_pr_closing_commit(self, pr_number: Annotated[int, "PR number"]) -> str:
        try:
            result = self.git_history_tools.find_pr_closing_commit(pr_number)
            return json.dumps(result, indent=2) if result else json.dumps({"error": f"No closing commit for PR #{pr_number}"})
        except Exception as e:
            logger.error(f"Error finding PR closing commit for PR #{pr_number}: {e}")
            return json.dumps({"error": str(e)})

    def get_issue_closing_info(
        self,
        issue_number: Annotated[int, "Issue number to get closing information for"]
    ) -> str:
        """Get detailed information about who closed an issue and with what commit/PR"""
        try:
            from ..git_tools import IssueClosingTools
            if hasattr(self, 'issue_closing_tools'):
                result = self.issue_closing_tools.get_issue_closing_info(issue_number)
            else:
                # Create temporary instance if not available
                issue_closing_tools = IssueClosingTools(str(self.repo_path), None)
                result = issue_closing_tools.get_issue_closing_info(issue_number)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error getting issue closing info: {e}")
            return json.dumps({"error": str(e)})

    def get_open_issues_related_to_commit(
        self,
        commit_sha: Annotated[str, "Commit SHA to find related open issues for"]
    ) -> str:
        """Find open issues that might be related to changes in a specific commit"""
        try:
            from ..git_tools import IssueClosingTools
            if hasattr(self, 'issue_closing_tools'):
                result = self.issue_closing_tools.get_open_issues_related_to_commit(commit_sha)
            else:
                # Create temporary instance if not available
                issue_closing_tools = IssueClosingTools(str(self.repo_path), None)
                result = issue_closing_tools.get_open_issues_related_to_commit(commit_sha)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error finding open issues for commit: {e}")
            return json.dumps({"error": str(e)})
            
    def find_when_feature_was_added(
        self,
        feature_search_term: Annotated[str, "Code pattern, function name, or content to find when it was first added"]
    ) -> str:
        """Find when a specific feature, function, or code pattern was first added to the codebase using git history"""
        try:
            # First try using commit index if available
            if self.commit_index_manager.is_initialized():
                try:
                    import asyncio
                    # Check if we're already in an event loop
                    current_loop = None
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        current_loop = None
                    
                    if current_loop:
                        # We're already in an async context, use concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.commit_index_manager.search_commits(
                                    query=f"add {feature_search_term}", k=10, sort_by_date=True
                                )
                            )
                            results = future.result(timeout=30)
                    else:
                        # No event loop running, safe to use asyncio.run
                        results = asyncio.run(
                            self.commit_index_manager.search_commits(
                                query=f"add {feature_search_term}", k=10, sort_by_date=True
                            )
                        )
                    
                    if results:
                        oldest_commit = results[0].commit  # Assuming sorted oldest first
                        return json.dumps({
                            "search_term": feature_search_term,
                            "introducing_commit": {
                                "sha": oldest_commit.sha,
                                "date": oldest_commit.commit_date,
                                "subject": oldest_commit.subject,
                                "author": oldest_commit.author_name
                            },
                            "method": "commit_index_search"
                        }, indent=2)
                except Exception as e:
                    logger.warning(f"Commit index search failed, falling back to git log: {e}")
            
            # Fallback to git log
            git_cmd = [
                "git", "log", "-S", feature_search_term, 
                "--pretty=format:%H|%an|%ae|%ad|%s", "--date=short", "--reverse"
            ]
            result = subprocess.run(git_cmd, capture_output=True, text=True, cwd=self.repo_path, check=True)
            
            if not result.stdout.strip():
                return json.dumps({
                    "search_term": feature_search_term,
                    "message": f"No commits found that introduced '{feature_search_term}'"
                })
            
            # Parse the first (oldest) commit
            lines = result.stdout.strip().split('\n')
            first_line = lines[0]
            parts = first_line.split('|', 4)
            
            if len(parts) < 5:
                return json.dumps({
                    "search_term": feature_search_term,
                    "error": "Could not parse git log output"
                })
            
            commit_hash, author_name, author_email, date, subject = parts
            
            # Get files changed in this commit
            show_cmd = ["git", "show", "--name-only", "--pretty=format:", commit_hash]
            show_result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            files_changed = []
            if show_result.returncode == 0:
                files_changed = [f.strip() for f in show_result.stdout.strip().split('\n') if f.strip()]
            
            return json.dumps({
                "search_term": feature_search_term,
                "introducing_commit": {
                    "sha": commit_hash,
                    "author": author_name,
                    "email": author_email,
                    "date": date,
                    "subject": subject,
                    "files_changed": files_changed
                },
                "method": "git_log_search"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in find_when_feature_was_added: {e}")
            return json.dumps({"error": str(e), "search_term": feature_search_term})

    def get_file_history(
        self,
        file_path: Annotated[str, "Path to the file to get history for, relative to repository root"]
    ) -> str:
        """Get the timeline of issues/PRs that touched a file"""
        try:
            # If we have an issue_rag_system available through dependencies, use it
            # For now, use git log to get commit history and try to link to issues/PRs
            log_cmd = [
                "git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso",
                "--", file_path
            ]
            result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=self.repo_path, check=True)
            
            if not result.stdout.strip():
                return json.dumps({
                    "file_path": file_path,
                    "message": f"No history found for '{file_path}'"
                })
            
            timeline = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split('|', 3)
                if len(parts) >= 4:
                    commit_hash, author, date, subject = parts
                    
                    # Try to extract issue/PR references from commit message
                    import re
                    issue_refs = re.findall(r'#(\d+)', subject)
                    pr_refs = re.findall(r'pull request #(\d+)', subject, re.IGNORECASE)
                    
                    def parse_date(dt):
                        try:
                            return datetime.fromisoformat(dt.replace('Z', '+00:00'))
                        except:
                            return dt
                    
                    timeline.append({
                        "commit": commit_hash,
                        "author": author,
                        "date": date,
                        "subject": subject,
                        "issue_refs": issue_refs,
                        "pr_refs": pr_refs
                    })
            
            # Sort by date (newest first)
            timeline.sort(key=lambda x: parse_date(x["date"]), reverse=True)
            
            return self._chunk_large_output(json.dumps({
                "file_path": file_path,
                "timeline": timeline[:50],  # Limit to last 50 commits
                "total_commits": len(timeline)
            }, indent=2))
            
        except Exception as e:
            logger.error(f"Error getting file history: {e}")
            return json.dumps({"error": str(e), "file_path": file_path})

    def summarize_feature_evolution(
        self,
        feature_query: Annotated[str, "Feature name, keyword, or description to trace evolution for"]
    ) -> str:
        """Summarize how a feature evolved over time by finding all related issues, PRs, and changes"""
        try:
            timeline = []
            
            # Try to find related commits using git log
            log_cmd = [
                "git", "log", "--grep=" + feature_query, "--oneline", "--date=short",
                "--pretty=format:%H|%ad|%s"
            ]
            result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        commit_hash, date, subject = parts
                        timeline.append({
                            "type": "commit",
                            "sha": commit_hash,
                            "date": date,
                            "subject": subject
                        })
            
            # Also search for code changes using -S flag
            search_cmd = [
                "git", "log", "-S", feature_query, "--oneline", "--date=short",
                "--pretty=format:%H|%ad|%s"
            ]
            search_result = subprocess.run(search_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if search_result.returncode == 0 and search_result.stdout.strip():
                for line in search_result.stdout.strip().split('\n'):
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        commit_hash, date, subject = parts
                        # Avoid duplicates
                        if not any(t.get("sha") == commit_hash for t in timeline):
                            timeline.append({
                                "type": "code_change",
                                "sha": commit_hash,
                                "date": date,
                                "subject": subject
                            })
            
            # Sort by date
            def parse_date(item):
                return item.get("date", "")
            
            timeline.sort(key=parse_date)
            
            if not timeline:
                return json.dumps({
                    "feature_query": feature_query,
                    "timeline": [],
                    "message": "No evolution history found for this feature."
                })
            
            return self._chunk_large_output(json.dumps({
                "feature_query": feature_query,
                "timeline": timeline,
                "count": len(timeline),
                "summary": {
                    "total_commits": len([t for t in timeline if t["type"] in ["commit", "code_change"]]),
                    "earliest_date": timeline[0]["date"] if timeline else None,
                    "latest_date": timeline[-1]["date"] if timeline else None
                }
            }, indent=2))
            
        except Exception as e:
            logger.error(f"Error summarizing feature evolution: {e}")
            return json.dumps({
                "feature_query": feature_query,
                "error": str(e)
            })

    def who_implemented_this(
        self,
        feature_name: Annotated[str, "Name of the feature, function, class, or code pattern to find the original implementation for"],
        file_path: Annotated[Optional[str], "Optional file path to search in. If not provided, searches the entire repository"] = None
    ) -> str:
        """Find who initially implemented a feature, function, or class using git history"""
        try:
            # Build the git log command
            if file_path:
                # Validate file path
                import os
                abs_path = os.path.join(self.repo_path, file_path)
                if not abs_path.startswith(os.path.realpath(self.repo_path)):
                    return json.dumps({"error": "File path outside repository"})
                
                log_cmd = [
                    "git", "log", "--reverse", "--pretty=format:%H|%an|%ae|%ad|%s",
                    "-S", feature_name, "--", file_path
                ]
            else:
                log_cmd = [
                    "git", "log", "--reverse", "--pretty=format:%H|%an|%ae|%ad|%s",
                    "-S", feature_name
                ]
            
            result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return json.dumps({
                    "feature_name": feature_name,
                    "file_path": file_path,
                    "error": f"Git log failed: {result.stderr}"
                })
            
            if not result.stdout.strip():
                return json.dumps({
                    "feature_name": feature_name,
                    "file_path": file_path,
                    "message": f"No commits found that introduced '{feature_name}'"
                })
            
            # Get the first commit (oldest)
            lines = result.stdout.strip().split('\n')
            first_line = lines[0]
            parts = first_line.split('|', 4)
            
            if len(parts) < 5:
                return json.dumps({
                    "feature_name": feature_name,
                    "error": "Could not parse git log output"
                })
            
            commit_hash, author_name, author_email, date, subject = parts
            
            # Get files changed in this commit
            show_cmd = ["git", "show", "--name-only", "--pretty=format:", commit_hash]
            show_result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            files_changed = []
            if show_result.returncode == 0:
                files_changed = [f.strip() for f in show_result.stdout.strip().split('\n') if f.strip()]
            
            # Find related issue/PR from commit message
            related_issue = None
            import re
            issue_refs = re.findall(r'#(\d+)', subject)
            
            # Get all contributors to this feature
            all_contributors_cmd = ["git", "log", "--pretty=format:%an", "-S", feature_name]
            if file_path:
                all_contributors_cmd.extend(["--", file_path])
            
            contributors_result = subprocess.run(all_contributors_cmd, capture_output=True, text=True, cwd=self.repo_path)
            contributors = []
            total_contributors = 0
            
            if contributors_result.returncode == 0 and contributors_result.stdout.strip():
                all_contributors = contributors_result.stdout.strip().split('\n')
                unique_contributors = list(set(all_contributors))
                contributors = unique_contributors[:5]  # Top 5 contributors
                total_contributors = len(unique_contributors)
            
            result_data = {
                "feature_name": feature_name,
                "file_path": file_path,
                "initial_implementation": {
                    "commit": {
                        "sha": commit_hash,
                        "author": author_name,
                        "email": author_email,
                        "date": date,
                        "subject": subject
                    },
                    "files_changed": files_changed,
                    "issue_references": issue_refs
                },
                "contributors": {
                    "implementer": author_name,
                    "all_contributors": contributors,
                    "total_contributors": total_contributors
                },
                "method": "git_log_analysis"
            }
            
            if related_issue:
                result_data["related_issue"] = related_issue
            
            return json.dumps(result_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error in who_implemented_this: {e}")
            return json.dumps({
                "feature_name": feature_name,
                "file_path": file_path,
                "error": str(e)
            })

    # Commit Index Manager dependent methods
    async def search_commits( # Made async to align with CommitIndexManager
        self,
        query: Annotated[str, "Search query for commit messages and metadata"],
        k: Annotated[int, "Number of commits to return (default: 10)"] = 10,
        author_filter: Annotated[Optional[str], "Filter by author name (optional)"] = None,
        file_filter: Annotated[Optional[str], "Filter by file path (optional)"] = None
    ) -> str:
        try:
            if not self.commit_index_manager.is_initialized():
                return self._search_commits_fallback(query, k, author_filter, file_filter)
            
            file_filter_list = [file_filter] if file_filter else None
            results = await self.commit_index_manager.search_commits(
                query, k=k, author_filter=author_filter, file_filter=file_filter_list
            )
            return json.dumps([r.to_dict() for r in results], indent=2) # Assuming SearchResult has to_dict
        except Exception as e:
            logger.error(f"Error in search_commits: {e}")
            return self._search_commits_fallback(query, k, author_filter, file_filter)

    def _search_commits_fallback(self, query: str, k: int, author_filter: Optional[str], file_filter: Optional[str]) -> str:
        """Fallback git log search when commit index is not available"""
        try:
            cmd = ["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=short", f"-{k}"]
            
            if author_filter:
                cmd.extend(["--author", author_filter])
            
            if file_filter:
                cmd.extend(["--", file_filter])
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            commits = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        sha, author, date, subject = parts
                        if query.lower() in subject.lower():
                            commits.append({
                                "sha": sha,
                                "author": author,
                                "date": date,
                                "subject": subject
                            })
            
            return json.dumps(commits[:k], indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_file_timeline(
        self,
        file_path: Annotated[str, "Path to the file to get timeline for"],
        limit: Annotated[int, "Maximum number of commits to return (default: 20)"] = 20
    ) -> str:
        try:
            if not self.commit_index_manager.is_initialized():
                # Fallback to git log
                cmd = ["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=short", f"-{limit}", "--", file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                timeline = []
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split('|', 3)
                        if len(parts) >= 4:
                            sha, author, date, subject = parts
                            timeline.append({
                                "sha": sha,
                                "author": author,
                                "date": date,
                                "subject": subject
                            })
                
                return json.dumps({"file_path": file_path, "timeline": timeline}, indent=2)
            
            timeline = self.commit_index_manager.get_file_timeline(file_path, limit=limit)
            return json.dumps(timeline, indent=2)
        except Exception as e:
            logger.error(f"Error in get_file_timeline: {e}")
            return json.dumps({"error": str(e)})

    def get_file_commit_statistics(
        self,
        file_path: Annotated[str, "Path to the file to get statistics for"]
    ) -> str:
        try:
            if not self.commit_index_manager.is_initialized():
                # Fallback git stats
                cmd = ["git", "log", "--pretty=format:%an", "--", file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                if result.returncode == 0:
                    authors = result.stdout.strip().split('\n')
                    author_counts = {}
                    for author in authors:
                        if author:
                            author_counts[author] = author_counts.get(author, 0) + 1
                    
                    return json.dumps({
                        "file_path": file_path,
                        "total_commits": len(authors),
                        "unique_authors": len(author_counts),
                        "top_contributors": dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                    }, indent=2)
                
                return json.dumps({"error": "Could not get git statistics"})
            
            stats = self.commit_index_manager.get_file_statistics(file_path)
            return json.dumps(stats, indent=2)
        except Exception as e:
            logger.error(f"Error in get_file_commit_statistics: {e}")
            return json.dumps({"error": str(e)})

    def get_commit_details(
        self,
        commit_sha: Annotated[str, "Commit SHA to get details for (can be short or full SHA)"]
    ) -> str:
        try:
            if self.commit_index_manager.is_initialized():
                commit_doc = self.commit_index_manager.get_commit_by_sha(commit_sha)
                if commit_doc:
                    return json.dumps(commit_doc.to_dict(), indent=2)
            
            return self._get_commit_details_fallback(commit_sha)
        except Exception as e:
            logger.error(f"Error in get_commit_details: {e}")
            return self._get_commit_details_fallback(commit_sha)

    def _get_commit_details_fallback(self, commit_sha: str) -> str:
        """Fallback method to get commit details using git show"""
        try:
            cmd = ["git", "show", "--pretty=format:%H|%an|%ae|%ad|%s|%b", "--name-status", commit_sha]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return json.dumps({"error": f"Commit {commit_sha} not found"})
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return json.dumps({"error": "No output from git show"})
            
            # Parse the first line with commit info
            parts = lines[0].split('|', 5)
            if len(parts) < 6:
                return json.dumps({"error": "Could not parse commit info"})
            
            sha, author, email, date, subject, body = parts
            
            # Parse file changes
            files_changed = []
            for line in lines[1:]:
                if '\t' in line:
                    status, file_path = line.split('\t', 1)
                    files_changed.append({"status": status, "file": file_path})
            
            return json.dumps({
                "sha": sha,
                "author": author,
                "email": email,
                "date": date,
                "subject": subject,
                "body": body,
                "files_changed": files_changed
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in fallback commit details: {e}")
            return json.dumps({"error": str(e)})

    def analyze_commit_patterns(
        self,
        analysis_type: Annotated[str, "Type of analysis: 'authors', 'files', 'messages', or 'general'"] = "general"
    ) -> str:
        try:
            if analysis_type == "authors":
                cmd = ["git", "shortlog", "-sn", "--all"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                if result.returncode == 0:
                    authors = []
                    for line in result.stdout.strip().split('\n'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            count, name = parts[0], '\t'.join(parts[1:])
                            authors.append({"name": name, "commits": int(count)})
                    
                    return json.dumps({"analysis_type": "authors", "top_contributors": authors[:10]}, indent=2)
            
            elif analysis_type == "files":
                cmd = ["git", "log", "--pretty=format:", "--name-only"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                if result.returncode == 0:
                    file_counts = {}
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            file_counts[line] = file_counts.get(line, 0) + 1
                    
                    top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    return json.dumps({"analysis_type": "files", "most_changed_files": [{"file": f, "changes": c} for f, c in top_files]}, indent=2)
            
            elif analysis_type == "messages":
                cmd = ["git", "log", "--pretty=format:%s", "-100"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
                
                if result.returncode == 0:
                    messages = result.stdout.strip().split('\n')
                    # Simple analysis - count common words
                    word_counts = {}
                    for msg in messages:
                        words = msg.lower().split()
                        for word in words:
                            if len(word) > 3:  # Skip short words
                                word_counts[word] = word_counts.get(word, 0) + 1
                    
                    common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    return json.dumps({"analysis_type": "messages", "common_words": [{"word": w, "count": c} for w, c in common_words]}, indent=2)
            
            else:  # general
                # Get basic repository statistics
                cmd1 = ["git", "rev-list", "--all", "--count"]
                cmd2 = ["git", "shortlog", "-sn", "--all"]
                
                total_commits = subprocess.run(cmd1, capture_output=True, text=True, cwd=self.repo_path)
                contributors = subprocess.run(cmd2, capture_output=True, text=True, cwd=self.repo_path)
                
                result = {"analysis_type": "general"}
                
                if total_commits.returncode == 0:
                    result["total_commits"] = int(total_commits.stdout.strip())
                
                if contributors.returncode == 0:
                    contributor_lines = contributors.stdout.strip().split('\n')
                    result["total_contributors"] = len(contributor_lines)
                    if contributor_lines:
                        top_contributor = contributor_lines[0].split('\t')
                        if len(top_contributor) >= 2:
                            result["top_contributor"] = {"name": top_contributor[1], "commits": int(top_contributor[0])}
                
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.error(f"Error in analyze_commit_patterns: {e}")
            return json.dumps({"error": str(e)})
