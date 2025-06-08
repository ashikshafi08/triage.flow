"""
Git History Tools
Provides enhanced git history functionality for tracking function and file changes over time
"""

import os
import json
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class GitHistoryTools:
    """Enhanced git history functionality for detailed code tracking"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def find_commits_touching_function(
        self,
        function_name: str,
        file_path: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find all commits that modified a specific function"""
        try:
            # Use git log with -L to track function changes
            log_cmd = [
                "git", "log",
                f"-L:{function_name}:{file_path}",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                f"-{limit}"
            ]
            
            result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                # Fallback to searching for the function name
                log_cmd = [
                    "git", "log", "-p", "-S", function_name,
                    "--pretty=format:COMMIT:%H|%an|%ae|%ad|%s",
                    "--", file_path,
                    f"-{limit}"
                ]
                result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            commits = []
            current_commit = None
            
            for line in result.stdout.split('\n'):
                if line.startswith('COMMIT:'):
                    if current_commit:
                        commits.append(current_commit)
                    
                    parts = line[7:].split('|', 4)
                    if len(parts) >= 5:
                        current_commit = {
                            "sha": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "date": parts[3],
                            "message": parts[4],
                            "changes": []
                        }
                elif current_commit and (line.startswith('+') or line.startswith('-')):
                    if function_name in line:
                        current_commit["changes"].append(line)
            
            if current_commit:
                commits.append(current_commit)
            
            return {
                "function_name": function_name,
                "file_path": file_path,
                "commits": commits,
                "total_found": len(commits)
            }
            
        except Exception as e:
            logger.error(f"Error finding commits for function: {e}")
            return {"error": str(e)}
    
    def get_function_evolution(
        self,
        function_name: str,
        file_path: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Get the evolution of a function over time with diff details"""
        try:
            commits_result = self.find_commits_touching_function(function_name, file_path, limit)
            
            if "error" in commits_result:
                return commits_result
            
            evolution = []
            commits = commits_result.get("commits", [])
            
            for i, commit in enumerate(commits):
                commit_sha = commit["sha"]
                
                # Get the function content at this commit
                function_content = self._get_function_at_commit(file_path, function_name, commit_sha)
                
                # Get diff from previous commit if available
                diff_info = None
                if i < len(commits) - 1:  # Not the oldest commit
                    prev_commit = commits[i + 1]["sha"]
                    diff_info = self._get_function_diff_between_commits(
                        file_path, function_name, prev_commit, commit_sha
                    )
                
                evolution.append({
                    "commit": commit,
                    "function_content": function_content,
                    "diff_from_previous": diff_info
                })
            
            return {
                "function_name": function_name,
                "file_path": file_path,
                "evolution": evolution,
                "total_commits": len(evolution)
            }
            
        except Exception as e:
            logger.error(f"Error getting function evolution: {e}")
            return {"error": str(e)}
    
    def find_pr_closing_commit(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get the merge commit information for a PR"""
        try:
            # Use git log to find the merge commit
            merge_commit_cmd = [
                "git", "log", "--grep", f"Merge pull request #{pr_number}",
                "--pretty=format:%H|%an|%ae|%ad|%s", "-1"
            ]
            
            result = subprocess.run(merge_commit_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|', 4)
                if len(parts) >= 5:
                    return {
                        "sha": parts[0],
                        "author_name": parts[1],
                        "author_email": parts[2],
                        "date": parts[3],
                        "message": parts[4]
                    }
            
            # Fallback: look for squash commits
            squash_commit_cmd = [
                "git", "log", "--grep", f"(#{pr_number})",
                "--pretty=format:%H|%an|%ae|%ad|%s", "-10"
            ]
            
            result = subprocess.run(squash_commit_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split('|', 4)
                    if len(parts) >= 5:
                        return {
                            "sha": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting PR closing commit: {e}")
            return None
    
    def get_commit_details(self, commit_sha: str) -> Dict[str, Any]:
        """Get detailed information about a commit"""
        try:
            # Get commit info
            show_cmd = ["git", "show", "--name-only", "--pretty=format:%H|%an|%ae|%ad|%s|%B", commit_sha]
            result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return {"error": f"Commit {commit_sha} not found"}
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return {"error": "No commit information found"}
            
            # Parse commit info
            commit_info = lines[0].split('|', 5)
            if len(commit_info) < 6:
                return {"error": "Invalid commit format"}
            
            # Find where the commit body ends and files begin
            body_lines = []
            files_changed = []
            in_files = False
            
            for line in lines[1:]:
                if not in_files and (not line.strip() or not any(c.isalpha() for c in line)):
                    # Empty line or no alphabetic chars - might be transition to files
                    if line.strip():
                        files_changed.append(line.strip())
                        in_files = True
                elif in_files:
                    if line.strip():
                        files_changed.append(line.strip())
                else:
                    body_lines.append(line)
            
            return {
                "sha": commit_info[0],
                "author_name": commit_info[1],
                "author_email": commit_info[2],
                "date": commit_info[3],
                "subject": commit_info[4],
                "body": '\n'.join(body_lines).strip(),
                "files_changed": files_changed
            }
            
        except Exception as e:
            logger.error(f"Error getting commit details: {e}")
            return {"error": str(e)}
    
    def find_related_commits_by_files(
        self,
        file_paths: List[str],
        since_date: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find commits that touched specific files"""
        try:
            cmd = ["git", "log", "--pretty=format:%H|%an|%ae|%ad|%s", f"-{limit}"]
            
            if since_date:
                cmd.extend(["--since", since_date])
            
            cmd.append("--")
            cmd.extend(file_paths)
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return {"error": f"Git log failed: {result.stderr}"}
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('|', 4)
                    if len(parts) >= 5:
                        commits.append({
                            "sha": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        })
            
            return {
                "file_paths": file_paths,
                "commits": commits,
                "total_found": len(commits)
            }
            
        except Exception as e:
            logger.error(f"Error finding related commits: {e}")
            return {"error": str(e)}
    
    def _get_function_at_commit(
        self,
        file_path: str,
        function_name: str,
        commit_sha: str
    ) -> Optional[str]:
        """Get the content of a function at a specific commit"""
        try:
            # Get file content at that commit
            show_cmd = ["git", "show", f"{commit_sha}:{file_path}"]
            result = subprocess.run(
                show_cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode != 0:
                return None
            
            content = result.stdout
            # Simple function extraction (can be enhanced)
            lines = content.split('\n')
            
            function_lines = []
            in_function = False
            brace_count = 0
            
            for line in lines:
                if not in_function and function_name in line:
                    # Basic patterns for different languages
                    if any(pattern in line for pattern in [
                        f"def {function_name}",  # Python
                        f"function {function_name}",  # JavaScript
                        f"{function_name}(",  # General function call pattern
                        f"func {function_name}",  # Go
                        f"fn {function_name}",  # Rust
                    ]):
                        in_function = True
                        function_lines.append(line)
                        brace_count = line.count('{') - line.count('}')
                
                elif in_function:
                    function_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    
                    # For Python, look for dedentation
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        if any(func_line.strip().startswith('def ') for func_line in function_lines):
                            break
                    
                    # For brace languages, when braces balance
                    elif brace_count <= 0 and line.strip().endswith('}'):
                        break
            
            return '\n'.join(function_lines) if function_lines else None
            
        except Exception as e:
            logger.error(f"Error getting function at commit: {e}")
            return None
    
    def _get_function_diff_between_commits(
        self,
        file_path: str,
        function_name: str,
        old_commit: str,
        new_commit: str
    ) -> Optional[Dict[str, Any]]:
        """Get diff of a function between two commits"""
        try:
            old_content = self._get_function_at_commit(file_path, function_name, old_commit)
            new_content = self._get_function_at_commit(file_path, function_name, new_commit)
            
            if not old_content or not new_content:
                return None
            
            # Simple diff calculation
            old_lines = old_content.split('\n')
            new_lines = new_content.split('\n')
            
            return {
                "old_commit": old_commit,
                "new_commit": new_commit,
                "old_lines_count": len(old_lines),
                "new_lines_count": len(new_lines),
                "lines_added": len(new_lines) - len(old_lines),
                "old_content": old_content,
                "new_content": new_content
            }
            
        except Exception as e:
            logger.error(f"Error getting function diff: {e}")
            return None 