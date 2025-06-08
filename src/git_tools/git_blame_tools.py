"""
Git Blame Tools
Provides enhanced git blame functionality including historical blame at specific commits
"""

import os
import json
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class GitBlameTools:
    """Enhanced git blame functionality for historical analysis"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def git_blame_at_commit(
        self,
        file_path: str,
        commit_sha: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get git blame information for a file at a specific commit"""
        try:
            # Build blame command
            blame_cmd = ["git", "blame", "-w", "-C"]
            
            if line_start and line_end:
                blame_cmd.extend(["-L", f"{line_start},{line_end}"])
            elif line_start:
                blame_cmd.extend(["-L", f"{line_start},{line_start}"])
            
            blame_cmd.extend([commit_sha, "--", file_path])
            
            result = subprocess.run(
                blame_cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.repo_path
            )
            
            if result.returncode != 0:
                return {
                    "error": f"Git blame failed: {result.stderr}",
                    "file_path": file_path,
                    "commit_sha": commit_sha
                }
            
            # Parse blame output
            blame_lines = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parsed_line = self._parse_blame_line(line)
                    if parsed_line:
                        blame_lines.append(parsed_line)
            
            return {
                "file_path": file_path,
                "commit_sha": commit_sha,
                "line_range": self._format_line_range(line_start, line_end),
                "blame_lines": blame_lines,
                "total_lines": len(blame_lines)
            }
            
        except Exception as e:
            logger.error(f"Error in git_blame_at_commit: {e}")
            return {"error": str(e)}
    
    def git_blame_function_at_commit(
        self,
        file_path: str,
        function_name: str,
        commit_sha: str
    ) -> Dict[str, Any]:
        """Get blame for a specific function at a specific commit"""
        try:
            # First, find the function in the file at that commit
            function_lines = self._find_function_lines_at_commit(
                file_path, function_name, commit_sha
            )
            
            if not function_lines:
                return {
                    "error": f"Function '{function_name}' not found in {file_path} at commit {commit_sha}"
                }
            
            # Get blame for those specific lines
            blame_result = self.git_blame_at_commit(
                file_path, 
                commit_sha, 
                function_lines["start"], 
                function_lines["end"]
            )
            
            if "error" in blame_result:
                return blame_result
            
            # Add function context
            blame_result.update({
                "function_name": function_name,
                "function_start_line": function_lines["start"],
                "function_end_line": function_lines["end"],
                "function_signature": function_lines.get("signature", "")
            })
            
            return blame_result
            
        except Exception as e:
            logger.error(f"Error in git_blame_function_at_commit: {e}")
            return {"error": str(e)}
    
    def compare_blame_across_commits(
        self,
        file_path: str,
        commit_sha1: str,
        commit_sha2: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compare blame information between two commits"""
        try:
            blame1 = self.git_blame_at_commit(file_path, commit_sha1, line_start, line_end)
            blame2 = self.git_blame_at_commit(file_path, commit_sha2, line_start, line_end)
            
            if "error" in blame1 or "error" in blame2:
                return {
                    "error": "Failed to get blame for one or both commits",
                    "blame1_error": blame1.get("error"),
                    "blame2_error": blame2.get("error")
                }
            
            # Analyze differences
            changes = self._analyze_blame_changes(
                blame1.get("blame_lines", []),
                blame2.get("blame_lines", [])
            )
            
            return {
                "file_path": file_path,
                "commit1": commit_sha1,
                "commit2": commit_sha2,
                "line_range": self._format_line_range(line_start, line_end),
                "blame1": blame1,
                "blame2": blame2,
                "changes": changes
            }
            
        except Exception as e:
            logger.error(f"Error comparing blame across commits: {e}")
            return {"error": str(e)}
    
    def _parse_blame_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single git blame line"""
        try:
            # Parse git blame format: commit_hash (author date line_num) code
            parts = line.split(')', 1)
            if len(parts) != 2:
                return None
            
            commit_info = parts[0] + ')'
            code = parts[1]
            
            # Extract commit hash (first part before space)
            commit_hash = commit_info.split()[0]
            
            # Extract author and date from parentheses
            paren_start = commit_info.find('(')
            paren_end = commit_info.rfind(')')
            
            if paren_start == -1 or paren_end == -1:
                return None
            
            paren_content = commit_info[paren_start + 1:paren_end]
            
            # Parse the parentheses content
            parts = paren_content.rsplit(' ', 2)  # Split from right to get line_num and date
            
            if len(parts) >= 3:
                line_num = parts[-1]
                date_time = parts[-2]
                author_info = ' '.join(parts[:-2])
            else:
                line_num = ""
                date_time = ""
                author_info = paren_content
            
            return {
                "commit": commit_hash,
                "author": author_info,
                "date": date_time,
                "line_number": line_num,
                "code": code.strip()
            }
            
        except Exception as e:
            logger.debug(f"Failed to parse blame line: {line}, error: {e}")
            return None
    
    def _find_function_lines_at_commit(
        self,
        file_path: str,
        function_name: str,
        commit_sha: str
    ) -> Optional[Dict[str, Any]]:
        """Find the line range of a function at a specific commit"""
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
            lines = content.split('\n')
            
            # Simple function detection (can be enhanced for different languages)
            function_start = None
            function_signature = ""
            brace_count = 0
            in_function = False
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Look for function definition
                if not in_function and function_name in line:
                    # Basic patterns for different languages
                    if any(pattern in line for pattern in [
                        f"def {function_name}",  # Python
                        f"function {function_name}",  # JavaScript
                        f"{function_name}(",  # General function call pattern
                        f"func {function_name}",  # Go
                        f"fn {function_name}",  # Rust
                    ]):
                        function_start = i
                        function_signature = line.strip()
                        in_function = True
                        brace_count = line.count('{') - line.count('}')
                
                # Track braces to find function end
                elif in_function:
                    brace_count += line.count('{') - line.count('}')
                    
                    # For Python, look for dedentation
                    if function_signature.startswith('def ') and stripped and not line.startswith(' ') and not line.startswith('\t'):
                        return {
                            "start": function_start,
                            "end": i - 1,
                            "signature": function_signature
                        }
                    
                    # For brace languages, when braces balance
                    elif brace_count <= 0 and stripped.endswith('}'):
                        return {
                            "start": function_start,
                            "end": i,
                            "signature": function_signature
                        }
            
            # If we're still in function at end of file
            if in_function and function_start:
                return {
                    "start": function_start,
                    "end": len(lines),
                    "signature": function_signature
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding function lines: {e}")
            return None
    
    def _format_line_range(self, start: Optional[int], end: Optional[int]) -> str:
        """Format line range for display"""
        if start and end:
            return f"{start}-{end}"
        elif start:
            return str(start)
        else:
            return "all"
    
    def _analyze_blame_changes(self, blame1: List[Dict], blame2: List[Dict]) -> Dict[str, Any]:
        """Analyze changes between two blame results"""
        try:
            changes = {
                "lines_changed": 0,
                "authors_changed": [],
                "commits_changed": [],
                "summary": ""
            }
            
            # Simple comparison by line number
            blame1_by_line = {b.get("line_number", ""): b for b in blame1}
            blame2_by_line = {b.get("line_number", ""): b for b in blame2}
            
            for line_num in set(blame1_by_line.keys()) | set(blame2_by_line.keys()):
                b1 = blame1_by_line.get(line_num)
                b2 = blame2_by_line.get(line_num)
                
                if not b1 or not b2:
                    changes["lines_changed"] += 1
                elif b1.get("commit") != b2.get("commit"):
                    changes["lines_changed"] += 1
                    if b1.get("author") != b2.get("author"):
                        changes["authors_changed"].extend([b1.get("author"), b2.get("author")])
                    changes["commits_changed"].extend([b1.get("commit"), b2.get("commit")])
            
            # Remove duplicates
            changes["authors_changed"] = list(set(filter(None, changes["authors_changed"])))
            changes["commits_changed"] = list(set(filter(None, changes["commits_changed"])))
            
            # Generate summary
            if changes["lines_changed"] == 0:
                changes["summary"] = "No changes detected in blame information"
            else:
                changes["summary"] = f"{changes['lines_changed']} lines changed blame information"
                if changes["authors_changed"]:
                    changes["summary"] += f", involving {len(changes['authors_changed'])} authors"
            
            return changes
            
        except Exception as e:
            logger.error(f"Error analyzing blame changes: {e}")
            return {"error": str(e)} 