"""
Issue Closing Tools
Provides enhanced issue closing analysis functionality
"""

import os
import json
import asyncio
import subprocess
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..issue_rag import IssueAwareRAG

logger = logging.getLogger(__name__)

class IssueClosingTools:
    """Enhanced issue closing analysis functionality"""
    
    def __init__(self, repo_path: str, issue_rag_system: Optional['IssueAwareRAG'] = None):
        self.repo_path = Path(repo_path)
        self.issue_rag_system = issue_rag_system
    
    def get_issue_closing_info(self, issue_number: int) -> Dict[str, Any]:
        """Get detailed information about who closed an issue and with what commit/PR"""
        try:
            if not self.issue_rag_system or not self.issue_rag_system.indexer:
                return {"error": "Issue RAG system not available"}
            
            # Get issue doc
            if issue_number not in self.issue_rag_system.indexer.issue_docs:
                return {"error": f"Issue #{issue_number} not found"}
            
            issue_doc = self.issue_rag_system.indexer.issue_docs[issue_number]
            
            # Get patch linkage info
            patch_builder = self.issue_rag_system.indexer.patch_builder
            links_by_issue = patch_builder.load_patch_links()
            
            result = {
                "issue_number": issue_number,
                "title": issue_doc.title,
                "state": issue_doc.state,
                "closed_at": issue_doc.closed_at
            }
            
            # Get the PR that closed this issue
            if issue_number in links_by_issue:
                pr_links = links_by_issue[issue_number]
                if pr_links:
                    # Get the closing PR info
                    closing_pr = pr_links[0]  # Usually there's only one
                    
                    result["closed_by"] = {
                        "pr_number": closing_pr.pr_number,
                        "pr_title": closing_pr.pr_title,
                        "pr_url": closing_pr.pr_url,
                        "merged_at": closing_pr.merged_at
                    }
                    
                    # Get the actual closing commit
                    closing_commit_info = self._get_pr_closing_commit(closing_pr.pr_number)
                    if closing_commit_info:
                        result["closing_commit"] = closing_commit_info
                    
                    # Get the diff
                    diff_docs = self.issue_rag_system.indexer.diff_docs
                    if closing_pr.pr_number in diff_docs:
                        diff_doc = diff_docs[closing_pr.pr_number]
                        result["files_changed"] = diff_doc.files_changed
                        result["diff_summary"] = diff_doc.diff_summary[:500] + "..." if len(diff_doc.diff_summary) > 500 else diff_doc.diff_summary
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting issue closing info: {e}")
            return {"error": str(e)}
    
    def get_open_issues_related_to_commit(self, commit_sha: str) -> Dict[str, Any]:
        """Find open issues that might be related to changes in a specific commit"""
        try:
            if not self.issue_rag_system:
                return {"error": "Issue RAG system not available"}
            
            # Get commit details
            show_cmd = ["git", "show", "--name-only", "--pretty=format:%s", commit_sha]
            result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return {"error": f"Commit {commit_sha} not found"}
            
            lines = result.stdout.strip().split('\n')
            commit_message = lines[0]
            files_changed = [f for f in lines[1:] if f.strip()]
            
            # Search for related open issues
            related_issues = []
            
            # Search by commit message keywords
            keywords = [word for word in commit_message.split() if len(word) > 3]
            query = ' '.join(keywords[:5])  # Use top 5 keywords
            
            # Use the issue RAG system to find related issues
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                issue_context = loop.run_until_complete(
                    self.issue_rag_system.get_issue_context(query, max_issues=10)
                )
                
                for search_result in issue_context.related_issues:
                    issue = search_result.issue
                    if issue.state == "open":
                        # Check if any of the commit's files are mentioned in the issue
                        files_mentioned = any(
                            file in issue.body or file in issue.title
                            for file in files_changed
                        )
                        
                        related_issues.append({
                            "number": issue.id,
                            "title": issue.title,
                            "state": issue.state,
                            "url": f"https://github.com/{self.issue_rag_system.repo_owner}/{self.issue_rag_system.repo_name}/issues/{issue.id}",
                            "similarity": round(search_result.similarity, 3),
                            "files_mentioned": files_mentioned,
                            "labels": issue.labels
                        })
                
            finally:
                loop.close()
            
            return {
                "commit_sha": commit_sha,
                "commit_message": commit_message,
                "files_changed": files_changed,
                "related_open_issues": related_issues,
                "total_found": len(related_issues)
            }
            
        except Exception as e:
            logger.error(f"Error finding issues related to commit: {e}")
            return {"error": str(e)}
    
    def analyze_issue_resolution_pattern(self, issue_numbers: List[int]) -> Dict[str, Any]:
        """Analyze patterns in how multiple issues were resolved"""
        try:
            if not self.issue_rag_system:
                return {"error": "Issue RAG system not available"}
            
            resolution_data = []
            
            for issue_num in issue_numbers:
                closing_info = self.get_issue_closing_info(issue_num)
                if "error" not in closing_info:
                    resolution_data.append(closing_info)
            
            if not resolution_data:
                return {"error": "No valid issue resolution data found"}
            
            # Analyze patterns
            patterns = {
                "total_issues": len(resolution_data),
                "resolved_by_pr": sum(1 for r in resolution_data if "closed_by" in r),
                "common_authors": {},
                "common_files": {},
                "resolution_timeline": []
            }
            
            # Count authors and files
            for resolution in resolution_data:
                if "closing_commit" in resolution:
                    author = resolution["closing_commit"].get("author_name", "unknown")
                    patterns["common_authors"][author] = patterns["common_authors"].get(author, 0) + 1
                
                for file_path in resolution.get("files_changed", []):
                    patterns["common_files"][file_path] = patterns["common_files"].get(file_path, 0) + 1
                
                patterns["resolution_timeline"].append({
                    "issue_number": resolution["issue_number"],
                    "closed_at": resolution.get("closed_at"),
                    "pr_number": resolution.get("closed_by", {}).get("pr_number")
                })
            
            # Sort by frequency
            patterns["common_authors"] = dict(sorted(
                patterns["common_authors"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            patterns["common_files"] = dict(sorted(
                patterns["common_files"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing resolution patterns: {e}")
            return {"error": str(e)}
    
    def find_issues_closed_by_author(self, author_name: str, limit: int = 10) -> Dict[str, Any]:
        """Find issues closed by a specific author"""
        try:
            if not self.issue_rag_system:
                return {"error": "Issue RAG system not available"}
            
            closed_issues = []
            
            # Get all closed issues
            for issue_id, issue_doc in self.issue_rag_system.indexer.issue_docs.items():
                if issue_doc.state == "closed":
                    closing_info = self.get_issue_closing_info(issue_id)
                    
                    if ("closing_commit" in closing_info and 
                        closing_info["closing_commit"].get("author_name") == author_name):
                        closed_issues.append({
                            "issue_number": issue_id,
                            "title": issue_doc.title,
                            "closed_at": issue_doc.closed_at,
                            "pr_number": closing_info.get("closed_by", {}).get("pr_number"),
                            "files_changed": closing_info.get("files_changed", [])
                        })
                        
                        if len(closed_issues) >= limit:
                            break
            
            return {
                "author_name": author_name,
                "closed_issues": closed_issues,
                "total_found": len(closed_issues)
            }
            
        except Exception as e:
            logger.error(f"Error finding issues closed by author: {e}")
            return {"error": str(e)}
    
    def _get_pr_closing_commit(self, pr_number: int) -> Optional[Dict[str, Any]]:
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