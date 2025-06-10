import json
from typing import Optional, List, Dict, Any
from .agent_tools import AgenticCodebaseExplorer, FunctionTool
from .issue_rag import IssueAwareRAG
from .new_rag import LocalRepoContextExtractor
from llama_index.core.agent import ReActAgent
from .chunk_store import ChunkStoreFactory
from .config import settings
import subprocess
import logging
import os
from functools import lru_cache
import contextlib
import io
import sys
import asyncio

logger = logging.getLogger(__name__)

class FoundingMemberAgent:
    def __init__(self, session_id: str, code_rag: LocalRepoContextExtractor, issue_rag: IssueAwareRAG):
        self.session_id = session_id
        self.code_rag = code_rag
        self.issue_rag = issue_rag
        self.explorer = AgenticCodebaseExplorer(session_id, code_rag.repo_info['repo_path'], issue_rag_system=issue_rag)
        self.tools = self._register_tools()
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.explorer.llm,
            memory=self.explorer.memory,
            verbose=True,
            max_iterations=settings.AGENTIC_MAX_ITERATIONS
        )
        self.chunk_store = ChunkStoreFactory.get_instance()
        # TODO: Add memory, reasoning loop, etc.

    def _register_tools(self) -> List[FunctionTool]:
        tools = self.explorer._create_tools()
        tools.extend([
            FunctionTool.from_defaults(
                fn=self.get_file_history,
                name="get_file_history",
                description="Get the timeline of issues/PRs that touched a file. Use this to understand how a file evolved over time."
            ),
            FunctionTool.from_defaults(
                fn=self.summarize_feature_evolution,
                name="summarize_feature_evolution",
                description="Summarize how a feature evolved over time. Use this to understand the development history of a specific feature or component."
            ),
            FunctionTool.from_defaults(
                fn=self.who_fixed_this,
                name="who_fixed_this",
                description="Find who/what last changed a line or function. Use this to identify the most recent changes to specific code."
            ),
            FunctionTool.from_defaults(
                fn=self.who_implemented_this,
                name="who_implemented_this",
                description="Find who initially implemented a class, function, or feature. Use this to identify the original author of code."
            ),
            FunctionTool.from_defaults(
                fn=self.regression_detector,
                name="regression_detector",
                description="Detect if a new issue is a regression of a past one. Use this to identify potential regressions in the codebase."
            ),
        ])
        return tools

    def _chunk_large_output(self, content: str) -> str:
        """Handle large outputs by chunking and storing in Redis."""
        try:
            chunk_size = settings.CHUNK_STORE_CONFIG.get("max_chunk_size", 8192)
            preview_size = getattr(settings, "MAX_CONTENT_PREVIEW_CHARS", 1000)
            if len(content) > chunk_size:
                chunk_id = self.chunk_store.store(content)
                return json.dumps({
                    "type": "chunked",
                    "chunk_id": chunk_id,
                    "preview": content[:preview_size] + "...",
                    "total_size": len(content),
                    "message": "Response was too large and has been chunked. Use the chunk_id to retrieve the full content."
                })
            return content
        except Exception as e:
            logger.error(f"Error chunking output: {e}")
            preview_size = getattr(settings, "MAX_CONTENT_PREVIEW_CHARS", 1000)
            return json.dumps({
                "error": "Error processing large output",
                "message": str(e),
                "preview": content[:preview_size] + "..." if len(content) > preview_size else content
            })

    # --- New tool implementations will be added here ---
    def get_file_history(self, file_path: str) -> str:
        """
        Returns a timeline of all PRs and issues that touched the given file, using patch linkage and diff docs.
        """
        if not self.issue_rag or not self.issue_rag.indexer.patch_builder:
            return json.dumps({"error": "Issue RAG or patch linkage not available."})
        patch_builder = self.issue_rag.indexer.patch_builder
        links_by_issue = patch_builder.load_patch_links()
        diff_docs = self.issue_rag.indexer.diff_docs if hasattr(self.issue_rag.indexer, 'diff_docs') else {}
        # Build a list of (pr, issue) for this file
        history = []
        for issue_id, pr_links in links_by_issue.items():
            for link in pr_links:
                # Check if this file is in the files_changed for this PR
                if file_path in link.files_changed:
                    # Try to get diff summary if available
                    diff_doc = diff_docs.get(link.pr_number)
                    diff_summary = diff_doc.diff_summary if diff_doc else None
                    # Try to get issue title
                    issue_title = None
                    issue_url = None
                    if issue_id in self.issue_rag.indexer.issue_docs:
                        issue_doc = self.issue_rag.indexer.issue_docs[issue_id]
                        issue_title = issue_doc.title
                        issue_url = f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/issues/{issue_id}"
                    history.append({
                        "pr_number": link.pr_number,
                        "pr_title": link.pr_title,
                        "pr_url": link.pr_url,
                        "merged_at": link.merged_at,
                        "issue_id": issue_id,
                        "issue_title": issue_title,
                        "issue_url": issue_url,
                        "diff_summary": diff_summary
                    })
        # Sort by merged_at (if available)
        def parse_date(dt):
            from datetime import datetime
            try:
                return datetime.fromisoformat(dt.replace('Z', '+00:00')) if dt else None
            except Exception:
                return None
        history.sort(key=lambda x: parse_date(x.get('merged_at')), reverse=False)
        if not history:
            return json.dumps({"file": file_path, "history": [], "message": "No PR or issue history found for this file."})
        # Chunk the output if it's too large
        result = json.dumps({"file": file_path, "history": history, "count": len(history)})
        return self._chunk_large_output(result)

    def summarize_feature_evolution(self, feature_query: str) -> str:
        """
        Summarizes the evolution of a feature by searching for all issues, PRs, and diffs related to the feature (by keyword, file, or folder).
        """
        # Find all issues matching the feature query
        if not self.issue_rag or not self.issue_rag.indexer:
            return json.dumps({"error": "Issue RAG not available."})
        # Use the issue retriever to find related issues
        retriever = self.issue_rag.retriever
        issues, patches = [], []
        try:
            # Use a high k to get a broad timeline
            issues, patches = self.issue_rag.indexer.issue_docs.values(), self.issue_rag.indexer.diff_docs.values()
            # Filter issues by keyword
            feature_lower = feature_query.lower()
            related_issues = [i for i in issues if feature_lower in i.title.lower() or feature_lower in i.body.lower()]
            # Find all PRs that mention the feature or touch related files
            related_patches = []
            for patch in patches:
                if feature_lower in patch.diff_summary.lower():
                    related_patches.append(patch)
                else:
                    # Also check file names
                    if any(feature_lower in f.lower() for f in patch.files_changed):
                        related_patches.append(patch)
            # Build a timeline
            timeline = []
            for issue in related_issues:
                timeline.append({
                    "type": "issue",
                    "id": issue.id,
                    "title": issue.title,
                    "url": f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/issues/{issue.id}",
                    "created_at": issue.created_at,
                    "closed_at": issue.closed_at,
                    "labels": issue.labels
                })
            for patch in related_patches:
                timeline.append({
                    "type": "pr",
                    "pr_number": patch.pr_number,
                    "issue_id": patch.issue_id,
                    "files_changed": patch.files_changed,
                    "diff_summary": patch.diff_summary,
                    "url": f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/pull/{patch.pr_number}"
                })
            # Sort by created/merged date if available
            def parse_date(item):
                from datetime import datetime
                if item["type"] == "issue":
                    return item["created_at"] or ""
                else:
                    return item.get("merged_at") or ""
            timeline.sort(key=parse_date)
            if not timeline:
                return json.dumps({"feature": feature_query, "timeline": [], "message": "No evolution history found for this feature."})
            # Chunk the output if it's too large
            result = json.dumps({"feature": feature_query, "timeline": timeline, "count": len(timeline)})
            return self._chunk_large_output(result)
        except Exception as e:
            return json.dumps({"feature": feature_query, "error": str(e)})

    def who_fixed_this(self, file_path: str, line_number: int = None) -> str:
        """
        Finds who/what last changed a file (and optionally a line).
        Uses patch linkage and diff docs, with git blame fallback for line-level changes.
        """
        repo_root = self.code_rag.repo_info["repo_path"]
        abs_path = os.path.realpath(os.path.join(repo_root, file_path))
        if not abs_path.startswith(os.path.realpath(repo_root)):
            return json.dumps({"error": "File outside repo"})
        if not self.issue_rag or not self.issue_rag.indexer:
            return json.dumps({"error": "Issue RAG not available."})
        
        diff_docs = self.issue_rag.indexer.diff_docs if hasattr(self.issue_rag.indexer, 'diff_docs') else {}
        # Find all diffs that touched this file
        candidates = []
        for diff_doc in diff_docs.values():
            if file_path in diff_doc.files_changed:
                if line_number is not None:
                    # Try to find the line in the diff summary
                    try:
                        if hasattr(diff_doc, 'diff_path') and diff_doc.diff_path:
                            with open(diff_doc.diff_path, 'r', encoding='utf-8', errors='ignore') as f:
                                diff_text = f.read()
                            # Look for the line number context (best effort: look for +/- and context lines)
                            line_str = f"+{line_number}"  # Not perfect, but a hint
                            if line_str in diff_text or f"-{line_number}" in diff_text:
                                candidates.append(diff_doc)
                                continue
                    except Exception:
                        pass
                candidates.append(diff_doc)
        
        # If no candidates found and we have a line number, try git blame
        if not candidates and line_number is not None:
            try:
                # Change to repo directory for git commands
                original_cwd = os.getcwd()
                os.chdir(repo_root)
                
                # Run git blame for the specific line
                blame_cmd = ["git", "blame", "-L", f"{line_number},{line_number}", file_path]
                result = subprocess.run(blame_cmd, capture_output=True, text=True, cwd=repo_root)
                
                if result.returncode == 0 and result.stdout:
                    # Parse git blame output
                    # Format: <commit-hash> (<author> <date>) <line>
                    blame_line = result.stdout.strip()
                    if blame_line:
                        commit_hash = blame_line.split()[0]
                        # Get commit details
                        show_cmd = ["git", "show", "--name-only", "--pretty=format:%an|%ad|%s", commit_hash]
                        show_result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=repo_root)
                        if show_result.returncode == 0:
                            lines = show_result.stdout.split('\n')
                            if lines:
                                parts = lines[0].split('|', 2)
                                if len(parts) >= 3:
                                    author, date, subject = parts
                                    return json.dumps({
                                        "file": file_path,
                                        "line": line_number,
                                        "type": "git_blame",
                                        "commit": {
                                            "hash": commit_hash,
                                            "author": author.strip(),
                                            "date": date.strip(),
                                            "subject": subject.strip()
                                        },
                                        "message": "Found using git blame (patch linkage had no matches)"
                                    })
                os.chdir(original_cwd)
            except Exception as e:
                logger.warning(f"Git blame fallback failed: {e}")
                if 'original_cwd' in locals():
                    os.chdir(original_cwd)
        
        if not candidates:
            return json.dumps({"file": file_path, "line": line_number, "message": "No PR found that changed this file/line."})
        
        # Pick the most recent by merged_at (if available)
        def parse_date(doc):
            from datetime import datetime
            try:
                merged_at = getattr(doc, 'merged_at', None)
                if merged_at:
                    return datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                return None
            except Exception:
                return None
        candidates.sort(key=parse_date, reverse=True)
        top = candidates[0]
        # Try to get PR info
        pr_number = top.pr_number
        pr_url = f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/pull/{pr_number}"
        # Author info is not always available in patch linkage, so we return what we have
        result = json.dumps({
            "file": file_path,
            "line": line_number,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "merged_at": getattr(top, 'merged_at', None),
            "files_changed": top.files_changed,
            "diff_summary": top.diff_summary
        })
        return self._chunk_large_output(result)

    def who_implemented_this(self, feature_name: str, file_path: Optional[str] = None) -> str:
        """
        Find who initially implemented a feature/class/function.
        Uses git log to find the first commit that introduced the feature.
        """
        try:
            repo_root = self.code_rag.repo_info["repo_path"]
            original_cwd = os.getcwd()
            os.chdir(repo_root)
            
            # If file_path is provided, search in that specific file
            if file_path:
                # Validate file path
                abs_path = os.path.realpath(os.path.join(repo_root, file_path))
                if not abs_path.startswith(os.path.realpath(repo_root)):
                    return json.dumps({"error": "File outside repo"})
                
                # Use git log with -S to find when the feature was added
                log_cmd = [
                    "git", "log", "--reverse", "--pretty=format:%H|%an|%ae|%ad|%s",
                    "-S", feature_name, "--", file_path
                ]
            else:
                # Search across the entire repository
                log_cmd = [
                    "git", "log", "--reverse", "--pretty=format:%H|%an|%ae|%ad|%s",
                    "-S", feature_name
                ]
            
            result = subprocess.run(log_cmd, capture_output=True, text=True, cwd=repo_root)
            
            if result.returncode == 0 and result.stdout:
                # Get the first commit (oldest)
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split('|', 4)
                    if len(parts) >= 5:
                        commit_hash, author_name, author_email, date, subject = parts
                        
                        # Get more details about the commit
                        show_cmd = ["git", "show", "--stat", "--format=fuller", commit_hash]
                        show_result = subprocess.run(show_cmd, capture_output=True, text=True, cwd=repo_root)
                        
                        # Extract files changed from the commit
                        files_changed = []
                        if show_result.returncode == 0:
                            lines = show_result.stdout.split('\n')
                            in_stats = False
                            for line in lines:
                                if line.strip() == '':
                                    in_stats = True
                                    continue
                                if in_stats and '|' in line:
                                    file_name = line.split('|')[0].strip()
                                    if file_name:
                                        files_changed.append(file_name)
                        
                        # Try to find related issue/PR
                        related_issue = None
                        if self.issue_rag and self.issue_rag.indexer:
                            # Search commit message for issue references
                            import re
                            issue_refs = re.findall(r'#(\d+)', subject)
                            for issue_num in issue_refs:
                                issue_id = int(issue_num)
                                if issue_id in self.issue_rag.indexer.issue_docs:
                                    issue_doc = self.issue_rag.indexer.issue_docs[issue_id]
                                    related_issue = {
                                        "id": issue_id,
                                        "title": issue_doc.title,
                                        "url": f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/issues/{issue_id}"
                                    }
                                    break
                        
                        os.chdir(original_cwd)
                        
                        result_data = {
                            "feature": feature_name,
                            "file_path": file_path,
                            "initial_implementation": {
                                "commit": {
                                    "hash": commit_hash.strip(),
                                    "author_name": author_name.strip(),
                                    "author_email": author_email.strip(),
                                    "date": date.strip(),
                                    "subject": subject.strip()
                                },
                                "files_changed": files_changed[:10],  # Limit to 10 files
                                "total_files_changed": len(files_changed)
                            }
                        }
                        
                        if related_issue:
                            result_data["related_issue"] = related_issue
                        
                        # Check if there were multiple contributors
                        all_contributors_cmd = [
                            "git", "log", "--pretty=format:%an", "-S", feature_name
                        ]
                        if file_path:
                            all_contributors_cmd.extend(["--", file_path])
                        
                        contributors_result = subprocess.run(all_contributors_cmd, capture_output=True, text=True, cwd=repo_root)
                        if contributors_result.returncode == 0:
                            contributors = list(set(contributors_result.stdout.strip().split('\n')))
                            result_data["total_contributors"] = len(contributors)
                            result_data["all_contributors"] = contributors[:5]  # Top 5 contributors
                        
                        return json.dumps(result_data)
            
            os.chdir(original_cwd)
            return json.dumps({
                "feature": feature_name,
                "file_path": file_path,
                "message": f"No commits found that introduced '{feature_name}'"
            })
            
        except Exception as e:
            logger.error(f"Error in who_implemented_this: {e}")
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            return json.dumps({
                "feature": feature_name,
                "error": str(e),
                "message": "Error searching for initial implementation"
            })

    async def regression_detector(self, issue_query: str) -> str:
        """Detect if a new issue is a regression of a past one."""
        try:
            # First, search for similar issues using the retriever
            issues, patches = await self.issue_rag.retriever.find_related_issues(
                issue_query, 
                k=10,  # Get more issues for regression analysis
                include_patches=True
            )
            
            if not issues:
                return json.dumps({
                    "query": issue_query,
                    "regression_candidates": [],
                    "message": "No similar issues found to analyze for regressions."
                })

            # Analyze each issue for potential regression patterns
            regression_candidates = []
            for issue_result in issues:
                issue = issue_result.issue
                # Skip if issue is still open
                if issue.state != "closed":
                    continue
                    
                # Check if this issue has a patch URL
                if not issue.patch_url:
                    continue
                    
                # Find the PR number from patch URL
                import re
                m = re.search(r'/pull/(\d+)', issue.patch_url)
                pr_number = int(m.group(1)) if m else None
                if not pr_number:
                    continue
                    
                # Get the files changed in that PR
                diff_docs = self.issue_rag.indexer.diff_docs
                diff_doc = diff_docs.get(pr_number)
                files_changed = diff_doc.files_changed if diff_doc else []
                
                # Check if any of these files have been changed by a newer PR
                newer_prs = []
                for dd in diff_docs.values():
                    if any(f in dd.files_changed for f in files_changed):
                        # If merged_at is newer, it's a candidate
                        from datetime import datetime
                        try:
                            old_date = datetime.fromisoformat(diff_doc.merged_at.replace('Z', '+00:00')) if diff_doc and diff_doc.merged_at else None
                            new_date = datetime.fromisoformat(dd.merged_at.replace('Z', '+00:00')) if dd.merged_at else None
                            if old_date and new_date and new_date > old_date:
                                newer_prs.append({
                                    "pr_number": dd.pr_number,
                                    "merged_at": dd.merged_at,
                                    "files_changed": dd.files_changed,
                                    "diff_summary": dd.diff_summary[:200] + "..." if len(dd.diff_summary) > 200 else dd.diff_summary
                                })
                        except Exception:
                            continue
                            
                # Only include if there are newer PRs that touched the same files
                if newer_prs:
                    regression_candidates.append({
                        "closed_issue": {
                            "id": issue.id,
                            "title": issue.title,
                            "url": f"https://github.com/{self.issue_rag.repo_owner}/{self.issue_rag.repo_name}/issues/{issue.id}",
                            "closed_at": issue.closed_at,
                            "similarity": issue_result.similarity
                        },
                        "fix_pr": {
                            "pr_number": pr_number,
                            "patch_url": issue.patch_url,
                            "files_changed": files_changed
                        },
                        "newer_prs": newer_prs[:5]  # Limit to 5 most recent
                    })
                    
            # Sort by similarity of the original issue
            regression_candidates.sort(key=lambda x: x["closed_issue"]["similarity"], reverse=True)
            
            # Chunk the output if it's too large
            result = json.dumps({
                "query": issue_query, 
                "regression_candidates": regression_candidates[:5],  # Limit to top 5
                "total_candidates": len(regression_candidates)
            })
            return self._chunk_large_output(result)
        except Exception as e:
            logger.error(f"Error in regression detection: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                "query": issue_query,
                "error": str(e),
                "message": "Error analyzing potential regressions"
            })

    @lru_cache(maxsize=4096)
    def _blame_line(self, path, line):
        # ... blame logic ...
        pass

    # --- Agentic reasoning loop (to be implemented) ---
    async def agentic_answer(self, user_query: str) -> str:
        """
        Async agentic reasoning loop using LlamaIndex ReActAgent (mirrors AgenticCodebaseExplorer.query).
        Returns structured JSON with steps, final_answer, etc.
        """
        try:
            # Capture stdout/stderr during agent execution
            @contextlib.contextmanager
            def capture_output():
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                try:
                    sys.stdout = stdout_buffer
                    sys.stderr = stderr_buffer
                    yield stdout_buffer, stderr_buffer
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            with capture_output() as (stdout_buffer, stderr_buffer):
                response = await self.agent.achat(user_query)

            captured_output = stdout_buffer.getvalue()
            if not captured_output.strip():
                captured_output = stderr_buffer.getvalue()

            # Parse ReAct steps (reuse explorer's parser if available)
            if hasattr(self.explorer, '_parse_react_steps'):
                steps, final_answer = self.explorer._parse_react_steps(captured_output)
            else:
                steps = []
                final_answer = str(response)

            # Fallback if no steps/final_answer
            if len(steps) == 0 and not final_answer:
                response_str = str(response).strip()
                if response_str and len(response_str) > 20:
                    final_answer = response_str

            # Format as structured JSON
            return json.dumps({
                "steps": steps,
                "final_answer": final_answer,
                "status": "complete"
            })
        except Exception as e:
            logger.error(f"Error in agentic_answer: {e}")
            return json.dumps({
                "error": str(e),
                "status": "error"
            }) 