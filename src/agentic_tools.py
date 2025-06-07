"""
Agentic Tools Implementation using LlamaIndex
Provides directory exploration, codebase search, and file analysis capabilities
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Annotated, AsyncGenerator
from pathlib import Path
import logging
import io
import sys
import contextlib
import re

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai import OpenAI

from .config import settings

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr during execution"""
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

class AgenticCodebaseExplorer:
    """
    Agentic system for exploring and analyzing codebases using LlamaIndex tools
    """
    
    def __init__(self, session_id: str, repo_path: str, issue_rag_system: Optional['IssueAwareRAG'] = None):
        self.session_id = session_id
        self.repo_path = Path(repo_path)
        self.issue_rag = issue_rag_system
        
        # Initialize LLM based on settings
        self.llm = self._get_llm()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create memory for the agent
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        
        # Initialize the ReAct agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            max_iterations=settings.AGENTIC_MAX_ITERATIONS
        )
    
    def _get_llm(self) -> LLM:
        """Get LLM instance based on settings"""
        if settings.llm_provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API key is required")
            return OpenRouter(
                api_key=settings.openrouter_api_key,
                model=settings.default_model,
                max_tokens=4096,
                temperature=0.7
            )
        else:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAI(
                api_key=settings.openai_api_key,
                model=settings.default_model,
                max_tokens=4096,
                temperature=0.7
            )
    
    def _create_tools(self) -> List[FunctionTool]:
        """Create the tools available to the agent"""
        tools = [
            FunctionTool.from_defaults(
                fn=self.explore_directory,
                name="explore_directory",
                description="Explore the contents of a directory in the repository to understand its structure and files"
            ),
            FunctionTool.from_defaults(
                fn=self.search_codebase,
                name="search_codebase", 
                description="Search through the codebase for specific patterns, functions, classes, or concepts"
            ),
            FunctionTool.from_defaults(
                fn=self.read_file,
                name="read_file",
                description="Read the complete contents of a specific file in the repository"
            ),
            FunctionTool.from_defaults(
                fn=self.analyze_file_structure,
                name="analyze_file_structure",
                description="Analyze the structure and components of a file or directory"
            ),
            FunctionTool.from_defaults(
                fn=self.find_related_files,
                name="find_related_files",
                description="Find files related to a specific file based on imports, references, and patterns"
            ),
            FunctionTool.from_defaults(
                fn=self.semantic_content_search,
                name="semantic_content_search",
                description="Search for content semantically across all files in the repository"
            ),
            FunctionTool.from_defaults(
                fn=self.generate_code_example,
                name="generate_code_example",
                description="Generate code examples based on repository patterns and context"
            ),
            FunctionTool.from_defaults(
                fn=self.analyze_github_issue,
                name="analyze_github_issue",
                description="Analyze a GitHub issue to understand the problem and provide solution approaches"
            ),
            FunctionTool.from_defaults(
                fn=self.find_issue_related_files,
                name="find_issue_related_files",
                description="Find files in the codebase that are related to a specific issue or problem description"
            ),
            FunctionTool.from_defaults(
                fn=self.related_issues,
                name="related_issues",
                description="Find similar past GitHub issues in the current repository that might provide context or solutions"
            ),
            FunctionTool.from_defaults(
                fn=self.get_pr_for_issue,
                name="get_pr_for_issue",
                description="Find the pull request associated with a given issue number"
            ),
            FunctionTool.from_defaults(
                fn=self.get_pr_diff,
                name="get_pr_diff",
                description="Retrieve the diff for a given pull request number"
            ),
            FunctionTool.from_defaults(
                fn=self.get_files_changed_in_pr,
                name="get_files_changed_in_pr",
                description="Lists all files that were modified, added, or deleted in a given pull request."
            ),
            FunctionTool.from_defaults(
                fn=self.get_pr_summary,
                name="get_pr_summary",
                description="Provides a concise summary of the changes made in a specific pull request, based on its diff."
            ),
            FunctionTool.from_defaults(
                fn=self.find_issues_related_to_file,
                name="find_issues_related_to_file",
                description="Finds issues whose resolution involved changes to the specified file path."
            ),
            FunctionTool.from_defaults(
                fn=self.get_issue_resolution_summary,
                name="get_issue_resolution_summary",
                description="Summarizes how a specific issue was resolved, including linked PRs and a summary of changes."
            ),
            FunctionTool.from_defaults(
                fn=self.check_issue_status_and_linked_pr,
                name="check_issue_status_and_linked_pr",
                description="Checks the current status (open/closed) of a GitHub issue and lists any directly linked Pull Requests."
            ),
            FunctionTool.from_defaults(
                fn=self.write_complete_code,
                name="write_complete_code",
                description="Write complete, untruncated code files based on specifications and reference files"
            )
        ]
        return tools
    
    def explore_directory(
        self, 
        directory_path: Annotated[str, "Path to the directory to explore, relative to repository root"]
    ) -> str:
        """Explore directory contents with metadata"""
        try:
            full_path = self.repo_path / directory_path if directory_path else self.repo_path
            
            if not full_path.exists() or not full_path.is_dir():
                return f"Directory {directory_path} does not exist or is not a directory"
            
            items = []
            
            # Get all items in directory
            for item in sorted(full_path.iterdir()):
                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime,
                        "path": str(item.relative_to(self.repo_path))
                    }
                    
                    # Add extra info for files
                    if item.is_file():
                        item_info["extension"] = item.suffix
                        # Get file preview for small files
                        if stat.st_size < 1000:
                            try:
                                with open(item, 'r', encoding='utf-8') as f:
                                    preview = f.read(200)
                                    item_info["preview"] = preview + ("..." if len(preview) == 200 else "")
                            except Exception:
                                item_info["preview"] = "Binary or unreadable file"
                    
                    items.append(item_info)
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    continue
            
            # Create summary
            total_files = len([i for i in items if i["type"] == "file"])
            total_dirs = len([i for i in items if i["type"] == "directory"])
            
            result = {
                "directory": directory_path or "root",
                "summary": f"{total_files} files, {total_dirs} directories",
                "items": items[:50]  # Limit to avoid overwhelming context
            }
            
            if len(items) > 50:
                result["note"] = f"Showing first 50 items out of {len(items)} total"
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error exploring directory {directory_path}: {e}")
            return f"Error exploring directory: {str(e)}"
    
    def search_codebase(
        self, 
        query: Annotated[str, "Search query - can be code patterns, function names, or concepts"],
        file_types: Annotated[Optional[List[str]], "File extensions to search (e.g., ['.py', '.js']). None for all files"] = None
    ) -> str:
        """Search through codebase files"""
        try:
            results = []
            search_count = 0
            max_results = 30
            
            # Define default file types if none specified
            if file_types is None:
                file_types = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt']
            
            for file_path in self.repo_path.rglob("*"):
                if search_count >= max_results:
                    break
                    
                if not file_path.is_file():
                    continue
                    
                if file_types and file_path.suffix not in file_types:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Simple text search - could be enhanced with regex
                    if query.lower() in content.lower():
                        # Find context around matches
                        lines = content.split('\n')
                        matches = []
                        
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                # Get context (3 lines before and after)
                                start = max(0, i - 3)
                                end = min(len(lines), i + 4)
                                context = '\n'.join(lines[start:end])
                                
                                matches.append({
                                    "line_number": i + 1,
                                    "line": line.strip(),
                                    "context": context
                                })
                                
                                if len(matches) >= 3:  # Limit matches per file
                                    break
                        
                        if matches:
                            results.append({
                                "file": str(file_path.relative_to(self.repo_path)),
                                "matches": matches
                            })
                            search_count += 1
                            
                except Exception as e:
                    continue
            
            return json.dumps({
                "query": query,
                "total_files_searched": search_count,
                "results": results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error searching codebase: {e}")
            return f"Error searching codebase: {str(e)}"
    
    def read_file(
        self, 
        file_path: Annotated[str, "Path to the file to read, relative to repository root"]
    ) -> str:
        """Read complete file contents with dynamic chunking"""
        try:
            full_path = self.repo_path / file_path
            
            if not full_path.exists() or not full_path.is_file():
                return f"File {file_path} does not exist or is not a file"
            
            # Get file stats
            stat = full_path.stat()
            
            # If file is too large, use chunked reading
            if stat.st_size > settings.MAX_AGENTIC_FILE_SIZE_BYTES and settings.ENABLE_DYNAMIC_CONTENT:
                return self._read_large_file(full_path, file_path)
            
            # Regular file reading for smaller files
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return json.dumps({
                "file": file_path,
                "size": stat.st_size,
                "lines": len(content.split('\n')),
                "content": content
            }, indent=2)
            
        except UnicodeDecodeError:
            return f"File {file_path} appears to be binary and cannot be read as text"
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

    def _read_large_file(self, file_path: Path, relative_path: str) -> str:
        """Read large files in chunks with smart content handling"""
        try:
            chunks = []
            total_lines = 0
            current_chunk = []
            current_chunk_size = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total_lines += 1
                    line_size = len(line.encode('utf-8'))
                    
                    # If adding this line would exceed chunk size, save current chunk
                    if current_chunk_size + line_size > settings.CONTENT_CHUNK_SIZE:
                        chunks.append(''.join(current_chunk))
                        current_chunk = []
                        current_chunk_size = 0
                        
                        # If we've reached max chunks, stop
                        if len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST:
                            break
                    
                    current_chunk.append(line)
                    current_chunk_size += line_size
            
            # Add final chunk if any
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            # If file was truncated, add note
            truncated_note = ""
            if len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST:
                truncated_note = f"\n\nNote: File was truncated after {settings.MAX_CHUNKS_PER_REQUEST} chunks. Total lines: {total_lines}"
            
            return json.dumps({
                "file": relative_path,
                "size": file_path.stat().st_size,
                "total_lines": total_lines,
                "chunks": len(chunks),
                "content": chunks,
                "truncated": len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST,
                "truncated_note": truncated_note
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error reading large file {file_path}: {e}")
            return f"Error reading large file: {str(e)}"

    async def stream_large_file(self, file_path: str) -> AsyncGenerator[str, None]:
        """Stream large file content in chunks"""
        try:
            full_path = self.repo_path / file_path
            
            if not full_path.exists() or not full_path.is_file():
                yield json.dumps({"error": f"File {file_path} does not exist or is not a file"})
                return
            
            # Get file stats
            stat = full_path.stat()
            
            # Send initial metadata
            yield json.dumps({
                "type": "metadata",
                "file": file_path,
                "size": stat.st_size,
                "chunk_size": settings.STREAM_CHUNK_SIZE
            })
            
            # Stream file content
            with open(full_path, 'r', encoding='utf-8') as f:
                buffer = []
                current_size = 0
                
                for line in f:
                    line_size = len(line.encode('utf-8'))
                    
                    # If adding this line would exceed chunk size, send current buffer
                    if current_size + line_size > settings.STREAM_CHUNK_SIZE:
                        yield json.dumps({
                            "type": "chunk",
                            "content": ''.join(buffer)
                        })
                        buffer = []
                        current_size = 0
                    
                    buffer.append(line)
                    current_size += line_size
                
                # Send final buffer if any
                if buffer:
                    yield json.dumps({
                        "type": "chunk",
                        "content": ''.join(buffer)
                    })
                
                # Send completion message
                yield json.dumps({
                    "type": "complete",
                    "message": "File streaming completed"
                })
                
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {e}")
            yield json.dumps({
                "type": "error",
                "error": str(e)
            })
    
    def analyze_file_structure(
        self, 
        target_path: Annotated[str, "Path to analyze - can be file or directory"] = ""
    ) -> str:
        """Analyze file structure and relationships"""
        try:
            full_path = self.repo_path / target_path if target_path else self.repo_path
            
            if not full_path.exists():
                return f"Path {target_path} does not exist"
            
            analysis = {
                "path": target_path or "root",
                "type": "directory" if full_path.is_dir() else "file"
            }
            
            if full_path.is_file():
                # Analyze single file
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                analysis.update({
                    "size": len(content),
                    "lines": len(content.split('\n')),
                    "extension": full_path.suffix,
                    "functions": self._extract_functions(content, full_path.suffix),
                    "classes": self._extract_classes(content, full_path.suffix)
                })
            else:
                # Analyze directory structure
                files_by_type = {}
                total_size = 0
                
                for file_path in full_path.rglob("*"):
                    if file_path.is_file():
                        ext = file_path.suffix or "no_extension"
                        if ext not in files_by_type:
                            files_by_type[ext] = {"count": 0, "total_size": 0}
                        
                        try:
                            size = file_path.stat().st_size
                            files_by_type[ext]["count"] += 1
                            files_by_type[ext]["total_size"] += size
                            total_size += size
                        except Exception:
                            continue
                
                analysis.update({
                    "total_size": total_size,
                    "files_by_type": files_by_type,
                    "structure_summary": f"Contains {len(files_by_type)} different file types"
                })
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error analyzing file structure: {e}")
            return f"Error analyzing structure: {str(e)}"
    
    def find_related_files(
        self, 
        file_path: Annotated[str, "The file path to find related files for"]
    ) -> str:
        """Find files related to a given file"""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return f"File does not exist: {file_path}"
            
            related_files = []
            file_stem = Path(file_path).stem
            file_dir = Path(file_path).parent
            
            # Search for files with similar names
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    file_obj = Path(file)
                    rel_path = Path(root).relative_to(self.repo_path) / file
                    
                    # Skip the original file
                    if str(rel_path) == file_path:
                        continue
                    
                    # Check for related patterns
                    if (file_stem in file_obj.stem or 
                        file_obj.stem in file_stem or
                        str(file_dir) in str(rel_path.parent)):
                        related_files.append(str(rel_path))
            
            # Also check for imports/references by reading the file
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for import statements and file references
                import_patterns = [
                    r'from\s+[\w\.]+\s+import',
                    r'import\s+[\w\.]+',
                    r'require\(["\']([^"\']+)["\']',
                    r'#include\s*[<"]([^>"]+)[>"]'
                ]
                
                for pattern in import_patterns:
                    import re
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Convert module paths to potential file paths
                        potential_paths = [
                            f"{match.replace('.', '/')}.py",
                            f"{match.replace('.', '/')}.js",
                            f"{match.replace('.', '/')}.ts",
                            f"{match}.py",
                            f"{match}.js",
                            f"{match}.ts"
                        ]
                        
                        for pot_path in potential_paths:
                            if (self.repo_path / pot_path).exists():
                                related_files.append(pot_path)
                                
            except:
                pass  # Ignore file reading errors
            
            return json.dumps({
                "original_file": file_path,
                "related_files": list(set(related_files))[:20],  # Limit to 20 results
                "total_found": len(set(related_files))
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error finding related files: {e}")
            return f"Error finding related files: {str(e)}"
    
    def _extract_functions(self, content: str, file_extension: str) -> List[str]:
        """Extract function definitions from file content"""
        functions = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if file_extension == '.py':
                if line.startswith('def ') and '(' in line:
                    functions.append(line)
            elif file_extension in ['.js', '.jsx', '.ts', '.tsx']:
                if ('function ' in line or '=>' in line) and ('(' in line):
                    functions.append(line)
        
        return functions[:15]  # Limit to avoid overwhelming context
    
    def _extract_classes(self, content: str, file_extension: str) -> List[str]:
        """Extract class definitions from file content"""
        classes = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if file_extension == '.py':
                if line.startswith('class ') and ':' in line:
                    classes.append(line)
            elif file_extension in ['.js', '.jsx', '.ts', '.tsx']:
                if line.startswith('class ') and ('{' in line or line.endswith('{')):
                    classes.append(line)
        
        return classes[:10]  # Limit to avoid overwhelming context
    
    def semantic_content_search(
        self,
        query: Annotated[str, "Search query for content across files"]
    ) -> str:
        """Search for content semantically across files"""
        try:
            results = []
            query_terms = query.lower().split()
            
            # Search through files for semantic matches
            for file_path in self.repo_path.rglob("*"):
                if not file_path.is_file():
                    continue
                
                # Skip binary files and large files
                try:
                    stat = file_path.stat()
                    if stat.st_size > settings.MAX_AGENTIC_FILE_SIZE_BYTES:  # Use configurable limit
                        continue
                    
                    # Only search text files
                    if file_path.suffix not in ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt', '.config', '.sh']:
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Calculate relevance score
                    content_lower = content.lower()
                    score = 0
                    matches = []
                    
                    for term in query_terms:
                        term_count = content_lower.count(term)
                        if term_count > 0:
                            score += term_count
                            matches.append(f"{term}: {term_count}")
                    
                    # Add bonus for exact phrase matches
                    if query.lower() in content_lower:
                        score += 10
                        matches.append(f"exact phrase: {content_lower.count(query.lower())}")
                    
                    rel_path = file_path.relative_to(self.repo_path)
                    
                    if score > 0:
                        # Get context around matches
                        context_snippets = []
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if any(term in line.lower() for term in query_terms):
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = '\n'.join(lines[start:end])
                                context_snippets.append(snippet)
                                if len(context_snippets) >= 3:  # Limit snippets
                                    break
                        
                        results.append({
                            "file": str(rel_path),
                            "score": score,
                            "matches": matches,
                            "context": context_snippets[:2],  # Limit context
                            "size": len(content)
                        })
                
                except:
                    continue  # Skip files that can't be read
        
            # Sort by score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:15]  # Limit to top 15 results
            
            return json.dumps({
                "query": query,
                "total_results": len(results),
                "results": results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in semantic content search: {e}")
            return f"Error in semantic content search: {str(e)}"

    def generate_code_example(
        self,
        description: Annotated[str, "Description of what kind of code example to generate"],
        context_files: Annotated[Optional[List[str]], "List of relevant files to base the example on"] = None
    ) -> str:
        """Generate practical, runnable code examples based on codebase analysis"""
        try:
            logger.info(f"[DEBUG] Generating code for: {description}")
            
            if not context_files:
                return json.dumps({
                    "message": "No context files provided",
                    "suggestion": "Use @filename to specify files to analyze",
                    "example": f"Try: 'Using @agents.py, show me how to build: {description}'"
                }, indent=2)
            
            # Detect the primary language from context files and repository
            primary_language = self._detect_primary_language_from_context(context_files)
            
            # Analyze the provided context files
            analysis = self._analyze_repository_context(context_files)
            analysis["detected_language"] = primary_language
            
            # Generate language-appropriate guidance and code
            return self._create_language_appropriate_example(description, analysis, context_files, primary_language)
            
        except Exception as e:
            logger.error(f"Error generating code example: {e}")
            return json.dumps({
                "error": f"Failed to generate code: {str(e)}",
                "description": description,
                "suggestion": "Try providing specific files to analyze"
            }, indent=2)
    
    def _detect_primary_language_from_context(self, context_files: List[str]) -> str:
        """Detect primary language from context files and repository structure"""
        
        # Language detection based on file extensions
        language_map = {
            '.py': 'python',
                '.js': 'javascript',
                '.jsx': 'javascript', 
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.java': 'java',
                '.kt': 'kotlin',
                '.scala': 'scala',
                '.go': 'go',
                '.rs': 'rust',
                '.rb': 'ruby',
                '.php': 'php',
                '.cs': 'csharp',
                '.cpp': 'cpp',
                '.cc': 'cpp',
                '.cxx': 'cpp',
            '.c': 'c',
                '.swift': 'swift',
                '.m': 'objective-c',
                '.ex': 'elixir',
                '.exs': 'elixir',
            '.ml': 'ocaml',
            '.hs': 'haskell',
            '.clj': 'clojure',
                '.dart': 'dart',
                '.lua': 'lua',
            '.r': 'r',
            '.jl': 'julia'
        }
        
        # Count languages from context files
        language_counts = {}
        for file_path in context_files:
            ext = Path(file_path).suffix.lower()
            if ext in language_map:
                lang = language_map[ext]
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # If we have a clear winner from context files, use it
        if language_counts:
            primary_lang = max(language_counts, key=language_counts.get)
            logger.info(f"[DEBUG] Detected primary language from context: {primary_lang}")
            return primary_lang
        
        # Fallback: check broader repository
        try:
            repo_language_info = self._detect_repository_languages()
            primary_lang = repo_language_info.get("primary_language", "generic")
            logger.info(f"[DEBUG] Using repository-wide language detection: {primary_lang}")
            return primary_lang
        except:
            logger.info(f"[DEBUG] Defaulting to generic language")
            return "generic"
    
    def _create_language_appropriate_example(self, description: str, analysis: Dict, context_files: List[str], primary_language: str) -> str:
        """Create code example appropriate for the detected language"""
        
        # Find most relevant class regardless of language
        relevant_class = self._find_most_relevant_class(description, analysis)
        
        # Generate language-specific explanation
        explanation = self._generate_language_explanation(description, analysis, relevant_class, context_files, primary_language)
        
        # Generate language-appropriate code example
        code_example = self._generate_language_specific_example(description, analysis, relevant_class, primary_language)
        
        return json.dumps({
            "detected_language": primary_language,
            "analysis": f"Analyzed {len(context_files)} files and found {len(analysis['classes'])} classes",
            "available_patterns": {
                "classes": [cls["name"] for cls in analysis["classes"]],
                "base_classes": analysis["base_classes"],
                "capabilities": analysis["capabilities"][:10]
            },
            "recommendation": explanation,
            "code_example": code_example,
            "implementation_guide": self._get_language_implementation_steps(relevant_class, analysis, primary_language)
        }, indent=2)
    
    def _find_most_relevant_class(self, description: str, analysis: Dict) -> Dict:
        """Find most relevant class regardless of language"""
        user_intent = description.lower()
        
        # Find class that matches description keywords
        for cls in analysis["classes"]:
            if any(keyword in cls["name"].lower() or keyword in cls["purpose"].lower() 
                   for keyword in user_intent.split()):
                return cls
        
        # Fallback to first class if available
        return analysis["classes"][0] if analysis["classes"] else None
    
    def _generate_language_explanation(self, description: str, analysis: Dict, relevant_class: Dict, context_files: List[str], language: str) -> str:
        """Generate explanation based on detected language"""
        
        explanation = f"## How to build: {description}\n\n"
        explanation += f"**Detected language:** {language.title()}\n"
        explanation += f"**Based on analysis of:** {', '.join(context_files)}\n\n"
        
        if relevant_class:
            explanation += f"**Recommended approach:** Use `{relevant_class['name']}` as a foundation\n"
            explanation += f"- **Purpose:** {relevant_class['purpose']}\n"
            explanation += f"- **Base class:** {relevant_class['base']}\n"
            explanation += f"- **File:** {relevant_class['file']}\n\n"
        
        if analysis["base_classes"]:
            explanation += f"**Available base classes:** {', '.join(analysis['base_classes'][:3])}\n\n"
        
        explanation += f"**Repository capabilities include:** "
        explanation += ", ".join(analysis["capabilities"][:8])
        
        return explanation
    
    def _generate_language_specific_example(self, description: str, analysis: Dict, relevant_class: Dict, language: str) -> str:
        """Generate code example in the appropriate language"""
        
        # Extract relevant patterns
        imports = analysis["imports"][:3]
        base_class = relevant_class["base"] if relevant_class and relevant_class["base"] else "BaseClass"
        
        # Generate class name
        words = description.split()
        class_name = "My" + "".join(word.capitalize() for word in words[:2] if word.isalpha())
        
        # Language-specific code generation
        if language == "javascript":
            return self._generate_javascript_example(description, class_name, base_class, relevant_class)
        elif language == "typescript":
            return self._generate_typescript_example(description, class_name, base_class, relevant_class)
        elif language == "java":
            return self._generate_java_example(description, class_name, base_class, relevant_class)
        elif language == "go":
            return self._generate_go_example(description, class_name, relevant_class)
        elif language == "rust":
            return self._generate_rust_example(description, class_name, relevant_class)
        elif language == "csharp":
            return self._generate_csharp_example(description, class_name, base_class, relevant_class)
        elif language == "ruby":
            return self._generate_ruby_example(description, class_name, base_class, relevant_class)
        elif language == "php":
            return self._generate_php_example(description, class_name, base_class, relevant_class)
        elif language == "python":
            return self._generate_python_example(description, class_name, base_class, relevant_class, imports)
        else:
            return self._generate_generic_example(description, class_name, base_class, relevant_class, language)
    
    def _generate_javascript_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate JavaScript example"""
        return f"""// {description} - JavaScript implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

class {class_name} extends {base_class} {{
    constructor() {{
        super();
        // Initialize your {description.lower()}
    }}
    
    async processRequest(inputData) {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        const result = await this.processData(inputData);
        return result;
    }}
    
    async processData(data) {{
        // Your implementation here
        return `Processed: ${{data}}`;
    }}
}}

// Usage example
const processor = new {class_name}();
processor.processRequest("your input here")
    .then(result => console.log(result))
    .catch(error => console.error(error));

// Export for use in other modules
module.exports = {class_name};"""

    def _generate_typescript_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate TypeScript example"""
        return f"""// {description} - TypeScript implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

interface ProcessorInput {{
    data: string;
    options?: Record<string, any>;
}}

interface ProcessorResult {{
    success: boolean;
    result: string;
    metadata?: Record<string, any>;
}}

class {class_name} extends {base_class} {{
    constructor() {{
        super();
        // Initialize your {description.lower()}
    }}
    
    async processRequest(input: ProcessorInput): Promise<ProcessorResult> {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        const result = await this.processData(input.data);
        return {{
            success: true,
            result: result
        }};
    }}
    
    private async processData(data: string): Promise<string> {{
        // Your implementation here
        return `Processed: ${{data}}`;
    }}
}}

// Usage example
const processor = new {class_name}();
processor.processRequest({{ data: "your input here" }})
    .then(result => console.log(result))
    .catch(error => console.error(error));

export default {class_name};"""

    def _generate_java_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate Java example"""
        return f"""// {description} - Java implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

import java.util.concurrent.CompletableFuture;
import java.util.HashMap;
import java.util.Map;

public class {class_name} extends {base_class} {{
    
    public {class_name}() {{
        super();
        // Initialize your {description.lower()}
    }}
    
    public CompletableFuture<String> processRequest(String inputData) {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        return CompletableFuture.supplyAsync(() -> {{
            try {{
                return processData(inputData);
            }} catch (Exception e) {{
                throw new RuntimeException("Processing failed", e);
            }}
        }});
    }}
    
    private String processData(String data) {{
        // Your implementation here
        return "Processed: " + data;
    }}
    
    // Main method for testing
    public static void main(String[] args) {{
        {class_name} processor = new {class_name}();
        processor.processRequest("your input here")
                .thenAccept(result -> System.out.println(result))
                .exceptionally(error -> {{
                    System.err.println("Error: " + error.getMessage());
                    return null;
                }});
    }}
}}"""

    def _generate_go_example(self, description: str, class_name: str, relevant_class: Dict) -> str:
        """Generate Go example"""
        struct_name = class_name
        return f"""// {description} - Go implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

package main

import (
    "fmt"
    "log"
)

// {struct_name} implements {description.lower()}
type {struct_name} struct {{
    // Add your fields here
}}

// New{struct_name} creates a new instance
func New{struct_name}() *{struct_name} {{
    return &{struct_name}{{
        // Initialize your {description.lower()}
    }}
}}

// ProcessRequest implements the core logic
func (p *{struct_name}) ProcessRequest(inputData string) (string, error) {{
    // Implement your core logic here
    // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
    
    result, err := p.processData(inputData)
    if err != nil {{
        return "", err
    }}
    
    return result, nil
}}

// processData handles the actual processing
func (p *{struct_name}) processData(data string) (string, error) {{
    // Your implementation here
    return fmt.Sprintf("Processed: %s", data), nil
}}

// Example usage
func main() {{
    processor := New{struct_name}()
    result, err := processor.ProcessRequest("your input here")
    if err != nil {{
        log.Fatalf("Error: %v", err)
    }}
    
    fmt.Println(result)
}}"""

    def _generate_rust_example(self, description: str, class_name: str, relevant_class: Dict) -> str:
        """Generate Rust example"""
        return f"""// {description} - Rust implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct {class_name} {{
    // Add your fields here
}}

#[derive(Debug)]
pub enum ProcessingError {{
    InvalidInput(String),
    ProcessingFailed(String),
}}

impl fmt::Display for ProcessingError {{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {{
        match self {{
            ProcessingError::InvalidInput(msg) => write!(f, "Invalid input: {{}}", msg),
            ProcessingError::ProcessingFailed(msg) => write!(f, "Processing failed: {{}}", msg),
        }}
    }}
}}

impl Error for ProcessingError {{}}

impl {class_name} {{
    /// Create a new instance
    pub fn new() -> Self {{
        Self {{
            // Initialize your {description.lower()}
        }}
    }}
    
    /// Process the request
    pub async fn process_request(&self, input_data: &str) -> Result<String, ProcessingError> {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        let result = self.process_data(input_data).await?;
        Ok(result)
    }}
    
    /// Process the actual data
    async fn process_data(&self, data: &str) -> Result<String, ProcessingError> {{
        // Your implementation here
        Ok(format!("Processed: {{}}", data))
    }}
}}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {{
    let processor = {class_name}::new();
    let result = processor.process_request("your input here").await?;
    println!("{{}}", result);
    Ok(())
}}"""

    def _generate_csharp_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate C# example"""
        return f"""// {description} - C# implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

using System;
using System.Threading.Tasks;

namespace YourNamespace
{{
    public class {class_name} : {base_class}
    {{
        public {class_name}()
        {{
            // Initialize your {description.lower()}
        }}
        
        public async Task<string> ProcessRequestAsync(string inputData)
        {{
            // Implement your core logic here
            // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
            
            var result = await ProcessDataAsync(inputData);
            return result;
        }}
        
        private async Task<string> ProcessDataAsync(string data)
        {{
            // Your implementation here
            await Task.Delay(100); // Simulate async work
            return $"Processed: {{data}}";
        }}
    }}
    
    // Usage example
    class Program
    {{
        static async Task Main(string[] args)
        {{
            var processor = new {class_name}();
            try
            {{
                var result = await processor.ProcessRequestAsync("your input here");
                Console.WriteLine(result);
            }}
            catch (Exception ex)
            {{
                Console.WriteLine($"Error: {{ex.Message}}");
            }}
        }}
    }}
}}"""

    def _generate_ruby_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate Ruby example"""
        return f"""# {description} - Ruby implementation
# Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

class {class_name} < {base_class}
  def initialize
    super
    # Initialize your {description.lower()}
  end
  
  def process_request(input_data)
    # Implement your core logic here
    # Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
    
    result = process_data(input_data)
    result
  end
  
  private
  
  def process_data(data)
    # Your implementation here
    "Processed: #{{data}}"
  end
end

# Usage example
if __FILE__ == $0
  processor = {class_name}.new
  result = processor.process_request("your input here")
  puts result
end"""

    def _generate_php_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict) -> str:
        """Generate PHP example"""
        return f"""<?php
// {description} - PHP implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

class {class_name} extends {base_class}
{{
    public function __construct()
    {{
        parent::__construct();
        // Initialize your {description.lower()}
    }}
    
    public function processRequest($inputData)
    {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        $result = $this->processData($inputData);
        return $result;
    }}
    
    private function processData($data)
    {{
        // Your implementation here
        return "Processed: " . $data;
    }}
}}

// Usage example
$processor = new {class_name}();
$result = $processor->processRequest("your input here");
echo $result . PHP_EOL;
?>"""

    def _generate_python_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict, imports: List[str]) -> str:
        """Generate Python example"""
        return f"""# {description} - Python implementation
# Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

{chr(10).join(imports[:3])}

class {class_name}({base_class}):
    \"\"\"
    Implementation of: {description}
    Based on patterns from: {relevant_class['name'] if relevant_class else 'repository analysis'}
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize based on existing patterns\"\"\"
        super().__init__()
        # Initialize your specific requirements here
        
    async def process_request(self, input_data):
        \"\"\"Main processing function\"\"\"
        # Implement your core logic here
        # Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        result = await self.process_data(input_data)
        return result
        
    async def process_data(self, data):
        \"\"\"Process data using repository patterns\"\"\"
        # Your implementation here
        return f"Processed: {{data}}"

# Usage example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        processor = {class_name}()
        result = await processor.process_request("your input here")
        print(result)
    
    asyncio.run(main())"""

    def _generate_generic_example(self, description: str, class_name: str, base_class: str, relevant_class: Dict, language: str) -> str:
        """Generate generic pseudocode example"""
        return f"""// {description} - {language.title()} implementation
// Based on: {relevant_class['name'] if relevant_class else 'repository analysis'}

// Note: This is pseudocode - adapt to your {language} syntax

class {class_name} extends {base_class} {{
    
    constructor() {{
        // Initialize your {description.lower()}
    }}
    
    function processRequest(inputData) {{
        // Implement your core logic here
        // Follow patterns from {relevant_class['name'] if relevant_class else 'analyzed classes'}
        
        result = this.processData(inputData)
        return result
    }}
    
    function processData(data) {{
        // Your implementation here
        return "Processed: " + data
    }}
}}

// Usage example
processor = new {class_name}()
result = processor.processRequest("your input here")
print(result)

// Adapt this pseudocode to proper {language} syntax and conventions"""
    
    def _get_language_implementation_steps(self, relevant_class: Dict, analysis: Dict, language: str) -> List[str]:
        """Get implementation steps specific to the language"""
        base_steps = [
            f"1. Study the existing {language} class structure and patterns",
            f"2. Follow {language} conventions and best practices",
            "3. Implement required methods based on the framework",
            "4. Add your specific functionality",
            "5. Test with simple examples first"
        ]
        
        # Language-specific guidance
        if language == "javascript":
            base_steps.append("6. Consider using async/await for asynchronous operations")
            base_steps.append("7. Add proper error handling with try/catch")
        elif language == "typescript":
            base_steps.append("6. Define proper interfaces for type safety")
            base_steps.append("7. Use generics where appropriate")
        elif language == "java":
            base_steps.append("6. Implement proper exception handling")
            base_steps.append("7. Consider using dependency injection")
        elif language == "go":
            base_steps.append("6. Handle errors explicitly with error return values")
            base_steps.append("7. Use goroutines for concurrent operations if needed")
        elif language == "rust":
            base_steps.append("6. Use Result<T, E> for error handling")
            base_steps.append("7. Leverage ownership and borrowing for memory safety")
        elif language == "python":
            base_steps.append("6. Use type hints for better code documentation")
            base_steps.append("7. Consider async/await for I/O operations")
        
        if analysis["capabilities"]:
            base_steps.append(f"8. Leverage existing capabilities: {', '.join(analysis['capabilities'][:3])}")
            
        return base_steps

    def _parse_react_steps(self, raw_response: str):
        """Parse ReAct steps from raw agent response"""
        logger.info(f"[DEBUG] Parsing raw response length: {len(raw_response)}")
        logger.info(f"[DEBUG] Raw response first 500 chars: {raw_response[:500]}")
        
        steps = []
        current_type = None
        current_content = []
        
        for line in raw_response.split('\n'):
            line_lower = line.strip().lower()
            
            # Check for ReAct control tokens
            if line_lower.startswith('thought:'):
                if current_type and current_content:
                    steps.append({
                        "type": current_type, 
                        "content": "\n".join(current_content).strip(),
                        "step": len(steps)
                    })
                current_type = "thought"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                logger.info(f"[DEBUG] Found Thought step")
                
            elif line_lower.startswith('action:'):
                if current_type and current_content:
                    steps.append({
                        "type": current_type, 
                        "content": "\n".join(current_content).strip(),
                        "step": len(steps)
                    })
                current_type = "action"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                logger.info(f"[DEBUG] Found Action step")
                
            elif line_lower.startswith('observation:'):
                if current_type and current_content:
                    steps.append({
                        "type": current_type, 
                        "content": "\n".join(current_content).strip(),
                        "step": len(steps)
                    })
                current_type = "observation"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                logger.info(f"[DEBUG] Found Observation step")
                
            elif line_lower.startswith('answer:'):
                if current_type and current_content:
                    steps.append({
                        "type": current_type, 
                        "content": "\n".join(current_content).strip(),
                        "step": len(steps)
                    })
                current_type = "answer"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                logger.info(f"[DEBUG] Found Answer step")
                
            else:
                # Continue accumulating content for current step
                if current_content is not None:
                    current_content.append(line)
        
        # Add final step if present
        if current_type and current_content:
            steps.append({
                "type": current_type, 
                "content": "\n".join(current_content).strip(),
                "step": len(steps)
            })
        
        # Extract final answer from the last answer step
        final_answer = None
        for step in reversed(steps):
            if step["type"] == "answer":
                final_answer = step["content"]
                break

        # If no ReAct steps found but we have meaningful content, treat entire response as final answer
        if len(steps) == 0 and raw_response.strip() and len(raw_response.strip()) > 50:
            # Check if it looks like a meaningful response (not just technical logs)
            if not any(pattern in raw_response.lower() for pattern in ['http request:', 'info:', 'debug:', 'error:', 'running step']):
                logger.info(f"[DEBUG] No ReAct steps found, but treating full response as final answer")
                final_answer = raw_response.strip()
        
        logger.info(f"[DEBUG] Parsed {len(steps)} steps, final_answer: {final_answer[:100] if final_answer else 'None'}")
        return steps, final_answer

    def _format_agentic_response(self, steps, final_answer=None, partial=False, suggestions=None):
        """Format agentic output as structured JSON for the frontend UI."""
        return json.dumps({
            "type": "final",
            "steps": steps,
            "final_answer": final_answer,
            "partial": partial,
            "suggestions": suggestions or []
        })

    async def query(self, user_message: str) -> str:
        """Main query method that uses the agent to respond (now returns structured output)."""
        try:
            logger.info(f"Starting agentic analysis: {user_message[:100]}...")
            
            # Capture the verbose output during agent execution
            with capture_output() as (stdout_buffer, stderr_buffer):
                response = await self.agent.achat(user_message)
            
            logger.info(f"Agentic analysis completed successfully")
            
            # Get the captured verbose output which contains the full ReAct trace
            captured_output = stdout_buffer.getvalue()
            if not captured_output.strip():
                captured_output = stderr_buffer.getvalue()
            
            # Clean the captured output to remove logging noise
            captured_output = self._clean_captured_output(captured_output)
            
            logger.info(f"[DEBUG] Captured output length: {len(captured_output)}")
            logger.info(f"[DEBUG] Captured output first 500 chars: {captured_output[:500]}")
            
            # Use captured output if it contains ReAct steps, otherwise fallback to response
            if "Thought:" in captured_output or "Action:" in captured_output:
                full_react_trace = captured_output
                logger.info(f"[DEBUG] Using captured output for ReAct trace")
            else:
                # Fallback to agent memory approach
                chat_history = self.agent.memory.get_all()
                logger.info(f"[DEBUG] Agent memory has {len(chat_history)} messages")
                
                full_react_trace = ""
                for msg in reversed(chat_history):
                    if hasattr(msg, 'role') and msg.role.value == "assistant":
                        full_react_trace = msg.content
                        break
                
                if not full_react_trace:
                    full_react_trace = str(response)
                logger.info(f"[DEBUG] Using agent memory/response for ReAct trace")
            
            logger.info(f"[DEBUG] Full ReAct trace length: {len(full_react_trace)}")
            logger.info(f"[DEBUG] Full ReAct trace first 500 chars: {full_react_trace[:500]}")
            
            # Parse steps from the full ReAct trace
            steps, final_answer = self._parse_react_steps(full_react_trace)
            
            # Fallback: If no answer, but a tool was called, try to get its output directly
            if not final_answer:
                import re
                # Check for get_pr_summary tool usage
                pr_summary_match = re.search(r'Action: get_pr_summary\\s*(\\d+)', full_react_trace)
                if pr_summary_match:
                    pr_number = int(pr_summary_match.group(1))
                    logger.info(f"[DEBUG] Fallback: Directly calling get_pr_summary({pr_number})")
                    try:
                        tool_output = self.get_pr_summary(pr_number)
                        final_answer = tool_output
                        # Add a synthetic answer step for UI consistency
                        steps.append({
                            "type": "answer",
                            "content": tool_output,
                            "step": len(steps)
                        })
                    except Exception as e:
                        logger.error(f"[DEBUG] Fallback get_pr_summary failed: {e}")
                
                # Check for get_pr_diff tool usage
                pr_diff_match = re.search(r'(?:Action: get_pr_diff|get_pr_diff|diff.*pr.*#?(\d+)|pr.*#?(\d+).*diff)', full_react_trace, re.IGNORECASE)
                if pr_diff_match:
                    # Extract PR number from the match
                    pr_number = None
                    for group in pr_diff_match.groups():
                        if group and group.isdigit():
                            pr_number = int(group)
                            break
                    
                    # If no number found in the action, try to extract from the user message
                    if not pr_number:
                        # Look for PR number in the original user message
                        user_pr_match = re.search(r'pr.*#?(\d+)|#(\d+)', user_message, re.IGNORECASE)
                        if user_pr_match:
                            for group in user_pr_match.groups():
                                if group and group.isdigit():
                                    pr_number = int(group)
                                    break
                    
                    if pr_number:
                        logger.info(f"[DEBUG] Fallback: Directly calling get_pr_diff({pr_number})")
                        try:
                            tool_output = self.get_pr_diff(pr_number)
                            final_answer = tool_output
                            # Add a synthetic answer step for UI consistency
                            steps.append({
                                "type": "answer",
                                "content": tool_output,
                                "step": len(steps)
                            })
                        except Exception as e:
                            logger.error(f"[DEBUG] Fallback get_pr_diff failed: {e}")
                    else:
                        logger.warning(f"[DEBUG] get_pr_diff action detected but no PR number found")
                
                # Add more tool-specific fallbacks here if needed
            
            # If no steps found and no final answer, try using the original response as final answer
            if len(steps) == 0 and not final_answer:
                response_str = str(response).strip()
                if response_str and len(response_str) > 20:
                    # Check if it looks like a meaningful response
                    if not any(pattern in response_str.lower() for pattern in ['http request:', 'info:', 'debug:', 'error:']):
                        logger.info(f"[DEBUG] Using original response as final answer")
                        final_answer = response_str
            
            # Extract suggestions from cleaned response 
            cleaned_response = self._extract_clean_answer(str(response))
            suggestions = []
            if "Would you like me to" in cleaned_response:
                suggestions = [s.strip("* ") for s in cleaned_response.split("\n") if s.strip().startswith("* ")]

            return self._format_agentic_response(steps, final_answer, partial=False, suggestions=suggestions)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in agentic query: {error_msg}")
            
            # Handle specific "max iterations" error more gracefully
            if "max iterations" in error_msg.lower() or "reached max" in error_msg.lower():
                logger.warning(f"Agent hit max iterations, providing partial result")
                # Try to extract any partial work from the agent's memory
                partial_steps = []
                try:
                    chat_history = self.agent.memory.get_all()
                    if chat_history:
                        for msg in reversed(chat_history):
                            if hasattr(msg, 'role') and msg.role.value == "assistant":
                                partial_content = msg.content
                                partial_steps, partial_answer = self._parse_react_steps(partial_content)
                                if partial_steps or partial_answer:
                                    return self._format_agentic_response(
                                        partial_steps, 
                                        partial_answer or "Analysis was complex and hit iteration limits. Here's what I found so far.",
                                        partial=True, 
                                        suggestions=["Try a more specific question", "Break down your request into smaller parts", "Ask about specific files or directories"]
                                    )
                                break
                except:
                    pass
                
                # Fallback for max iterations
                return self._format_agentic_response(
                    [], 
                    final_answer="This analysis required more steps than allowed. Try asking a more specific question, like 'Show me files in src/' or 'What's in the main directory?'",
                    partial=True, 
                    suggestions=["Ask about specific directories", "Request file listings", "Break down complex queries"]
                )
            
            # Fallback: return partial/structured error
            return self._format_agentic_response([], final_answer=None, partial=True, suggestions=["Try a more specific question", "Explore a directory", "Ask about a file"])
    
    def _extract_clean_answer(self, raw_response: str) -> str:
        """Extract clean final answer from ReAct agent response"""
        try:
            # Look for the final answer after "Answer:" 
            if "Answer:" in raw_response:
                # Split on "Answer:" and take the last part
                answer_parts = raw_response.split("Answer:")
                if len(answer_parts) > 1:
                    clean_answer = answer_parts[-1].strip()
                    if clean_answer:
                        return clean_answer
            
            # If no "Answer:" found, try to extract meaningful content
            # Remove ReAct framework artifacts
            lines = raw_response.split('\n')
            clean_lines = []
            
            skip_patterns = [
                'Thought:',
                'Action:',
                'Action Input:',
                'Observation:',
                '> Running step',
                'INFO:',
                'HTTP Request:',
                'Step input:',
                '{',  # JSON objects from tool calls
                '}',
                'Pandas Instructions:',
                'Pandas Output:'
            ]
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip lines that contain ReAct framework artifacts
                should_skip = False
                for pattern in skip_patterns:
                    if pattern in line:
                        should_skip = True
                        break
                
                if not should_skip:
                    clean_lines.append(line)
            
            # Join clean lines and return
            if clean_lines:
                cleaned = '\n'.join(clean_lines)
                
                # Remove any remaining framework artifacts
                cleaned = cleaned.replace('```', '')
                cleaned = '\n'.join([line for line in cleaned.split('\n') if line.strip()])
                
                if cleaned.strip():
                    return cleaned.strip()
            
            # Fallback - return a cleaned version of the original
            return self._basic_cleanup(raw_response)
            
        except Exception as e:
            logger.error(f"Error extracting clean answer: {e}")
            return self._basic_cleanup(raw_response)
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic cleanup of response text"""
        # Remove obvious ReAct artifacts
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith('Thought:') and 
                not line.startswith('Action:') and 
                not line.startswith('Action Input:') and 
                not line.startswith('Observation:') and 
                not line.startswith('> Running step') and 
                not line.startswith('INFO:') and
                not line.startswith('HTTP Request:')):
                filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines).strip()
        return result if result else "I analyzed your request but encountered some formatting issues. Please try asking in a different way."
    
    def _get_natural_exploration_suggestions(self, original_query: str) -> str:
        """Provide natural exploration suggestions instead of exposing iteration limits"""
        
        # Analyze the original query to provide relevant suggestions
        query_lower = original_query.lower()
        
        if "explore" in query_lower and "directory" in query_lower:
            return """## Codebase Exploration

I started exploring your codebase and can see it has an interesting structure! Let me help you discover it step by step.

**I can help you with:**

📂 **Directory Structure** - "What's in the src directory?"  
📄 **Key Files** - "Show me the main Python files"  
🏗️ **Architecture** - "How is this project organized?"  
🔧 **Specific Components** - "Explain the agents.py file"  

**What would you like to explore first?**"""
        
        elif "analyze" in query_lower:
            return """## Code Analysis Ready

I'm ready to analyze your codebase! I work best when you give me specific areas to focus on.

**Try asking me to:**

🔍 **Analyze specific files** - "What does main.py do?"  
📋 **Understand structure** - "How are the modules organized?"  
🔗 **Find relationships** - "What files are related to authentication?"  
💡 **Explain patterns** - "Show me the design patterns used"  

**What aspect of the code interests you most?**"""
        
        else:
            return """## Let's Explore Your Code Together

I'm here to help you understand your codebase! I work best with focused questions.

**Popular exploration patterns:**

🏠 **Project Overview** - "What is this project about?"  
📁 **Directory Exploration** - "What's in the [directory] folder?"  
📄 **File Analysis** - "Explain the [filename] file"  
🔍 **Find Functionality** - "Where is [feature] implemented?"  

**What would you like to discover?**"""
    
    def _get_natural_error_recovery(self, original_query: str, error_msg: str) -> str:
        """Provide natural error recovery without exposing technical details"""
        
        # Don't expose raw error messages to users
        logger.error(f"Providing natural error recovery for: {error_msg}")
        
        return """## Let's Try a Different Approach

I had some trouble with that analysis. Let me help you explore your codebase with a more focused approach.

**Try these patterns:**

🎯 **Specific Questions** - "What files are in the src directory?"  
📖 **File Reading** - "Show me the contents of main.py"  
🔍 **Targeted Search** - "Find all Python files with 'agent' in the name"  
📂 **Step-by-step** - "First show me the project structure"  

**What specific part of your code would you like to explore?**"""
    
    def reset_memory(self):
        """Reset the agent's memory"""
        self.memory.reset()

    async def stream_query(self, user_message: str):
        """Async generator that streams agentic steps as JSON lines for real-time UI updates."""
        try:
            logger.info(f"[stream] Starting agentic analysis: {user_message[:100]}...")
            
            # Yield initial status
            yield json.dumps({
                "type": "status",
                "content": "Starting analysis...",
                "step": 0
            })
            
            # Capture the verbose output during agent execution
            with capture_output() as (stdout_buffer, stderr_buffer):
                response = await self.agent.achat(user_message)
            
            logger.info(f"[stream] Agentic analysis completed successfully")
            
            # Get the captured verbose output which contains the full ReAct trace
            captured_output = stdout_buffer.getvalue()
            if not captured_output.strip():
                captured_output = stderr_buffer.getvalue()
            
            # Clean the captured output to remove logging noise
            captured_output = self._clean_captured_output(captured_output)
            
            logger.info(f"[DEBUG] Captured output length: {len(captured_output)}")
            logger.info(f"[DEBUG] Captured output first 500 chars: {captured_output[:500]}")
            
            # Use captured output if it contains ReAct steps, otherwise fallback to response
            if "Thought:" in captured_output or "Action:" in captured_output:
                full_react_trace = captured_output
                logger.info(f"[DEBUG] Using captured output for ReAct trace")
            else:
                # Fallback to agent memory approach
                chat_history = self.agent.memory.get_all()
                logger.info(f"[DEBUG] Agent memory has {len(chat_history)} messages")
                
                full_react_trace = ""
                for msg in reversed(chat_history):
                    if hasattr(msg, 'role') and msg.role.value == "assistant":
                        full_react_trace = msg.content
                    break

                if not full_react_trace:
                    full_react_trace = str(response)
                logger.info(f"[DEBUG] Using agent memory/response for ReAct trace")
            
            logger.info(f"[DEBUG] Full ReAct trace length: {len(full_react_trace)}")
            logger.info(f"[DEBUG] Full ReAct trace first 500 chars: {full_react_trace[:500]}")
            
            # Parse steps from the full ReAct trace
            steps, final_answer = self._parse_react_steps(full_react_trace)
            
            # Fallback: If no answer, but a tool was called, try to get its output directly
            if not final_answer:
                import re
                # Check for get_pr_summary tool usage
                pr_summary_match = re.search(r'Action: get_pr_summary\\s*(\\d+)', full_react_trace)
                if pr_summary_match:
                    pr_number = int(pr_summary_match.group(1))
                    logger.info(f"[DEBUG] Stream fallback: Directly calling get_pr_summary({pr_number})")
                    try:
                        tool_output = self.get_pr_summary(pr_number)
                        final_answer = tool_output
                        # Add a synthetic answer step for UI consistency
                        steps.append({
                            "type": "answer",
                            "content": tool_output,
                            "step": len(steps)
                        })
                    except Exception as e:
                        logger.error(f"[DEBUG] Stream fallback get_pr_summary failed: {e}")
                
                # Check for get_pr_diff tool usage
                pr_diff_match = re.search(r'(?:Action: get_pr_diff|get_pr_diff|diff.*pr.*#?(\d+)|pr.*#?(\d+).*diff)', full_react_trace, re.IGNORECASE)
                if pr_diff_match:
                    # Extract PR number from the match
                    pr_number = None
                    for group in pr_diff_match.groups():
                        if group and group.isdigit():
                            pr_number = int(group)
                            break
                    
                    # If no number found in the action, try to extract from the user message
                    if not pr_number:
                        # Look for PR number in the original user message
                        user_pr_match = re.search(r'pr.*#?(\d+)|#(\d+)', user_message, re.IGNORECASE)
                        if user_pr_match:
                            for group in user_pr_match.groups():
                                if group and group.isdigit():
                                    pr_number = int(group)
                                    break
                    
                    if pr_number:
                        logger.info(f"[DEBUG] Stream fallback: Directly calling get_pr_diff({pr_number})")
                        try:
                            tool_output = self.get_pr_diff(pr_number)
                            final_answer = tool_output
                            # Add a synthetic answer step for UI consistency
                            steps.append({
                                "type": "answer",
                                "content": tool_output,
                                "step": len(steps)
                            })
                        except Exception as e:
                            logger.error(f"[DEBUG] Stream fallback get_pr_diff failed: {e}")
                    else:
                        logger.warning(f"[DEBUG] get_pr_diff action detected but no PR number found")
            
            # If no steps found and no final answer, try using the original response as final answer
            if len(steps) == 0 and not final_answer:
                response_str = str(response).strip()
                if response_str and len(response_str) > 20:
                    # Check if it looks like a meaningful response
                    if not any(pattern in response_str.lower() for pattern in ['http request:', 'info:', 'debug:', 'error:']):
                        logger.info(f"[DEBUG] Using original response as final answer")
                        final_answer = response_str
            
            # Extract suggestions from cleaned response 
            cleaned_response = self._extract_clean_answer(str(response))
            suggestions = []
            if "Would you like me to" in cleaned_response:
                suggestions = [s.strip("* ") for s in cleaned_response.split("\n") if s.strip().startswith("* ")]

            # Yield each step incrementally
            for i, step in enumerate(steps):
                try:
                    # Ensure step has content field for logging
                    content_len = len(step.get('content', '')) if step.get('content') else 0
                    logger.info(f"[DEBUG] Yielding step {i}: type={step.get('type', 'unknown')}, content_len={content_len}")
                    yield json.dumps({"type": "step", "step": step})
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                except Exception as step_error:
                    logger.error(f"[DEBUG] Error yielding step {i}: {step_error}")
                    # Yield a safe fallback step
                    safe_step = {
                        "type": step.get("type", "unknown"),
                        "content": f"Step {i}: {step.get('type', 'unknown')} (error in serialization)",
                        "step": i
                    }
                    yield json.dumps({"type": "step", "step": safe_step})

            # Yield a final event with all steps, final answer, and suggestions
            try:
                final_payload = {
                    "type": "final",
                    "final": True,
                    "steps": steps,
                    "final_answer": final_answer,
                    "partial": False,
                    "suggestions": suggestions or [],
                    "total_steps": len(steps)
                }
                logger.info(f"[DEBUG] Yielding final payload with {len(steps)} steps and final_answer: {bool(final_answer)}")
                yield json.dumps(final_payload)
            except Exception as final_error:
                logger.error(f"[DEBUG] Error yielding final payload: {final_error}")
                # Yield a safe fallback final payload
                safe_final_payload = {
                    "type": "final",
                    "final": True,
                    "steps": [],
                    "final_answer": str(final_answer) if final_answer else "Analysis completed with errors",
                    "partial": True,
                    "suggestions": [],
                    "total_steps": len(steps),
                    "error": "Error in final payload serialization"
                }
                yield json.dumps(safe_final_payload)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[stream] Error in agentic stream_query: {error_msg}")
            
            # Handle specific "max iterations" error more gracefully
            if "max iterations" in error_msg.lower() or "reached max" in error_msg.lower():
                logger.warning(f"[stream] Agent hit max iterations, providing partial result")
                # Try to extract any partial work from the agent's memory
                try:
                    chat_history = self.agent.memory.get_all()
                    partial_steps = []
                    partial_answer = None
                    
                    if chat_history:
                        for msg in reversed(chat_history):
                            if hasattr(msg, 'role') and msg.role.value == "assistant":
                                partial_content = msg.content
                                partial_steps, partial_answer = self._parse_react_steps(partial_content)
                                break
                    
                    # Yield partial steps if found
                    if partial_steps:
                        for step in partial_steps:
                            yield json.dumps({"type": "step", "step": step})
                    
                    # Yield max iterations error with helpful message
                    yield json.dumps({
                        "type": "error",
                        "final": True,
                        "steps": partial_steps,
                        "final_answer": partial_answer or "Analysis hit complexity limits. Try a more specific question.",
                        "partial": True,
                        "suggestions": ["Ask about specific directories", "Request file listings", "Break down complex queries"],
                        "error": "Analysis required too many reasoning steps. Please try a more focused question."
                    })
                    return
                except:
                    pass
            
            # Yield a structured error/partial result
            yield json.dumps({
                "type": "error",
                "final": True,
                "steps": [],
                "final_answer": None,
                "partial": True,
                "suggestions": ["Try a more specific question", "Explore a directory", "Ask about a file"],
                "error": error_msg
            })

    def _clean_captured_output(self, captured_output: str) -> str:
        """Clean captured output to remove logging noise but preserve ReAct trace."""
        if not captured_output:
            return ""
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_output = ansi_escape.sub('', captured_output)
        
        # Remove common escape sequences
        cleaned_output = re.sub(r'\033\[[0-9;]*m', '', cleaned_output)
        cleaned_output = re.sub(r'\[0m', '', cleaned_output) # Common reset sequence
        cleaned_output = re.sub(r'\[1;3;[0-9]+m', '', cleaned_output) # Example colored output

        lines = cleaned_output.split('\n')
        preserved_lines = []
        
        # More specific patterns for LlamaIndex and httpx logging
        skip_log_patterns = [
            re.compile(r'^\s*DEBUG:'),
            re.compile(r'^\s*INFO:'),
            re.compile(r'^\s*WARNING:'),
            re.compile(r'^\s*ERROR:'),
            re.compile(r'^\s*INFO:httpx:HTTP Request:'),
            re.compile(r'^\s*⚡'), # LlamaIndex lightning bolt
            re.compile(r'^\s*> Running step'), # LlamaIndex step execution log
            re.compile(r'^\s*Step \w+ produced event'), # LlamaIndex event log
        ]

        for line in lines:
            # Don't strip lines globally yet, preserve original form for parsing
            
            # Skip specific logging patterns
            if any(pattern.search(line) for pattern in skip_log_patterns):
                continue
            
            # Skip lines that are purely terminal formatting artifacts after initial cleaning
            if re.match(r'^[\[\]0-9;m\s]*$', line.strip()): 
                continue
            
            preserved_lines.append(line) # Keep the original line, not stripped
        
        return '\n'.join(preserved_lines)
    
    def _parse_react_steps(self, raw_response: str):
        """Parse ReAct steps from raw agent response into a structured format."""
        logger.info(f"[DEBUG] Parsing ReAct trace (length: {len(raw_response)}): {raw_response[:500]}")
        
        steps = []
        lines = raw_response.split('\n')
        i = 0
        step_counter = 0

        while i < len(lines):
            line_content_stripped = lines[i].strip() # For prefix checking
            original_line = lines[i] # Keep original for content extraction
            
            if not line_content_stripped: # Skip empty lines
                i += 1
                continue

            current_step_data = {"step": step_counter}

            if line_content_stripped.startswith("Thought:"):
                current_step_data["type"] = "thought"
                # Content starts after "Thought:"
                content_lines = [original_line.split("Thought:", 1)[1].strip()]
                i += 1
                # Collect subsequent lines until a new ReAct keyword or end of lines
                while i < len(lines) and not re.match(r"^(Thought:|Action:|Action Input:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                    content_lines.append(lines[i]) # Keep original spacing/indentation
                    i += 1
                current_step_data["content"] = "\n".join(content_lines).strip()
                steps.append(current_step_data)
                step_counter += 1
            
            elif line_content_stripped.startswith("Action:"):
                current_step_data["type"] = "action"
                current_step_data["tool_name"] = original_line.split("Action:", 1)[1].strip()
                current_step_data["content"] = f"Calling tool: {current_step_data['tool_name']}"  # Ensure content exists
                i += 1
                # Look for Action Input immediately following
                if i < len(lines) and lines[i].strip().startswith("Action Input:"):
                    action_input_header = lines[i].strip()
                    action_input_content_start_index = i
                    
                    # The content of Action Input is everything after "Action Input:"
                    # It might be single-line or multi-line JSON.
                    input_json_lines = [lines[i].split("Action Input:", 1)[1].strip()]
                    i += 1
                    
                    # Heuristic to capture multi-line JSON:
                    # Continue if line doesn't start with another ReAct keyword
                    # and we haven't hit a balanced JSON structure (if it looks like JSON)
                    # This is tricky because Action Input might not always be JSON.
                    
                    # Try to detect if it's JSON
                    first_input_line_stripped = input_json_lines[0]
                    is_likely_json = first_input_line_stripped.startswith('{') or first_input_line_stripped.startswith('[')

                    if is_likely_json:
                        open_braces = first_input_line_stripped.count('{') + first_input_line_stripped.count('[')
                        close_braces = first_input_line_stripped.count('}') + first_input_line_stripped.count(']')
                        
                        while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                            current_input_line = lines[i] # Keep original
                            input_json_lines.append(current_input_line)
                            open_braces += current_input_line.count('{') + current_input_line.count('[')
                            close_braces += current_input_line.count('}') + current_input_line.count(']')
                            i += 1
                            if open_braces > 0 and open_braces == close_braces: # Balanced braces/brackets
                                break
                    else: # Not starting like JSON, assume single line or simple multi-line text
                         while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                            input_json_lines.append(lines[i])
                            i += 1
                    
                    action_input_str = "\n".join(input_json_lines).strip()
                    try:
                        current_step_data["tool_input"] = json.loads(action_input_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Action Input for {current_step_data['tool_name']} not JSON: {action_input_str[:100]}")
                        current_step_data["tool_input"] = action_input_str 
                else: # No "Action Input:" line found, tool might not take input or input is implicit
                    current_step_data["tool_input"] = None 
                    # Do not increment i here, let the main loop handle the next line

                steps.append(current_step_data)
                step_counter += 1

            elif line_content_stripped.startswith("Observation:"):
                current_step_data["type"] = "observation"
                if steps and steps[-1]["type"] == "action": # Associate with the last action
                    current_step_data["observed_tool_name"] = steps[-1].get("tool_name", "unknown_tool")

                obs_content_lines = [original_line.split("Observation:", 1)[1].strip()]
                i += 1
                
                first_obs_line_stripped = obs_content_lines[0]
                is_likely_json_obs = first_obs_line_stripped.startswith('{') or first_obs_line_stripped.startswith('[')

                if is_likely_json_obs:
                    open_braces_obs = first_obs_line_stripped.count('{') + first_obs_line_stripped.count('[')
                    close_braces_obs = first_obs_line_stripped.count('}') + first_obs_line_stripped.count(']')
                    while i < len(lines) and not re.match(r"^(Thought:|Action:|Answer:|Final Answer:|Observation:)", lines[i].strip()):
                        current_obs_line = lines[i]
                        obs_content_lines.append(current_obs_line)
                        open_braces_obs += current_obs_line.count('{') + current_obs_line.count('[')
                        close_braces_obs += current_obs_line.count('}') + current_obs_line.count(']')
                        i += 1
                        if open_braces_obs > 0 and open_braces_obs == close_braces_obs:
                            break
                else: # Not JSON, just text
                    while i < len(lines) and not re.match(r"^(Thought:|Action:|Answer:|Final Answer:|Observation:)", lines[i].strip()):
                        obs_content_lines.append(lines[i])
                        i += 1
                
                observation_str = "\n".join(obs_content_lines).strip()
                try:
                    parsed_observation = json.loads(observation_str)
                    current_step_data["content"] = parsed_observation
                    if isinstance(parsed_observation, dict):
                        current_step_data["tool_output_preview"] = f"JSON data (Keys: {list(parsed_observation.keys())[:3]}...)"
                    elif isinstance(parsed_observation, list):
                        current_step_data["tool_output_preview"] = f"JSON array (Length: {len(parsed_observation)}, First item: {str(parsed_observation[0])[:50]}...)" if parsed_observation else "Empty JSON array"
                    else:
                        current_step_data["tool_output_preview"] = observation_str[:200] + "..." if len(observation_str) > 200 else observation_str
                except json.JSONDecodeError:
                    current_step_data["content"] = observation_str
                    current_step_data["tool_output_preview"] = observation_str[:200] + "..." if len(observation_str) > 200 else observation_str
                
                steps.append(current_step_data)
                step_counter += 1

            elif line_content_stripped.startswith("Answer:") or line_content_stripped.startswith("Final Answer:"):
                current_step_data["type"] = "answer"
                if line_content_stripped.startswith("Final Answer:"):
                    content_lines = [original_line.split("Final Answer:", 1)[1].strip()]
                else:
                    content_lines = [original_line.split("Answer:", 1)[1].strip()]
                i += 1
                while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:)", lines[i].strip()):
                    content_lines.append(lines[i]) # Keep original spacing
                    i += 1
                current_step_data["content"] = "\n".join(content_lines).strip()
                steps.append(current_step_data)
                step_counter += 1
            else:
                # Line doesn't match known ReAct prefixes.
                # It could be part of a multi-line thought or answer, or an unexpected line.
                # If the last step was a thought or answer, append to it.
                if steps and steps[-1]["type"] in ["thought", "answer"] and isinstance(steps[-1]["content"], str):
                    logger.debug(f"Appending to previous {steps[-1]['type']}: {original_line.strip()}")
                    steps[-1]["content"] += "\n" + original_line # Append with original spacing
                    steps[-1]["content"] = steps[-1]["content"].strip() # Strip at the end
                else:
                    logger.debug(f"Skipping unexpected line: {original_line.strip()}")
                i += 1 # Move to next line

        final_answer_obj = next((step for step in reversed(steps) if step["type"] == "answer"), None)
        final_answer_content = final_answer_obj["content"] if final_answer_obj else None

        # If no specific "Answer:" step, but the agent produced a raw response (not ReAct like)
        if not final_answer_content and not steps and raw_response.strip():
            # Check if it's not just leftover logging
            if not any(pattern.search(raw_response) for pattern in [
                re.compile(r'^\s*(DEBUG|INFO|WARNING|ERROR)'), 
                re.compile(r'HTTP Request:'),
                re.compile(r'> Running step')
            ]):
                logger.info(f"[DEBUG] No ReAct steps, using raw_response as final answer.")
                final_answer_content = raw_response.strip()
                steps.append({"type": "answer", "content": final_answer_content, "step": 0})

        # Ensure all steps have required fields
        for step in steps:
            if 'content' not in step:
                step['content'] = f"Step {step.get('step', 0)}: {step.get('type', 'unknown')} executed"
            if 'type' not in step:
                step['type'] = 'unknown'
            if 'step' not in step:
                step['step'] = 0
        
        logger.info(f"[DEBUG] Parsed {len(steps)} steps. Final answer derived: {bool(final_answer_content)}")
        return steps, final_answer_content
    
    def _detect_repository_languages(self) -> Dict[str, str]:
        """Detect languages in the repository by analyzing file extensions"""
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.swift': 'swift',
            '.m': 'objective-c',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.ml': 'ocaml',
            '.hs': 'haskell',
            '.clj': 'clojure',
                '.dart': 'dart',
                '.lua': 'lua',
            '.r': 'r',
            '.jl': 'julia'
        }
        
        language_counts = {}
        total_files = 0
        
        # Walk through repository and count files by language
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                ext = file_path.suffix.lower()
                if ext in language_map:
                    lang = language_map[ext]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    total_files += 1
        
        if not language_counts:
            return {"primary_language": "generic", "languages": {}}
        
        # Find primary language (most common)
        primary_language = max(language_counts, key=language_counts.get)
        
        # Calculate percentages
        language_percentages = {}
        for lang, count in language_counts.items():
            language_percentages[lang] = round((count / total_files) * 100, 1)
        
        return {
            "primary_language": primary_language,
            "languages": language_percentages,
            "total_files": total_files
        }
    
    def _analyze_repository_context(self, context_files: List[str]) -> Dict[str, Any]:
        """Analyze context files to understand available patterns (language-agnostic)"""
        analysis = {
            "classes": [],
            "base_classes": [],
            "imports": [],
            "capabilities": [],
            "patterns": {}
        }
        
        for file_path in context_files:
            try:
                full_path = self.repo_path / file_path
                if not full_path.exists():
                    continue
                    
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detect file language
                file_ext = Path(file_path).suffix.lower()
                language = self._get_language_from_extension(file_ext)
                
                # Extract patterns based on language
                if language == "python":
                    self._extract_python_patterns(content, file_path, analysis)
                elif language in ["javascript", "typescript"]:
                    self._extract_js_ts_patterns(content, file_path, analysis)
                elif language == "java":
                    self._extract_java_patterns(content, file_path, analysis)
                elif language == "go":
                    self._extract_go_patterns(content, file_path, analysis)
                elif language == "rust":
                    self._extract_rust_patterns(content, file_path, analysis)
                elif language == "csharp":
                    self._extract_csharp_patterns(content, file_path, analysis)
                elif language == "ruby":
                    self._extract_ruby_patterns(content, file_path, analysis)
                elif language == "php":
                    self._extract_php_patterns(content, file_path, analysis)
                else:
                    # Generic pattern extraction for unknown languages
                    self._extract_generic_patterns(content, file_path, analysis)
                
                # Extract capabilities mentioned in the file (language-agnostic)
                analysis["capabilities"].extend(self._extract_file_capabilities(content))
                        
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                continue
        
        # Remove duplicates
        analysis["base_classes"] = list(set(analysis["base_classes"]))
        analysis["capabilities"] = list(set(analysis["capabilities"]))
        
        return analysis
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Get language from file extension"""
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.swift': 'swift',
            '.m': 'objective-c',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.ml': 'ocaml'
        }
        return language_map.get(ext, 'generic')
    
    def _extract_python_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract Python-specific patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Python imports
            if line.startswith(('import ', 'from ')):
                analysis["imports"].append(line)
            
            # Python classes
            elif line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                base_class = ""
                
                if '(' in line and ')' in line:
                    base_class = line.split('(')[1].split(')')[0].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_js_ts_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract JavaScript/TypeScript patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # JavaScript/TypeScript imports
            if line.startswith(('import ', 'const ', 'let ', 'var ')) and ('require(' in line or 'import' in line):
                analysis["imports"].append(line)
            
            # Classes
            elif line.startswith('class ') or line.startswith('export class '):
                class_match = line.replace('export ', '').replace('class ', '')
                class_name = class_match.split(' ')[0].split('(')[0].split('{')[0].strip()
                
                base_class = ""
                if ' extends ' in line:
                    base_class = line.split(' extends ')[1].split(' ')[0].split('{')[0].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_java_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract Java patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Java imports
            if line.startswith('import '):
                analysis["imports"].append(line)
            
            # Java classes
            elif line.startswith('public class ') or line.startswith('class '):
                class_line = line.replace('public ', '').replace('class ', '')
                class_name = class_line.split(' ')[0].split('<')[0].strip()
                
                base_class = ""
                if ' extends ' in line:
                    base_class = line.split(' extends ')[1].split(' ')[0].split('<')[0].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_go_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract Go patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Go imports
            if line.startswith('import ') or (line.startswith('"') and line.endswith('"')):
                analysis["imports"].append(line)
            
            # Go structs (closest to classes)
            elif line.startswith('type ') and ' struct ' in line:
                struct_name = line.split('type ')[1].split(' struct')[0].strip()
                
                class_info = {
                    "name": struct_name,
                    "base": "",  # Go doesn't have inheritance
                    "file": file_path,
                    "purpose": self._extract_struct_purpose(content, struct_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_rust_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract Rust patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Rust uses/imports
            if line.startswith('use '):
                analysis["imports"].append(line)
            
            # Rust structs
            elif line.startswith('pub struct ') or line.startswith('struct '):
                struct_line = line.replace('pub ', '').replace('struct ', '')
                struct_name = struct_line.split(' ')[0].split('<')[0].strip()
                
                class_info = {
                    "name": struct_name,
                    "base": "",  # Rust uses traits instead of inheritance
                    "file": file_path,
                    "purpose": self._extract_struct_purpose(content, struct_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_csharp_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract C# patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # C# using statements
            if line.startswith('using '):
                analysis["imports"].append(line)
            
            # C# classes
            elif 'class ' in line and (line.startswith('public ') or line.startswith('internal ') or line.startswith('class ')):
                class_part = line.split('class ')[1]
                class_name = class_part.split(' ')[0].split(':')[0].split('<')[0].strip()
                
                base_class = ""
                if ' : ' in line:
                    base_class = line.split(' : ')[1].split(' ')[0].split(',')[0].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_ruby_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract Ruby patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Ruby requires
            if line.startswith('require ') or line.startswith('require_relative '):
                analysis["imports"].append(line)
            
            # Ruby classes
            elif line.startswith('class '):
                class_line = line.replace('class ', '')
                class_name = class_line.split(' ')[0].split('<')[0].strip()
                
                base_class = ""
                if ' < ' in line:
                    base_class = line.split(' < ')[1].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_php_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract PHP patterns"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # PHP includes/requires
            if line.startswith(('require ', 'require_once ', 'include ', 'include_once ', 'use ')):
                analysis["imports"].append(line)
            
            # PHP classes
            elif line.startswith('class ') or line.startswith('abstract class ') or line.startswith('final class '):
                class_line = line.replace('abstract ', '').replace('final ', '').replace('class ', '')
                class_name = class_line.split(' ')[0].strip()
                
                base_class = ""
                if ' extends ' in line:
                    base_class = line.split(' extends ')[1].split(' ')[0].strip()
                    analysis["base_classes"].append(base_class)
                
                class_info = {
                    "name": class_name,
                    "base": base_class,
                    "file": file_path,
                    "purpose": self._extract_class_purpose(content, class_name)
                }
                analysis["classes"].append(class_info)
    
    def _extract_generic_patterns(self, content: str, file_path: str, analysis: Dict):
        """Extract patterns from unknown languages using generic heuristics"""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Generic import-like patterns
            if any(keyword in line.lower() for keyword in ['import', 'require', 'include', 'use', '#include']):
                analysis["imports"].append(line)
            
            # Generic class-like patterns
            if any(keyword in line.lower() for keyword in ['class', 'struct', 'type', 'interface']):
                # Try to extract something that looks like a class name
                words = line.split()
                for i, word in enumerate(words):
                    if word.lower() in ['class', 'struct', 'type', 'interface'] and i + 1 < len(words):
                        class_name = words[i + 1].split('(')[0].split('{')[0].split(':')[0].strip()
                        
                        class_info = {
                            "name": class_name,
                            "base": "",
                            "file": file_path,
                            "purpose": "Generic class or structure"
                        }
                        analysis["classes"].append(class_info)
                        break
    
    def _extract_class_purpose(self, content: str, class_name: str) -> str:
        """Extract what a class is designed to do (works for multiple languages)"""
        lines = content.split('\n')
        
        # Find class definition
        for i, line in enumerate(lines):
            if class_name in line and any(keyword in line.lower() for keyword in ['class', 'struct', 'type']):
                # Look for documentation in next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    doc_line = lines[j].strip()
                    
                    # Python/JavaScript/TypeScript docstrings
                    if doc_line.startswith('"""') or doc_line.startswith("'''") or doc_line.startswith('/**'):
                        quote_char = '"""' if doc_line.startswith('"""') else ("'''" if doc_line.startswith("'''") else "*")
                        if doc_line.count(quote_char) >= 2 or doc_line.endswith('*/'):
                            return doc_line.replace('"""', '').replace("'''", '').replace('/**', '').replace('*/', '').strip()
                        else:
                            # Multi-line docstring
                            docstring = doc_line.replace('"""', '').replace("'''", '').replace('/**', '').replace('*', '').strip()
                            for k in range(j + 1, min(j + 5, len(lines))):
                                next_line = lines[k].strip()
                                if '"""' in next_line or "'''" in next_line or '*/' in next_line:
                                    docstring += ' ' + next_line.replace('"""', '').replace("'''", '').replace('*/', '').replace('*', '').strip()
                                    break
                                docstring += ' ' + next_line.replace('*', '').strip()
                            return docstring.strip()
                    
                    # Comments (// or # style)
                    elif doc_line.startswith('//') or doc_line.startswith('#'):
                        return doc_line.replace('//', '').replace('#', '').strip()
                break
        
        return "No description available"
    
    def _extract_struct_purpose(self, content: str, struct_name: str) -> str:
        """Extract purpose for Go/Rust structs"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if struct_name in line and ('struct' in line or 'type' in line):
                # Look for comments above the struct
                for j in range(max(0, i - 3), i):
                    comment_line = lines[j].strip()
                    if comment_line.startswith('//') or comment_line.startswith('///'):
                        return comment_line.replace('//', '').replace('///', '').strip()
                
                # Look for comments after the struct
                for j in range(i + 1, min(i + 3, len(lines))):
                    comment_line = lines[j].strip()
                    if comment_line.startswith('//') or comment_line.startswith('///'):
                        return comment_line.replace('//', '').replace('///', '').strip()
                break
        
        return "No description available"
    
    def _extract_file_capabilities(self, content: str) -> List[str]:
        """Extract capabilities from file content (language-agnostic)"""
        capabilities = []
        
        # Look for action words in docstrings and comments
        action_words = [
            "search", "analyze", "generate", "create", "build", "process",
            "execute", "explore", "find", "parse", "validate", "extract",
            "tool", "agent", "model", "chat", "query", "run", "handle",
            "manage", "connect", "authenticate", "request", "response"
        ]
        
        content_lower = content.lower()
        for word in action_words:
            if word in content_lower:
                capabilities.append(word)
        
        return capabilities 

    def analyze_github_issue(
        self,
        issue_identifier: Annotated[str, "Issue number (#123) or full GitHub issue URL to analyze"]
    ) -> str:
        """Analyze a GitHub issue comprehensively to understand requirements and complexity"""
        try:
            # Import GitHub client here to avoid circular imports
            from .github_client import GitHubIssueClient
            
            # Initialize GitHub client
            github_client = GitHubIssueClient()
            
            # If it's just a number, construct the URL based on repo path
            if issue_identifier.startswith('#') or issue_identifier.isdigit():
                issue_number = issue_identifier.lstrip('#')
                # Try to get repo URL from session or repo path
                repo_url = self._get_repo_url_from_path()
                if not repo_url:
                    return json.dumps({
                        "error": "Cannot determine repository URL",
                        "suggestion": "Please provide the full GitHub issue URL instead of just the issue number"
                    }, indent=2)
                issue_url = f"{repo_url}/issues/{issue_number}"
            else:
                issue_url = issue_identifier
            
            # Fetch issue data
            issue_response = asyncio.run(github_client.get_issue(issue_url))
            
            if issue_response.status != "success" or not issue_response.data:
                return json.dumps({
                    "error": f"Failed to fetch issue: {issue_response.error}",
                    "issue_identifier": issue_identifier
                }, indent=2)
            
            issue = issue_response.data
            
            # Perform comprehensive analysis
            analysis = self._perform_issue_analysis(issue)
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error analyzing GitHub issue: {e}")
            return json.dumps({
                "error": f"Failed to analyze issue: {str(e)}",
                "issue_identifier": issue_identifier
            }, indent=2)
    
    def find_issue_related_files(
        self,
        issue_description: Annotated[str, "Description of the issue or feature to find related files for"],
        search_depth: Annotated[str, "Search depth: 'surface' for obvious matches, 'deep' for comprehensive analysis"] = "surface"
    ) -> str:
        """Find files in the repository that are likely relevant to solving a specific issue"""
        try:
            # Extract keywords and technical terms from issue description
            search_terms = self._extract_issue_keywords(issue_description)
            
            # Perform multi-stage search
            relevant_files = []
            
            # Stage 1: Direct keyword search
            for term in search_terms['primary']:
                search_results = json.loads(self.search_codebase(term, ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.rb']))
                parsed_results = json.loads(search_results) if isinstance(search_results, str) else search_results
                
                for result in parsed_results.get('results', []):
                    file_path = result['file']
                    relevance_score = len(result['matches']) * 2  # Base score
                    
                    # Boost score for certain file types or locations
                    if any(keyword in file_path.lower() for keyword in search_terms['primary']):
                        relevance_score += 5
                    if file_path.startswith(('src/', 'lib/', 'app/')):
                        relevance_score += 2
                    if file_path.endswith(('_test.py', '.test.js', '.spec.js')):
                        relevance_score -= 1  # Lower priority for test files initially
                    
                    relevant_files.append({
                        "file": file_path,
                        "relevance_score": relevance_score,
                        "match_reason": f"Contains '{term}'",
                        "matches": result['matches'][:3]  # Limit matches
                    })
            
            # Stage 2: Semantic/contextual search if deep analysis requested
            if search_depth == "deep":
                for context_term in search_terms['contextual']:
                    semantic_results = json.loads(self.semantic_content_search(context_term))
                    parsed_semantic = json.loads(semantic_results) if isinstance(semantic_results, str) else semantic_results
                    
                    for result in parsed_semantic.get('results', []):
                        file_path = result['file']
                        # Add files that weren't found in direct search
                        if not any(rf['file'] == file_path for rf in relevant_files):
                            relevant_files.append({
                                "file": file_path,
                                "relevance_score": result['score'],
                                "match_reason": f"Semantic match for '{context_term}'",
                                "context": result.get('context', [])[:2]
                            })
            
            # Stage 3: Find related configuration and test files
            config_files = self._find_configuration_files(search_terms)
            test_files = self._find_related_test_files([rf['file'] for rf in relevant_files])
            
            # Combine and deduplicate
            all_files = relevant_files + config_files + test_files
            unique_files = {}
            for file_info in all_files:
                file_path = file_info['file']
                if file_path not in unique_files or file_info['relevance_score'] > unique_files[file_path]['relevance_score']:
                    unique_files[file_path] = file_info
            
            # Sort by relevance score
            sorted_files = sorted(unique_files.values(), key=lambda x: x['relevance_score'], reverse=True)
            
            # Limit results and categorize
            top_files = sorted_files[:15]
            
            analysis = {
                "issue_description": issue_description,
                "search_depth": search_depth,
                "search_terms_used": search_terms,
                "total_files_found": len(sorted_files),
                "top_relevant_files": top_files,
                "file_categories": self._categorize_files(top_files),
                "recommendations": self._generate_file_recommendations(top_files, search_terms)
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error finding issue-related files: {e}")
            return json.dumps({
                "error": f"Failed to find related files: {str(e)}",
                "issue_description": issue_description
            }, indent=2)
    
    def _get_repo_url_from_path(self) -> Optional[str]:
        """Try to determine the repository URL from the repo path"""
        try:
            # Check if there's a .git directory
            git_dir = self.repo_path / '.git'
            if git_dir.exists():
                # Try to read the origin URL
                config_file = git_dir / 'config'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        content = f.read()
                        # Look for origin URL
                        import re
                        match = re.search(r'url = (https://github\.com/[^/]+/[^/\s]+)', content)
                        if match:
                            return match.group(1).rstrip('.git')
            return None
        except Exception:
            return None
    
    def _perform_issue_analysis(self, issue) -> Dict[str, Any]:
        """Perform comprehensive analysis of a GitHub issue"""
        analysis = {
            "issue_metadata": {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at.isoformat() if hasattr(issue.created_at, 'isoformat') else str(issue.created_at),
                "labels": issue.labels,
                "assignees": issue.assignees,
                "comments_count": len(issue.comments)
            },
            "issue_classification": self._classify_issue_type(issue),
            "complexity_assessment": self._assess_issue_complexity(issue),
            "technical_requirements": self._extract_technical_requirements(issue),
            "suggested_approach": self._suggest_approach(issue),
            "estimated_effort": self._estimate_effort(issue),
            "related_keywords": self._extract_issue_keywords(issue.body)
        }
        
        # Add comments analysis if there are comments
        if issue.comments:
            analysis["comments_analysis"] = self._analyze_comments(issue.comments)
        
        return analysis
    
    def _classify_issue_type(self, issue) -> Dict[str, Any]:
        """Classify the type of issue based on title, labels, and content"""
        issue_types = {
            "bug": ["bug", "fix", "error", "broken", "issue", "problem", "not working"],
            "feature": ["feature", "enhancement", "add", "implement", "new", "support"],
            "documentation": ["documentation", "docs", "readme", "comment", "explain"],
            "refactor": ["refactor", "cleanup", "improve", "optimize", "restructure"],
            "test": ["test", "testing", "spec", "coverage", "unit test"],
            "security": ["security", "vulnerability", "exploit", "auth", "permission"],
            "performance": ["performance", "slow", "speed", "optimize", "memory", "cpu"]
        }
        
        text_to_analyze = f"{issue.title} {issue.body}".lower()
        label_text = " ".join(issue.labels).lower()
        
        scores = {}
        for issue_type, keywords in issue_types.items():
            score = 0
            for keyword in keywords:
                score += text_to_analyze.count(keyword) * 2
                score += label_text.count(keyword) * 5  # Labels are more definitive
            scores[issue_type] = score
        
        primary_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
        
        return {
            "primary_type": primary_type,
            "confidence_scores": scores,
            "detected_labels": [label for label in issue.labels if label.lower() in sum(issue_types.values(), [])]
        }
    
    def _assess_issue_complexity(self, issue) -> Dict[str, Any]:
        """Assess the complexity of an issue"""
        complexity_indicators = {
            "high": ["architecture", "refactor", "breaking change", "migration", "multiple", "complex", "system"],
            "medium": ["feature", "enhancement", "modify", "update", "improve", "several"],
            "low": ["fix", "simple", "minor", "typo", "documentation", "small"]
        }
        
        text = f"{issue.title} {issue.body}".lower()
        length_factor = len(issue.body) / 500  # Longer descriptions might indicate complexity
        comments_factor = len(issue.comments) / 10  # More discussion might indicate complexity
        
        complexity_score = 0
        for level, indicators in complexity_indicators.items():
            weight = {"high": 3, "medium": 2, "low": 1}[level]
            for indicator in indicators:
                complexity_score += text.count(indicator) * weight
        
        # Adjust based on length and discussion
        complexity_score += length_factor + comments_factor
        
        if complexity_score >= 6:
            level = "high"
        elif complexity_score >= 3:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": complexity_score,
            "factors": {
                "description_length": len(issue.body),
                "comments_count": len(issue.comments),
                "label_indicators": [label for label in issue.labels if any(label.lower() in indicators for indicators in complexity_indicators.values())]
            }
        }
    
    def _extract_technical_requirements(self, issue) -> Dict[str, Any]:
        """Extract technical requirements from issue description"""
        import re
        
        text = f"{issue.title} {issue.body}"
        
        # Look for technical patterns
        tech_patterns = {
            "programming_languages": re.findall(r'\b(python|javascript|java|go|rust|ruby|php|c\+\+|c#|swift|kotlin)\b', text, re.IGNORECASE),
            "frameworks": re.findall(r'\b(react|angular|vue|django|flask|spring|rails|laravel|express)\b', text, re.IGNORECASE),
            "databases": re.findall(r'\b(mysql|postgresql|mongodb|redis|sqlite|oracle)\b', text, re.IGNORECASE),
            "technologies": re.findall(r'\b(api|rest|graphql|websocket|docker|kubernetes|aws|azure|gcp)\b', text, re.IGNORECASE),
            "file_extensions": re.findall(r'\.(\w+)\b', text),
            "file_paths": re.findall(r'[/\w\.-]+\.(py|js|jsx|ts|tsx|java|go|rs|rb|php|html|css)', text),
            "error_messages": re.findall(r'error[:\s]+(.*?)(?:\n|$)', text, re.IGNORECASE),
            "version_numbers": re.findall(r'v?\d+\.\d+(?:\.\d+)?', text)
        }
        
        # Clean up and deduplicate
        requirements = {}
        for category, items in tech_patterns.items():
            if items:
                requirements[category] = list(set([item.lower() for item in items if len(item) > 1]))
        
        return requirements
    
    def _suggest_approach(self, issue) -> List[str]:
        """Suggest an approach for tackling the issue"""
        issue_type = self._classify_issue_type(issue)["primary_type"]
        complexity = self._assess_issue_complexity(issue)["level"]
        
        approaches = {
            "bug": [
                "1. Reproduce the issue locally",
                "2. Identify the root cause through debugging",
                "3. Write a test case to verify the fix",
                "4. Implement the fix",
                "5. Verify the fix doesn't break existing functionality"
            ],
            "feature": [
                "1. Break down the feature into smaller components",
                "2. Design the API/interface",
                "3. Implement core functionality",
                "4. Add comprehensive tests",
                "5. Update documentation"
            ],
            "documentation": [
                "1. Identify what needs to be documented",
                "2. Gather technical details",
                "3. Write clear, concise documentation",
                "4. Add examples where appropriate",
                "5. Review for accuracy and completeness"
            ]
        }
        
        base_approach = approaches.get(issue_type, approaches["feature"])
        
        # Modify based on complexity
        if complexity == "high":
            base_approach.insert(1, "1.5. Create a detailed design document")
            base_approach.append("6. Plan for gradual rollout/migration")
        elif complexity == "low":
            base_approach = [step for step in base_approach if "design" not in step.lower()]
        
        return base_approach
    
    def _estimate_effort(self, issue) -> Dict[str, Any]:
        """Estimate the effort required for the issue"""
        complexity = self._assess_issue_complexity(issue)
        issue_type = self._classify_issue_type(issue)["primary_type"]
        
        # Base estimates in hours
        base_estimates = {
            "bug": {"low": 2, "medium": 8, "high": 24},
            "feature": {"low": 8, "medium": 24, "high": 80},
            "documentation": {"low": 2, "medium": 6, "high": 16},
            "refactor": {"low": 4, "medium": 16, "high": 40},
            "test": {"low": 2, "medium": 8, "high": 20}
        }
        
        complexity_level = complexity["level"]
        base_hours = base_estimates.get(issue_type, base_estimates["feature"])[complexity_level]
        
        return {
            "estimated_hours": base_hours,
            "estimated_days": round(base_hours / 8, 1),
            "confidence": "medium" if complexity_level == "medium" else "low",
            "factors_considered": [
                f"Issue type: {issue_type}",
                f"Complexity: {complexity_level}",
                f"Description length: {len(issue.body)} chars",
                f"Comments: {len(issue.comments)}"
            ]
        }
    
    def _analyze_comments(self, comments) -> Dict[str, Any]:
        """Analyze comments for additional insights"""
        if not comments:
            return {}
        
        total_comments = len(comments)
        unique_users = len(set(comment.user for comment in comments))
        
        # Look for solution indicators in comments
        solution_indicators = ["fix", "solution", "resolve", "workaround", "patch"]
        potential_solutions = []
        
        for comment in comments:
            comment_text = comment.body.lower()
            if any(indicator in comment_text for indicator in solution_indicators):
                potential_solutions.append({
                    "user": comment.user,
                    "snippet": comment.body[:200] + "..." if len(comment.body) > 200 else comment.body
                })
        
        return {
            "total_comments": total_comments,
            "unique_participants": unique_users,
            "potential_solutions_mentioned": len(potential_solutions),
            "solution_snippets": potential_solutions[:3],  # Limit to top 3
            "discussion_level": "high" if total_comments > 10 else "medium" if total_comments > 3 else "low"
        }
    
    def _extract_issue_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords from issue text for searching"""
        import re
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those"}
        
        # Extract technical terms (camelCase, snake_case, file extensions, etc.)
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b|\b\w+_\w+\b|\b\w+\.\w+\b', text)
        
        # Extract quoted strings (often error messages or specific values)
        quoted_terms = re.findall(r'["\']([^"\']+)["\']', text)
        
        # Extract individual words, filter stop words
        words = re.findall(r'\b\w+\b', text.lower())
        significant_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Extract file paths and extensions
        file_patterns = re.findall(r'[/\w\.-]*\.(?:py|js|jsx|ts|tsx|java|go|rs|rb|php|html|css|json|yaml|yml|md|txt)', text)
        
        return {
            "primary": list(set(technical_terms + quoted_terms))[:10],  # Most important terms
            "contextual": list(set(significant_words))[:15],  # Supporting context
            "file_patterns": list(set(file_patterns)),
            "all_terms": list(set(technical_terms + quoted_terms + significant_words))
        }
    
    def _find_configuration_files(self, search_terms: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Find configuration files that might be relevant"""
        config_files = []
        config_patterns = [
            "package.json", "requirements.txt", "Gemfile", "go.mod", "Cargo.toml",
            "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml",
            "config.py", "settings.py", "config.js", "webpack.config.js",
            ".env", ".gitignore", "README.md"
        ]
        
        try:
            for pattern in config_patterns:
                for file_path in self.repo_path.rglob(pattern):
                    rel_path = str(file_path.relative_to(self.repo_path))
                    config_files.append({
                        "file": rel_path,
                        "relevance_score": 1,
                        "match_reason": "Configuration file",
                        "file_type": "configuration"
                    })
                    
        except Exception as e:
            logger.error(f"Error finding configuration files: {e}")
        
        return config_files[:5]  # Limit results
    
    def _find_related_test_files(self, source_files: List[str]) -> List[Dict[str, Any]]:
        """Find test files related to the source files"""
        test_files = []
        
        for source_file in source_files:
            # Common test file patterns
            base_name = Path(source_file).stem
            test_patterns = [
                f"test_{base_name}.py",
                f"{base_name}_test.py",
                f"{base_name}.test.js",
                f"{base_name}.spec.js",
                f"test{base_name.title()}.java"
            ]
            
            for pattern in test_patterns:
                try:
                    for test_path in self.repo_path.rglob(pattern):
                        rel_path = str(test_path.relative_to(self.repo_path))
                        test_files.append({
                            "file": rel_path,
                            "relevance_score": 2,
                            "match_reason": f"Test file for {source_file}",
                            "file_type": "test"
                        })
                except Exception:
                    continue
        
        return test_files
    
    def _categorize_files(self, files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize files by type"""
        categories = {
            "source_code": [],
            "tests": [],
            "configuration": [],
            "documentation": [],
            "other": []
        }
        
        for file_info in files:
            file_path = file_info["file"]
            file_ext = Path(file_path).suffix.lower()
            
            if any(test_indicator in file_path.lower() for test_indicator in ["test", "spec", "__test__"]):
                categories["tests"].append(file_path)
            elif file_ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php"]:
                categories["source_code"].append(file_path)
            elif file_ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".env"] or file_path.endswith("Dockerfile"):
                categories["configuration"].append(file_path)
            elif file_ext in [".md", ".txt", ".rst"]:
                categories["documentation"].append(file_path)
            else:
                categories["other"].append(file_path)
        
        return {k: v for k, v in categories.items() if v}  # Remove empty categories
    
    def _generate_file_recommendations(self, files: List[Dict[str, Any]], search_terms: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations for working with the found files"""
        recommendations = []
        
        if not files:
            return ["No relevant files found. Consider expanding search terms or checking if the repository structure matches the issue description."]
        
        # Analyze file distribution
        categories = self._categorize_files(files)
        
        if "source_code" in categories:
            recommendations.append(f"🔍 Start by examining {len(categories['source_code'])} source files: {', '.join(categories['source_code'][:3])}")
        
        if "tests" in categories:
            recommendations.append(f"🧪 Review {len(categories['tests'])} test files to understand expected behavior")
        
        if "configuration" in categories:
            recommendations.append(f"⚙️ Check {len(categories['configuration'])} configuration files for environment setup")
        
        # High relevance files
        high_relevance = [f for f in files if f.get("relevance_score", 0) > 5]
        if high_relevance:
            recommendations.append(f"⭐ Focus on high-relevance files: {', '.join([f['file'] for f in high_relevance[:3]])}")
        
        # Search term specific advice
        if search_terms.get("file_patterns"):
            recommendations.append(f"📁 Pay attention to files matching patterns: {', '.join(search_terms['file_patterns'][:3])}")
        
        return recommendations[:5]  # Limit recommendations

    def related_issues(
        self,
        query: Annotated[str, "Issue title, bug description, or error message to find similar past issues for"],
        k: Annotated[int, "Number of similar issues to return (default: 5)"] = 5,
        state: Annotated[str, "Issue state filter: 'open', 'closed', or 'all' (default: 'all')"] = "all",
        similarity: Annotated[float, "Minimum similarity threshold from 0.0 to 1.0 (default: 0.75)"] = 0.75
    ) -> str:
        """
        Find similar past GitHub issues in the current repository.
        
        This tool searches through the repository's issue history to find similar problems,
        bug reports, or feature requests that might provide context, solutions, or patches.
        
        Args:
            query: The issue description, bug report, or feature request to search for
            k: Number of similar issues to return (1-10)
            state: Filter by issue state ('open', 'closed', 'all')
            similarity: Minimum similarity score (0.0-1.0, higher = more similar)
        
        Returns:
            JSON string with similar issues including titles, URLs, states, and patch links
        """
        try:
            # Use the existing issue RAG instance instead of creating a new one
            if not self.issue_rag or not self.issue_rag.is_initialized():
                return json.dumps({
                    "error": "Issue RAG system not available or not initialized for this session",
                    "related_issues": [],
                    "suggestion": "The issue search system is still loading. Please try again in a moment."
                })
            
            # Use asyncio to run async function
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Get issue context using the existing, already-initialized instance
                issue_context = loop.run_until_complete(
                    self.issue_rag.get_issue_context(query, max_issues=k)
                )
                
                # Get repo info from the existing instance
                repo_info = self.issue_rag.indexer.repo_owner, self.issue_rag.indexer.repo_name
                owner, repo = repo_info
                
                # Format results
                results = {
                    "query": query,
                    "total_found": issue_context.total_found,
                    "processing_time": issue_context.processing_time,
                    "related_issues": []
                }
                
                for search_result in issue_context.related_issues:
                    issue = search_result.issue
                    issue_info = {
                        "number": issue.id,
                        "title": issue.title,
                        "state": issue.state,
                        "url": f"https://github.com/{owner}/{repo}/issues/{issue.id}",
                        "similarity": round(search_result.similarity, 3),
                        "labels": issue.labels,
                        "created_at": issue.created_at,
                        "body_preview": issue.body[:200] + "..." if len(issue.body) > 200 else issue.body,
                        "match_reasons": search_result.match_reasons
                    }
                    
                    # Add patch URL if available
                    if issue.patch_url:
                        issue_info["patch_url"] = issue.patch_url
                    
                    results["related_issues"].append(issue_info)
                
                return json.dumps(results, indent=2)
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in related_issues tool: {e}")
            return json.dumps({
                "error": f"Failed to search related issues: {str(e)}",
                "related_issues": []
            })

    def get_pr_for_issue(
        self,
        issue_number: Annotated[int, "The issue number to find the corresponding pull request for."]
    ) -> str:
        """
        Finds the pull request (PR) associated with a given issue number.
        This tool looks at the pre-built patch linkage data to find the PR that closed or fixed the issue.
        """
        if not self.issue_rag or not self.issue_rag.indexer.patch_builder:
            return json.dumps({"error": "Issue RAG system or PatchLinkageBuilder not available."})

        patch_builder = self.issue_rag.indexer.patch_builder
        links_by_issue = patch_builder.load_patch_links()

        if issue_number not in links_by_issue:
            return json.dumps({
                "error": f"No associated PR found for issue #{issue_number} in the patch linkage data."
            })

        pr_links = links_by_issue[issue_number]
        
        results = []
        for link in pr_links:
            results.append({
                "pr_number": link.pr_number,
                "pr_title": link.pr_title,
                "pr_url": link.pr_url,
                "merged_at": link.merged_at
            })
            
        return json.dumps({
            "issue_number": issue_number,
            "found_prs": results
        })

    def get_pr_diff(
        self,
        pr_number: Annotated[int, "The pull request number to get the diff for."]
    ) -> str:
        """
        Retrieves the diff for a given pull request (PR) number.
        This tool reads the cached diff file that was downloaded during the patch linkage build process.
        """
        if not self.issue_rag or not self.issue_rag.indexer.diff_docs:
            return json.dumps({"error": "Issue RAG system or diff documents not available."})

        diff_docs = self.issue_rag.indexer.diff_docs
        
        if pr_number not in diff_docs:
            return json.dumps({
                "error": f"No cached diff found for PR #{pr_number}. The diff may not have been downloaded or linked."
            })
            
        diff_doc = diff_docs[pr_number]
        
        diff_path = Path(diff_doc.diff_path)
        if not diff_path.exists():
            return json.dumps({
                "error": f"Diff file not found at {diff_doc.diff_path}, although metadata exists."
            })
            
        try:
            diff_text = diff_path.read_text(encoding='utf-8', errors='ignore')
            return json.dumps({
                "pr_number": pr_number,
                "issue_id": diff_doc.issue_id,
                "files_changed": diff_doc.files_changed,
                "diff_summary": diff_doc.diff_summary,
                "full_diff": diff_text
            })
        except Exception as e:
            return json.dumps({"error": f"Error reading diff file for PR #{pr_number}: {e}"})

    def get_files_changed_in_pr(
        self,
        pr_number: Annotated[int, "The pull request number to list changed files for."]
    ) -> str:
        """
        Lists all files that were modified, added, or deleted in a given pull request.
        This tool accesses the cached diff information.
        """
        if not self.issue_rag or not self.issue_rag.indexer.diff_docs:
            return json.dumps({"error": "Issue RAG system or diff documents not available."})

        diff_docs = self.issue_rag.indexer.diff_docs
        
        if pr_number not in diff_docs:
            return json.dumps({
                "error": f"No cached diff information found for PR #{pr_number}."
            })
            
        diff_doc = diff_docs[pr_number]
        
        return json.dumps({
            "pr_number": pr_number,
            "issue_id": diff_doc.issue_id,
            "files_changed": diff_doc.files_changed,
            "total_files_changed": len(diff_doc.files_changed)
        })

    def get_pr_summary(
        self,
        pr_number: Annotated[int, "The pull request number to summarize."]
    ) -> str:
        """
        Provides a concise summary of the changes made in a specific pull request, based on its diff.
        Uses an LLM to generate the summary.
        """
        if not self.issue_rag or not self.issue_rag.indexer.diff_docs:
            return json.dumps({"error": "Issue RAG system or diff documents not available."})

        diff_docs = self.issue_rag.indexer.diff_docs
        
        if pr_number not in diff_docs:
            return json.dumps({
                "error": f"No cached diff information found for PR #{pr_number}."
            })
            
        diff_doc = diff_docs[pr_number]
        diff_path = Path(diff_doc.diff_path)

        if not diff_path.exists():
            return json.dumps({
                "error": f"Diff file not found at {diff_doc.diff_path} for PR #{pr_number}, although metadata exists."
            })

        try:
            diff_text = diff_path.read_text(encoding='utf-8', errors='ignore')
            
            # Truncate diff if too long for the LLM prompt
            max_diff_length = 15000  # Max characters for the diff in the prompt
            if len(diff_text) > max_diff_length:
                diff_text_for_summary = diff_text[:max_diff_length] + "\n... [diff truncated for summary] ..."
            else:
                diff_text_for_summary = diff_text

            prompt = f"""Please summarize the following pull request diff. 
Focus on the main purpose of the changes, key files affected, and the overall impact.
Be concise and aim for a 2-3 sentence summary.

Diff for PR #{pr_number}:
---
{diff_text_for_summary}
---

Summary:"""

            # Use a smaller, faster LLM for summarization if possible, or the default agent LLM
            try:
                # Attempt to use a specific summarization model if configured, else default
                summary_llm = OpenRouter(
                    api_key=settings.openrouter_api_key,
                    model=settings.summarization_model or "mistralai/mistral-7b-instruct", # Default to a small model
                    max_tokens=200,
                    temperature=0.5
                )
                response = summary_llm.complete(prompt)
                summary = response.text.strip()
            except Exception as llm_error:
                logger.warning(f"Failed to use specific summarization LLM for PR #{pr_number} ({llm_error}), falling back to agent's LLM.")
                response = self.llm.complete(prompt) # Fallback to the main agent LLM
                summary = response.text.strip()
            
            return json.dumps({
                "pr_number": pr_number,
                "issue_id": diff_doc.issue_id,
                "summary": summary,
                "files_changed": diff_doc.files_changed,
                "diff_preview": diff_doc.diff_summary # The pre-extracted summary for quick view
            })

        except Exception as e:
            logger.error(f"Error summarizing PR #{pr_number}: {e}")
            return json.dumps({"error": f"Error summarizing PR #{pr_number}: {str(e)}"})

    def find_issues_related_to_file(
        self,
        file_path: Annotated[str, "The file path to find related issues for."]
    ) -> str:
        """
        Finds issues whose resolution involved changes to the specified file path.
        This tool uses the cached patch linkage and diff information.
        """
        if not self.issue_rag or not self.issue_rag.indexer.patch_builder or not self.issue_rag.indexer.diff_docs:
            return json.dumps({"error": "Issue RAG system, PatchLinkageBuilder, or diff documents not available."})

        patch_builder = self.issue_rag.indexer.patch_builder
        diff_docs = self.issue_rag.indexer.diff_docs
        
        # Normalize the input file path to be relative to the repo root and use forward slashes
        normalized_file_path = Path(file_path).as_posix()

        related_issues = {} # Use a dict to store unique issues with their PRs

        for pr_number, diff_doc in diff_docs.items():
            # Normalize paths in diff_doc.files_changed as well
            normalized_files_changed = [Path(f).as_posix() for f in diff_doc.files_changed]
            if normalized_file_path in normalized_files_changed:
                if diff_doc.issue_id: # Ensure there's an associated issue
                    if diff_doc.issue_id not in related_issues:
                        related_issues[diff_doc.issue_id] = {
                            "issue_number": diff_doc.issue_id,
                            "related_prs": []
                        }
                    related_issues[diff_doc.issue_id]["related_prs"].append({
                        "pr_number": pr_number,
                        "pr_title": diff_doc.pr_title if hasattr(diff_doc, 'pr_title') else f"PR #{pr_number}", # Assuming pr_title is on diff_doc
                        "pr_url": f"https://github.com/{patch_builder.repo_owner}/{patch_builder.repo_name}/pull/{pr_number}"
                    })
        
        if not related_issues:
            return json.dumps({
                "file_path": normalized_file_path,
                "message": "No issues found whose resolution directly involved changes to this file based on cached PR data."
            })

        return json.dumps({
            "file_path": normalized_file_path,
            "related_issues_count": len(related_issues),
            "issues": list(related_issues.values())
        })

    def get_issue_resolution_summary(
        self,
        issue_number: Annotated[int, "The issue number to get the resolution summary for."]
    ) -> str:
        """
        Summarizes how a specific issue was resolved, including linked PRs,
        and a summary of changes from those PRs.
        """
        if not self.issue_rag:
            return json.dumps({"error": "Issue RAG system not available."})

        pr_info_str = self.get_pr_for_issue(issue_number)
        pr_info = json.loads(pr_info_str)

        if "error" in pr_info or not pr_info.get("found_prs"):
            return json.dumps({
                "issue_number": issue_number,
                "summary": "No directly linked PRs found for this issue in the cached data. Resolution details are unavailable.",
                "linked_prs": []
            })

        resolution_details = []
        for pr_data in pr_info["found_prs"]:
            pr_num = pr_data["pr_number"]
            summary_str = self.get_pr_summary(pr_num)
            summary_data = json.loads(summary_str)
            
            detail = {
                "pr_number": pr_num,
                "pr_title": pr_data["pr_title"],
                "pr_url": pr_data["pr_url"],
                "merged_at": pr_data["merged_at"],
                "change_summary": summary_data.get("summary", "Summary not available."),
                "files_changed_count": len(summary_data.get("files_changed", []))
            }
            resolution_details.append(detail)
        
        overall_summary = f"Issue #{issue_number} was resolved by {len(resolution_details)} PR(s). "
        if resolution_details:
            overall_summary += f"Key changes involved PR #{resolution_details[0]['pr_number']} ('{resolution_details[0]['pr_title']}')."
            if len(resolution_details) > 1:
                overall_summary += f" Additional changes in other PRs."
        else: # Should not happen if found_prs was true, but as a safeguard
            overall_summary = f"Issue #{issue_number} has linked PRs, but details could not be fully summarized."


        return json.dumps({
            "issue_number": issue_number,
            "overall_summary": overall_summary,
            "linked_prs_details": resolution_details
        })

    def check_issue_status_and_linked_pr(
        self,
        issue_identifier: Annotated[str, "Issue number (#123) or full GitHub issue URL to check."]
    ) -> str:
        """
        Checks the current status (open/closed) of a GitHub issue and lists any directly linked Pull Requests.
        Combines issue analysis with PR linkage information.
        """
        # First, analyze the issue to get its current status and details
        issue_analysis_str = self.analyze_github_issue(issue_identifier)
        issue_analysis = json.loads(issue_analysis_str)

        if "error" in issue_analysis:
            # If issue analysis failed, return that error
            return issue_analysis_str
        
        issue_number = issue_analysis.get("issue_metadata", {}).get("number")
        if not issue_number:
            return json.dumps({
                "error": "Could not determine issue number from analysis.",
                "analysis_response": issue_analysis
            })

        # Then, get linked PRs for this issue number
        pr_info_str = self.get_pr_for_issue(issue_number)
        pr_info = json.loads(pr_info_str)

        # Combine the information
        combined_result = {
            "issue_number": issue_number,
            "issue_title": issue_analysis.get("issue_metadata", {}).get("title"),
            "current_status": issue_analysis.get("issue_metadata", {}).get("state"),
            "classification": issue_analysis.get("issue_classification", {}).get("primary_type"),
            "complexity": issue_analysis.get("complexity_assessment", {}).get("level"),
            "linked_prs": pr_info.get("found_prs", []) # Default to empty list if no PRs or error in pr_info
        }
        
        if "error" in pr_info and not combined_result["linked_prs"]: # Add PR error only if no PRs were found
            combined_result["pr_linkage_error"] = pr_info["error"]


        return json.dumps(combined_result, indent=2)

    def write_complete_code(
        self,
        description: Annotated[str, "Detailed description of what code to write"],
        context_files: Annotated[Optional[List[str]], "List of reference files to base the code on"] = None,
        language: Annotated[Optional[str], "Programming language (auto-detected if not specified)"] = None,
        output_format: Annotated[str, "Output format: 'markdown' or 'raw'"] = "markdown"
    ) -> str:
        """
        Write complete, untruncated code based on specifications and reference files.
        Produces properly formatted code in backticks for easy parsing and display.
        """
        try:
            logger.info(f"[DEBUG] Writing complete code for: {description}")
            
            # Detect language if not provided
            if not language and context_files:
                language = self._detect_primary_language_from_context(context_files)
            elif not language:
                language = "python"  # Default fallback
            
            # Read and analyze context files if provided
            context_content = ""
            patterns = {}
            if context_files:
                for file_path in context_files:
                    try:
                        full_path = self.repo_path / file_path
                        if full_path.exists():
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                context_content += f"\n\n=== {file_path} ===\n{content}"
                    except Exception as e:
                        logger.warning(f"Could not read context file {file_path}: {e}")
                
                # Extract patterns from context
                analysis = self._analyze_repository_context(context_files)
                patterns = {
                    "classes": [cls["name"] for cls in analysis.get("classes", [])],
                    "imports": analysis.get("imports", []),
                    "base_classes": analysis.get("base_classes", []),
                    "functions": analysis.get("functions", [])[:10]  # Limit to avoid token overflow
                }
            
            # Create comprehensive prompt for complete code generation
            prompt = self._build_complete_code_prompt(description, language, context_content, patterns, output_format)
            
            # Use LLM to generate complete code
            try:
                # Use LLMClient's OpenRouter LLM for proper API key handling
                from .llm_client import LLMClient
                llm_client = LLMClient()
                
                # Get the OpenRouter LLM instance with proper configuration
                openrouter_llm = llm_client._get_openrouter_llm("google/gemini-2.5-flash-preview-05-20")
                
                # Use the LLM directly for synchronous completion
                response = openrouter_llm.complete(prompt)
                code_response = response.text.strip()
                
                # Clean and format the response
                formatted_response = self._format_complete_code_response(
                    code_response, description, language, output_format
                )
                
                return formatted_response
                
            except Exception as e:
                logger.error(f"Error generating code: {e}")
                return self._generate_error_response(description, language, str(e))
                
        except Exception as e:
            logger.error(f"Error in write_complete_code: {e}")
            return f"Error writing code: {str(e)}"
    
    def _build_complete_code_prompt(
        self, 
        description: str, 
        language: str, 
        context_content: str, 
        patterns: Dict,
        output_format: str
    ) -> str:
        """Build a comprehensive prompt for complete code generation"""
        
        prompt = f"""You are an expert {language} developer. Write COMPLETE, PRODUCTION-READY code based on the following requirements.

REQUIREMENTS:
{description}

LANGUAGE: {language}

IMPORTANT INSTRUCTIONS:
1. Write COMPLETE, UNTRUNCATED code - never cut off or abbreviate
2. Include ALL necessary imports, classes, functions, and logic
3. Make the code production-ready and fully functional
4. Add comprehensive comments explaining key parts
5. Follow best practices for {language}
6. If multiple files are needed, provide them all
7. ALWAYS wrap code in proper markdown code blocks with language specification

"""

        # Add context if available
        if context_content:
            prompt += f"""
REFERENCE CODE CONTEXT:
{context_content[:8000]}  # Limit context to avoid token overflow

"""

        # Add patterns if available
        if patterns.get("classes"):
            prompt += f"""
AVAILABLE PATTERNS TO FOLLOW:
- Classes: {', '.join(patterns['classes'][:5])}
- Base classes: {', '.join(patterns['base_classes'][:3])}
- Common imports: {', '.join(patterns['imports'][:5])}

"""

        # Add language-specific guidance
        language_guidance = self._get_language_specific_guidance(language)
        prompt += f"""
{language_guidance}

OUTPUT FORMAT:
"""
        
        if output_format == "markdown":
            prompt += f"""
Provide the code in markdown format with proper code blocks:

```{language}
# Complete code here
```

Include explanations before and after the code blocks as needed.
"""
        else:
            prompt += """
Provide only the raw code without markdown formatting.
"""

        prompt += """
CRITICAL: Ensure the code is COMPLETE and FUNCTIONAL. Do not truncate or abbreviate any part.
"""

        return prompt
    
    def _get_language_specific_guidance(self, language: str) -> str:
        """Get language-specific coding guidance"""
        guidance_map = {
            "python": """
PYTHON BEST PRACTICES:
- Use proper imports and module structure
- Follow PEP 8 style guidelines
- Include docstrings for classes and functions
- Use type hints where appropriate
- Handle exceptions properly
- Use context managers for resources
""",
            "javascript": """
JAVASCRIPT BEST PRACTICES:
- Use ES6+ features appropriately
- Proper error handling with try/catch
- Use const/let instead of var
- Include JSDoc comments
- Follow async/await patterns
- Use proper module imports/exports
""",
            "typescript": """
TYPESCRIPT BEST PRACTICES:
- Define proper interfaces and types
- Use generic types where appropriate
- Include comprehensive type annotations
- Follow strict TypeScript configuration
- Use proper import/export syntax
- Handle null/undefined safely
""",
            "java": """
JAVA BEST PRACTICES:
- Follow Java naming conventions
- Use proper package structure
- Include Javadoc comments
- Handle exceptions appropriately
- Use generics and collections properly
- Follow SOLID principles
""",
            "go": """
GO BEST PRACTICES:
- Use proper package naming
- Follow Go formatting conventions
- Include proper error handling
- Use interfaces appropriately
- Follow Go concurrency patterns
- Include godoc comments
""",
            "rust": """
RUST BEST PRACTICES:
- Use proper ownership and borrowing
- Handle Result and Option types
- Include comprehensive error handling
- Use traits and generics appropriately
- Follow Rust naming conventions
- Include rustdoc comments
""",
            "csharp": """
C# BEST PRACTICES:
- Follow C# naming conventions
- Use proper namespace structure
- Include XML documentation
- Use async/await patterns
- Handle exceptions properly
- Use LINQ where appropriate
"""
        }
        
        return guidance_map.get(language, """
GENERAL BEST PRACTICES:
- Follow language conventions
- Include proper documentation
- Handle errors appropriately
- Use clear, descriptive names
- Follow established patterns
""")
    
    def _format_complete_code_response(
        self, 
        code_response: str, 
        description: str, 
        language: str, 
        output_format: str
    ) -> str:
        """Format the code response to ensure proper presentation"""
        
        if output_format == "raw":
            # For raw output, just return the LLM's response directly
            return code_response
        
        # For markdown output:
        
        def ensure_lang_in_block(match):
            opening_ticks = match.group(1) # ```
            lang_spec = match.group(2) # language specifier or empty string
            code_content = match.group(3) # content
            closing_ticks = match.group(4) # ```

            # If lang_spec is empty or just whitespace, replace with the default language
            if not lang_spec.strip():
                return f"{opening_ticks}{language}\n{code_content}{closing_ticks}"
            else:
                # Language is already specified, return as is
                return match.group(0)

        # Regex to find code blocks: ```optional_lang\n content ```
        # It captures the opening ```, the optional language, the content, and closing ```
        code_block_pattern = re.compile(r"(```)(.*?)\n(.*?)(```)", re.DOTALL)
        
        if code_block_pattern.search(code_response):
            # If code blocks are present, ensure they have language specifiers
            processed_response = code_block_pattern.sub(ensure_lang_in_block, code_response)
        else:
            # No ``` found. If the LLM was supposed to generate code,
            # wrap the entire response in a code block as a fallback.
            processed_response = f"# Code for: {description}\n\n```{language}\n{code_response}\n```"
            logger.warning("LLM did not use markdown code blocks for write_complete_code. Wrapping entire response.")

        # Add a general title if the response doesn't start with one (heuristic)
        # Check if the response (after potential wrapping) starts with a Markdown heading.
        # Use lstrip() to handle potential leading whitespace.
        if not processed_response.lstrip().startswith("#"):
             processed_response = f"# Generated Code: {description}\n\n{processed_response}"
        
        return processed_response
    
    def _generate_error_response(self, description: str, language: str, error: str) -> str:
        """Generate a helpful error response"""
        return f"""# Error: Could not generate code

**Description:** {description}
**Language:** {language}
**Error:** {error}

Please try:
1. Simplifying the description
2. Providing more specific requirements
3. Including relevant context files with @ syntax

Example: `@agents.py write a news aggregation agent`
"""
