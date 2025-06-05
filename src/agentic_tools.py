"""
Agentic Tools Implementation using LlamaIndex
Provides directory exploration, codebase search, and file analysis capabilities
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Annotated
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
    
    def __init__(self, session_id: str, repo_path: str):
        self.session_id = session_id
        self.repo_path = Path(repo_path)
        
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
        """Create all the tools for the agent"""
        tools = []
        
        # Directory exploration tool
        explore_dir_tool = FunctionTool.from_defaults(
            fn=self.explore_directory,
            name="explore_directory",
            description="Explore the contents of a directory, showing files and subdirectories with metadata"
        )
        tools.append(explore_dir_tool)
        
        # Codebase search tool
        search_tool = FunctionTool.from_defaults(
            fn=self.search_codebase,
            name="search_codebase", 
            description="Search through the entire codebase for specific terms, patterns, or concepts"
        )
        tools.append(search_tool)
        
        # File reading tool
        read_file_tool = FunctionTool.from_defaults(
            fn=self.read_file,
            name="read_file",
            description="Read and analyze the complete contents of a specific file"
        )
        tools.append(read_file_tool)
        
        # File analysis tool
        analyze_file_tool = FunctionTool.from_defaults(
            fn=self.analyze_file_structure,
            name="analyze_file_structure",
            description="Analyze file structure, dependencies, and relationships in the codebase"
        )
        tools.append(analyze_file_tool)
        
        # Find related files tool
        find_related_tool = FunctionTool.from_defaults(
            fn=self.find_related_files,
            name="find_related_files",
            description="Find files related to a given file based on imports, references, or naming patterns"
        )
        tools.append(find_related_tool)
        
        # Simple content search tool (replaces RAG for now)
        content_search_tool = FunctionTool.from_defaults(
            fn=self.semantic_content_search,
            name="semantic_content_search",
            description="Search for content semantically across files using keyword and context matching"
        )
        tools.append(content_search_tool)
        
        # Code generation tool
        code_gen_tool = FunctionTool.from_defaults(
            fn=self.generate_code_example,
            name="generate_code_example",
            description="Generate practical, runnable code examples based on the codebase analysis. Creates complete working examples that users can execute."
        )
        tools.append(code_gen_tool)
        
        # GitHub Issue Analysis tool
        issue_analysis_tool = FunctionTool.from_defaults(
            fn=self.analyze_github_issue,
            name="analyze_github_issue",
            description="Analyze a GitHub issue comprehensively, providing insights on complexity, type, requirements, and suggested approach"
        )
        tools.append(issue_analysis_tool)
        
        # Find Issue Related Files tool
        find_issue_files_tool = FunctionTool.from_defaults(
            fn=self.find_issue_related_files,
            name="find_issue_related_files",
            description="Find files in the repository that are relevant to a specific GitHub issue using intelligent search patterns"
        )
        tools.append(find_issue_files_tool)
        
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
        """Read complete file contents"""
        try:
            full_path = self.repo_path / file_path
            
            if not full_path.exists() or not full_path.is_file():
                return f"File {file_path} does not exist or is not a file"
            
            # Check file size to avoid reading huge files
            stat = full_path.stat()
            if stat.st_size > 100000:  # 100KB limit
                return f"File {file_path} is too large ({stat.st_size} bytes). Use explore_directory for file info."
            
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
                    if stat.st_size > 100000:  # 100KB limit
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
                logger.info(f"[DEBUG] Yielding step {i}: type={step['type']}, content_len={len(step['content'])}")
                yield json.dumps({"type": "step", "step": step})
                await asyncio.sleep(0.01)  # Small delay for streaming effect

            # Yield a final event with all steps, final answer, and suggestions
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
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[stream] Error in agentic stream_query: {error_msg}")
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
        """Clean captured output to only include ReAct steps with proper formatting"""
        if not captured_output:
            return ""
        
        # First, remove all ANSI color codes and terminal formatting
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_output = ansi_escape.sub('', captured_output)
        
        # Also remove common escape sequences
        cleaned_output = re.sub(r'\033\[[0-9;]*m', '', cleaned_output)
        cleaned_output = re.sub(r'\[0m', '', cleaned_output)
        cleaned_output = re.sub(r'\[1;3;[0-9]+m', '', cleaned_output)
        
        lines = cleaned_output.split('\n')
        cleaned_lines = []
        current_step_type = None
        current_step_content = []
        skip_json_block = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip common logging/debug patterns
            skip_patterns = [
                'Running step',
                'Step input:',
                'INFO:httpx:',
                'HTTP Request:',
                'DEBUG:',
                'INFO:',
                'WARNING:',
                'ERROR:',
                'Step new_user_msg produced event',
                'Step prepare_chat_history produced event',
                'Step handle_llm_input produced event',
                '⚡',  # Remove lightning emoji
                'action',  # Remove standalone 'action' text
                'Observation:',  # Remove duplicate observation markers
            ]
            
            # Check if this line should be skipped
            should_skip = False
            for pattern in skip_patterns:
                if pattern in line:
                    should_skip = True
                    break
            
            if should_skip:
                continue
            
            # Skip lines that look like terminal formatting artifacts
            if re.match(r'^[\[\]0-9;m\s]+$', line):
                continue
            
            # Detect start of JSON blocks and skip them
            if line.startswith('{') or ('{' in line and '"' in line):
                skip_json_block = True
                brace_count = line.count('{') - line.count('}')
                continue
            
            # Continue skipping JSON content
            if skip_json_block:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    skip_json_block = False
                continue
            
            # Detect ReAct control tokens (case insensitive)
            line_lower = line.lower()
            if line_lower.startswith('thought:'):
                # Save previous step if exists
                if current_step_type and current_step_content:
                    cleaned_lines.append(f"{current_step_type}: {' '.join(current_step_content)}")
                current_step_type = "Thought"
                current_step_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                
            elif line_lower.startswith('action:'):
                # Save previous step if exists
                if current_step_type and current_step_content:
                    cleaned_lines.append(f"{current_step_type}: {' '.join(current_step_content)}")
                current_step_type = "Action"
                current_step_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                
            elif line_lower.startswith('action input:'):
                # Skip Action Input as it's usually verbose JSON
                continue
                
            elif line_lower.startswith('observation:'):
                # Save previous step if exists
                if current_step_type and current_step_content:
                    cleaned_lines.append(f"{current_step_type}: {' '.join(current_step_content)}")
                current_step_type = "Observation"
                # For observations, just add a summary instead of full content
                current_step_content = ["Tool executed successfully with results."]
                
            elif line_lower.startswith('answer:'):
                # Save previous step if exists
                if current_step_type and current_step_content:
                    cleaned_lines.append(f"{current_step_type}: {' '.join(current_step_content)}")
                current_step_type = "Answer"
                current_step_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                
            elif current_step_type and not skip_json_block:
                # Continue accumulating content for current step, but filter out JSON-like content
                if not (line.startswith('{') or line.startswith('}') or '"' in line and ':' in line):
                    # Clean the line of any remaining artifacts
                    cleaned_line = re.sub(r'[\[\]0-9;m]', '', line).strip()
                    if cleaned_line:  # Only add non-empty lines
                        # For Answer steps, don't limit length as users expect full responses
                        if current_step_type == "Answer":
                            current_step_content.append(cleaned_line)
                        # For other steps, use a more generous limit
                        elif len(' '.join(current_step_content + [cleaned_line])) < 2000:
                            current_step_content.append(cleaned_line)
        
        # Add final step if present
        if current_step_type and current_step_content:
            cleaned_lines.append(f"{current_step_type}: {' '.join(current_step_content)}")
        
        return '\n'.join(cleaned_lines)
    
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