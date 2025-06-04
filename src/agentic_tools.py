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

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai import OpenAI

from .config import settings

logger = logging.getLogger(__name__)

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
            # Detect the primary language(s) in the repository
            language_info = self._detect_repository_languages()
            primary_language = language_info.get("primary_language", "python")
            all_languages = language_info.get("languages", ["python"])
            
            # Generate language-appropriate code example
            code_example = self._generate_language_specific_code(description, primary_language, context_files)
            
            example_data = {
                "description": description,
                "generated_at": "Generated by AgenticCodebaseExplorer",
                "detected_languages": all_languages,
                "primary_language": primary_language,
                "example_type": "code_snippet",
                "code": code_example,
                "usage_instructions": self._get_language_usage_instructions(primary_language),
                "notes": f"This is a working {primary_language} example based on your repository structure. Customize based on your specific needs."
            }
            
            # If context files provided, include them in the metadata
            if context_files:
                example_data["based_on_files"] = context_files
                # Try to extract patterns from context files
                patterns = self._extract_patterns_from_files(context_files)
                if patterns:
                    example_data["extracted_patterns"] = patterns
            
            return json.dumps(example_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error generating code example: {e}")
            return f"Error generating code example: {str(e)}"

    def _detect_repository_languages(self) -> Dict[str, Any]:
        """Detect the primary programming languages in the repository"""
        try:
            language_counts = {}
            file_extensions = {
                # Web Technologies
                '.js': 'javascript',
                '.jsx': 'javascript', 
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.html': 'html',
                '.css': 'css',
                '.scss': 'scss',
                '.sass': 'sass',
                '.vue': 'vue',
                '.svelte': 'svelte',
                
                # Backend Languages
                '.py': 'python',
                '.java': 'java',
                '.kt': 'kotlin',
                '.scala': 'scala',
                '.go': 'go',
                '.rs': 'rust',
                '.rb': 'ruby',
                '.php': 'php',
                '.cs': 'csharp',
                '.fs': 'fsharp',
                '.vb': 'vb.net',
                
                # Systems Programming
                '.c': 'c',
                '.cpp': 'cpp',
                '.cc': 'cpp',
                '.cxx': 'cpp',
                '.h': 'c',
                '.hpp': 'cpp',
                '.swift': 'swift',
                '.m': 'objective-c',
                '.mm': 'objective-c',
                
                # Functional Languages
                '.hs': 'haskell',
                '.elm': 'elm',
                '.clj': 'clojure',
                '.cljs': 'clojure',
                '.ml': 'ocaml',
                '.ex': 'elixir',
                '.exs': 'elixir',
                
                # Data & Config
                '.sql': 'sql',
                '.sh': 'bash',
                '.ps1': 'powershell',
                '.r': 'r',
                '.jl': 'julia',
                '.dart': 'dart',
                '.lua': 'lua',
                
                # Config files
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.toml': 'toml',
                '.xml': 'xml',
            }
            
            # Count files by language
            total_files = 0
            for file_path in self.repo_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    ext = file_path.suffix.lower()
                    if ext in file_extensions:
                        language = file_extensions[ext]
                        language_counts[language] = language_counts.get(language, 0) + 1
                        total_files += 1
            
            if not language_counts:
                return {"primary_language": "python", "languages": ["python"], "confidence": "low"}
            
            # Sort by count to find primary language
            sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
            primary_language = sorted_languages[0][0]
            all_languages = [lang for lang, count in sorted_languages if count > 0]
            
            # Calculate confidence based on distribution
            primary_count = sorted_languages[0][1]
            confidence = "high" if primary_count > total_files * 0.6 else "medium" if primary_count > total_files * 0.3 else "low"
            
            return {
                "primary_language": primary_language,
                "languages": all_languages[:5],  # Top 5 languages
                "language_distribution": dict(sorted_languages[:10]),
                "confidence": confidence,
                "total_files": total_files
            }
            
        except Exception as e:
            logger.error(f"Error detecting languages: {e}")
            return {"primary_language": "python", "languages": ["python"], "confidence": "low"}

    def _generate_language_specific_code(self, description: str, language: str, context_files: Optional[List[str]] = None) -> str:
        """Generate code example in the specified language"""
        
        # Language-specific templates
        templates = {
            "python": f'''# {description}
# Python implementation

def example_function():
    """
    {description}
    
    This function demonstrates the requested functionality.
    """
    print(f"Example implementation for: {description}")
    return "success"

if __name__ == "__main__":
    result = example_function()
    print(f"Result: {{result}}")''',

            "javascript": f'''// {description}
// JavaScript implementation

function exampleFunction() {{
    /**
     * {description}
     * 
     * This function demonstrates the requested functionality.
     */
    console.log(`Example implementation for: {description}`);
    return "success";
}}

// Usage example
const result = exampleFunction();
console.log(`Result: ${{result}}`);''',

            "typescript": f'''// {description}
// TypeScript implementation

function exampleFunction(): string {{
    /**
     * {description}
     * 
     * This function demonstrates the requested functionality.
     */
    console.log(`Example implementation for: {description}`);
    return "success";
}}

// Usage example
const result: string = exampleFunction();
console.log(`Result: ${{result}}`);''',

            "java": f'''// {description}
// Java implementation

public class ExampleClass {{
    
    /**
     * {description}
     * 
     * This method demonstrates the requested functionality.
     */
    public static String exampleMethod() {{
        System.out.println("Example implementation for: {description}");
        return "success";
    }}
    
    public static void main(String[] args) {{
        String result = exampleMethod();
        System.out.println("Result: " + result);
    }}
}}''',

            "go": f'''// {description}
// Go implementation

package main

import "fmt"

// exampleFunction demonstrates the requested functionality
func exampleFunction() string {{
    fmt.Printf("Example implementation for: {description}\\n")
    return "success"
}}

func main() {{
    result := exampleFunction()
    fmt.Printf("Result: %s\\n", result)
}}''',

            "rust": f'''// {description}
// Rust implementation

fn example_function() -> String {{
    println!("Example implementation for: {description}");
    "success".to_string()
}}

fn main() {{
    let result = example_function();
    println!("Result: {{}}", result);
}}''',

            "csharp": f'''// {description}
// C# implementation

using System;

public class ExampleClass 
{{
    /// <summary>
    /// {description}
    /// 
    /// This method demonstrates the requested functionality.
    /// </summary>
    public static string ExampleMethod() 
    {{
        Console.WriteLine($"Example implementation for: {description}");
        return "success";
    }}
    
    public static void Main() 
    {{
        string result = ExampleMethod();
        Console.WriteLine($"Result: {{result}}");
    }}
}}''',

            "ruby": f'''# {description}
# Ruby implementation

def example_method
  # {description}
  # 
  # This method demonstrates the requested functionality.
  puts "Example implementation for: {description}"
  "success"
end

# Usage example
result = example_method
puts "Result: #{{result}}"''',

            "php": f'''<?php
// {description}
// PHP implementation

/**
 * {description}
 * 
 * This function demonstrates the requested functionality.
 */
function exampleFunction() {{
    echo "Example implementation for: {description}\\n";
    return "success";
}}

// Usage example
$result = exampleFunction();
echo "Result: $result\\n";
?>''',

            "swift": f'''// {description}
// Swift implementation

import Foundation

func exampleFunction() -> String {{
    print("Example implementation for: {description}")
    return "success"
}}

// Usage example
let result = exampleFunction()
print("Result: \\(result)")''',

            "kotlin": f'''// {description}
// Kotlin implementation

fun exampleFunction(): String {{
    println("Example implementation for: {description}")
    return "success"
}}

fun main() {{
    val result = exampleFunction()
    println("Result: $result")
}}''',
        }
        
        # Get template or default to python
        template = templates.get(language, templates["python"])
        
        # If we have context files, try to incorporate patterns
        if context_files:
            # This could be enhanced to analyze the context files and
            # generate more sophisticated examples based on existing patterns
            template += f"\n\n// Based on analysis of: {', '.join(context_files[:3])}"
        
        return template

    def _get_language_usage_instructions(self, language: str) -> List[str]:
        """Get language-specific usage instructions"""
        instructions = {
            "python": [
                "Save this code to a .py file",
                "Run with: python filename.py",
                "Install dependencies with: pip install <package>"
            ],
            "javascript": [
                "Save this code to a .js file", 
                "Run with: node filename.js",
                "Install dependencies with: npm install <package>"
            ],
            "typescript": [
                "Save this code to a .ts file",
                "Compile with: tsc filename.ts",
                "Run with: node filename.js",
                "Install dependencies with: npm install <package>"
            ],
            "java": [
                "Save this code to ExampleClass.java",
                "Compile with: javac ExampleClass.java", 
                "Run with: java ExampleClass"
            ],
            "go": [
                "Save this code to main.go",
                "Run with: go run main.go",
                "Build with: go build"
            ],
            "rust": [
                "Save this code to main.rs",
                "Run with: cargo run",
                "Or compile with: rustc main.rs && ./main"
            ],
            "csharp": [
                "Save this code to Program.cs",
                "Run with: dotnet run",
                "Or compile with: csc Program.cs"
            ],
            "ruby": [
                "Save this code to example.rb",
                "Run with: ruby example.rb",
                "Install gems with: gem install <gem>"
            ],
            "php": [
                "Save this code to example.php",
                "Run with: php example.php"
            ],
            "swift": [
                "Save this code to example.swift",
                "Run with: swift example.swift"
            ],
            "kotlin": [
                "Save this code to Example.kt",
                "Compile with: kotlinc Example.kt -include-runtime -d example.jar",
                "Run with: java -jar example.jar"
            ]
        }
        
        return instructions.get(language, [
            f"Save this code to a {language} source file",
            f"Follow {language} compilation/execution procedures",
            "Install any required dependencies"
        ])

    def _extract_patterns_from_files(self, context_files: List[str]) -> Dict[str, Any]:
        """Extract patterns from context files to inform code generation"""
        try:
            patterns = {
                "imports": [],
                "functions": [],
                "classes": [],
                "common_patterns": []
            }
            
            for file_path in context_files[:3]:  # Limit to first 3 files
                try:
                    full_path = self.repo_path / file_path
                    if not full_path.exists():
                        continue
                        
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract imports/includes
                    lines = content.split('\n')
                    for line in lines[:20]:  # Check first 20 lines for imports
                        line = line.strip()
                        if (line.startswith('import ') or 
                            line.startswith('from ') or 
                            line.startswith('#include') or
                            line.startswith('const ') and 'require(' in line):
                            patterns["imports"].append(line)
                    
                    # Extract function/method patterns
                    ext = full_path.suffix
                    patterns["functions"].extend(self._extract_functions(content, ext))
                    patterns["classes"].extend(self._extract_classes(content, ext))
                    
                except Exception:
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return {}
    
    async def query(self, user_message: str) -> str:
        """Main query method that uses the agent to respond"""
        try:
            logger.info(f"Starting agentic analysis: {user_message[:100]}...")
            
            # Use the ReAct agent to process the query
            response = await self.agent.achat(user_message)
            
            logger.info(f"Agentic analysis completed successfully")
            
            # Extract clean final answer from ReAct agent response
            response_text = str(response)
            
            # Clean up the response to remove ReAct framework artifacts
            cleaned_response = self._extract_clean_answer(response_text)
            
            return cleaned_response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in agentic query: {error_msg}")
            
            # If it's a max iterations error, provide natural continuation instead of exposing limits
            if "max iterations" in error_msg.lower() or "reached max iterations" in error_msg.lower():
                logger.info("Converting iteration limit to natural continuation prompt")
                
                # Try to extract what was accomplished from memory
                try:
                    chat_history = self.memory.get_all()
                    
                    if chat_history:
                        # Get the last meaningful response
                        assistant_messages = [msg for msg in chat_history if hasattr(msg, 'role') and msg.role == 'assistant']
                        
                        if assistant_messages and len(assistant_messages) > 0:
                            last_response = assistant_messages[-1].content
                            
                            # Clean and return partial results with natural continuation
                            cleaned_partial = self._extract_clean_answer(last_response)
                            
                            return f"""{cleaned_partial}

---

**I've made good progress exploring your codebase! Would you like me to:**

🔍 **Dive deeper** into specific files or modules  
📁 **Explore** a particular directory in more detail  
🏗️ **Analyze** the architecture and relationships  
💡 **Explain** how specific components work  

*Just ask about what interests you most, and I'll focus my analysis there.*"""
                
                except Exception as memory_error:
                    logger.error(f"Error accessing agent memory: {memory_error}")
                
                # Fallback - provide natural exploration suggestions
                return self._get_natural_exploration_suggestions(user_message)
            
            # For other errors, provide helpful guidance without technical details
            return self._get_natural_error_recovery(user_message, error_msg)
    
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