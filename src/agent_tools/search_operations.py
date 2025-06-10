# src/agent_tools/search_operations.py

import os
import json
import logging
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Annotated

# Assuming settings might be needed for MAX_AGENTIC_FILE_SIZE_BYTES
try:
    from ..config import settings
except ImportError:
    class MockSettings:
        MAX_AGENTIC_FILE_SIZE_BYTES = 1024 * 1024  # 1MB
    settings = MockSettings()

logger = logging.getLogger(__name__)

class SearchOperations:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def search_codebase(
        self, 
        query: Annotated[str, "Search query - can be code patterns, function names, or concepts"],
        file_types: Annotated[Optional[List[str]], "File extensions to search (e.g., ['.py', '.js']). None for all files"] = None
    ) -> str:
        """Search through codebase files"""
        try:
            results = []
            search_count = 0 # Counts files added to results
            processed_file_count = 0 # Counts files actually read and processed
            max_results_files = 30 # Max files to include in results
            
            if file_types is None:
                file_types = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt']
            
            for file_path_obj in self.repo_path.rglob("*"):
                if not file_path_obj.is_file():
                    continue
                
                processed_file_count +=1
                if file_types and file_path_obj.suffix not in file_types:
                    continue
                
                try:
                    # Skip very large files early
                    if file_path_obj.stat().st_size > settings.MAX_AGENTIC_FILE_SIZE_BYTES * 2: # A bit more lenient for search
                        continue

                    with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    if query.lower() in content.lower():
                        if search_count >= max_results_files: # Check before adding to results
                            break 

                        lines = content.split('\n')
                        matches_in_file = []
                        for i, line_text in enumerate(lines):
                            if query.lower() in line_text.lower():
                                start = max(0, i - 3)
                                end = min(len(lines), i + 4)
                                context = '\n'.join(lines[start:end])
                                matches_in_file.append({
                                    "line_number": i + 1,
                                    "line": line_text.strip(),
                                    "context": context
                                })
                                if len(matches_in_file) >= 3:  # Limit matches per file
                                    break
                        
                        if matches_in_file:
                            results.append({
                                "file": str(file_path_obj.relative_to(self.repo_path)),
                                "matches": matches_in_file
                            })
                            search_count += 1
                except Exception: # Skip files that can't be read or processed
                    continue
            
            return json.dumps({
                "query": query,
                "files_with_matches": search_count,
                "total_files_processed": processed_file_count,
                "results": results
            }, indent=2)
        except Exception as e:
            logger.error(f"Error searching codebase: {e}")
            return f"Error searching codebase: {str(e)}"

    def find_related_files(
        self, 
        file_path_str: Annotated[str, "The file path to find related files for"]
    ) -> str:
        """Find files related to a given file"""
        try:
            target_full_path = self.repo_path / file_path_str
            if not target_full_path.exists() or not target_full_path.is_file():
                return f"Target file does not exist: {file_path_str}"
            
            related_files_set = set()
            file_stem = target_full_path.stem
            file_dir = target_full_path.parent

            for item_path in self.repo_path.rglob("*"):
                if not item_path.is_file() or item_path.name.startswith('.'):
                    continue
                
                # Skip the original file itself
                if item_path.resolve() == target_full_path.resolve():
                    continue

                rel_path_str = str(item_path.relative_to(self.repo_path))

                # Check for similar names or same directory
                if (file_stem in item_path.stem or 
                    item_path.stem in file_stem or
                    item_path.parent == file_dir):
                    related_files_set.add(rel_path_str)

            # Check for imports/references by reading the target file
            try:
                with open(target_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                import_patterns = [
                    r'from\s+[\.\w]+\s+import', r'import\s+[\.\w]+', # Python
                    r'require\(["\']([^"\']+)["\']', # JS/TS CJS
                    r'import\s+.*\s+from\s+["\']([^"\']+)["\']', # JS/TS ESM
                    r'#include\s*[<"]([^>"]+)[>"]' # C/C++
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    for match_group in matches:
                        # Handle cases where re.findall returns tuples for multiple capture groups
                        match = match_group if isinstance(match_group, str) else match_group[-1]
                        
                        # Basic attempt to resolve module to file path
                        # This is highly language/project specific and may need improvement
                        potential_paths_variations = [
                            match,
                            match.replace('.', '/'),
                            f"{match.replace('.', '/')}.py",
                            f"{match.replace('.', '/')}.js",
                            f"{match.replace('.', '/')}.ts",
                            # Add more common extensions or logic as needed
                        ]
                        for pp_var in potential_paths_variations:
                            # Check relative to file_dir first, then repo_path
                            if (file_dir / pp_var).exists():
                                related_files_set.add(str((file_dir / pp_var).relative_to(self.repo_path)))
                            elif (self.repo_path / pp_var).exists():
                                 related_files_set.add(pp_var)
            except Exception as e:
                logger.warning(f"Could not read or parse imports for {file_path_str}: {e}")
            
            final_related_files = list(related_files_set)
            return json.dumps({
                "original_file": file_path_str,
                "related_files": final_related_files[:20],
                "total_found": len(final_related_files)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error finding related files for {file_path_str}: {e}")
            return f"Error finding related files: {str(e)}"

    def semantic_content_search(
        self,
        query: Annotated[str, "Search query for content across files"]
    ) -> str:
        """Search for content semantically across files"""
        try:
            results = []
            query_terms = query.lower().split()
            
            searchable_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml', '.md', '.txt', '.config', '.sh', '.java', '.cs', '.cpp', '.c', '.h', '.html', '.css']

            for file_path_obj in self.repo_path.rglob("*"):
                if not file_path_obj.is_file():
                    continue
                
                try:
                    if file_path_obj.stat().st_size > settings.MAX_AGENTIC_FILE_SIZE_BYTES:
                        continue
                    if file_path_obj.suffix.lower() not in searchable_extensions:
                        continue
                    
                    with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    content_lower = content.lower()
                    score = 0
                    match_details = []
                    
                    for term in query_terms:
                        term_count = content_lower.count(term)
                        if term_count > 0:
                            score += term_count
                            match_details.append(f"{term}: {term_count}x")
                    
                    if query.lower() in content_lower: # Exact phrase bonus
                        score += 10 * content_lower.count(query.lower())
                        match_details.append(f"exact_phrase: {content_lower.count(query.lower())}x")
                    
                    if score > 0:
                        context_snippets = []
                        lines = content.split('\n')
                        for i, line_text in enumerate(lines):
                            if any(term in line_text.lower() for term in query_terms) or query.lower() in line_text.lower():
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = '\n'.join(lines[start:end])
                                context_snippets.append(snippet)
                                if len(context_snippets) >= 2: # Limit snippets per file
                                    break
                        
                        results.append({
                            "file": str(file_path_obj.relative_to(self.repo_path)),
                            "score": score,
                            "match_details": match_details,
                            "context_snippets": context_snippets,
                            "size_bytes": len(content.encode('utf-8'))
                        })
                except Exception:
                    continue 
        
            results.sort(key=lambda x: x['score'], reverse=True)
            return json.dumps({
                "query": query,
                "total_results_found": len(results),
                "results_shown": len(results[:15]),
                "results": results[:15]
            }, indent=2)
        except Exception as e:
            logger.error(f"Error in semantic content search: {e}")
            return f"Error in semantic content search: {str(e)}"
