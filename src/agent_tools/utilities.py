# src/agent_tools/utilities.py

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Assuming 'settings' might be needed by some utilities, e.g., _chunk_large_output
# For now, let's assume it's passed if needed or accessed via a global config.
# from ..config import settings # If needed globally
# from ..chunk_store import ChunkStoreFactory # If needed globally

logger = logging.getLogger(__name__)

def get_current_head_sha(repo_path: Path) -> Optional[str]:
    """Get the current HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting current HEAD SHA: {e}")
        return None

def extract_repo_info(repo_path: Path) -> tuple[str, str]:
    """Extract repository owner and name from various sources."""
    try:
        # Method 1: Try to get from git remote
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
            import re
            github_match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', remote_url)
            if github_match:
                owner, name = github_match.groups()
                name = name.replace('.git', '')
                return owner, name
        
        # Method 2: Try to extract from README or other metadata files
        for readme_file in ['README.md', 'readme.md', 'README.txt', 'package.json']:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8')
                    import re
                    github_match = re.search(r'github\.com/([^/]+)/([^/\s)]+)', content)
                    if github_match:
                        owner, name = github_match.groups()
                        name = name.replace('.git', '')
                        return owner, name
                except Exception:
                    continue
        
        return "unknown", "unknown"
    except Exception as e:
        logger.warning(f"Error extracting repo info: {e}")
        return "unknown", "unknown"

@lru_cache(maxsize=4096)
def blame_line_cached(repo_path: Path, file_path: str, line_number: int) -> Optional[Dict[str, Any]]:
    """Cached version of blame line for performance."""
    try:
        abs_file_path = repo_path / file_path # Ensure file_path is relative to repo_path
        if not abs_file_path.exists():
            logger.warning(f"File not found for blame: {abs_file_path}")
            return None
        
        blame_cmd = ["git", "blame", "-w", "-C", "-L", f"{line_number},{line_number}", str(abs_file_path)]
        result = subprocess.run(blame_cmd, capture_output=True, text=True, cwd=repo_path)
        
        if result.returncode != 0:
            logger.warning(f"Git blame failed for {file_path}:{line_number}: {result.stderr}")
            return None
        
        blame_line = result.stdout.strip()
        if not blame_line:
            return None
        
        parts = blame_line.split(')', 1)
        if len(parts) != 2:
            return None
        
        commit_info_part = parts[0] + ')'
        code = parts[1].strip()
        
        commit_hash_match = re.match(r'^([0-9a-fA-F]+)', commit_info_part)
        commit_hash = commit_hash_match.group(1) if commit_hash_match else "unknown_hash"

        author_date_match = re.search(r'\((.*?)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s+\d+\)', commit_info_part)
        if author_date_match:
            author = author_date_match.group(1).strip()
            date = author_date_match.group(2).strip()
        else:
            author = "unknown_author"
            date = "unknown_date"
            
        return {
            "commit": commit_hash,
            "author": author,
            "date": date,
            "code": code
        }
    except Exception as e:
        logger.error(f"Error in blame_line_cached for {file_path}:{line_number}: {e}")
        return None

def chunk_large_output(content: str, chunk_store_instance: Optional[Any] = None, chunk_size: int = 8192, preview_size: int = 1000) -> str:
    """Handle large outputs by chunking and storing in cache if available."""
    try:
        if chunk_store_instance and hasattr(chunk_store_instance, 'store'):
            if len(content) > chunk_size: # Use passed chunk_size
                chunk_id = chunk_store_instance.store(content)
                return json.dumps({
                    "type": "chunked",
                    "chunk_id": chunk_id,
                    "preview": content[:preview_size] + "...",
                    "total_size": len(content),
                    "message": "Response was too large and has been chunked. Use the chunk_id to retrieve the full content."
                })
        
        # Fallback if no chunk_store or content is not large enough
        return content
            
    except Exception as e:
        logger.warning(f"Error chunking output: {e}")
        if len(content) > preview_size:
            return json.dumps({
                "type": "truncated",
                "preview": content[:preview_size] + "...",
                "total_size": len(content),
                "error": "Could not chunk large response",
                "message": str(e)
            })
        return content

def extract_functions(content: str, file_extension: str) -> List[str]:
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

def extract_classes(content: str, file_extension: str) -> List[str]:
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

def get_repo_url_from_path(repo_path: Path) -> Optional[str]:
    """Try to determine the repository URL from the repo path."""
    try:
        git_dir = repo_path / '.git'
        if git_dir.exists() and git_dir.is_dir():
            config_file = git_dir / 'config'
            if config_file.exists() and config_file.is_file():
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                import re # Import re here as it's used locally
                match = re.search(r'url = (https://github\.com/[^/]+/[^/\s]+)', content)
                if match:
                    return match.group(1).rstrip('.git')
        return None
    except Exception as e:
        logger.warning(f"Could not determine repo URL from path {repo_path}: {e}")
        return None
