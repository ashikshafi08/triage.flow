# src/agent_tools/file_operations.py

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated, AsyncGenerator

# Assuming settings and ChunkStoreFactory are accessible.
# If they are part of the main AgenticCodebaseExplorer instance, they'll be passed in.
# For now, direct import for settings, chunk_store will be passed.
try:
    from ..config import settings
except ImportError:
    # Fallback for isolated testing or different structure
    class MockSettings:
        MAX_AGENTIC_FILE_SIZE_BYTES = 1024 * 1024  # 1MB
        ENABLE_DYNAMIC_CONTENT = True
        CONTENT_CHUNK_SIZE = 1024 * 4 # 4KB
        MAX_CHUNKS_PER_REQUEST = 100
        STREAM_CHUNK_SIZE = 1024 * 8 # 8KB
    settings = MockSettings()

from .utilities import extract_functions, extract_classes # Moved from agentic_tools.py

logger = logging.getLogger(__name__)

class FileOperations:
    def __init__(self, repo_path: Path, chunk_store_instance: Optional[Any]):
        self.repo_path = repo_path
        self.chunk_store = chunk_store_instance # For _chunk_large_output, if used by these methods

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
            for item in sorted(full_path.iterdir()):
                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime, # Consider converting to ISO format string
                        "path": str(item.relative_to(self.repo_path))
                    }
                    if item.is_file():
                        item_info["extension"] = item.suffix
                        if stat.st_size < 1000: # Small file preview
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
            
            total_files = len([i for i in items if i["type"] == "file"])
            total_dirs = len([i for i in items if i["type"] == "directory"])
            
            result = {
                "directory": directory_path or "root",
                "summary": f"{total_files} files, {total_dirs} directories",
                "items": items[:50] 
            }
            if len(items) > 50:
                result["note"] = f"Showing first 50 items out of {len(items)} total"
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error exploring directory {directory_path}: {e}")
            return f"Error exploring directory: {str(e)}"

    def read_file(
        self, 
        file_path: Annotated[str, "Path to the file to read, relative to repository root"]
    ) -> str:
        """Read complete file contents with dynamic chunking"""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists() or not full_path.is_file():
                return f"File {file_path} does not exist or is not a file"
            
            stat = full_path.stat()
            if stat.st_size > settings.MAX_AGENTIC_FILE_SIZE_BYTES and settings.ENABLE_DYNAMIC_CONTENT:
                return self._read_large_file(full_path, file_path)
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return json.dumps({
                "file": file_path, "size": stat.st_size,
                "lines": len(content.split('\n')), "content": content
            }, indent=2)
        except UnicodeDecodeError:
            return f"File {file_path} appears to be binary and cannot be read as text"
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

    def _read_large_file(self, file_path_obj: Path, relative_path: str) -> str:
        """Read large files in chunks with smart content handling (helper method)"""
        try:
            chunks = []
            total_lines = 0
            current_chunk_lines = []
            current_chunk_size_bytes = 0
            
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                for line_text in f:
                    total_lines += 1
                    line_size_bytes = len(line_text.encode('utf-8'))
                    
                    if current_chunk_size_bytes + line_size_bytes > settings.CONTENT_CHUNK_SIZE:
                        if current_chunk_lines: # Ensure chunk is not empty
                           chunks.append(''.join(current_chunk_lines))
                        current_chunk_lines = []
                        current_chunk_size_bytes = 0
                        if len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST:
                            break
                    
                    current_chunk_lines.append(line_text)
                    current_chunk_size_bytes += line_size_bytes
            
            if current_chunk_lines: # Add final chunk
                chunks.append(''.join(current_chunk_lines))
            
            truncated_note = ""
            if len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST and total_lines > sum(len(c.split('\n')) for c in chunks):
                 truncated_note = f"\n\nNote: File was truncated after {settings.MAX_CHUNKS_PER_REQUEST} chunks. Total lines in file: {total_lines}"

            return json.dumps({
                "file": relative_path, "size": file_path_obj.stat().st_size,
                "total_lines": total_lines, "chunks_returned": len(chunks),
                "content_is_chunked": True, "content": chunks, # 'content' now holds list of chunks
                "truncated": len(chunks) >= settings.MAX_CHUNKS_PER_REQUEST,
                "note": truncated_note
            }, indent=2)
        except Exception as e:
            logger.error(f"Error reading large file {file_path_obj}: {e}")
            return f"Error reading large file: {str(e)}"

    async def stream_large_file(self, file_path: str) -> AsyncGenerator[str, None]:
        """Stream large file content in chunks"""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists() or not full_path.is_file():
                yield json.dumps({"error": f"File {file_path} does not exist or is not a file"})
                return
            
            stat = full_path.stat()
            yield json.dumps({
                "type": "metadata", "file": file_path, "size": stat.st_size,
                "chunk_size_bytes": settings.STREAM_CHUNK_SIZE
            })
            
            with open(full_path, 'r', encoding='utf-8') as f:
                buffer_lines = []
                current_size_bytes = 0
                for line_text in f:
                    line_size_bytes = len(line_text.encode('utf-8'))
                    if current_size_bytes + line_size_bytes > settings.STREAM_CHUNK_SIZE:
                        if buffer_lines:
                            yield json.dumps({"type": "chunk", "content": ''.join(buffer_lines)})
                        buffer_lines = []
                        current_size_bytes = 0
                    buffer_lines.append(line_text)
                    current_size_bytes += line_size_bytes
                if buffer_lines: # Send final buffer
                    yield json.dumps({"type": "chunk", "content": ''.join(buffer_lines)})
            yield json.dumps({"type": "complete", "message": "File streaming completed"})
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {e}")
            yield json.dumps({"type": "error", "error": str(e)})
    
    def analyze_file_structure(
        self, 
        target_path: Annotated[str, "Path to analyze - can be file or directory"] = ""
    ) -> str:
        """Analyze file structure and relationships"""
        try:
            full_path = self.repo_path / target_path if target_path else self.repo_path
            if not full_path.exists():
                return f"Path {target_path} does not exist"
            
            analysis = {"path": target_path or "root", "type": "directory" if full_path.is_dir() else "file"}
            
            if full_path.is_file():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                analysis.update({
                    "size": len(content), "lines": len(content.split('\n')),
                    "extension": full_path.suffix,
                    "functions": extract_functions(content, full_path.suffix), # Using imported utility
                    "classes": extract_classes(content, full_path.suffix)     # Using imported utility
                })
            else: # Directory analysis
                files_by_type = {}
                total_size = 0
                for item_path in full_path.rglob("*"):
                    if item_path.is_file():
                        ext = item_path.suffix or "no_extension"
                        files_by_type.setdefault(ext, {"count": 0, "total_size": 0})
                        try:
                            size = item_path.stat().st_size
                            files_by_type[ext]["count"] += 1
                            files_by_type[ext]["total_size"] += size
                            total_size += size
                        except Exception: continue # Skip files we can't stat
                analysis.update({
                    "total_size": total_size, "files_by_type": files_by_type,
                    "structure_summary": f"Contains {len(files_by_type)} different file types"
                })
            return json.dumps(analysis, indent=2)
        except Exception as e:
            logger.error(f"Error analyzing file structure for {target_path}: {e}")
            return f"Error analyzing structure: {str(e)}"
