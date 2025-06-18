"""
Enhanced Persistence System for RAG Indices
Provides robust index persistence with file change detection and incremental updates
"""

import os
import json
import hashlib
import time
import logging

from pathlib import Path
from datetime import datetime
import faiss
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class FileChecksum:
    """File metadata for change detection"""
    path: str
    size: int
    mtime: float
    checksum: str
    
    def __post_init__(self):
        """Ensure path is always relative"""
        if os.path.isabs(self.path):
            # Convert to relative path if absolute
            self.path = os.path.relpath(self.path)

@dataclass 
class IndexMetadata:
    """Metadata for index persistence"""
    repo_url: str
    branch: str
    owner: str
    repo: str
    total_files: int
    total_nodes: int
    created_at: str
    last_updated: str
    index_version: str
    embedding_model: str
    file_checksums: Dict[str, FileChecksum]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert FileChecksum objects to dicts
        result['file_checksums'] = {
            path: asdict(checksum) for path, checksum in self.file_checksums.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary"""
        # Convert file_checksums back to FileChecksum objects
        file_checksums = {
            path: FileChecksum(**checksum_data) 
            for path, checksum_data in data.get('file_checksums', {}).items()
        }
        data['file_checksums'] = file_checksums
        return cls(**data)

class EnhancedPersistenceManager:
    """Enhanced persistence manager with change detection and incremental updates"""
    
    def __init__(self, base_dir: str = ".index_cache"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def get_index_dir(self, repo_url: str, branch: str = "main") -> Path:
        """Get persistent index directory for a repository"""
        # Create a safe directory name from repo URL
        repo_hash = hashlib.md5(f"{repo_url}#{branch}".encode()).hexdigest()[:12]
        url_parts = repo_url.replace(".git", "").split('/')
        if len(url_parts) >= 2:
            safe_name = f"{url_parts[-2]}_{url_parts[-1]}_{branch}_{repo_hash}"
        else:
            safe_name = f"repo_{repo_hash}"
        
        index_dir = self.base_dir / "repositories" / safe_name
        index_dir.mkdir(parents=True, exist_ok=True)
        return index_dir
    
    def calculate_file_checksum(self, file_path: Path) -> FileChecksum:
        """Calculate checksum for a file"""
        try:
            stat = file_path.stat()
            
            # For large files, use size + mtime for speed
            if stat.st_size > 1024 * 1024:  # 1MB
                checksum = hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()
            else:
                # For smaller files, use actual content hash
                with open(file_path, 'rb') as f:
                    checksum = hashlib.md5(f.read()).hexdigest()
            
            return FileChecksum(
                path=str(file_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                checksum=checksum
            )
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return FileChecksum(
                path=str(file_path),
                size=0,
                mtime=0,
                checksum=""
            )
    
    def scan_repository_files(self, repo_path: Path, extensions: List[str]) -> Dict[str, FileChecksum]:
        """Scan repository and calculate checksums for all relevant files"""
        file_checksums = {}
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                if extensions and file_path.suffix not in extensions:
                    continue
                
                # Make path relative to repo root
                try:
                    rel_path = file_path.relative_to(repo_path)
                    checksum = self.calculate_file_checksum(file_path)
                    checksum.path = str(rel_path)  # Store as relative path
                    file_checksums[str(rel_path)] = checksum
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
        
        return file_checksums
    
    def detect_changes(self, 
                      old_metadata: IndexMetadata, 
                      repo_path: Path, 
                      extensions: List[str]) -> Dict[str, Any]:
        """Detect what files have changed since last indexing"""
        logger.info(f"Detecting changes in {repo_path}")
        
        # Get current file state
        current_checksums = self.scan_repository_files(repo_path, extensions)
        old_checksums = old_metadata.file_checksums
        
        # Find changes
        added_files = set(current_checksums.keys()) - set(old_checksums.keys())
        removed_files = set(old_checksums.keys()) - set(current_checksums.keys())
        
        modified_files = set()
        for path in set(current_checksums.keys()) & set(old_checksums.keys()):
            if current_checksums[path].checksum != old_checksums[path].checksum:
                modified_files.add(path)
        
        changes = {
            "added_files": list(added_files),
            "removed_files": list(removed_files), 
            "modified_files": list(modified_files),
            "total_changes": len(added_files) + len(removed_files) + len(modified_files),
            "current_checksums": current_checksums
        }
        
        logger.info(f"Changes detected: {changes['total_changes']} total "
                   f"({len(added_files)} added, {len(removed_files)} removed, {len(modified_files)} modified)")
        
        return changes
    
    def should_rebuild_index(self, 
                           old_metadata: IndexMetadata, 
                           repo_path: Path, 
                           extensions: List[str],
                           force_rebuild: bool = False) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if index should be rebuilt and return change details"""
        if force_rebuild:
            return True, {"reason": "force_rebuild", "total_changes": -1}
        
        # Check if embedding model changed
        current_model = "text-embedding-3-small"  # Your current model
        if old_metadata.embedding_model != current_model:
            return True, {"reason": "embedding_model_changed", "total_changes": -1}
        
        # Check file changes
        changes = self.detect_changes(old_metadata, repo_path, extensions)
        
        # Rebuild if significant changes (>10% of files or >50 files)
        total_files = len(old_metadata.file_checksums)
        change_threshold = max(50, total_files * 0.1)  # 10% or 50 files, whichever is larger
        
        if changes["total_changes"] > change_threshold:
            changes["reason"] = "significant_changes"
            return True, changes
        elif changes["total_changes"] > 0:
            changes["reason"] = "minor_changes"  
            return False, changes  # Could do incremental update
        else:
            changes["reason"] = "no_changes"
            return False, changes
    
    def save_metadata(self, index_dir: Path, metadata: IndexMetadata) -> None:
        """Save index metadata"""
        metadata_file = index_dir / "enhanced_metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved enhanced metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_metadata(self, index_dir: Path) -> Optional[IndexMetadata]:
        """Load index metadata"""
        metadata_file = index_dir / "enhanced_metadata.json"
        if not metadata_file.exists():
            logger.info(f"No enhanced metadata found at {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return IndexMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
            return None
    
    def validate_index_integrity(self, index_dir: Path) -> bool:
        """Validate that all required index files exist and are valid"""
        required_files = [
            "enhanced_metadata.json",
            "docstore.json", 
            "index_store.json",
            "default__vector_store.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = index_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
            elif file_path.stat().st_size == 0:
                missing_files.append(f"{file_name} (empty)")
        
        if missing_files:
            logger.warning(f"Index integrity check failed. Missing/empty files: {missing_files}")
            return False
        
        # Additional validation: check if docstore has reasonable content
        try:
            docstore_file = index_dir / "docstore.json"
            with open(docstore_file, 'r') as f:
                docstore_data = json.load(f)
            
            if not docstore_data.get("docstore/data", {}):
                logger.warning("Docstore appears to be empty")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to validate docstore: {e}")
            return False
        
        logger.info("Index integrity check passed")
        return True
    
    def cleanup_corrupted_index(self, index_dir: Path) -> None:
        """Clean up corrupted index files"""
        logger.info(f"Cleaning up corrupted index at {index_dir}")
        
        files_to_remove = [
            "enhanced_metadata.json",
            "docstore.json",
            "index_store.json", 
            "default__vector_store.json",
            "graph_store.json"
        ]
        
        for file_name in files_to_remove:
            file_path = index_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Removed corrupted file: {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_name}: {e}")

    def detect_issue_changes(self, 
                           old_metadata: Dict[str, Any], 
                           repo_owner: str, 
                           repo_name: str) -> Dict[str, Any]:
        """Detect changes in GitHub issues and PRs since last sync"""
        logger.info(f"Detecting issue/PR changes for {repo_owner}/{repo_name}")
        
        changes = {
            "new_issues": [],
            "updated_issues": [],
            "new_prs": [],
            "updated_prs": [],
            "total_changes": 0,
            "last_issue_sync": old_metadata.get("last_issue_sync"),
            "last_pr_sync": old_metadata.get("last_pr_sync"),
            "current_sync": datetime.now().isoformat()
        }
        
        return changes
    
    def should_sync_issues(self, 
                          old_metadata: Dict[str, Any], 
                          force_sync: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Determine if issues/PRs should be synced"""
        if force_sync:
            return True, {"reason": "force_sync", "total_changes": -1}
        
        # Check if we have never synced before
        if not old_metadata.get("last_issue_sync"):
            return True, {"reason": "never_synced", "total_changes": -1}
        
        # Check time since last sync (sync if >24 hours)
        try:
            last_sync = datetime.fromisoformat(old_metadata["last_issue_sync"])
            time_since_sync = datetime.now() - last_sync
            
            if time_since_sync.total_seconds() > 24 * 3600:  # 24 hours
                return True, {"reason": "time_threshold", "hours_since_sync": time_since_sync.total_seconds() / 3600}
            else:
                return False, {"reason": "recent_sync", "hours_since_sync": time_since_sync.total_seconds() / 3600}
                
        except Exception as e:
            logger.warning(f"Error parsing last sync time: {e}")
            return True, {"reason": "invalid_sync_time", "total_changes": -1}
    
    def save_sync_metadata(self, 
                          index_dir: Path, 
                          repo_owner: str, 
                          repo_name: str,
                          issues_synced: int = 0,
                          prs_synced: int = 0) -> None:
        """Save sync metadata for incremental updates"""
        sync_metadata_file = index_dir / "sync_metadata.json"
        
        try:
            # Load existing sync metadata if it exists
            existing_metadata = {}
            if sync_metadata_file.exists():
                with open(sync_metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # Update with current sync info
            sync_metadata = {
                **existing_metadata,
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "last_full_sync": datetime.now().isoformat(),
                "last_issue_sync": datetime.now().isoformat(),
                "last_pr_sync": datetime.now().isoformat(),
                "issues_synced": issues_synced,
                "prs_synced": prs_synced,
                "sync_count": existing_metadata.get("sync_count", 0) + 1
            }
            
            with open(sync_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(sync_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved sync metadata: {issues_synced} issues, {prs_synced} PRs")
            
        except Exception as e:
            logger.error(f"Failed to save sync metadata: {e}")
    
    def load_sync_metadata(self, index_dir: Path) -> Dict[str, Any]:
        """Load sync metadata for incremental updates"""
        sync_metadata_file = index_dir / "sync_metadata.json"
        
        if not sync_metadata_file.exists():
            return {}
        
        try:
            with open(sync_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load sync metadata: {e}")
            return {}

# Global instance
persistence_manager = EnhancedPersistenceManager() 