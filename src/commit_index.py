"""
Commit Metadata Index Module
Lightweight commit-meta index that complements the existing PR-diff RAG system.
Stores commit metadata for semantic search over commit messages while deferring diff embeddings.
"""

import os
import json
import asyncio
import logging
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import re

import faiss
import numpy as np
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from tqdm.auto import tqdm

from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class CommitMeta:
    """Represents commit metadata for lightweight indexing"""
    sha: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    commit_date: str
    author_date: str
    subject: str
    body: str
    files_changed: List[str]
    files_added: List[str]
    files_modified: List[str] 
    files_deleted: List[str]
    insertions: int
    deletions: int
    is_merge: bool
    parent_shas: List[str]
    branch_info: Optional[str] = None
    pr_number: Optional[int] = None  # Extracted from commit message if available

@dataclass
class CommitSearchResult:
    """Represents a commit search result with relevance scoring"""
    commit: CommitMeta
    similarity: float
    match_reasons: List[str]
    file_relevance: float = 0.0  # For file-specific searches

class CommitIndexer:
    """Handles indexing and storage of Git commit metadata"""
    
    def __init__(self, repo_path: str, repo_owner: str = None, repo_name: str = None):
        self.repo_path = Path(repo_path)
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_key = f"{repo_owner}/{repo_name}" if repo_owner and repo_name else "unknown"
        
        logger.info(f"CommitIndexer __init__: Received repo_path='{repo_path}', owner='{repo_owner}', name='{repo_name}', resulting repo_key='{self.repo_key}'")

        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key
        )
        
        # Setup storage paths using a persistent base directory
        safe_repo_key = self.repo_key.replace('/', '_')
        # Use a fixed base path within the project's CWD for persistence
        persistent_base_path = Path(".") / ".index_cache" / "commit_indexes"
        self.index_dir = persistent_base_path / safe_repo_key
        self.index_dir.mkdir(parents=True, exist_ok=True) # Ensure parent dirs are created
        
        logger.info(f"CommitIndexer: Using persistent index directory: {self.index_dir.resolve()}")
        
        self.commits_file = self.index_dir / "commits.jsonl"
        self.metadata_file = self.index_dir / "metadata.json"
        self.file_stats_file = self.index_dir / "file_stats.json"
        # self.faiss_index_file = self.index_dir / "index.faiss" # Not directly used for LlamaIndex StorageContext persistence

        # Initialize index components
        # self.faiss_index = None # Not directly managed
        self.vector_index = None
        self.commit_metas = {}  # sha -> CommitMeta
        self.bm25_retriever = None
        self.file_touch_stats = {}  # file_path -> {touches, authors, commits}
        
    async def build_commit_index(
        self, 
        max_commits: Optional[int] = None,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        force_rebuild: bool = False
    ) -> None:
        """Build commit metadata index from git history"""
        
        max_commits = max_commits or getattr(settings, 'MAX_COMMITS_TO_PROCESS', 5000)
        
        logger.info(f"Building commit index for {self.repo_path} (max_commits={max_commits}, force_rebuild={force_rebuild})")
        
        if force_rebuild:
            logger.info(f"Force rebuilding commit index - clearing existing cache in {self.index_dir}")
            self._clear_cache()
        elif not force_rebuild and await self.load_existing_index():
            logger.info("Using existing commit index")
            return
        
        # Ensure we have complete git history for commit indexing
        from .local_repo_loader import unshallow_repository
        unshallow_repository(str(self.repo_path))
        
        # Extract commit metadata from git
        commits = await self._extract_commit_metadata(max_commits, since_date, until_date)
        logger.info(f"Extracted {len(commits)} commits")
        
        if not commits:
            logger.warning("No commits found to index")
            return
        
        # Build file touch statistics
        await self._build_file_statistics(commits)
        
        # Verify file statistics were built
        if self.file_touch_stats:
            logger.info(f"✅ Successfully built file statistics for {len(self.file_touch_stats)} files")
        else:
            logger.warning("⚠️ No file statistics were built - this may indicate an issue with git log parsing")
        
        # Create searchable documents for embedding
        documents = self._create_commit_documents(commits)
        
        # Build FAISS index
        await self._build_faiss_index(documents)
        
        # Build BM25 index for keyword search
        await self._build_bm25_index(documents)
        
        # Save non-vector-store components
        await self._save_commits(commits)
        await self._save_file_statistics()
        await self._save_metadata(len(commits))
        
        logger.info(f"Commit index built successfully with {len(commits)} commits")
        
        # Log the contents of the index directory after saving
        try:
            if self.index_dir.exists() and self.index_dir.is_dir():
                dir_contents = [str(p.name) for p in self.index_dir.iterdir()]
                logger.info(f"CommitIndexer: Contents of index directory '{self.index_dir.resolve()}' after build: {dir_contents}")
            else:
                logger.warning(f"CommitIndexer: Index directory '{self.index_dir.resolve()}' does not exist after build.")
        except Exception as e_log:
            logger.error(f"CommitIndexer: Error listing contents of index directory: {e_log}")

    def _clear_cache(self) -> None:
        """Clear existing cache files"""
        import shutil
        if self.index_dir.exists():
            try:
                shutil.rmtree(self.index_dir)
                logger.info(f"Cleared cache directory: {self.index_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear cache directory {self.index_dir}: {e}")
        self.index_dir.mkdir(exist_ok=True)
    
    async def _extract_commit_metadata(
        self, 
        max_commits: int,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None
    ) -> List[CommitMeta]:
        """Extract commit metadata using git log"""
        
        # Use the improved batch approach
        return await self._extract_with_batch_approach(max_commits, since_date, until_date)
    
    async def _extract_with_batch_approach(
        self, 
        max_commits: int,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None
    ) -> List[CommitMeta]:
        """Extract commits using a more reliable batch approach"""
        
        logger.info("Using batch approach for commit extraction")
        
        # First, get all commit SHAs
        cmd = ["git", "rev-list", "--all", f"-{max_commits}"]
        
        if since_date:
            cmd.extend(["--since", since_date])
        if until_date:
            cmd.extend(["--until", until_date])
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=self.repo_path,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git rev-list failed: {result.stderr}")
        
        shas = result.stdout.strip().split('\n')
        shas = [sha for sha in shas if sha.strip()]
        
        logger.info(f"Found {len(shas)} commits to process")
        
        commits = []
        
        # Process commits in batches
        batch_size = 50
        for i in tqdm(range(0, len(shas), batch_size), desc="Processing commit batches"):
            batch_shas = shas[i:i + batch_size]
            batch_commits = await self._process_commit_batch(batch_shas)
            commits.extend(batch_commits)
            
            # Log progress for first few batches
            if i < 100:
                files_count = sum(len(c.files_changed) for c in batch_commits)
                logger.debug(f"Batch {i//batch_size + 1}: Processed {len(batch_commits)} commits with {files_count} total file changes")
        
        return commits
    
    async def _process_commit_batch(self, shas: List[str]) -> List[CommitMeta]:
        """Process a batch of commits"""
        commits = []
        
        for sha in shas:
            # Get commit details with a custom format
            cmd = [
                "git", "show",
                "--pretty=format:%H%n%an%n%ae%n%cn%n%ce%n%ci%n%ai%n%P%n%s%n%b%nEND_MESSAGE",
                "--name-status",
                sha
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to get details for commit {sha[:8]}: {result.stderr}")
                continue
            
            commit = self._parse_single_commit_output(result.stdout)
            if commit:
                commits.append(commit)
                self.commit_metas[sha] = commit
        
        return commits
    
    def _parse_single_commit_output(self, output: str) -> Optional[CommitMeta]:
        """Parse output from git show for a single commit"""
        lines = output.split('\n')
        
        if len(lines) < 9:
            return None
        
        try:
            # Parse commit metadata
            sha = lines[0].strip()
            author_name = lines[1].strip()
            author_email = lines[2].strip()
            committer_name = lines[3].strip()
            committer_email = lines[4].strip()
            commit_date = lines[5].strip()
            author_date = lines[6].strip()
            parent_shas = lines[7].strip().split() if lines[7].strip() else []
            subject = lines[8].strip()
            
            # Parse body (everything until END_MESSAGE)
            body_lines = []
            i = 9
            while i < len(lines) and lines[i] != "END_MESSAGE":
                body_lines.append(lines[i])
                i += 1
            body = '\n'.join(body_lines).strip()
            
            # Parse file changes (everything after END_MESSAGE)
            files_changed = []
            files_added = []
            files_modified = []
            files_deleted = []
            
            # Skip the END_MESSAGE line and any empty lines
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Now parse file changes
            while i < len(lines):
                line = lines[i].strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        status, file_path = parts
                        # Clean the file path
                        file_path = file_path.strip()
                        if file_path:
                            files_changed.append(file_path)
                            
                            if status.startswith('A'):
                                files_added.append(file_path)
                            elif status.startswith('M'):
                                files_modified.append(file_path)
                            elif status.startswith('D'):
                                files_deleted.append(file_path)
                            elif status.startswith('R'):
                                # Handle renames (R100, etc.)
                                if ' ' in file_path:
                                    # Format: old_name -> new_name
                                    old_name, new_name = file_path.split(' -> ', 1)
                                    files_changed[-1] = new_name  # Replace with new name
                                    files_deleted.append(old_name)
                                    files_added.append(new_name)
                i += 1
            
            # Get stats for insertions/deletions
            stats_cmd = ["git", "show", "--stat", "--format=", sha]
            stats_result = subprocess.run(
                stats_cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=2
            )
            
            insertions = 0
            deletions = 0
            
            if stats_result.returncode == 0:
                for line in stats_result.stdout.strip().split('\n'):
                    if 'insertion' in line:
                        match = re.search(r'(\d+) insertion', line)
                        if match:
                            insertions = int(match.group(1))
                    if 'deletion' in line:
                        match = re.search(r'(\d+) deletion', line)
                        if match:
                            deletions = int(match.group(1))
            
            # Debug logging for first few commits
            if len(self.commit_metas) < 5:
                logger.debug(f"Commit {sha[:8]}: {len(files_changed)} files changed")
                if files_changed:
                    logger.debug(f"  Sample files: {files_changed[:3]}")
            
            return CommitMeta(
                sha=sha,
                author_name=author_name,
                author_email=author_email,
                committer_name=committer_name,
                committer_email=committer_email,
                commit_date=commit_date,
                author_date=author_date,
                subject=subject,
                body=body,
                files_changed=files_changed,
                files_added=files_added,
                files_modified=files_modified,
                files_deleted=files_deleted,
                insertions=insertions,
                deletions=deletions,
                is_merge=len(parent_shas) > 1,
                parent_shas=parent_shas,
                pr_number=self._extract_pr_number(subject + " " + body)
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse commit output: {e}")
            return None
    
    async def _extract_with_complex_format(
        self, 
        max_commits: int,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None
    ) -> List[CommitMeta]:
        """Extract using complex git log format (deprecated - keeping for reference)"""
        
        # This method had issues with parsing, replaced with batch approach
        raise NotImplementedError("Use _extract_with_batch_approach instead")
    
    async def _extract_with_simple_format(
        self, 
        max_commits: int,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None
    ) -> List[CommitMeta]:
        """Extract using simple git log format (deprecated - keeping for reference)"""
        
        # This method was inefficient, replaced with batch approach
        raise NotImplementedError("Use _extract_with_batch_approach instead")
    
    def _parse_git_log_output(self, output: str) -> List[CommitMeta]:
        """Parse git log output into CommitMeta objects (deprecated)"""
        # This method had parsing issues, replaced with _parse_single_commit_output
        raise NotImplementedError("Use _parse_single_commit_output instead")
    
    def _is_merge_commit(self, sha: str) -> bool:
        """Check if commit is a merge commit"""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--parents", "-n", "1", sha],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            if result.returncode == 0:
                # If there are more than 2 space-separated values, it's a merge
                return len(result.stdout.strip().split()) > 2
        except Exception:
            pass
        return False
    
    def _get_parent_commits(self, sha: str) -> List[str]:
        """Get parent commit SHAs"""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--parents", "-n", "1", sha],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                return parts[1:] if len(parts) > 1 else []
        except Exception:
            pass
        return []
    
    def _extract_pr_number(self, text: str) -> Optional[int]:
        """Extract PR number from commit message"""
        # Common patterns for PR references
        patterns = [
            r'#(\d+)',  # #1234
            r'pull request #(\d+)',
            r'PR #(\d+)',
            r'merge pull request #(\d+)',
            r'\(#(\d+)\)'  # (#1234)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    async def _enrich_commit_stats(self, commit: CommitMeta) -> None:
        """Add insertion/deletion statistics to commit (no longer needed with new approach)"""
        # Stats are now gathered in _parse_single_commit_output
        pass
    
    async def _build_file_statistics(self, commits: List[CommitMeta]) -> None:
        """Build file touch statistics for timeline/heat-map features"""
        self.file_touch_stats = {}
        
        logger.info(f"Building file statistics from {len(commits)} commits")
        
        # Count files across all commits for debugging
        total_files_across_commits = 0
        
        for commit in commits:
            if not commit.files_changed:
                continue
            
            total_files_across_commits += len(commit.files_changed)
                
            for file_path in commit.files_changed:
                # Skip empty or invalid file paths
                if not file_path or file_path.strip() == "":
                    continue
                    
                if file_path not in self.file_touch_stats:
                    self.file_touch_stats[file_path] = {
                        "touch_count": 0,
                        "authors": set(),
                        "commits": [],
                        "first_seen": commit.commit_date,
                        "last_seen": commit.commit_date,
                        "additions": 0,
                        "deletions": 0
                    }
                
                stats = self.file_touch_stats[file_path]
                stats["touch_count"] += 1
                stats["authors"].add(commit.author_email)
                stats["commits"].append({
                    "sha": commit.sha,
                    "date": commit.commit_date,
                    "author": commit.author_name,
                    "subject": commit.subject
                })
                
                # Update date range
                if commit.commit_date < stats["first_seen"]:
                    stats["first_seen"] = commit.commit_date
                if commit.commit_date > stats["last_seen"]:
                    stats["last_seen"] = commit.commit_date
                
                # Add change statistics
                stats["additions"] += commit.insertions
                stats["deletions"] += commit.deletions
        
        # Convert sets to lists for JSON serialization
        for file_path in self.file_touch_stats:
            self.file_touch_stats[file_path]["authors"] = list(self.file_touch_stats[file_path]["authors"])
        
        logger.info(f"Built file statistics for {len(self.file_touch_stats)} unique files")
        logger.info(f"Total file changes across all commits: {total_files_across_commits}")
    
    def _create_commit_documents(self, commits: List[CommitMeta]) -> List[Document]:
        """Create searchable documents from commit metadata"""
        documents = []
        
        for commit in commits:
            # Create searchable content focusing on commit message and context
            content_parts = [
                f"Commit: {commit.sha[:12]}",
                f"Author: {commit.author_name} <{commit.author_email}>",
                f"Date: {commit.commit_date}",
                f"Subject: {commit.subject}",
            ]
            
            if commit.body.strip():
                content_parts.append(f"Body: {commit.body}")
            
            if commit.files_changed:
                content_parts.append(f"Files changed: {', '.join(commit.files_changed[:10])}")
                if len(commit.files_changed) > 10:
                    content_parts.append(f"... and {len(commit.files_changed) - 10} more files")
            
            if commit.pr_number:
                content_parts.append(f"Pull Request: #{commit.pr_number}")
            
            if commit.is_merge:
                content_parts.append("Type: Merge commit")
            
            content_parts.append(f"Changes: +{commit.insertions} -{commit.deletions}")
            
            searchable_content = "\n".join(content_parts)
            
            doc = Document(
                text=searchable_content,
                metadata={
                    "commit_sha": commit.sha,
                    "author": commit.author_name,
                    "date": commit.commit_date,
                    "type": "commit",
                    "pr_number": commit.pr_number,
                    "file_count": len(commit.files_changed)
                }
            )
            
            documents.append(doc)
        
        return documents
    
    async def _build_faiss_index(self, documents: List[Document]) -> None:
        """Build FAISS vector index for semantic search"""
        if not documents:
            return
        
        logger.info(f"Building FAISS index with {len(documents)} commit documents...")
        
        # Create node parser
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=2048,  # Smaller than issue chunks since commit messages are shorter
            chunk_overlap=100
        )
        
        # Get embedding dimensions
        vec_size = 1536  # text-embedding-3-small
        
        # Create FAISS index
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(vec_size))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Build index
        self.vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        # Save FAISS index
        self.vector_index.storage_context.persist(str(self.index_dir))
        
        logger.info("FAISS index built and saved")
    
    async def _build_bm25_index(self, documents: List[Document]) -> None:
        """Build BM25 index for keyword search"""
        if not documents:
            return
        
        logger.info("Building BM25 index for commit search...")
        
        # Create nodes for BM25
        node_parser = SimpleNodeParser.from_defaults(chunk_size=2048)
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Build BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=50,  # Get more candidates for reranking
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )
        
        logger.info("BM25 index built")
    
    async def load_existing_index(self) -> bool:
        """Load existing commit index if available with robust error handling"""
        
        # Check if essential files exist
        essential_files = [self.commits_file, self.metadata_file, self.file_stats_file]
        missing_files = [f for f in essential_files if not f.exists()]
        
        if missing_files:
            logger.info(f"CommitIndexer.load_existing_index: Missing essential files {[f.name for f in missing_files]} from {self.index_dir}")
            return False
        
        # Check if vector store files exist
        vector_store_files = [
            self.index_dir / "default__vector_store.json",
            self.index_dir / "docstore.json",
            self.index_dir / "index_store.json"
        ]
        
        vector_store_exists = all(f.exists() for f in vector_store_files)
        logger.info(f"CommitIndexer.load_existing_index: Vector store files exist: {vector_store_exists}")
        
        # Load metadata to check cache age and validity
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_commits = metadata.get("total_commits", 0)
            created_at = metadata.get("created_at", "unknown")
            logger.info(f"CommitIndexer.load_existing_index: Found cache with {total_commits} commits created at {created_at}")
            
            if total_commits < 5:
                logger.warning(f"CommitIndexer.load_existing_index: Cache has suspiciously low commit count ({total_commits}), will rebuild")
                return False
                
        except Exception as e:
            logger.warning(f"CommitIndexer.load_existing_index: Failed to read metadata: {e}")
            return False
        
        try:
            # Always load commits and file stats (these are essential and fast)
            logger.info(f"Loading commits and file stats from {self.index_dir}")
            await self._load_commits()
            await self._load_file_statistics()
            
            commits_loaded = len(self.commit_metas)
            files_loaded = len(self.file_touch_stats)
            logger.info(f"Loaded {commits_loaded} commits and {files_loaded} file statistics")
            
            if commits_loaded == 0:
                logger.warning("No commits loaded from cache, will rebuild")
                return False
            
            # Try to load vector store if files exist
            if vector_store_exists:
                try:
                    logger.info(f"Loading VectorStoreIndex from {self.index_dir}")
                    # Ensure embed_model is set in Settings for from_persist_dir
                    Settings.embed_model = self.embed_model
                    storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
                    self.vector_index = VectorStoreIndex.from_storage(storage_context)
                    logger.info(f"Successfully loaded VectorStoreIndex with {len(self.commit_metas)} commits")
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"Vector store corrupted (UTF-8 decode error): {e}. Cleaning up and will rebuild vector store only.")
                    self._cleanup_corrupted_vector_store()
                    self.vector_index = None
                    
                except Exception as e:
                    logger.warning(f"Failed to load VectorStoreIndex from {self.index_dir}: {e}. Will rebuild vector store only.")
                    self._cleanup_corrupted_vector_store()
                    self.vector_index = None
            else:
                logger.info(f"Vector store files missing, will rebuild vector store from existing commits")
                self.vector_index = None
            
            # Rebuild vector store if needed (using existing commit data)
            if self.vector_index is None and self.commit_metas:
                logger.info(f"Rebuilding vector store from {len(self.commit_metas)} existing commits")
                documents = self._create_commit_documents(list(self.commit_metas.values()))
                await self._build_faiss_index(documents)
                logger.info(f"Vector store rebuilt successfully")

            # Always rebuild BM25 (it's fast and doesn't persist)
            if self.commit_metas:
                documents = self._create_commit_documents(list(self.commit_metas.values()))
                await self._build_bm25_index(documents)
            
            logger.info(f"Successfully loaded commit index with {len(self.commit_metas)} commits and {len(self.file_touch_stats)} file stats")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing commit index: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _cleanup_corrupted_vector_store(self):
        """Clean up corrupted vector store files"""
        corrupted_files = [
            "default__vector_store.json",
            "docstore.json", 
            "index_store.json",
            "graph_store.json",
            "image__vector_store.json"
        ]
        
        for file_name in corrupted_files:
            file_path = self.index_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Removed corrupted vector store file: {file_name}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove corrupted file {file_name}: {cleanup_error}")
    
    async def _load_commits(self) -> None:
        """Load commits from storage"""
        self.commit_metas = {}
        
        with open(self.commits_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                commit = CommitMeta(**data)
                self.commit_metas[commit.sha] = commit
    
    async def _load_file_statistics(self) -> None:
        """Load file statistics from storage"""
        with open(self.file_stats_file, 'r', encoding='utf-8') as f:
            self.file_touch_stats = json.load(f)
    
    async def _load_faiss_index(self) -> None:
        """Load LlamaIndex VectorStoreIndex from storage (this method is now part of load_existing_index)"""
        # This specific method is effectively replaced by the logic within load_existing_index
        # which uses VectorStoreIndex.from_storage(StorageContext.from_defaults(persist_dir=...))
        # Kept for structural reference if direct FAISS interaction was ever re-introduced, but
        # for LlamaIndex's default persistence, direct FAISS file loading is not the primary path.
        logger.debug("_load_faiss_index is deprecated; loading handled by StorageContext in load_existing_index.")
        pass

    async def _save_commits(self, commits: List[CommitMeta]) -> None:
        """Save commits to storage"""
        with open(self.commits_file, 'w', encoding='utf-8') as f:
            for commit in commits:
                f.write(json.dumps(commit.__dict__, ensure_ascii=False) + '\n')
    
    async def _save_file_statistics(self) -> None:
        """Save file statistics to storage"""
        with open(self.file_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_touch_stats, f, indent=2, ensure_ascii=False)
    
    async def _save_metadata(self, total_commits: int) -> None:
        """Save index metadata"""
        metadata = {
            "total_commits": total_commits,
            "created_at": datetime.now().isoformat(),
            "repo_path": str(self.repo_path),
            "repo_key": self.repo_key,
            "embedding_model": "text-embedding-3-small"
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def get_recent_commits(self, limit: int = 100) -> List[CommitMeta]:
        """Get recent commits without search, sorted by date"""
        if not self.commit_metas:
            return []
        
        # Sort commits by date (newest first) and return the requested number
        sorted_commits = sorted(
            self.commit_metas.values(),
            key=lambda c: c.commit_date,
            reverse=True
        )
        
        return sorted_commits[:limit]


class CommitRetriever:
    """Handles commit search and retrieval"""
    
    def __init__(self, indexer: CommitIndexer):
        self.indexer = indexer
    
    async def search_commits(
        self, 
        query: str,
        k: int = 10,
        author_filter: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        file_filter: Optional[List[str]] = None,
        pr_filter: Optional[int] = None,
        include_merges: bool = True
    ) -> List[CommitSearchResult]:
        """Search commits using hybrid dense + sparse retrieval"""
        
        if not self.indexer.bm25_retriever:
            logger.warning("Commit index not initialized")
            return []
        
        # Dense (semantic) search if available
        dense_results = []
        if self.indexer.vector_index:
            dense_results = await self._dense_search(query, k * 2)
        
        # Sparse (keyword) search
        sparse_results = await self._sparse_search(query, k * 2)
        
        # Combine and deduplicate results
        combined_results = self._combine_results(dense_results, sparse_results)
        
        # Apply filters
        filtered_results = self._apply_filters(
            combined_results,
            author_filter=author_filter,
            date_range=date_range,
            file_filter=file_filter,
            pr_filter=pr_filter,
            include_merges=include_merges
        )
        
        # Format results
        search_results = self._format_commit_results(filtered_results[:k])
        
        return search_results
    
    async def _dense_search(self, query: str, k: int) -> List[Dict]:
        """Perform dense vector search"""
        try:
            # Handle empty or whitespace-only queries
            if not query or not query.strip():
                logger.debug("Dense search: Empty query provided, returning empty results")
                return []
            
            retriever = self.indexer.vector_index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                commit_sha = node.metadata.get("commit_sha")
                if commit_sha and commit_sha in self.indexer.commit_metas:
                    results.append({
                        "commit": self.indexer.commit_metas[commit_sha],
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "source": "dense",
                        "node": node
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    async def _sparse_search(self, query: str, k: int) -> List[Dict]:
        """Perform sparse BM25 search"""
        try:
            # Handle empty or whitespace-only queries
            if not query or not query.strip():
                logger.debug("Sparse search: Empty query provided, returning empty results")
                return []
            
            nodes = self.indexer.bm25_retriever.retrieve(query)
            
            results = []
            for node in nodes:
                commit_sha = node.metadata.get("commit_sha")
                if commit_sha and commit_sha in self.indexer.commit_metas:
                    results.append({
                        "commit": self.indexer.commit_metas[commit_sha],
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "source": "sparse",
                        "node": node
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate dense and sparse results"""
        combined = {}
        
        # Add dense results
        for result in dense_results:
            sha = result["commit"].sha
            if sha not in combined:
                combined[sha] = result
                combined[sha]["combined_score"] = result["score"] * 0.7  # Weight dense results
            else:
                # Boost score if found in both
                combined[sha]["combined_score"] += result["score"] * 0.3
        
        # Add sparse results
        for result in sparse_results:
            sha = result["commit"].sha
            if sha not in combined:
                combined[sha] = result
                combined[sha]["combined_score"] = result["score"] * 0.5  # Lower weight for sparse only
            else:
                # Boost score if found in both
                combined[sha]["combined_score"] += result["score"] * 0.3
        
        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        
        return sorted_results
    
    def _apply_filters(
        self,
        results: List[Dict],
        author_filter: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        file_filter: Optional[List[str]] = None,
        pr_filter: Optional[int] = None,
        include_merges: bool = True
    ) -> List[Dict]:
        """Apply various filters to search results"""
        
        filtered = []
        
        for result in results:
            commit = result["commit"]
            
            # Author filter
            if author_filter and author_filter.lower() not in commit.author_name.lower():
                continue
            
            # Date range filter
            if date_range:
                try:
                    commit_date = datetime.fromisoformat(commit.commit_date.replace('Z', '+00:00'))
                    start_date = datetime.fromisoformat(date_range[0]) if date_range[0] else None
                    end_date = datetime.fromisoformat(date_range[1]) if date_range[1] else None
                    
                    if start_date and commit_date < start_date:
                        continue
                    if end_date and commit_date > end_date:
                        continue
                except Exception:
                    continue
            
            # File filter
            if file_filter:
                file_match = any(f in commit.files_changed for f in file_filter)
                if not file_match:
                    continue
            
            # PR filter
            if pr_filter and commit.pr_number != pr_filter:
                continue
            
            # Merge commit filter
            if not include_merges and commit.is_merge:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _format_commit_results(self, results: List[Dict]) -> List[CommitSearchResult]:
        """Format results into CommitSearchResult objects"""
        
        formatted_results = []
        
        for result in results:
            commit = result["commit"]
            
            # Calculate file relevance if query mentions files
            file_relevance = 0.0  # Could be enhanced based on file mentions in query
            
            # Determine match reasons
            match_reasons = []
            if result["source"] == "dense":
                match_reasons.append("semantic similarity")
            if result["source"] == "sparse":
                match_reasons.append("keyword match")
            if "combined_score" in result:
                match_reasons.append("hybrid ranking")
            
            search_result = CommitSearchResult(
                commit=commit,
                similarity=result.get("combined_score", result["score"]),
                match_reasons=match_reasons,
                file_relevance=file_relevance
            )
            
            formatted_results.append(search_result)
        
        return formatted_results
    
    def get_file_timeline(
        self,
        file_path: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get timeline of commits that touched a specific file"""
        
        timeline = []
        
        for commit in self.indexer.commit_metas.values():
            if file_path in commit.files_changed:
                change_type = "unknown"
                if file_path in commit.files_added:
                    change_type = "added"
                elif file_path in commit.files_modified:
                    change_type = "modified"
                elif file_path in commit.files_deleted:
                    change_type = "deleted"
                
                timeline.append({
                    "sha": commit.sha,
                    "author": commit.author_name,
                    "date": commit.commit_date,
                    "subject": commit.subject,
                    "change_type": change_type,
                    "insertions": commit.insertions,
                    "deletions": commit.deletions,
                    "pr_number": commit.pr_number
                })
        
        # Sort by date (newest first)
        timeline.sort(key=lambda x: x["date"], reverse=True)
        
        return timeline[:limit]
    
    def get_file_statistics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific file"""
        return self.indexer.file_touch_stats.get(file_path)
    
    def get_commit_by_sha(self, sha: str) -> Optional[CommitMeta]:
        """Get commit metadata by SHA"""
        return self.indexer.commit_metas.get(sha)


class CommitIndexManager:
    """High-level manager for commit indexing functionality"""
    
    def __init__(self, repo_path: str, repo_owner: str = None, repo_name: str = None):
        self.indexer = CommitIndexer(repo_path, repo_owner, repo_name)
        self.retriever = CommitRetriever(self.indexer)
        self._initialized = False
    
    async def initialize(
        self,
        max_commits: Optional[int] = None,
        force_rebuild: bool = False,
        since_date: Optional[str] = None
    ) -> None:
        """Initialize the commit index"""
        await self.indexer.build_commit_index(
            max_commits=max_commits,
            force_rebuild=force_rebuild,
            since_date=since_date
        )
        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the commit index is initialized"""
        return self._initialized and bool(self.indexer.commit_metas)
    
    async def search_commits(self, query: str, **kwargs) -> List[CommitSearchResult]:
        """Search commits (proxy to retriever)"""
        if not self.is_initialized():
            return []
        return await self.retriever.search_commits(query, **kwargs)
    
    def get_file_timeline(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Get file timeline (proxy to retriever)"""
        if not self.is_initialized():
            return []
        return self.retriever.get_file_timeline(file_path, **kwargs)
    
    def get_file_statistics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file statistics (proxy to retriever)"""
        if not self.is_initialized():
            return None
        return self.retriever.get_file_statistics(file_path)
    
    def get_commit_by_sha(self, sha: str) -> Optional[CommitMeta]:
        """Get commit by SHA (proxy to retriever)"""
        if not self.is_initialized():
            return None
        return self.retriever.get_commit_by_sha(sha)
    
    def get_recent_commits(self, limit: int = 100) -> List[CommitMeta]:
        """Get recent commits without search, sorted by date"""
        if not self._initialized:
            return []
        return self.indexer.get_recent_commits(limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall index statistics"""
        if not self.is_initialized():
            return {"initialized": False}
        
        total_commits = len(self.indexer.commit_metas)
        total_files = len(self.indexer.file_touch_stats)
        
        # Calculate some aggregate stats
        total_insertions = sum(c.insertions for c in self.indexer.commit_metas.values())
        total_deletions = sum(c.deletions for c in self.indexer.commit_metas.values())
        merge_commits = sum(1 for c in self.indexer.commit_metas.values() if c.is_merge)
        
        authors = set(c.author_email for c in self.indexer.commit_metas.values())
        
        return {
            "initialized": True,
            "total_commits": total_commits,
            "total_files_touched": total_files,
            "total_authors": len(authors),
            "merge_commits": merge_commits,
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "repo_path": str(self.indexer.repo_path),
            "index_path": str(self.indexer.index_dir)
        }
