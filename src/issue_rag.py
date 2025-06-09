"""
Issue-Aware RAG System
Integrates GitHub issue history into the existing RAG pipeline for enhanced context
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import re
import faiss
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
import Stemmer
from tqdm.auto import tqdm # Import tqdm

from src.config import settings
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .models import IssueDoc, IssueSearchResult, IssueContextResponse, PatchSearchResult
from .cache_manager import rag_cache
from .patch_linkage import PatchLinkageBuilder
from .commit_index import CommitIndexManager

logger = logging.getLogger(__name__)

def to_int(issue_id: Any) -> Optional[int]:
    """Safely convert issue_id to integer, returning None if conversion fails"""
    try:
        return int(issue_id)
    except (ValueError, TypeError):
        return None

class IssueIndexer:
    """Handles indexing and storage of GitHub issues"""
    
    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_key = f"{repo_owner}/{repo_name}"
        self.github_client = GitHubIssueClient()
        self.llm_client = LLMClient()
        
        # Initialize patch linkage builder
        self.patch_builder = PatchLinkageBuilder(repo_owner, repo_name)
        
        # Initialize commit index manager (optional enhancement)
        self.commit_index_manager = CommitIndexManager(
            repo_path=".",  # Assume we're running from repo root
            repo_owner=repo_owner,
            repo_name=repo_name
        )
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",  # Changed from large to small
            api_key=settings.openai_api_key
        )
        
        # Setup storage paths
        self.index_dir = Path(f".faiss_issues_{repo_owner}_{repo_name}")
        self.index_dir.mkdir(exist_ok=True)
        
        self.issues_file = self.index_dir / "issues.jsonl"
        self.metadata_file = self.index_dir / "metadata.json"
        self.faiss_index_file = self.index_dir / "index.faiss"
        self.faiss_nodes_file = self.index_dir / "nodes.json"
        
        # Initialize index components
        self.faiss_index = None
        self.vector_index = None
        self.issue_docs = {}  # id -> IssueDoc
        self.diff_docs = {} # pr_number -> DiffDoc
        self.bm25_retriever = None
        self.bm25_score_cache = {}  # Cache for BM25 score normalization
        
    async def crawl_and_index_issues(
        self, 
        max_issues: Optional[int] = None,  # Changed from 1000 to None
        force_rebuild_dependencies: bool = False,
        max_issues_for_patch_linkage: Optional[int] = None
    ) -> None:
        """Crawl issues from GitHub and build a new FAISS/BM25 index"""
        # Use settings.MAX_ISSUES_TO_PROCESS if max_issues is None
        max_issues = max_issues or settings.MAX_ISSUES_TO_PROCESS
        
        logger.info(
            f"Starting issue crawl for {self.repo_key} (max_issues={max_issues}, "
            f"force_rebuild_dependencies={force_rebuild_dependencies}, "
            f"max_issues_for_patch_linkage={max_issues_for_patch_linkage})"
        )
        
        # Ensure we have patch docs before indexing
        diff_file = self.patch_builder.index_dir / "diff_docs.jsonl"
        
        # Determine max_issues for patch_builder
        actual_max_issues_for_patches = max_issues_for_patch_linkage if max_issues_for_patch_linkage is not None else max_issues

        if force_rebuild_dependencies or not diff_file.exists() or diff_file.stat().st_size == 0:
            if force_rebuild_dependencies:
                logger.info(f"Forcing rebuild of patch linkage dependencies with max_issues={actual_max_issues_for_patches}...")
            else:
                logger.info(
                    "No cached patch docs found or they are empty, "
                    f"building patch linkage first with max_issues={actual_max_issues_for_patches}..."
                )
            await self.patch_builder.build_patch_linkage(max_issues=actual_max_issues_for_patches)
        else:
            logger.info(f"Using existing patch docs from {diff_file}")

        # 1. Fetch all closed issues
        repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}"
        issues = await self.github_client.list_issues(repo_url, state="all", max_pages=100)
        
        # Limit issue count (pull requests are already filtered out by github_client.list_issues)
        issues = issues[:max_issues]
        logger.info(f"Fetched {len(issues)} issues to process")
        
        # 2. Convert issues to llama_index Documents
        issue_documents = []
        logger.info(f"Converting {len(issues)} issues to LlamaIndex documents...")
        for issue in tqdm(issues, desc="Processing issues", unit="issue"):
            issue_doc = self._create_issue_doc(issue)
            # Use issue.number as the key for issue_docs, as IssueDoc.id is derived from issue.number
            self.issue_docs[issue.number] = issue_doc 
            
            # Create a better formatted document for indexing with enhanced searchability
            # Combine title, body, and key metadata for better semantic matching
            searchable_content = self._create_searchable_content(issue)
            
            # Keep metadata minimal to avoid chunk size issues
            # Move detailed information to the main text content
            doc = Document(
                text=searchable_content,
                metadata={
                    "issue_id": issue.number,
                    "state": issue.state,
                    "type": "issue",
                },
            )
            issue_documents.append(doc)

        # 3. Load diff documents and add them to the index
        patch_documents = []
        try:
            loaded_diffs = self.patch_builder.load_diff_docs()
            self.diff_docs = {diff.pr_number: diff for diff in loaded_diffs}
            logger.info(f"Loaded {len(loaded_diffs)} diff documents to add to the index.")
            for diff in loaded_diffs:
                # Create detailed text content that includes all information
                # Move file paths and detailed info to text content rather than metadata
                detailed_content = self._create_patch_content(diff)
                
                doc = Document(
                    text=detailed_content,
                    metadata={
                        "issue_id": diff.issue_id,
                        "pr_number": diff.pr_number,
                        "type": "patch",
                    },
                )
                patch_documents.append(doc)
        except FileNotFoundError:
            logger.warning("diff_docs.jsonl not found. Index will be built without patches.")
        except Exception as e:
            logger.error(f"Error loading diff docs: {e}")

        all_documents = issue_documents + patch_documents
        
        # 4. Build FAISS index for vector search with larger chunk size
        logger.info(f"Building FAISS index with {len(all_documents)} total documents...")
        
        # Use a node parser with larger chunk size to accommodate rich content
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=4096,  # Increased from default 1024
            chunk_overlap=200  # Add overlap to preserve context
        )
        
        # Get embedding dimensions for the small model (1536 for text-embedding-3-small)
        embedding_dimensions = self.embed_model.dimensions
        if embedding_dimensions is None:
            # text-embedding-3-small has 1536 dimensions
            vec_size = 1536
        else:
            vec_size = int(embedding_dimensions)
        
        logger.info(f"Using embedding dimension: {vec_size}")
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(vec_size))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create nodes with the larger chunk size
        nodes = node_parser.get_nodes_from_documents(all_documents)
        
        self.vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        # 5. Save index components
        self._save_faiss_index(nodes)
        
        # 6. Build and cache BM25 retriever
        logger.info("Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=50,
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )
        await self._compute_bm25_statistics(nodes)
        
        # 7. Save issues and metadata
        await self._save_issues()
        await self._save_metadata(total_issues=len(issues), total_docs=len(all_documents))
        logger.info("Successfully built and saved new issue and patch index.")
    
    async def _compute_bm25_statistics(self, nodes: List) -> None:
        """Pre-compute BM25 score statistics for better normalization"""
        if not self.bm25_retriever:
            return
        
        try:
            # Sample diverse queries to establish score ranges
            sample_queries = [
                "bug fix error",
                "performance optimization",
                "feature request enhancement",
                "memory leak issue",
                "user interface problem",
                "documentation update",
                "test case failure",
                "security vulnerability"
            ]
            
            all_scores = []
            for query in sample_queries:
                try:
                    results = self.bm25_retriever.retrieve(query)
                    scores = [getattr(node, 'score', 0.0) for node in results if hasattr(node, 'score')]
                    all_scores.extend(scores)
                except Exception:
                    continue
            
            if all_scores:
                self.bm25_score_cache = {
                    "max_score": max(all_scores),
                    "min_score": min(all_scores),
                    "avg_score": sum(all_scores) / len(all_scores),
                    "score_range": max(all_scores) - min(all_scores)
                }
                logger.info(f"BM25 score stats: max={self.bm25_score_cache['max_score']:.2f}, "
                           f"avg={self.bm25_score_cache['avg_score']:.2f}, "
                           f"range={self.bm25_score_cache['score_range']:.2f}")
            else:
                # Fallback values
                self.bm25_score_cache = {
                    "max_score": 10.0,
                    "min_score": 0.0,
                    "avg_score": 5.0,
                    "score_range": 10.0
                }
                
        except Exception as e:
            logger.warning(f"Failed to compute BM25 statistics: {e}")
            # Set safe defaults
            self.bm25_score_cache = {
                "max_score": 10.0,
                "min_score": 0.0,
                "avg_score": 5.0,
                "score_range": 10.0
            }
    
    def _normalize_bm25_score(self, raw_score: float) -> float:
        """Normalize BM25 score to 0-1 range using pre-computed statistics"""
        if not self.bm25_score_cache:
            # Fallback to simple normalization if no stats available
            return min(0.8, max(0.1, raw_score / 15.0))  # More conservative scaling
        
        try:
            max_score = self.bm25_score_cache["max_score"]
            min_score = self.bm25_score_cache["min_score"]
            score_range = self.bm25_score_cache["score_range"]
            
            if score_range == 0:
                return 0.4  # Lower default similarity if no range
            
            # Normalize to 0-1, then map to a more conservative 0.1-0.8 range
            normalized = (raw_score - min_score) / score_range
            return 0.1 + (normalized * 0.7)  # Maps [0,1] to [0.1, 0.8] instead of [0.6, 0.95]
            
        except Exception:
            # Fallback to more conservative method
            return min(0.8, max(0.1, raw_score / 15.0))
    
    async def load_existing_index(self) -> bool:
        """Load existing issue index if available with robust error handling"""
        if self.faiss_index_file.exists() and self.faiss_nodes_file.exists():
            logger.info(f"Loading existing binary FAISS index from {self.index_dir}")
            try:
                # Load FAISS index from disk
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                
                # Load nodes (documents)
                with open(self.faiss_nodes_file, 'r', encoding='utf-8') as f:
                    nodes_dict = json.load(f)
                    nodes = [Document(**doc) for doc in nodes_dict]
                
                # Reconstruct VectorStoreIndex
                if self.faiss_index.ntotal == len(nodes):
                    vector_store = FaissVectorStore(faiss_index=self.faiss_index)
                    self.vector_index = VectorStoreIndex(nodes=nodes, vector_store=vector_store, embed_model=self.embed_model)
                    
                    # Load issues and metadata
                    await self._load_issues()
                    await self._load_metadata()

                    # Load diff docs
                    try:
                        loaded_diffs = self.patch_builder.load_diff_docs()
                        self.diff_docs = {diff.pr_number: diff for diff in loaded_diffs}
                        logger.info(f"Loaded {len(self.diff_docs)} diff documents from cache.")
                    except FileNotFoundError:
                        logger.info("No cached diff documents found to load.")
                    except Exception as e:
                        logger.warning(f"Could not load diff documents: {e}")
                    
                    # Rebuild BM25 retriever
                    try:
                        self.bm25_retriever = BM25Retriever.from_defaults(
                            nodes=nodes,
                            similarity_top_k=50,
                            stemmer=Stemmer.Stemmer("english"),
                            language="english"
                        )
                        # Recompute BM25 statistics
                        await self._compute_bm25_statistics(nodes)
                    except Exception as e:
                        logger.warning(f"Failed to rebuild BM25 retriever: {e}")
                        self.bm25_retriever = None
                    
                    # Clean up legacy JSON store files from pre-1.1 versions to prevent UTF-8 decode errors
                    for stale in ["default__vector_store.json", "docstore.json",
                                  "index_store.json", "graph_store.json"]:
                        p = self.index_dir / stale
                        if p.exists():
                            p.unlink()
                            logger.info(f"Deleted legacy store file {stale}")
                    
                    logger.info(f"Successfully loaded issue index with {len(nodes)} nodes")
                    return True
                    
            except Exception as e:
                logger.warning(f"Failed to load binary FAISS index: {e}")
                # Delete corrupted index files and fall back to rebuild
                self._cleanup_corrupted_index()
                return False
        
        # Fallback to old storage context method (if binary loading failed)
        try:
            Settings.embed_model = self.embed_model
            storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
            self.vector_index = VectorStoreIndex.from_documents([], storage_context=storage_context)
            
            logger.info(f"Loaded existing issue index for {self.repo_key} with {len(self.issue_docs)} issues")
            return True
        except Exception as e:
            logger.warning(f"Failed to load storage context: {e}")
            # Delete corrupted files and force rebuild
            self._cleanup_corrupted_index()
            return False
                
    def _cleanup_corrupted_index(self):
        """Clean up corrupted index files to force a fresh rebuild"""
        try:
            if self.faiss_index_file.exists():
                self.faiss_index_file.unlink()
                logger.info("Removed corrupted FAISS index file")
            
            if self.faiss_nodes_file.exists():
                self.faiss_nodes_file.unlink()
                logger.info("Removed corrupted nodes file")
            
            # Clean up old storage context files
            storage_files = [
                "default__vector_store.json",
                "docstore.json", 
                "index_store.json",
                "graph_store.json"
            ]
            
            for filename in storage_files:
                file_path = self.index_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed corrupted {filename}")
                    
        except Exception as e:
            logger.warning(f"Error during index cleanup: {e}")

    def _save_faiss_index(self, nodes):
        """Save the FAISS index and node data to disk."""
        # Access the internal _faiss_index attribute of FaissVectorStore
        faiss.write_index(self.vector_index.vector_store._faiss_index, str(self.faiss_index_file))
        with open(self.faiss_nodes_file, "w") as f:
            json.dump([n.dict() for n in nodes], f)

    async def _save_metadata(self, total_issues: int, total_docs: int) -> None:
        """Save metadata about the index"""
        metadata = {
            "repo": self.repo_key,
            "total_issues": total_issues,
            "total_documents": total_docs,
            "index_contains_patches": total_docs > total_issues,
            "last_updated": datetime.now().isoformat(),
            "index_version": "1.5"
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def _save_issues(self) -> None:
        """Save the issue documents to a JSONL file"""
        with open(self.issues_file, 'w', encoding='utf-8') as f:
            for issue_doc in self.issue_docs.values():
                f.write(issue_doc.model_dump_json() + '\n')

    async def _load_issues(self) -> None:
        """Load issue documents from the JSONL file."""
        if not self.issues_file.exists():
            logger.warning("issues.jsonl not found. No issues loaded.")
            return
        self.issue_docs = {}
        with open(self.issues_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # First try to parse as JSON to handle backward compatibility
                    import json
                    data = json.loads(line.strip())
                    
                    # Add default values for new fields if they don't exist
                    if 'closed_by_commit' not in data:
                        data['closed_by_commit'] = None
                    if 'closed_by_pr' not in data:
                        data['closed_by_pr'] = None
                    if 'closed_by_author' not in data:
                        data['closed_by_author'] = None
                    if 'closed_event_data' not in data:
                        data['closed_event_data'] = None
                    
                    # Now validate with the full model
                    issue = IssueDoc.model_validate(data)
                    self.issue_docs[issue.id] = issue
                except Exception as e:
                    logger.warning(f"Skipping malformed line in issues.jsonl: {e}")

    async def _load_metadata(self) -> None:
        """Load metadata from the JSON file."""
        if not self.metadata_file.exists():
            logger.warning("metadata.json not found. No metadata loaded.")
            return
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                # You can store parts of metadata in the instance if needed
                logger.info(f"Loaded index metadata: {metadata}")
        except Exception as e:
            logger.error(f"Failed to load metadata.json: {e}")

    def _create_issue_doc(self, issue: Any) -> IssueDoc:
        """Create an IssueDoc from a GitHub issue object"""
        patch_url = self.patch_builder.get_patch_url_for_issue(issue.number) # Use issue.number here
        
        return IssueDoc(
            id=issue.number, # And ensure this uses issue.number as well
            state=issue.state,
            title=issue.title,
            body=issue.body or "",
            comments=[c.body for c in issue.comments],
            labels=issue.labels,
            created_at=issue.created_at.isoformat(),
            closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
            patch_url=patch_url,
            repo=self.repo_key
        )

    def _create_searchable_content(self, issue: Any) -> str:
        """Create enhanced searchable content from issue data"""
        parts = []
        
        # Main title and description
        parts.append(f"Title: {issue.title}")
        
        # Clean and format body
        body = issue.body or ""
        if body:
            # Clean markdown formatting for better text search
            import re
            body = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', body)  # Replace code blocks
            body = re.sub(r'`([^`]+)`', r'\1', body)  # Remove inline code formatting
            body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)  # Extract link text
            body = re.sub(r'#+\s*', '', body)  # Remove markdown headers
            body = re.sub(r'\s+', ' ', body).strip()  # Normalize whitespace
            
        parts.append(f"Description: {body}")
        
        # Add important metadata
        parts.append(f"State: {issue.state}")
        
        if issue.labels:
            # Format labels for better matching
            labels_text = " ".join(issue.labels)
            parts.append(f"Labels: {labels_text}")
            
            # Also add labels as individual searchable terms
            category_labels = [f'category-{label}' for label in issue.labels]
            parts.append(f"Categories: {' '.join(category_labels)}")
        
        # Add issue type context
        if any(label in ['bug', 'error', 'issue'] for label in issue.labels):
            parts.append("Type: Bug report or error")
        elif any(label in ['enhancement', 'feature'] for label in issue.labels):
            parts.append("Type: Feature request or enhancement")
        elif any(label in ['question', 'help'] for label in issue.labels):
            parts.append("Type: Question or help request")
        
        # Combine with good separation for embedding
        return "\n\n".join(parts)

    def _create_patch_content(self, diff: Any) -> str:
        """Create detailed text content for a patch"""
        parts = []
        
        # Add summary and details
        parts.append(f"Summary: {diff.diff_summary}")
        parts.append(f"Files changed: {', '.join(diff.files_changed)}")
        
        # Add detailed information
        parts.append(f"Issue ID: {diff.issue_id}")
        parts.append(f"PR Number: {diff.pr_number}")
        parts.append(f"Merged at: {diff.merged_at}")
        
        # Combine with good separation for embedding
        return "\n\n".join(parts)


class IssueReranker:
    """Reranks issue candidates using a cheaper LLM"""
    
    def __init__(self, llm_client: LLMClient, indexer: 'IssueIndexer'):
        self.llm_client = llm_client
        self.indexer = indexer
        self.cache = {}  # Simple in-memory cache for reranking results
    
    def _get_cache_key(self, query: str, candidates: List[Dict]) -> str:
        """Generate a cache key for the reranking results"""
        # Create a deterministic key from query and candidate IDs
        candidate_ids = [str(c.get("issue_id", c.get("pr_number", ""))) for c in candidates]
        return f"{query}::{','.join(candidate_ids)}"
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict],
        max_candidates: int = 5
    ) -> List[Dict]:
        """Rerank candidates using LLM"""
        if not candidates:
            return []
            
        # Check cache first
        cache_key = self._get_cache_key(query, candidates)
        if cache_key in self.cache:
            logger.info("Using cached reranking results")
            return self.cache[cache_key]
        
        # Prepare candidate summaries
        summaries = []
        for candidate in candidates:
            if "issue_id" in candidate:
                issue_doc = self.indexer.issue_docs[candidate["issue_id"]]
                summary = f"#{issue_doc.id}: {issue_doc.title}\n{issue_doc.body[:200]}..."
            else:
                diff_doc = self.indexer.diff_docs[candidate["pr_number"]]
                summary = f"PR #{diff_doc.pr_number} for Issue #{diff_doc.issue_id}: {diff_doc.diff_summary[:200]}..."
            summaries.append(summary)
        
        # Extract issue numbers for reference
        issue_numbers = []
        for candidate in candidates:
            if "issue_id" in candidate:
                issue_numbers.append(candidate["issue_id"])
            else:
                issue_numbers.append(candidate["pr_number"])
        
        # Create improved prompt for reranking
        prompt = f"""You are a helpful assistant that ranks GitHub issues by relevance to a user query.

User Query: {query}

Available Issues (rank these by relevance):
{chr(10).join(f"{i+1}. Issue #{issue_numbers[i]}: {s}" for i, s in enumerate(summaries))}

Instructions:
- Rank ALL {len(issue_numbers)} issues by relevance to the user query
- Return ONLY a valid JSON object with the "ranked_ids" field
- Include ALL issue numbers in your ranking
- Most relevant first, least relevant last

Example format:
{{"ranked_ids": [1210, 468, 554, 960, 676]}}

Your ranking (JSON only):"""

        try:
            # Use a cheaper model for reranking
            llm = self.llm_client._get_openrouter_llm(
                model="google/gemini-2.5-flash-preview-05-20"
            )
            response = llm.complete(prompt)
            
            # Log the raw response for debugging
            raw_response = response.text.strip()
            logger.debug(f"Raw reranker response: {raw_response}")
            
            # Try to extract JSON from the response
            import json
            import re
            
            # First try direct parsing
            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to find JSON within the response
                json_match = re.search(r'\{[^}]*"ranked_ids"[^}]*\}', raw_response)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Try to extract just the array part
                    array_match = re.search(r'\[[0-9,\s]+\]', raw_response)
                    if array_match:
                        ranked_ids = json.loads(array_match.group())
                        result = {"ranked_ids": ranked_ids}
                    else:
                        raise ValueError("No valid JSON found in response")
            
            ranked_ids = result.get("ranked_ids", [])
            
            # Convert to strings for matching
            ranked_ids = [str(id_val) for id_val in ranked_ids]
            
            # Reorder candidates based on LLM ranking
            id_to_candidate = {
                str(c.get("issue_id", c.get("pr_number", ""))): c 
                for c in candidates
            }
            
            reranked = []
            for id_str in ranked_ids:
                if id_str in id_to_candidate:
                    reranked.append(id_to_candidate[id_str])
            
            # Add any missing candidates at the end
            missing = [c for c in candidates if str(c.get("issue_id", c.get("pr_number", ""))) not in ranked_ids]
            reranked.extend(missing)
            
            # Cache results
            self.cache[cache_key] = reranked[:max_candidates]
            
            logger.info(f"Successfully reranked {len(ranked_ids)} issues")
            return reranked[:max_candidates]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fall back to original ranking
            return candidates[:max_candidates]


class IssueRetriever:
    """Handles retrieval of similar issues"""
    
    def __init__(self, indexer: IssueIndexer):
        self.indexer = indexer
        self.reranker = IssueReranker(indexer.llm_client, indexer)
        
    async def find_related_issues(
        self, 
        query: str, 
        k: int = 5,
        state_filter: str = "all",
        similarity_threshold: float = 0.3,
        label_filter: Optional[List[str]] = None,
        include_patches: bool = False
    ) -> Tuple[List[IssueSearchResult], List['PatchSearchResult']]:
        """Find issues and patches similar to the query"""
        
        if not self.indexer.vector_index:
            return [], []
        
        try:
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)
            
            # Get more candidates initially for reranking
            initial_k = 50  # Get more candidates for reranking
            
            # Get both dense and sparse retrieval results
            dense_results = await self._dense_search(processed_query, initial_k)
            sparse_results = await self._sparse_search(processed_query, initial_k)
            
            # Separate results by type (issue vs patch)
            dense_issues = [r for r in dense_results if r.get("type") == "issue"]
            dense_patches = [r for r in dense_results if r.get("type") == "patch"]
            sparse_issues = [r for r in sparse_results if r.get("type") == "issue"]
            sparse_patches = [r for r in sparse_results if r.get("type") == "patch"]
            
            # Process and combine issue results
            combined_issues = self._combine_results(dense_issues, sparse_issues)
            filtered_issues = self._apply_filters(
                combined_issues, 
                state_filter, 
                similarity_threshold,
                label_filter,
                processed_query
            )
            
            # Rerank issues
            reranked_issues = await self.reranker.rerank(
                query=processed_query,
                candidates=filtered_issues,
                max_candidates=k
            )
            
            issue_search_results = self._format_issue_results(reranked_issues)
            
            # Process patch results if requested
            patch_search_results = []
            if include_patches:
                combined_patches = self._combine_results(dense_patches, sparse_patches)
                filtered_patches = [p for p in combined_patches if p["similarity"] >= similarity_threshold * 0.8]
                
                # Rerank patches
                reranked_patches = await self.reranker.rerank(
                    query=processed_query,
                    candidates=filtered_patches,
                    max_candidates=k
                )
                
                patch_search_results = self._format_patch_results(reranked_patches, similarity_threshold)
            
            return issue_search_results, patch_search_results
            
        except Exception as e:
            logger.error(f"Error in issue retrieval: {e}")
            return [], []
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to better match indexed content format"""
        import re
        
        # Start with the original query
        processed = query.strip()
        
        # If it looks like a GitHub issue title format (starts with [BUG], [FEATURE], etc.)
        # enhance it for better matching
        if re.match(r'^\[([^\]]+)\]', processed):
            # Extract the tag and content
            match = re.match(r'^\[([^\]]+)\]\s*(.+)', processed)
            if match:
                tag, content = match.groups()
                tag_lower = tag.lower()
                
                # Add explicit type context for better matching
                if any(t in tag_lower for t in ['bug', 'error', 'issue']):
                    processed = f"Bug report: {content}. Type: Bug report or error"
                elif any(t in tag_lower for t in ['feature', 'enhancement']):
                    processed = f"Feature request: {content}. Type: Feature request or enhancement"
                elif any(t in tag_lower for t in ['question', 'help']):
                    processed = f"Question: {content}. Type: Question or help request"
                else:
                    processed = f"Title: {content}"
        else:
            # For queries without brackets, add title context
            processed = f"Title: {processed}"
        
        # Clean up common formatting that might interfere with matching
        processed = re.sub(r'`([^`]+)`', r'\1', processed)  # Remove backticks
        processed = re.sub(r'"""([^"]+)"""', r'\1', processed)  # Remove triple quotes
        processed = re.sub(r'"([^"]+)"', r'\1', processed)  # Remove quotes
        
        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    async def _dense_search(self, query: str, k: int) -> List[Dict]:
        """Semantic vector search"""
        retriever = self.indexer.vector_index.as_retriever(similarity_top_k=k)
        nodes = await retriever.aretrieve(query)
        
        results = []
        for node_with_score in nodes:
            # Handle both NodeWithScore and direct node objects
            if hasattr(node_with_score, 'node'):
                # This is a NodeWithScore object
                node = node_with_score.node
                raw_score = getattr(node_with_score, 'score', None)
            else:
                # This is a direct node object
                node = node_with_score
                raw_score = getattr(node, 'score', None)
            
            metadata = node.metadata or {}
            doc_type = metadata.get("type", "issue") # Default to issue if type not present
            
            # Extract similarity score properly
            similarity_score = 0.0
            if raw_score is not None:
                # For FAISS, score could be distance (lower is better) or similarity (higher is better)
                # Check if it's a reasonable similarity score (0-1 range) or distance (could be large)
                if 0 <= raw_score <= 1:
                    # Likely already a similarity score
                    similarity_score = float(raw_score)
                    logger.debug(f"Dense search: using raw similarity score={similarity_score}")
                else:
                    # Likely a distance score, convert to similarity
                    similarity_score = max(0.0, min(1.0, 1.0 / (1.0 + abs(raw_score))))
                    logger.debug(f"Dense search: converted distance {raw_score} to similarity={similarity_score}")
            else:
                similarity_score = 0.3  # Lower neutral score if not available
                logger.debug(f"Dense search: no score available, using default={similarity_score}")
            
            result_item = {
                "similarity": similarity_score,
                "dense_similarity": similarity_score,  # Add this for debugging
                "match_type": "semantic",
                "match_reasons": ["semantic similarity"],
                "type": doc_type
            }
            
            if doc_type == "issue":
                issue_id = to_int(metadata.get("issue_id"))
                if issue_id and issue_id in self.indexer.issue_docs:
                    result_item["issue_id"] = issue_id
                    results.append(result_item)
                    logger.debug(f"Dense search: added issue {issue_id} with similarity {similarity_score}")
            elif doc_type == "patch":
                pr_number = to_int(metadata.get("pr_number"))
                if pr_number and pr_number in self.indexer.diff_docs:  # Check if patch exists
                    result_item["pr_number"] = pr_number
                    results.append(result_item)
                
        return results
    
    async def _sparse_search(self, query: str, k: int) -> List[Dict]:
        """Keyword-based BM25 search with improved score normalization"""
        if not self.indexer.bm25_retriever:
            return []
            
        nodes = self.indexer.bm25_retriever.retrieve(query)
        
        results = []
        for node in nodes[:k]:
            metadata = node.metadata or {}
            doc_type = metadata.get("type", "issue")
            
            # Use improved normalization
            bm25_score = getattr(node, 'score', 1.0)
            normalized_score = self.indexer._normalize_bm25_score(bm25_score)
            logger.debug(f"Sparse search: raw BM25 score={bm25_score}, normalized={normalized_score}")
            
            result_item = {
                "similarity": normalized_score,
                "match_type": "keyword",
                "match_reasons": ["keyword match"],
                "type": doc_type
            }
            
            if doc_type == "issue":
                issue_id = to_int(metadata.get("issue_id"))
                if issue_id and issue_id in self.indexer.issue_docs:
                    result_item["issue_id"] = issue_id
                    results.append(result_item)
                    logger.debug(f"Sparse search: added issue {issue_id} with similarity {normalized_score}")
            elif doc_type == "patch":
                pr_number = to_int(metadata.get("pr_number"))
                if pr_number and pr_number in self.indexer.diff_docs:  # Check if patch exists
                    result_item["pr_number"] = pr_number
                    results.append(result_item)
                
        return results
    
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate search results for both issues and patches with improved scoring"""
        seen_issues = set()
        seen_patches = set()
        combined = []
        
        # Create a lookup for combining scores from both search methods
        score_lookup = {}
        
        # Process dense results first (prioritize semantic similarity)
        for result in dense_results:
            doc_type = result.get("type", "issue")
            
            if doc_type == "issue":
                item_id = to_int(result.get("issue_id"))
                if not item_id or item_id not in self.indexer.issue_docs:
                    continue
                    
                if item_id not in seen_issues:
                    seen_issues.add(item_id)
                    result["issue_id"] = item_id
                    # Mark as dense result for potential boosting
                    result["has_dense"] = True
                    result["dense_similarity"] = result.get("similarity", 0.0)
                    # Ensure sparse fields are initialized
                    result["has_sparse"] = False
                    result["sparse_similarity"] = 0.0
                    score_lookup[("issue", item_id)] = result
                    combined.append(result)
                    
            elif doc_type == "patch":
                item_id = to_int(result.get("pr_number"))
                if not item_id or item_id not in self.indexer.diff_docs:
                    continue
                    
                if item_id not in seen_patches:
                    seen_patches.add(item_id)
                    result["pr_number"] = item_id
                    result["has_dense"] = True
                    result["dense_similarity"] = result.get("similarity", 0.0)
                    # Ensure sparse fields are initialized
                    result["has_sparse"] = False
                    result["sparse_similarity"] = 0.0
                    score_lookup[("patch", item_id)] = result
                    combined.append(result)
        
        # Add sparse results and combine scores for items found in both
        for result in sparse_results:
            doc_type = result.get("type", "issue")
            
            if doc_type == "issue":
                item_id = to_int(result.get("issue_id"))
                if not item_id or item_id not in self.indexer.issue_docs:
                    continue
                
                key = ("issue", item_id)
                if item_id in seen_issues:
                    # Item found in both searches - combine scores with weighted average
                    existing = score_lookup[key]
                    existing["has_sparse"] = True
                    existing["sparse_similarity"] = result["similarity"]
                    # Weighted combination: 60% dense, 40% sparse
                    existing["similarity"] = (0.6 * existing.get("dense_similarity", 0.0) + 
                                            0.4 * result.get("similarity", 0.0))
                    existing["match_reasons"].append("keyword match")
                else:
                    # Only found in sparse search
                    seen_issues.add(item_id)
                    result["issue_id"] = item_id
                    result["has_sparse"] = True
                    result["sparse_similarity"] = result.get("similarity", 0.0)
                    # Ensure dense fields are initialized
                    result["has_dense"] = False
                    result["dense_similarity"] = 0.0
                    score_lookup[key] = result
                    combined.append(result)
                    
            elif doc_type == "patch":
                item_id = to_int(result.get("pr_number"))
                if not item_id or item_id not in self.indexer.diff_docs:
                    continue
                
                key = ("patch", item_id)
                if item_id in seen_patches:
                    # Combine scores
                    existing = score_lookup[key]
                    existing["has_sparse"] = True
                    existing["sparse_similarity"] = result["similarity"]
                    existing["similarity"] = (0.6 * existing.get("dense_similarity", 0.0) + 
                                            0.4 * result.get("similarity", 0.0))
                    existing["match_reasons"].append("keyword match")
                else:
                    seen_patches.add(item_id)
                    result["pr_number"] = item_id
                    result["has_sparse"] = True
                    result["sparse_similarity"] = result.get("similarity", 0.0)
                    # Ensure dense fields are initialized
                    result["has_dense"] = False
                    result["dense_similarity"] = 0.0
                    score_lookup[key] = result
                    combined.append(result)
            
        return combined
    
    def _apply_filters(
        self, 
        results: List[Dict], 
        state_filter: str,
        similarity_threshold: float,
        label_filter: Optional[List[str]],
        query: str = ""  # Add query parameter for title matching
    ) -> List[Dict]:
        """Apply filters to search results"""
        filtered = []
        
        for result in results:
            doc_type = result.get("type", "issue")
            
            # Apply similarity threshold
            if result["similarity"] < similarity_threshold:
                continue
            
            if doc_type == "issue":
                issue_id = to_int(result.get("issue_id"))
                if issue_id and issue_id in self.indexer.issue_docs:
                    issue_doc = self.indexer.issue_docs[issue_id]
                    
                    # Apply state filter
                    if state_filter != "all" and issue_doc.state != state_filter:
                        continue
                    
                    # Apply label filter
                    if label_filter:
                        if not any(label in issue_doc.labels for label in label_filter):
                            continue
                    
                    result["issue_id"] = issue_id
                    filtered.append(result)
            
            elif doc_type == "patch":
                # Patches don't have state or labels, so they always pass this filter
                filtered.append(result)
        
        # Sort by similarity with MODEST label boosting (preserve original similarity ranking)
        def calculate_final_score(item):
            base_similarity = item["similarity"]
            
            if item.get("type") == "issue":
                issue_id = item["issue_id"]
                if issue_id in self.indexer.issue_docs:
                    issue_doc = self.indexer.issue_docs[issue_id]
                    
                    # Start with base similarity
                    final_score = base_similarity
                    
                    # Add exact/partial title matching boost
                    query_lower = query.lower() if query else ""
                    title_lower = issue_doc.title.lower()
                    
                    # Extract content from query (remove [BUG], [FEATURE] etc.)
                    import re
                    clean_query = re.sub(r'^\[([^\]]+)\]\s*', '', query_lower).strip()
                    clean_query = re.sub(r'["`]', '', clean_query).strip()
                    
                    # Check for exact matches or high overlap
                    if clean_query and len(clean_query) > 5:  # Only for substantial queries
                        if clean_query in title_lower:
                            final_score += 0.15  # Significant boost for substring match
                        else:
                            # Check for word overlap
                            query_words = set(clean_query.split())
                            title_words = set(title_lower.split())
                            
                            if query_words and title_words:
                                overlap = len(query_words & title_words) / len(query_words)
                                if overlap > 0.6:  # If >60% of query words appear in title
                                    final_score += 0.08
                                elif overlap > 0.4:  # If >40% of query words appear in title
                                    final_score += 0.04
                    
                    # Boost for items found in both dense and sparse search
                    if item.get("has_dense") and item.get("has_sparse"):
                        final_score += 0.03
                    
                    # Apply SMALL label boost to preserve ranking integrity
                    label_boost = 0.0
                    high_value_labels = ['bug', 'enhancement', 'performance', 'memory-leak', 'hooks', 'ssr']
                    
                    for label in issue_doc.labels:
                        if any(hvl in label.lower() for hvl in high_value_labels):
                            label_boost += 0.01  # Much smaller boost: 0.01 instead of 0.05
                    
                    # Small boost for recently closed issues
                    if issue_doc.state == "closed" and issue_doc.closed_at:
                        try:
                            closed_date = datetime.fromisoformat(issue_doc.closed_at.replace('Z', '+00:00'))
                            days_since_closed = (datetime.now() - closed_date.replace(tzinfo=None)).days
                            if days_since_closed < 365:
                                label_boost += 0.005  # Very small boost: 0.005 instead of 0.03
                        except:
                            pass
                    
                    final_score += label_boost
                    
                    # Cap the total boost to maintain ranking order
                    return min(0.99, final_score)
            
            return base_similarity
        
        # Calculate final scores but don't overwrite the similarity field
        scored_results = []
        for result in filtered:
            final_score = calculate_final_score(result)
            result_copy = result.copy()
            result_copy["final_score"] = final_score
            scored_results.append(result_copy)
        
        # Sort by final score but keep original similarity
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Remove the temporary final_score field
        for result in scored_results:
            result.pop("final_score", None)
        
        return scored_results

    def _format_issue_results(self, results: List[Dict]) -> List[IssueSearchResult]:
        """Convert raw results to IssueSearchResult objects"""
        search_results = []
        for result in results:
            issue_id = result["issue_id"]
            if issue_id in self.indexer.issue_docs:
                issue_doc = self.indexer.issue_docs[issue_id]
                search_result = IssueSearchResult(
                    issue=issue_doc,
                    similarity=result["similarity"],
                    match_reasons=result.get("match_reasons", [])
                )
                search_results.append(search_result)
        return search_results

    def _format_patch_results(self, results: List[Dict], similarity_threshold: float) -> List['PatchSearchResult']:
        """Format patch search results"""
        patch_results = []
        
        for result in results:
            pr_number = to_int(result.get("pr_number"))
            if not pr_number or pr_number not in self.indexer.diff_docs:
                continue
                
            # Use a lower threshold for patches since they're more technical
            if result["similarity"] < similarity_threshold * 0.8:  # 20% lower threshold for patches
                continue
                
            diff_doc = self.indexer.diff_docs[pr_number]
            patch_results.append(PatchSearchResult(
                patch=diff_doc,
                similarity=result["similarity"],
                match_reasons=result.get("match_reasons", [])
            ))
            
        return patch_results


class IssueAwareRAG:
    """Main interface for issue-aware RAG functionality"""
    
    def __init__(self, repo_owner: str, repo_name: str, progress_callback: Optional[Callable] = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.progress_callback = progress_callback
        self.indexer = IssueIndexer(repo_owner, repo_name)
        self.retriever = IssueRetriever(self.indexer)
        self._initialized = False
        self._pr_cache = {}  # Cache for PR data
    
    async def initialize(
        self, 
        force_rebuild: bool = False,
        max_issues_for_patch_linkage: Optional[int] = None,
        max_prs_for_patch_linkage: Optional[int] = None
    ) -> None:
        """Initialize the RAG system by building or loading the index"""
        self.indexer = IssueIndexer(self.repo_owner, self.repo_name)
        
        # Check if we can load existing index FIRST (before expensive operations)
        if not force_rebuild and await self.indexer.load_existing_index():
            logger.info(f"Issue-aware RAG loaded from cache for {self.repo_owner}/{self.repo_name}.")
            self.retriever = IssueRetriever(self.indexer)
            self._initialized = True
            return
        
        # Only do expensive building if we need to rebuild or no cache exists
        logger.info(f"Building new index for {self.repo_owner}/{self.repo_name} (force_rebuild={force_rebuild})")
        
        # Build patch linkage (expensive)
        builder = PatchLinkageBuilder(self.repo_owner, self.repo_name, self.progress_callback)
        await builder.build_patch_linkage(
            max_issues=max_issues_for_patch_linkage,
            max_prs=max_prs_for_patch_linkage,
            download_diffs=True
        )
        
        # Build the issue index (expensive)
        await self.indexer.crawl_and_index_issues(
            max_issues=max_issues_for_patch_linkage,
            force_rebuild_dependencies=force_rebuild,
            max_issues_for_patch_linkage=max_issues_for_patch_linkage
        )
        
        self.retriever = IssueRetriever(self.indexer)
        self._initialized = True
        logger.info(f"Issue-aware RAG initialized for {self.repo_owner}/{self.repo_name} successfully.")
    
    async def get_issue_context(
        self, 
        query: str,
        max_issues: int = 5,
        include_patches: bool = True
    ) -> IssueContextResponse:
        """Get issue context for a query"""
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        # Analyze query to determine search strategy
        query_analysis = self._analyze_query(query)
        
        # Search for related issues and patches
        related_issues, related_patches = await self.retriever.find_related_issues(
            query,
            k=max_issues,
            state_filter=query_analysis.get("preferred_state", "all"),
            similarity_threshold=0.3,
            label_filter=query_analysis.get("relevant_labels"),
            include_patches=include_patches
        )
        
        processing_time = time.time() - start_time
        
        return IssueContextResponse(
            related_issues=related_issues,
            patches=related_patches,
            total_found=len(related_issues) + len(related_patches),
            query_analysis=query_analysis,
            processing_time=processing_time
        )
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to optimize search strategy"""
        query_lower = query.lower()
        analysis = {
            "query_type": "general",
            "preferred_state": "all",
            "relevant_labels": None,
            "urgency": "normal"
        }
        
        # Detect query type
        if any(word in query_lower for word in ["bug", "error", "issue", "problem", "broken"]):
            analysis["query_type"] = "bug_report"
            analysis["relevant_labels"] = ["bug", "error"]
        elif any(word in query_lower for word in ["feature", "enhancement", "request"]):
            analysis["query_type"] = "feature_request"
            analysis["relevant_labels"] = ["enhancement", "feature"]
        elif any(word in query_lower for word in ["performance", "slow", "optimization"]):
            analysis["query_type"] = "performance"
            analysis["relevant_labels"] = ["performance", "optimization"]
        
        # Detect urgency
        if any(word in query_lower for word in ["urgent", "critical", "breaking", "blocker"]):
            analysis["urgency"] = "high"
            analysis["preferred_state"] = "open"  # Focus on open issues for urgent queries
        
        return analysis
    
    def is_initialized(self) -> bool:
        """Check if the system is initialized"""
        return self._initialized
    
    async def update_index(self) -> None:
        """Update the issue index with new/changed issues"""
        if self._initialized:
            await self.indexer.crawl_and_index_issues()
            self.retriever = IssueRetriever(self.indexer)
    
    async def initialize_commit_index(
        self,
        max_commits: Optional[int] = None,
        force_rebuild: bool = False
    ) -> bool:
        """Initialize the commit index for enhanced commit-level analysis"""
        try:
            await self.commit_index_manager.initialize(
                max_commits=max_commits,
                force_rebuild=force_rebuild
            )
            logger.info("Commit index initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize commit index: {e}")
            return False
    
    async def get_prs(
        self,
        state: str = "all",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of PRs from the index"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Load diff docs which contain PR information
            diff_docs = self.indexer.patch_builder.load_diff_docs()
            
            # Filter by state if needed
            if state != "all":
                diff_docs = [doc for doc in diff_docs if doc.merged_at is not None]
            
            # Sort by PR number
            diff_docs.sort(key=lambda x: x.pr_number, reverse=True)
            
            # Convert to dict format
            prs = []
            for doc in diff_docs[:limit]:
                pr = {
                    "number": doc.pr_number,
                    "title": doc.pr_title if hasattr(doc, 'pr_title') else f"PR #{doc.pr_number}",
                    "merged_at": doc.merged_at,
                    "files_changed": doc.files_changed,
                    "issue_id": doc.issue_id
                }
                prs.append(pr)
            
            return prs
            
        except Exception as e:
            logger.error(f"Error getting PRs: {e}")
            return []
    
    async def get_pr_details(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific PR"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check cache first
            if pr_number in self._pr_cache:
                return self._pr_cache[pr_number]
            
            # Load diff docs
            diff_docs = self.indexer.patch_builder.load_diff_docs()
            
            # Find the PR
            pr_doc = next((doc for doc in diff_docs if doc.pr_number == pr_number), None)
            if not pr_doc:
                return None
            
            # Build detailed PR info
            pr_details = {
                "number": pr_doc.pr_number,
                "title": pr_doc.pr_title if hasattr(pr_doc, 'pr_title') else f"PR #{pr_doc.pr_number}",
                "merged_at": pr_doc.merged_at,
                "files_changed": pr_doc.files_changed,
                "issue_id": pr_doc.issue_id,
                "diff_summary": pr_doc.diff_summary,
                "diff_path": pr_doc.diff_path
            }
            
            # Cache the result
            self._pr_cache[pr_number] = pr_details
            
            return pr_details
            
        except Exception as e:
            logger.error(f"Error getting PR details: {e}")
            return None
    
    async def get_pr_diff(self, pr_number: int) -> Optional[str]:
        """Get the full diff for a specific PR"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get PR details first
            pr_details = await self.get_pr_details(pr_number)
            if not pr_details:
                return None
            
            # Load the diff file
            diff_path = pr_details["diff_path"]
            if not os.path.exists(diff_path):
                return None
            
            with open(diff_path, 'r', encoding='utf-8') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Error getting PR diff: {e}")
            return None
    
    async def search_prs(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search PRs using the RAG system"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use the retriever to search for relevant PRs
            results = await self.retriever.find_related_issues(
                query=query,
                k=limit,
                include_patches=True  # This will include PR diffs in the search
            )
            
            # Format results
            pr_results = []
            for result in results:
                if hasattr(result, 'pr_number'):  # This is a PR result
                    pr_details = await self.get_pr_details(result.pr_number)
                    if pr_details:
                        pr_results.append({
                            **pr_details,
                            "similarity_score": result.similarity_score
                        })
            
            return pr_results
            
        except Exception as e:
            logger.error(f"Error searching PRs: {e}")
            return []
    
    def clear_pr_cache(self) -> None:
        """Clear the PR cache"""
        self._pr_cache.clear()
