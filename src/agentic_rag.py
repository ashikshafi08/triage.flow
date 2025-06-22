"""
Agentic RAG Integration
Enhanced multi-index retrieval system with intelligent query routing and composite retrieval capabilities.
Combines LocalRepoContextExtractor with AgenticCodebaseExplorer for enhanced context retrieval.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Literal
import time
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict

from .new_rag import LocalRepoContextExtractor
from .issue_rag import IssueAwareRAG

logger = logging.getLogger(__name__)

# Type definitions for backwards compatibility
ContextChunk = Dict[str, Any]
RetrievalMode = Literal["chunks", "files_via_metadata", "files_via_content", "auto_routed"]
IndexType = Literal["code", "issues", "prs", "docs", "tests", "configs"]


class QueryComplexity(Enum):
    """Query complexity levels for routing decisions"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class CompositeConfig:
    """Configuration for composite agentic retriever"""
    # Chunk sizes by index type
    chunk_sizes: Dict[str, int] = None
    
    # Fusion weights for different retrieval modes
    fusion_weights: Dict[str, float] = None
    
    # Routing thresholds
    routing_thresholds: Dict[str, float] = None
    
    # Performance settings
    max_concurrent_queries: int = 5
    max_results_per_index: int = 10
    enable_reranking: bool = True
    cache_routing_decisions: bool = True
    
    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = {
                "code": 2000,
                "issues": 1500,
                "prs": 1500,
                "docs": 1000,
                "tests": 1500,
                "configs": 800
            }
        
        if self.fusion_weights is None:
            self.fusion_weights = {
                "dense": 0.6,
                "sparse": 0.3,
                "agentic": 0.1
            }
        
        if self.routing_thresholds is None:
            self.routing_thresholds = {
                "complexity_simple": 0.3,
                "complexity_moderate": 0.6,
                "complexity_complex": 0.8,
                "confidence_min": 0.4,
                "agentic_threshold": 0.5
            }


def extract_repo_info_from_url(repo_url: str) -> Dict[str, str]:
    """Extract owner and repo name from GitHub URL"""
    # Remove .git suffix and trailing slashes
    clean_url = repo_url.rstrip('/').replace('.git', '')
    parts = clean_url.split('/')
    
    if len(parts) >= 2:
        owner = parts[-2]
        repo = parts[-1]
        return {"owner": owner, "repo": repo}
    else:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")


class CompositeAgenticRetriever:
    """
    Multi-index retrieval system with intelligent routing and fusion capabilities.
    Manages specialized indices for different content types and retrieval modes.
    """
    
    def __init__(self, session_id: str, config: Optional[CompositeConfig] = None):
        self.session_id = session_id
        self.config = config or CompositeConfig()
        self.indices: Dict[str, Any] = {}
        self.routing_cache: Dict[str, Any] = {}
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "routing_decisions": defaultdict(int),
            "index_usage": defaultdict(int),
            "fusion_applied": 0,
            "parallel_retrievals": 0,
            "avg_processing_time": 0.0
        }
        
        # Redis cache for routing decisions
        self._redis_cache = None
        self._initialize_redis_cache()
    
    def _initialize_redis_cache(self) -> None:
        """Initialize Redis cache for routing decisions"""
        try:
            if self.config.cache_routing_decisions:
                # Try to import Redis cache but don't fail if not available
                try:
                    from .cache.redis_cache import redis_cache
                    self._redis_cache = redis_cache
                    logger.debug("Redis cache initialized for routing decisions")
                except ImportError:
                    logger.debug("Redis cache not available, using memory only")
                    self._redis_cache = None
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self._redis_cache = None
    
    async def initialize_indices(
        self,
        repo_path: str,
        rag_extractor: LocalRepoContextExtractor,
        issue_rag: Optional[IssueAwareRAG] = None
    ) -> None:
        """Initialize specialized indices for different content types"""
        start_time = time.time()
        
        try:
            logger.info(f"Initializing composite indices for session {self.session_id}")
            
            # Primary code index (reuse existing RAG extractor)
            self.indices["code"] = rag_extractor
            
            # Issue index (if available)
            if issue_rag and issue_rag.is_initialized():
                self.indices["issues"] = issue_rag
                logger.info("Issue index initialized")
            
            processing_time = time.time() - start_time
            logger.info(f"Composite indices initialized in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize composite indices: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        retrieval_mode: RetrievalMode = "auto_routed",
        max_results: int = 15
    ) -> Dict[str, Any]:
        """
        Main retrieval method with intelligent routing and fusion
        """
        start_time = time.time()
        self._stats["total_queries"] += 1
        
        try:
            # Step 1: Analyze query and determine routing (simplified)
            query_analysis = await self._analyze_query_simple(query)
            
            # Step 2: Route to appropriate indices (simplified)
            target_indices = self._select_target_indices_simple(query)
            
            # Step 3: Parallel retrieval from indices for better performance
            all_chunks = await self._retrieve_from_indices_parallel(query, target_indices, max_results)
            
            # Step 4: Intelligent fusion with weighted scoring
            fused_chunks = await self._apply_intelligent_fusion(all_chunks, query, query_analysis)
            
            # Step 5: Create response
            total_time = time.time() - start_time
            
            result = {
                "context_chunks": fused_chunks[:max_results],
                "query_analysis": query_analysis,
                "total_processing_time": total_time,
                "cache_hits": self._stats["cache_hits"],
                "fusion_applied": len(target_indices) > 1,
                "reranking_applied": True,
                "indices_queried": target_indices,
                "fusion_strategy": "intelligent_weighted"
            }
            
            logger.debug(f"Composite retrieval completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Composite retrieval failed: {e}")
            # Fallback to simple code index
            return await self._fallback_retrieve(query, max_results)
    
    async def _analyze_query_simple(self, query: str) -> Dict[str, Any]:
        """Simplified query analysis"""
        query_lower = query.lower()
        
        # Determine complexity
        word_count = len(query.split())
        if word_count > 20:
            complexity = QueryComplexity.COMPLEX
        elif word_count > 10:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        # Check for agentic patterns
        agentic_patterns = [
            "explain", "analyze", "how does", "implement", "create", "find all",
            "comprehensive", "detailed", "step by step"
        ]
        should_use_agentic = any(pattern in query_lower for pattern in agentic_patterns)
        
        return {
            "query_type": "general",
            "complexity": complexity,
            "should_use_agentic": should_use_agentic,
            "confidence": 0.7,
            "processing_time": 0.01
        }
    
    def _select_target_indices_simple(self, query: str) -> List[str]:
        """Simple index selection"""
        target_indices = ["code"]  # Always include code
        
        query_lower = query.lower()
        
        # Add issue index for bug/problem queries
        if any(word in query_lower for word in ["issue", "bug", "problem", "error", "fix"]):
            target_indices.append("issues")
        
        return target_indices
    
    async def _retrieve_from_indices_parallel(self, query: str, target_indices: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Retrieve from multiple indices in parallel for better performance"""
        import asyncio
        
        tasks = []
        
        # Code index task
        if "code" in self.indices:
            async def get_code_chunks():
                try:
                    code_context = await self.indices["code"].get_relevant_context(query, None)
                    code_chunks = code_context.get("sources", [])
                    for chunk in code_chunks:
                        chunk["source_index"] = "code"
                        chunk["index_type"] = "code"
                        chunk["relevance_score"] = chunk.get("similarity", chunk.get("score", 0.5))
                    return code_chunks[:max_results//2]
                except Exception as e:
                    logger.warning(f"Error querying code index: {e}")
                    return []
            
            tasks.append(get_code_chunks())
        
        # Issue index task
        if "issues" in self.indices and "issues" in target_indices:
            async def get_issue_chunks():
                try:
                    issue_context = await self.indices["issues"].get_issue_context(query, max_issues=max_results//3)
                    issue_chunks = []
                    if issue_context.related_issues:
                        for result in issue_context.related_issues:
                            issue_chunk = {
                                "content": f"Issue #{result.issue.id}: {result.issue.title}\n{result.issue.body[:500]}",
                                "file": f"issue_{result.issue.id}",
                                "similarity": result.similarity,
                                "relevance_score": result.similarity,
                                "type": "issue",
                                "source_index": "issues",
                                "index_type": "issue",
                                "issue_id": result.issue.id,
                                "issue_state": getattr(result.issue, 'state', 'unknown')
                            }
                            issue_chunks.append(issue_chunk)
                    return issue_chunks
                except Exception as e:
                    logger.warning(f"Error querying issue index: {e}")
                    return []
            
            tasks.append(get_issue_chunks())
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_chunks = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Index query failed: {result}")
                    continue
                if isinstance(result, list):
                    all_chunks.extend(result)
            
            return all_chunks
        
        return []
    
    async def _apply_intelligent_fusion(self, chunks: List[Dict[str, Any]], query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply intelligent fusion algorithm with weighted scoring"""
        if not chunks:
            return []
        
        # Define weights based on query type and content type
        fusion_weights = self._get_fusion_weights(query, query_analysis)
        
        # Apply fusion scoring
        for chunk in chunks:
            base_score = chunk.get("relevance_score", 0.5)
            index_type = chunk.get("index_type", "unknown")
            
            # Apply content type weight
            content_weight = fusion_weights.get(index_type, 1.0)
            
            # Apply recency boost for issues
            recency_boost = 1.0
            if index_type == "issue":
                issue_state = chunk.get("issue_state", "unknown")
                if issue_state == "open":
                    recency_boost = 1.2  # Boost open issues
                elif issue_state == "closed":
                    recency_boost = 0.9  # Slightly lower closed issues
            
            # Apply query-content alignment boost
            alignment_boost = self._calculate_content_alignment(chunk, query)
            
            # Calculate final fusion score
            fusion_score = base_score * content_weight * recency_boost * alignment_boost
            chunk["fusion_score"] = fusion_score
            chunk["fusion_details"] = {
                "base_score": base_score,
                "content_weight": content_weight,
                "recency_boost": recency_boost,
                "alignment_boost": alignment_boost
            }
        
        # Sort by fusion score
        chunks.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        
        # Apply diversity filtering to avoid redundant results
        diverse_chunks = self._apply_diversity_filtering(chunks)
        
        return diverse_chunks
    
    def _get_fusion_weights(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Get fusion weights based on query characteristics"""
        query_lower = query.lower()
        
        # Default weights
        weights = {
            "code": 1.0,
            "issue": 0.8,
            "pr": 0.9,
            "docs": 0.7
        }
        
        # Adjust weights based on query content
        if any(word in query_lower for word in ["bug", "error", "problem", "issue", "broken"]):
            weights["issue"] = 1.3  # Boost issues for bug-related queries
            weights["code"] = 0.9
        
        elif any(word in query_lower for word in ["implement", "function", "class", "method"]):
            weights["code"] = 1.2  # Boost code for implementation queries
            weights["issue"] = 0.7
        
        elif any(word in query_lower for word in ["review", "change", "diff", "pull request"]):
            weights["pr"] = 1.3  # Boost PRs for review-related queries
            weights["code"] = 0.8
        
        return weights
    
    def _calculate_content_alignment(self, chunk: Dict[str, Any], query: str) -> float:
        """Calculate how well the chunk content aligns with the query"""
        content = chunk.get("content", "")
        if not content:
            return 1.0
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(content_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 1.0
        
        # Base alignment score
        alignment = overlap / total_query_words
        
        # Boost for exact phrase matches
        if query.lower() in content.lower():
            alignment *= 1.5
        
        # Normalize to reasonable range
        return min(max(alignment, 0.5), 2.0)
    
    def _apply_diversity_filtering(self, chunks: List[Dict[str, Any]], max_similar: int = 3) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid too many similar results"""
        if len(chunks) <= max_similar:
            return chunks
        
        diverse_chunks = []
        seen_files = set()
        file_count = {}
        
        for chunk in chunks:
            file_path = chunk.get("file", "unknown")
            
            # Limit results per file
            if file_path not in file_count:
                file_count[file_path] = 0
            
            if file_count[file_path] < 2:  # Max 2 chunks per file
                diverse_chunks.append(chunk)
                file_count[file_path] += 1
                seen_files.add(file_path)
            
            # Stop if we have enough diverse results
            if len(diverse_chunks) >= len(chunks):
                break
        
        return diverse_chunks
    
    def _generate_cache_key(self, query: str, retrieval_mode: str, max_results: int) -> str:
        """Generate cache key for query results"""
        import hashlib
        
        # Create a hash of the query parameters
        key_data = f"{query}_{retrieval_mode}_{max_results}_{self.session_id}"
        cache_key = f"composite_rag_{hashlib.md5(key_data.encode()).hexdigest()}"
        return cache_key
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        try:
            # Check memory cache first
            if cache_key in self.routing_cache:
                cached_data = self.routing_cache[cache_key]
                # Check if cache is still valid (5 minutes)
                if time.time() - cached_data.get("timestamp", 0) < 300:
                    return cached_data.get("result")
                else:
                    # Remove expired cache
                    del self.routing_cache[cache_key]
            
            # Check Redis cache if available
            if self._redis_cache:
                cached_result = await self._redis_cache.get(cache_key)
                if cached_result:
                    # Store in memory cache for faster access
                    self.routing_cache[cache_key] = {
                        "result": cached_result,
                        "timestamp": time.time()
                    }
                    return cached_result
            
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache the result for future use"""
        try:
            # Store in memory cache
            self.routing_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Limit memory cache size
            if len(self.routing_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.routing_cache.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )
                for old_key, _ in sorted_cache[:20]:  # Remove 20 oldest
                    del self.routing_cache[old_key]
            
            # Store in Redis cache if available
            if self._redis_cache:
                await self._redis_cache.set(cache_key, result, ttl=300)  # 5 minute TTL
            
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for optimization insights"""
        total_queries = self._stats["total_queries"]
        cache_hit_rate = (self._stats["cache_hits"] / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "fusion_applied": self._stats["fusion_applied"],
            "parallel_retrievals": self._stats["parallel_retrievals"],
            "avg_processing_time": f"{self._stats['avg_processing_time']:.3f}s",
            "routing_decisions": dict(self._stats["routing_decisions"]),
            "index_usage": dict(self._stats["index_usage"])
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.routing_cache.clear()
        logger.info("Composite RAG cache cleared")
    
    async def _fallback_retrieve(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback to simple code index retrieval"""
        try:
            if "code" in self.indices:
                code_index = self.indices["code"]
                context = await code_index.get_relevant_context(query, None)
                
                chunks = context.get("sources", [])[:max_results]
                
                return {
                    "context_chunks": chunks,
                    "query_analysis": {"query_type": "general", "complexity": QueryComplexity.SIMPLE},
                    "total_processing_time": 0.1,
                    "cache_hits": 0,
                    "fusion_applied": False,
                    "reranking_applied": False
                }
                
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
        
        # Ultimate fallback
        return {
            "context_chunks": [],
            "query_analysis": {"query_type": "error", "complexity": QueryComplexity.SIMPLE},
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "fusion_applied": False,
            "reranking_applied": False
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "total_queries": self._stats["total_queries"],
            "cache_hit_rate": self._stats["cache_hits"] / max(1, self._stats["total_queries"]),
            "routing_decisions": dict(self._stats["routing_decisions"]),
            "index_usage": dict(self._stats["index_usage"]),
            "available_indices": list(self.indices.keys())
        }


class AgenticRAGSystem:
    """
    Enhanced RAG system that combines semantic retrieval with agentic tool capabilities
    for more intelligent context extraction and query processing.
    
    Now features composite multi-index retrieval with intelligent routing.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.rag_extractor = None
        self.agentic_explorer = None
        self.founding_member_agent = None  # New: Advanced analysis agent
        self.issue_rag = None
        self.repo_path = None
        self.repo_info = None
        self._query_cache = {}
        
        # New composite retriever
        self.composite_retriever = CompositeAgenticRetriever(session_id)
        self._use_composite = False  # Flag to enable composite retrieval
        
        # Enhanced logging
        self._setup_enhanced_logging()
    
    def _setup_enhanced_logging(self) -> None:
        """Setup enhanced structured logging"""
        self.logger = logging.getLogger(f"agentic_rag.{self.session_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def initialize_core_systems(self, repo_url: str, branch: str = "main") -> None:
        """Initialize core RAG systems without issue RAG (that comes later async)"""
        try:
            self.logger.info(f"Initializing AgenticRAG core systems for session {self.session_id}")
            
            # Check if this repo is already loaded in memory cache
            repo_info_from_url = extract_repo_info_from_url(repo_url)
            repo_key = f"{repo_info_from_url['owner']}/{repo_info_from_url['repo']}"
            
            # Import the global cache to check if instance already exists
            from .api.dependencies import agentic_rag_cache
            if repo_key in agentic_rag_cache:
                existing_instance = agentic_rag_cache[repo_key]
                self.logger.info(f"Found existing AgenticRAG instance for {repo_key}, reusing core components")
                
                # Copy existing components instead of rebuilding
                self.rag_extractor = existing_instance.rag_extractor
                self.repo_path = existing_instance.repo_path
                self.repo_info = existing_instance.repo_info
                self.agentic_explorer = existing_instance.agentic_explorer
                self.issue_rag = existing_instance.issue_rag
                
                # More robust check for RAG extractor validity
                rag_is_valid = (
                    self.rag_extractor and 
                    hasattr(self.rag_extractor, 'query_engine') and 
                    self.rag_extractor.query_engine is not None and
                    hasattr(self.rag_extractor, 'vector_store') and
                    self.rag_extractor.vector_store is not None
                )
                
                if not rag_is_valid:
                    self.logger.warning(f"RAG extractor from cache missing critical components, will need fresh initialization")
                    # Instead of setting to None, try to reinitialize with existing repo path
                    if existing_instance.repo_path and hasattr(existing_instance, 'repo_path') and existing_instance.repo_path:
                        import os
                        if os.path.exists(existing_instance.repo_path):
                            self.logger.info(f"Attempting to reinitialize RAG extractor using existing repo at {existing_instance.repo_path}")
                            try:
                                code_rag = LocalRepoContextExtractor()
                                code_rag.current_repo_path = existing_instance.repo_path
                                # Try to rebuild indices from existing repository without re-cloning
                                await code_rag._rebuild_indices_from_existing_repo()
                                self.rag_extractor = code_rag
                                # Update the cache with the fixed rag_extractor
                                existing_instance.rag_extractor = code_rag
                                self.logger.info(f"Successfully reinitialized RAG extractor for {repo_key} from existing repo")
                            except Exception as reinit_error:
                                self.logger.error(f"Failed to reinitialize RAG extractor: {reinit_error}")
                                # Fall back to full re-cloning only as last resort
                                self.rag_extractor = None
                        else:
                            self.logger.warning(f"Existing repo path doesn't exist: {existing_instance.repo_path}")
                            self.rag_extractor = None
                    else:
                        self.logger.warning(f"No existing repo path available")
                        self.rag_extractor = None
                
                # If rag_extractor is still missing or invalid, create a fresh one
                if not self.rag_extractor:
                    self.logger.info(f"Creating fresh RAG extractor for cached instance {repo_key}")
                    code_rag = LocalRepoContextExtractor()
                    await code_rag.load_repository(repo_url, branch)
                    self.rag_extractor = code_rag
                    # Update the cache with the fresh rag_extractor
                    existing_instance.rag_extractor = code_rag
                    # Also update repo_path in case it changed
                    existing_instance.repo_path = code_rag.current_repo_path
                    self.repo_path = code_rag.current_repo_path
                
                # Initialize composite retriever with existing components
                if self.rag_extractor and self.repo_path:
                    await self._initialize_composite_retriever()
                
                self.logger.info(f"AgenticRAG core systems reused from cache for session {self.session_id}")
                return
            
            # If not in cache, proceed with initialization
            self.logger.info(f"No cached instance found for {repo_key}, initializing fresh")
            
            # Load repository locally  
            code_rag = LocalRepoContextExtractor()
            await code_rag.load_repository(repo_url, branch)
            
            self.rag_extractor = code_rag
            self.repo_path = code_rag.current_repo_path
            self.repo_info = repo_info_from_url
            
            # Initialize AgenticCodebaseExplorer with the repo path
            try:
                from .agent_tools.core import AgenticCodebaseExplorer
                self.agentic_explorer = AgenticCodebaseExplorer(
                    self.session_id, 
                    self.repo_path, 
                    issue_rag_system=None  # Will be set later
                )
            except ImportError as e:
                self.logger.warning(f"AgenticCodebaseExplorer not available: {e}")
                self.agentic_explorer = None
            
            # Initialize commit index - this should load existing cache when available
            if self.agentic_explorer:
                try:
                    self.logger.info(f"Initializing commit index for session {self.session_id}")
                    # force_rebuild=False means it will load existing cache if available
                    await self.agentic_explorer.initialize_commit_index(force_rebuild=False)
                    
                    # Verify initialization
                    if hasattr(self.agentic_explorer, 'commit_index_manager'):
                        stats = self.agentic_explorer.commit_index_manager.get_statistics()
                        self.logger.info(f"Commit index initialized for session {self.session_id}: {stats}")
                        
                        # Check if we got reasonable data from cache
                        if stats.get('total_commits', 0) < 10:
                            self.logger.warning(f"Low commit count ({stats.get('total_commits', 0)}) for session {self.session_id} - may indicate cache miss or small repo")
                    else:
                        self.logger.warning(f"Commit index manager not available for session {self.session_id}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to initialize commit index for session {self.session_id}: {e}")
                    import traceback
                    self.logger.warning(f"Full traceback: {traceback.format_exc()}")
                    # Continue without commit index - other tools will still work
            
            # Initialize composite retriever
            await self._initialize_composite_retriever()
            
            # Initialize FoundingMemberAgent for advanced analysis (but only after basic systems are ready)
            await self._initialize_founding_member_agent()
            
            self.logger.info(f"AgenticRAG core systems initialized for session {self.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AgenticRAG core systems: {e}")
            raise
    
    async def _initialize_composite_retriever(self) -> None:
        """Initialize the composite retriever with current components"""
        try:
            if self.rag_extractor and self.repo_path:
                await self.composite_retriever.initialize_indices(
                    self.repo_path,
                    self.rag_extractor,
                    self.issue_rag
                )
                self._use_composite = True
                self.logger.info("Composite retriever initialized successfully")
            else:
                self.logger.warning("Cannot initialize composite retriever: missing core components")
        except Exception as e:
            self.logger.warning(f"Failed to initialize composite retriever: {e}")
            self._use_composite = False
    
    async def _initialize_founding_member_agent(self) -> None:
        """Initialize the FoundingMemberAgent for advanced repository analysis"""
        try:
            if self.rag_extractor and self.repo_path:
                # Import FoundingMemberAgent
                from .founding_member_agent import FoundingMemberAgent
                
                # Initialize with code RAG and issue RAG (if available)
                self.founding_member_agent = FoundingMemberAgent(
                    session_id=self.session_id,
                    code_rag=self.rag_extractor,
                    issue_rag=self.issue_rag  # May be None initially, updated later
                )
                
                self.logger.info("FoundingMemberAgent initialized successfully")
            else:
                self.logger.warning("Cannot initialize FoundingMemberAgent: missing core components")
        except Exception as e:
            self.logger.warning(f"Failed to initialize FoundingMemberAgent: {e}")
            self.founding_member_agent = None

    async def initialize_issue_rag_async(self, session: Dict[str, Any]) -> None:
        """Initializes the IssueAwareRAG system asynchronously and updates session status."""
        from .patch_linkage import ProgressUpdate
        
        if not self.repo_info:
            self.logger.error(f"Session {self.session_id}: Cannot initialize IssueAwareRAG: repo_info is missing.")
            session["metadata"]["status"] = "error_repo_info_missing"
            session["metadata"]["message"] = "Error: Repository information missing, cannot load issue context."
            return

        owner = self.repo_info.get("owner")
        repo_name = self.repo_info.get("repo")

        if not (owner and repo_name):
            self.logger.error(f"Session {self.session_id}: Cannot initialize IssueAwareRAG: owner or repo name is missing from repo_info.")
            session["metadata"]["status"] = "error_repo_details_missing"
            session["metadata"]["message"] = "Error: Repository owner/name missing, cannot load issue context."
            return

        def progress_callback(update: ProgressUpdate):
            """Update session metadata with detailed progress information"""
            try:
                # Update session metadata with detailed progress
                session["metadata"]["status"] = "issue_linking"
                session["metadata"]["progress_stage"] = update.stage
                session["metadata"]["progress_step"] = update.current_step
                session["metadata"]["progress_percentage"] = update.progress_percentage
                session["metadata"]["progress_items_processed"] = update.items_processed
                session["metadata"]["progress_total_items"] = update.total_items
                session["metadata"]["progress_current_item"] = update.current_item
                session["metadata"]["progress_estimated_time"] = update.estimated_time_remaining
                session["metadata"]["progress_details"] = update.details
                
                # Create user-friendly message
                if update.current_item:
                    message = f"{update.current_step}: {update.current_item}"
                else:
                    message = f"{update.current_step} ({update.items_processed}/{update.total_items})"
                
                if update.estimated_time_remaining:
                    minutes = update.estimated_time_remaining // 60
                    seconds = update.estimated_time_remaining % 60
                    if minutes > 0:
                        message += f" - ~{minutes}m {seconds}s remaining"
                    else:
                        message += f" - ~{seconds}s remaining"
                
                session["metadata"]["message"] = message
                
                self.logger.info(f"Progress: {update.stage} - {update.current_step} ({update.progress_percentage:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")

        try:
            self.logger.info(f"Session {self.session_id}: Starting IssueAwareRAG initialization for {owner}/{repo_name}.")
            session["metadata"]["status"] = "issue_linking" 
            session["metadata"]["message"] = f"Starting issue linking and indexing for {owner}/{repo_name}..."
            
            # Create fresh IssueAwareRAG with progress callback to avoid coroutine reuse
            self.issue_rag = IssueAwareRAG(owner, repo_name, progress_callback)
            
            # Initialize with explicit error handling for coroutine reuse
            try:
                await self.issue_rag.initialize(force_rebuild=False) 
            except RuntimeError as re:
                if "cannot reuse already awaited coroutine" in str(re):
                    self.logger.warning(f"Session {self.session_id}: Coroutine reuse detected, retrying with force rebuild...")
                    # Create a completely fresh instance and force rebuild
                    self.issue_rag = IssueAwareRAG(owner, repo_name, progress_callback)
                    await self.issue_rag.initialize(force_rebuild=True)
                else:
                    raise re
            
            if self.agentic_explorer: 
                self.agentic_explorer.issue_rag_system = self.issue_rag
                # Also update the sub-components that depend on issue_rag_system
                if hasattr(self.agentic_explorer, 'pr_ops'):
                    self.agentic_explorer.pr_ops.issue_rag_system = self.issue_rag
                if hasattr(self.agentic_explorer, 'issue_ops'):
                    self.agentic_explorer.issue_ops.issue_rag_system = self.issue_rag
            
            # Update composite retriever with issue RAG
            if self._use_composite and self.composite_retriever:
                self.composite_retriever.indices["issues"] = self.issue_rag
                self.logger.info("Updated composite retriever with issue RAG")
            
            # Update FoundingMemberAgent with issue RAG
            if self.founding_member_agent:
                self.founding_member_agent.issue_rag = self.issue_rag
                # Reinitialize the explorer in FoundingMemberAgent to use the new issue_rag
                from .agent_tools.core import AgenticCodebaseExplorer
                self.founding_member_agent.explorer = AgenticCodebaseExplorer(
                    self.session_id, 
                    self.repo_path, 
                    issue_rag_system=self.issue_rag
                )
                self.logger.info("Updated FoundingMemberAgent with issue RAG")
            
            self.logger.info(f"Session {self.session_id}: IssueAwareRAG for {owner}/{repo_name} initialized successfully.")
            session["metadata"]["issue_rag_ready"] = True
            session["metadata"]["status"] = "ready" 
            session["metadata"]["message"] = "All systems ready. Full repository context available."
            
            # Clear progress fields since we're done
            for key in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", 
                       "progress_total_items", "progress_current_item", "progress_estimated_time", "progress_details"]:
                session["metadata"].pop(key, None)

        except Exception as e:
            self.logger.error(f"Session {self.session_id}: Failed to initialize IssueAwareRAG for {owner}/{repo_name}: {e}")
            
            # Log more detailed error information for debugging
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
            self.issue_rag = None # Ensure it's None if init failed
            session["metadata"]["issue_rag_ready"] = False
            session["metadata"]["status"] = "warning_issue_rag_failed"
            session["metadata"]["message"] = f"Core chat is ready. Issue context for {owner}/{repo_name} failed to load: {str(e)}"
            session["metadata"]["error_details_issue_rag"] = str(e) # Store specific error
            
            # Clear progress fields on error
            for key in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", 
                       "progress_total_items", "progress_current_item", "progress_estimated_time", "progress_details"]:
                session["metadata"].pop(key, None)
    
    async def get_enhanced_context(
        self, 
        query: str, 
        restrict_files: Optional[List[str]] = None,
        use_agentic_tools: bool = True,
        include_issue_context: bool = True
    ) -> List[ContextChunk]:
        """
        Get enhanced context using both RAG and agentic capabilities with issue awareness.
        
        This is the main public API method that now transparently decides between
        legacy single-index path and new composite multi-index path.
        """
        if not self.rag_extractor:
            raise ValueError("AgenticRAG not properly initialized")
        
        try:
            # Decide whether to use composite retrieval based on query complexity
            use_composite = self._should_use_composite_retrieval(query)
            
            if use_composite and self._use_composite:
                # Use new composite multi-index retrieval
                return await self._get_enhanced_context_composite(
                    query, restrict_files, use_agentic_tools, include_issue_context
                )
            else:
                # Use legacy single-index path for backward compatibility
                return await self._get_enhanced_context_legacy(
                    query, restrict_files, use_agentic_tools, include_issue_context
                )
                
        except Exception as e:
            self.logger.error(f"Error in enhanced context retrieval: {e}")
            # Fallback to base RAG
            try:
                base_context = await self.rag_extractor.get_relevant_context(query, restrict_files)
                return base_context.get("sources", [])
            except Exception as fallback_error:
                self.logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []
    
    def _should_use_composite_retrieval(self, query: str) -> bool:
        """Decide whether to use composite retrieval based on query characteristics"""
        if not self._use_composite:
            return False
        
        # Use composite for complex queries
        query_lower = query.lower()
        
        # Multi-domain queries
        domain_indicators = ["issue", "test", "config", "document", "readme"]
        domain_count = sum(1 for indicator in domain_indicators if indicator in query_lower)
        
        if domain_count >= 2:
            return True
        
        # Complex exploration queries
        complex_patterns = [
            "comprehensive", "detailed", "thorough", "all files", "entire",
            "architecture", "structure", "relationship", "dependency"
        ]
        
        if any(pattern in query_lower for pattern in complex_patterns):
            return True
        
        # Long queries likely benefit from multi-index
        if len(query.split()) > 15:
            return True
        
        return False
    
    async def _get_enhanced_context_composite(
        self,
        query: str,
        restrict_files: Optional[List[str]] = None,
        use_agentic_tools: bool = True,
        include_issue_context: bool = True
    ) -> List[ContextChunk]:
        """Enhanced context retrieval using composite multi-index system"""
        
        try:
            # Step 1: Use composite retriever
            composite_result = await self.composite_retriever.retrieve(
                query, retrieval_mode="auto_routed", max_results=15
            )
            
            # Step 2: Convert to legacy format for backward compatibility
            context_chunks = composite_result.get("context_chunks", [])
            
            # Step 3: Log composite retrieval statistics
            stats = self.composite_retriever.get_statistics()
            self.logger.info(f"Composite retrieval stats: {stats}")
            
            return context_chunks
            
        except Exception as e:
            self.logger.error(f"Composite retrieval failed: {e}")
            # Fallback to legacy path
            return await self._get_enhanced_context_legacy(
                query, restrict_files, use_agentic_tools, include_issue_context
            )
    
    async def _get_enhanced_context_legacy(
        self,
        query: str,
        restrict_files: Optional[List[str]] = None,
        use_agentic_tools: bool = True,
        include_issue_context: bool = True
    ) -> List[ContextChunk]:
        """Legacy enhanced context retrieval (original implementation)"""
        
        try:
            # Get base RAG context
            base_context = await self.rag_extractor.get_relevant_context(query, restrict_files)
            
            # Get issue context if enabled and available
            if include_issue_context and self.issue_rag and self.issue_rag.is_initialized():
                try:
                    issue_context = await self.issue_rag.get_issue_context(query, max_issues=3)
                    if issue_context.related_issues:
                        # Convert issue context to chunk format
                        issue_chunks = [
                            {
                                "content": f"Issue #{result.issue.id}: {result.issue.title}\n{result.issue.body[:500]}...",
                                "file": f"issue_{result.issue.id}",
                                "similarity": result.similarity,
                                "type": "issue",
                                "url": f"https://github.com/{self.repo_info.get('owner')}/{self.repo_info.get('repo')}/issues/{result.issue.id}" if self.repo_info else ""
                            }
                            for result in issue_context.related_issues
                        ]
                        
                        # Add issue chunks to sources
                        sources = base_context.get("sources", [])
                        sources.extend(issue_chunks)
                        base_context["sources"] = sources
                        
                        self.logger.info(f"Added {len(issue_context.related_issues)} related issues to context")
                except Exception as e:
                    self.logger.warning(f"Failed to get issue context: {e}")
            
            # Return chunks
            return base_context.get("sources", [])
                
        except Exception as e:
            self.logger.error(f"Legacy context retrieval failed: {e}")
            # Ultimate fallback
            try:
                base_context = await self.rag_extractor.get_relevant_context(query, restrict_files)
                return base_context.get("sources", [])
            except Exception:
                return []
    
    def get_repo_info(self) -> Optional[Dict[str, Any]]:
        """Get repository information"""
        return self.repo_info
    
    def get_repo_path(self) -> Optional[str]:
        """Get repository path"""
        return self.repo_path
    
    def get_composite_statistics(self) -> Optional[Dict[str, Any]]:
        """Get composite retriever statistics"""
        if self.composite_retriever:
            return self.composite_retriever.get_statistics()
        return None
    
    # FoundingMemberAgent integration methods
    async def get_file_history(self, file_path: str) -> str:
        """Get the timeline of all PRs and issues that touched a given file"""
        if self.founding_member_agent:
            return self.founding_member_agent.get_file_history(file_path)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def summarize_feature_evolution(self, feature_query: str) -> str:
        """Summarize how a feature evolved over time by searching issues, PRs, and diffs"""
        if self.founding_member_agent:
            return self.founding_member_agent.summarize_feature_evolution(feature_query)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def who_fixed_this(self, file_path: str, line_number: int = None) -> str:
        """Find who/what last changed a file (and optionally a line)"""
        if self.founding_member_agent:
            return self.founding_member_agent.who_fixed_this(file_path, line_number)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def who_implemented_this(self, feature_name: str, file_path: Optional[str] = None) -> str:
        """Find who initially implemented a feature/class/function"""
        if self.founding_member_agent:
            return self.founding_member_agent.who_implemented_this(feature_name, file_path)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def regression_detector(self, issue_query: str) -> str:
        """Detect if a new issue is a regression of a past one"""
        if self.founding_member_agent:
            return await self.founding_member_agent.regression_detector(issue_query)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def agentic_analysis(self, user_query: str) -> str:
        """Perform advanced agentic analysis using FoundingMemberAgent"""
        if self.founding_member_agent:
            return await self.founding_member_agent.agentic_answer(user_query)
        import json
        return json.dumps({"error": "FoundingMemberAgent not available"})
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.agentic_explorer:
                self.agentic_explorer.reset_memory()
            
            # Clear caches
            self._query_cache.clear()
            
            # Clear composite retriever caches
            if self.composite_retriever:
                self.composite_retriever.routing_cache.clear()
            
            self.logger.info(f"AgenticRAG cleaned up for session {self.session_id}")
            
        except Exception as e:
            self.logger.warning(f"Error during AgenticRAG cleanup: {e}")