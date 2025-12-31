"""Agentic RAG Integration - tinygrad-style rewrite (1,149→~400 lines)"""
import asyncio, logging, time, hashlib, json
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .new_rag import LocalRepoContextExtractor
from .issue_rag import IssueAwareRAG

logger = logging.getLogger(__name__)

ContextChunk = Dict[str, Any]
RetrievalMode = Literal["chunks", "files_via_metadata", "files_via_content", "auto_routed"]


class QueryComplexity(Enum):
    SIMPLE = "simple"; MODERATE = "moderate"; COMPLEX = "complex"; EXPERT = "expert"


@dataclass
class CompositeConfig:
    chunk_sizes: Dict[str, int] = field(default_factory=lambda: {"code": 2000, "issues": 1500, "prs": 1500, "docs": 1000, "tests": 1500, "configs": 800})
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {"dense": 0.6, "sparse": 0.3, "agentic": 0.1})
    routing_thresholds: Dict[str, float] = field(default_factory=lambda: {"complexity_simple": 0.3, "complexity_moderate": 0.6, "complexity_complex": 0.8, "confidence_min": 0.4, "agentic_threshold": 0.5})
    max_concurrent_queries: int = 5; max_results_per_index: int = 10; enable_reranking: bool = True; cache_routing_decisions: bool = True


def extract_repo_info_from_url(repo_url: str) -> Dict[str, str]:
    clean = repo_url.rstrip('/').replace('.git', '')
    parts = clean.split('/')
    if len(parts) < 2: raise ValueError(f"Invalid GitHub URL: {repo_url}")
    return {"owner": parts[-2], "repo": parts[-1]}


class CompositeAgenticRetriever:
    """Multi-index retrieval with intelligent routing and fusion"""

    def __init__(self, session_id: str, config: Optional[CompositeConfig] = None):
        self.session_id, self.config = session_id, config or CompositeConfig()
        self.indices, self.routing_cache = {}, {}
        self._stats = {"total_queries": 0, "cache_hits": 0, "routing_decisions": defaultdict(int), "index_usage": defaultdict(int), "fusion_applied": 0, "parallel_retrievals": 0, "avg_processing_time": 0.0}
        self._redis_cache = None
        if self.config.cache_routing_decisions:
            try:
                from .cache.redis_cache import redis_cache
                self._redis_cache = redis_cache
            except: pass

    async def initialize_indices(self, repo_path: str, rag_extractor: LocalRepoContextExtractor, issue_rag: Optional[IssueAwareRAG] = None) -> None:
        self.indices["code"] = rag_extractor
        if issue_rag and issue_rag.is_initialized(): self.indices["issues"] = issue_rag

    async def retrieve(self, query: str, retrieval_mode: RetrievalMode = "auto_routed", max_results: int = 15) -> Dict[str, Any]:
        start = time.time()
        self._stats["total_queries"] += 1

        try:
            analysis = self._analyze_query(query)
            targets = ["code"] + (["issues"] if any(w in query.lower() for w in ["issue", "bug", "problem", "error", "fix"]) else [])
            chunks = await self._retrieve_parallel(query, targets, max_results)
            fused = self._apply_fusion(chunks, query, analysis)

            return {"context_chunks": fused[:max_results], "query_analysis": analysis, "total_processing_time": time.time() - start,
                    "cache_hits": self._stats["cache_hits"], "fusion_applied": len(targets) > 1, "reranking_applied": True,
                    "indices_queried": targets, "fusion_strategy": "intelligent_weighted"}
        except Exception as e:
            logger.error(f"Composite retrieval failed: {e}")
            return await self._fallback(query, max_results)

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        words = len(query.split())
        complexity = QueryComplexity.COMPLEX if words > 20 else (QueryComplexity.MODERATE if words > 10 else QueryComplexity.SIMPLE)
        agentic = any(p in query.lower() for p in ["explain", "analyze", "how does", "implement", "create", "find all", "comprehensive", "detailed", "step by step"])
        return {"query_type": "general", "complexity": complexity, "should_use_agentic": agentic, "confidence": 0.7, "processing_time": 0.01}

    async def _retrieve_parallel(self, query: str, targets: List[str], max_results: int) -> List[Dict[str, Any]]:
        tasks = []
        if "code" in self.indices:
            async def get_code():
                try:
                    ctx = await self.indices["code"].get_relevant_context(query, None)
                    chunks = ctx.get("sources", [])[:max_results//2]
                    for c in chunks: c.update({"source_index": "code", "index_type": "code", "relevance_score": c.get("similarity", c.get("score", 0.5))})
                    return chunks
                except: return []
            tasks.append(get_code())

        if "issues" in self.indices and "issues" in targets:
            async def get_issues():
                try:
                    ctx = await self.indices["issues"].get_issue_context(query, max_issues=max_results//3)
                    return [{"content": f"Issue #{r.issue.id}: {r.issue.title}\n{r.issue.body[:500]}", "file": f"issue_{r.issue.id}",
                            "similarity": r.similarity, "relevance_score": r.similarity, "type": "issue", "source_index": "issues",
                            "index_type": "issue", "issue_id": r.issue.id, "issue_state": getattr(r.issue, 'state', 'unknown')}
                           for r in (ctx.related_issues or [])]
                except: return []
            tasks.append(get_issues())

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        return [c for r in results if isinstance(r, list) for c in r]

    def _apply_fusion(self, chunks: List[Dict], query: str, analysis: Dict) -> List[Dict]:
        if not chunks: return []
        q = query.lower()
        weights = {"code": 1.0, "issue": 0.8, "pr": 0.9, "docs": 0.7}
        if any(w in q for w in ["bug", "error", "problem"]): weights.update({"issue": 1.3, "code": 0.9})
        elif any(w in q for w in ["implement", "function", "class"]): weights.update({"code": 1.2, "issue": 0.7})

        for c in chunks:
            base = c.get("relevance_score", 0.5)
            itype = c.get("index_type", "unknown")
            state_boost = 1.2 if c.get("issue_state") == "open" else (0.9 if c.get("issue_state") == "closed" else 1.0)
            qw, cw = set(q.split()), set(c.get("content", "").lower().split())
            align = min(max(len(qw & cw) / max(len(qw), 1), 0.5), 2.0) * (1.5 if q in c.get("content", "").lower() else 1.0)
            c["fusion_score"] = base * weights.get(itype, 1.0) * state_boost * align

        chunks.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        seen, diverse = {}, []
        for c in chunks:
            fp = c.get("file", "unknown")
            if seen.get(fp, 0) < 2: diverse.append(c); seen[fp] = seen.get(fp, 0) + 1
        return diverse

    async def _fallback(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            if "code" in self.indices:
                ctx = await self.indices["code"].get_relevant_context(query, None)
                return {"context_chunks": ctx.get("sources", [])[:max_results], "query_analysis": {"query_type": "general", "complexity": QueryComplexity.SIMPLE},
                        "total_processing_time": 0.1, "cache_hits": 0, "fusion_applied": False, "reranking_applied": False}
        except: pass
        return {"context_chunks": [], "query_analysis": {"query_type": "error", "complexity": QueryComplexity.SIMPLE}, "total_processing_time": 0.0, "cache_hits": 0, "fusion_applied": False, "reranking_applied": False}

    def get_statistics(self) -> Dict[str, Any]:
        return {"total_queries": self._stats["total_queries"], "cache_hit_rate": self._stats["cache_hits"] / max(1, self._stats["total_queries"]),
                "routing_decisions": dict(self._stats["routing_decisions"]), "index_usage": dict(self._stats["index_usage"]), "available_indices": list(self.indices.keys())}

    def clear_cache(self) -> None: self.routing_cache.clear()


class AgenticRAGSystem:
    """Enhanced RAG system with agentic capabilities and composite retrieval"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.rag_extractor = self.agentic_explorer = self.founding_member_agent = self.issue_rag = self.repo_path = self.repo_info = None
        self._query_cache = {}
        self.composite_retriever = CompositeAgenticRetriever(session_id)
        self._use_composite = False
        self.logger = logging.getLogger(f"agentic_rag.{session_id}")

    async def initialize_core_systems(self, repo_url: str, branch: str = "main") -> None:
        self.logger.info(f"Initializing AgenticRAG for session {self.session_id}")
        repo_info = extract_repo_info_from_url(repo_url)
        repo_key = f"{repo_info['owner']}/{repo_info['repo']}"

        try:
            from .api.dependencies import agentic_rag_cache
            if repo_key in agentic_rag_cache:
                existing = agentic_rag_cache[repo_key]
                self.rag_extractor, self.repo_path, self.repo_info = existing.rag_extractor, existing.repo_path, existing.repo_info
                self.agentic_explorer, self.issue_rag = existing.agentic_explorer, existing.issue_rag

                if not (self.rag_extractor and hasattr(self.rag_extractor, 'query_engine') and self.rag_extractor.query_engine):
                    self.rag_extractor = None
                    if existing.repo_path and __import__('os').path.exists(existing.repo_path):
                        try:
                            code_rag = LocalRepoContextExtractor()
                            code_rag.current_repo_path = existing.repo_path
                            await code_rag._rebuild_indices_from_existing_repo()
                            self.rag_extractor = existing.rag_extractor = code_rag
                        except: self.rag_extractor = None

                if not self.rag_extractor:
                    code_rag = LocalRepoContextExtractor()
                    await code_rag.load_repository(repo_url, branch)
                    self.rag_extractor = existing.rag_extractor = code_rag
                    self.repo_path = existing.repo_path = code_rag.current_repo_path

                if self.rag_extractor and self.repo_path: await self._init_composite()
                return
        except Exception as e: logger.debug(f"Session reuse failed, creating new: {e}")

        code_rag = LocalRepoContextExtractor()
        await code_rag.load_repository(repo_url, branch)
        self.rag_extractor, self.repo_path, self.repo_info = code_rag, code_rag.current_repo_path, repo_info

        try:
            from .agent_tools.core import AgenticCodebaseExplorer
            self.agentic_explorer = AgenticCodebaseExplorer(self.session_id, self.repo_path, issue_rag_system=None)
        except Exception as e: logger.debug(f"AgenticCodebaseExplorer init failed: {e}"); self.agentic_explorer = None

        if self.agentic_explorer:
            try: await self.agentic_explorer.initialize_commit_index(force_rebuild=False)
            except Exception as e: logger.debug(f"Commit index init failed: {e}")

        await self._init_composite()
        await self._init_founding_member()

    async def _init_composite(self) -> None:
        if self.rag_extractor and self.repo_path:
            try:
                await self.composite_retriever.initialize_indices(self.repo_path, self.rag_extractor, self.issue_rag)
                self._use_composite = True
            except: self._use_composite = False

    async def _init_founding_member(self) -> None:
        if not (self.rag_extractor and self.repo_path): return
        try:
            from .founding_member_agent import FoundingMemberAgent
            self.founding_member_agent = FoundingMemberAgent(session_id=self.session_id, code_rag=self.rag_extractor, issue_rag=self.issue_rag)
        except: self.founding_member_agent = None

    async def initialize_issue_rag_async(self, session: Dict[str, Any]) -> None:
        if not self.repo_info:
            session["metadata"].update({"status": "error_repo_info_missing", "message": "Error: Repository information missing"})
            return

        owner, repo = self.repo_info.get("owner"), self.repo_info.get("repo")
        if not (owner and repo):
            session["metadata"].update({"status": "error_repo_details_missing", "message": "Error: Repository owner/name missing"})
            return

        def progress_cb(update):
            try:
                session["metadata"].update({"status": "issue_linking", "progress_stage": update.stage, "progress_step": update.current_step,
                                           "progress_percentage": update.progress_percentage, "progress_items_processed": update.items_processed,
                                           "progress_total_items": update.total_items, "progress_current_item": update.current_item})
                msg = f"{update.current_step}: {update.current_item}" if update.current_item else f"{update.current_step} ({update.items_processed}/{update.total_items})"
                if (eta := update.estimated_time_remaining): msg += f" - ~{eta//60}m {eta%60}s remaining" if eta >= 60 else f" - ~{eta}s remaining"
                session["metadata"]["message"] = msg
            except: pass

        try:
            session["metadata"].update({"status": "issue_linking", "message": f"Starting issue linking for {owner}/{repo}..."})
            self.issue_rag = IssueAwareRAG(owner, repo, progress_cb)
            try: await self.issue_rag.initialize(force_rebuild=False)
            except RuntimeError as e:
                if "cannot reuse" in str(e):
                    self.issue_rag = IssueAwareRAG(owner, repo, progress_cb)
                    await self.issue_rag.initialize(force_rebuild=True)
                else: raise

            if self.agentic_explorer:
                self.agentic_explorer.issue_rag_system = self.issue_rag
                for attr in ['pr_ops', 'issue_ops']:
                    if hasattr(self.agentic_explorer, attr): setattr(getattr(self.agentic_explorer, attr), 'issue_rag_system', self.issue_rag)

            if self._use_composite: self.composite_retriever.indices["issues"] = self.issue_rag
            if self.founding_member_agent:
                self.founding_member_agent.issue_rag = self.issue_rag
                from .agent_tools.core import AgenticCodebaseExplorer
                self.founding_member_agent.explorer = AgenticCodebaseExplorer(self.session_id, self.repo_path, issue_rag_system=self.issue_rag)

            session["metadata"].update({"issue_rag_ready": True, "status": "ready", "message": "All systems ready."})
            for k in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", "progress_total_items", "progress_current_item"]:
                session["metadata"].pop(k, None)

        except Exception as e:
            self.issue_rag = None
            session["metadata"].update({"issue_rag_ready": False, "status": "warning_issue_rag_failed", "message": f"Core chat ready. Issue context failed: {e}", "error_details_issue_rag": str(e)})
            for k in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", "progress_total_items", "progress_current_item"]:
                session["metadata"].pop(k, None)

    async def get_enhanced_context(self, query: str, restrict_files: Optional[List[str]] = None, use_agentic_tools: bool = True, include_issue_context: bool = True) -> List[ContextChunk]:
        if not self.rag_extractor: raise ValueError("AgenticRAG not initialized")

        try:
            q = query.lower()
            use_composite = self._use_composite and (len(query.split()) > 15 or sum(1 for i in ["issue", "test", "config", "document", "readme"] if i in q) >= 2
                                                     or any(p in q for p in ["comprehensive", "detailed", "thorough", "architecture", "structure"]))

            if use_composite:
                result = await self.composite_retriever.retrieve(query, retrieval_mode="auto_routed", max_results=15)
                return result.get("context_chunks", [])
            else:
                base = await self.rag_extractor.get_relevant_context(query, restrict_files)
                if include_issue_context and self.issue_rag and self.issue_rag.is_initialized():
                    try:
                        ictx = await self.issue_rag.get_issue_context(query, max_issues=3)
                        if ictx.related_issues:
                            issue_chunks = [{"content": f"Issue #{r.issue.id}: {r.issue.title}\n{r.issue.body[:500]}...", "file": f"issue_{r.issue.id}",
                                           "similarity": r.similarity, "type": "issue",
                                           "url": f"https://github.com/{self.repo_info['owner']}/{self.repo_info['repo']}/issues/{r.issue.id}" if self.repo_info else ""}
                                          for r in ictx.related_issues]
                            base["sources"] = base.get("sources", []) + issue_chunks
                    except: pass
                return base.get("sources", [])
        except Exception as e:
            self.logger.error(f"Enhanced context failed: {e}")
            try: return (await self.rag_extractor.get_relevant_context(query, restrict_files)).get("sources", [])
            except: return []

    def get_repo_info(self) -> Optional[Dict[str, Any]]: return self.repo_info
    def get_repo_path(self) -> Optional[str]: return self.repo_path
    def get_composite_statistics(self) -> Optional[Dict[str, Any]]: return self.composite_retriever.get_statistics() if self.composite_retriever else None

    def _fm_wrap(self, method: str, *args, **kwargs):
        if not self.founding_member_agent: return json.dumps({"error": "FoundingMemberAgent not available"})
        return getattr(self.founding_member_agent, method)(*args, **kwargs)

    async def get_file_history(self, file_path: str) -> str: return self._fm_wrap("get_file_history", file_path)
    async def summarize_feature_evolution(self, feature_query: str) -> str: return self._fm_wrap("summarize_feature_evolution", feature_query)
    async def who_fixed_this(self, file_path: str, line_number: int = None) -> str: return self._fm_wrap("who_fixed_this", file_path, line_number)
    async def who_implemented_this(self, feature_name: str, file_path: Optional[str] = None) -> str: return self._fm_wrap("who_implemented_this", feature_name, file_path)

    async def regression_detector(self, issue_query: str) -> str:
        return await self.founding_member_agent.regression_detector(issue_query) if self.founding_member_agent else json.dumps({"error": "FoundingMemberAgent not available"})

    async def agentic_analysis(self, user_query: str) -> str:
        return await self.founding_member_agent.agentic_answer(user_query) if self.founding_member_agent else json.dumps({"error": "FoundingMemberAgent not available"})

    async def cleanup(self):
        try:
            if self.agentic_explorer: self.agentic_explorer.reset_memory()
            self._query_cache.clear()
            if self.composite_retriever: self.composite_retriever.routing_cache.clear()
        except: pass
