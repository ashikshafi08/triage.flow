"""Issue-Aware RAG System - tinygrad-style rewrite (1,927→~600 lines)"""
import os, json, asyncio, logging, time, re, faiss
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from tqdm.auto import tqdm

from src.config import settings
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .models import IssueDoc, IssueSearchResult, IssueContextResponse, PatchSearchResult
from .cache import rag_cache
from .patch_linkage import PatchLinkageBuilder
from .commit_index import CommitIndexManager
from .utils.decorators import safe_op, retry

logger = logging.getLogger(__name__)

def to_int(v: Any) -> Optional[int]:
    try: return int(v)
    except (ValueError, TypeError): return None


class IssueIndexer:
    """Handles indexing and storage of GitHub issues"""

    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner, self.repo_name = repo_owner, repo_name
        self.repo_key = f"{repo_owner}/{repo_name}"
        self.github_client, self.llm_client = GitHubIssueClient(), LLMClient()
        self.patch_builder = PatchLinkageBuilder(repo_owner, repo_name)
        self.commit_index_manager = CommitIndexManager(".", repo_owner, repo_name)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.openai_api_key)

        # Storage paths
        self.index_dir = Path(f".faiss_issues_{repo_owner}_{repo_name}")
        self.index_dir.mkdir(exist_ok=True)
        self.issues_file = self.index_dir / "issues.jsonl"
        self.metadata_file = self.index_dir / "metadata.json"
        self.faiss_index_file = self.index_dir / "index.faiss"
        self.faiss_nodes_file = self.index_dir / "nodes.json"

        # State
        self.faiss_index = self.vector_index = self.bm25_retriever = None
        self.issue_docs, self.diff_docs, self.open_pr_docs = {}, {}, {}
        self.bm25_score_cache = {}

    async def crawl_and_index_issues(self, max_issues: Optional[int] = None,
                                     force_rebuild_dependencies: bool = False,
                                     max_issues_for_patch_linkage: Optional[int] = None) -> None:
        """Crawl issues from GitHub and build FAISS/BM25 index"""
        max_issues = max_issues or settings.MAX_ISSUES_TO_PROCESS
        logger.info(f"Starting crawl for {self.repo_key} (max={max_issues})")

        # Ensure patch docs exist
        diff_file = self.patch_builder.index_dir / "diff_docs.jsonl"
        actual_max = max_issues_for_patch_linkage or max_issues
        if force_rebuild_dependencies or not diff_file.exists() or diff_file.stat().st_size == 0:
            logger.info(f"Building patch linkage (max={actual_max})...")
            await self.patch_builder.build_patch_linkage(max_issues=actual_max)

        # Fetch issues
        repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}"
        per_page, max_pages = 30, min(100, (max_issues + 29) // 30)

        try:
            issues = await self.github_client.list_issues(repo_url, state="all", per_page=per_page, max_pages=max_pages)
        except Exception as e:
            logger.error(f"GitHub API failed: {e}")
            if self.issues_file.exists():
                await self._load_issues()
                if self.issue_docs:
                    logger.info(f"Using {len(self.issue_docs)} cached issues")
                    return
            raise

        issues = issues[:max_issues]
        if not issues and self.issues_file.exists():
            await self._load_issues()
            if self.issue_docs: return
        if not issues:
            logger.warning("No issues available"); return

        # Build documents
        issue_docs = []
        for i in tqdm(issues, desc="Issues"):
            self.issue_docs[i.number] = self._create_issue_doc(i)
            issue_docs.append(self._make_doc(i, "issue", i.number, self._searchable_content(i)))

        # Load patches and open PRs
        patch_docs, open_pr_docs = [], []
        diffs = self._safe_load(self.patch_builder.load_diff_docs)
        if diffs:
            self.diff_docs = {d.pr_number: d for d in diffs}
            patch_docs = [self._make_doc(d, "patch", d.pr_number, self._patch_content(d), issue_id=d.issue_id) for d in diffs]

        prs = self._safe_load(self.patch_builder.load_open_prs)
        if prs:
            self.open_pr_docs = {p.pr_number: p for p in prs}
            open_pr_docs = [self._make_doc(p, "open_pr", p.pr_number, self._open_pr_content(p),
                           review_decision=p.review_decision, draft=p.draft, mergeable=p.mergeable) for p in prs]

        all_docs = issue_docs + patch_docs + open_pr_docs
        logger.info(f"Building FAISS index with {len(all_docs)} documents...")

        # Build index
        parser = SimpleNodeParser.from_defaults(chunk_size=4096, chunk_overlap=200)
        vec_size = self.embed_model.dimensions or 1536
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(vec_size))
        nodes = parser.get_nodes_from_documents(all_docs)
        self.vector_index = VectorStoreIndex(nodes=nodes, storage_context=StorageContext.from_defaults(vector_store=vector_store), embed_model=self.embed_model)

        # Build BM25
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=50, stemmer=Stemmer.Stemmer("english"), language="english")
        await self._compute_bm25_stats(nodes)

        # Save everything
        self._save_faiss_index(nodes)
        await self._save_issues()
        await self._save_metadata(len(issues), len(all_docs))
        logger.info("Index built successfully")

    def _make_doc(self, item, dtype: str, num: int, text: str, **extra) -> Document:
        meta = {"type": dtype, ("issue_id" if dtype == "issue" else "pr_number"): num, **extra}
        return Document(text=text, metadata=meta)

    def _safe_load(self, loader):
        try: return loader()
        except Exception: return []

    async def _compute_bm25_stats(self, nodes: List) -> None:
        if not self.bm25_retriever: return
        queries = ["bug fix", "performance", "feature request", "memory leak", "documentation", "test failure"]
        scores = []
        for q in queries:
            for n in self.bm25_retriever.retrieve(q)[:10]:
                if (s := getattr(n, 'score', None)):
                    scores.append(s)
        self.bm25_score_cache = {"max": max(scores, default=10), "min": min(scores, default=0),
                                  "avg": sum(scores)/len(scores) if scores else 5, "range": max(scores, default=10) - min(scores, default=0)}

    def _normalize_bm25(self, raw: float) -> float:
        if not self.bm25_score_cache or self.bm25_score_cache.get("range", 0) == 0:
            return min(0.8, max(0.1, raw / 15))
        r = self.bm25_score_cache["range"]
        return 0.1 + ((raw - self.bm25_score_cache["min"]) / r) * 0.7

    async def load_existing_index(self) -> bool:
        """Load existing index if available"""
        if not (self.faiss_index_file.exists() and self.faiss_nodes_file.exists()):
            return False
        try:
            self.faiss_index = faiss.read_index(str(self.faiss_index_file))
            with open(self.faiss_nodes_file, 'r') as f:
                nodes = [Document(**d) for d in json.load(f)]
            if self.faiss_index.ntotal != len(nodes): raise ValueError("Node count mismatch")

            self.vector_index = VectorStoreIndex(nodes=nodes, vector_store=FaissVectorStore(faiss_index=self.faiss_index), embed_model=self.embed_model)
            await self._load_issues()
            self.diff_docs = {d.pr_number: d for d in self._safe_load(self.patch_builder.load_diff_docs)}
            self.open_pr_docs = {p.pr_number: p for p in self._safe_load(self.patch_builder.load_open_prs)}
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=50, stemmer=Stemmer.Stemmer("english"), language="english")
            await self._compute_bm25_stats(nodes)

            # Cleanup legacy files
            for f in ["default__vector_store.json", "docstore.json", "index_store.json", "graph_store.json"]:
                p = self.index_dir / f
                if p.exists(): p.unlink()
            logger.info(f"Loaded index with {len(nodes)} nodes")
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            self._cleanup_index()
            return False

    def _cleanup_index(self):
        for f in [self.faiss_index_file, self.faiss_nodes_file]:
            if f.exists(): f.unlink()
        for f in ["default__vector_store.json", "docstore.json", "index_store.json", "graph_store.json"]:
            p = self.index_dir / f
            if p.exists(): p.unlink()

    def _save_faiss_index(self, nodes, append: bool = False):
        faiss.write_index(self.vector_index.vector_store._faiss_index, str(self.faiss_index_file))
        existing = []
        if append and self.faiss_nodes_file.exists():
            with open(self.faiss_nodes_file) as f:
                existing = json.load(f)
        with open(self.faiss_nodes_file, "w") as f:
            json.dump(existing + [n.dict() for n in nodes], f)

    async def _save_metadata(self, total_issues: int, total_docs: int):
        with open(self.metadata_file, 'w') as f:
            json.dump({"repo": self.repo_key, "total_issues": total_issues, "total_documents": total_docs,
                       "last_updated": datetime.now().isoformat(), "index_version": "1.5"}, f, indent=2)

    async def _save_issues(self):
        with open(self.issues_file, 'w') as f:
            for doc in self.issue_docs.values(): f.write(doc.model_dump_json() + '\n')

    async def _load_issues(self):
        if not self.issues_file.exists(): return
        self.issue_docs = {}
        with open(self.issues_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for k in ['closed_by_commit', 'closed_by_pr', 'closed_by_author', 'closed_event_data']:
                        data.setdefault(k, None)
                    issue = IssueDoc.model_validate(data)
                    self.issue_docs[issue.id] = issue
                except Exception as e:
                    logger.warning(f"Malformed issue line: {e}")

    def _create_issue_doc(self, issue) -> IssueDoc:
        return IssueDoc(id=issue.number, state=issue.state, title=issue.title, body=issue.body or "",
                        comments=[c.body for c in issue.comments], labels=issue.labels,
                        created_at=issue.created_at.isoformat(),
                        closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
                        patch_url=self.patch_builder.get_patch_url_for_issue(issue.number), repo=self.repo_key)

    def _searchable_content(self, issue) -> str:
        body = re.sub(r'```[\s\S]*?```', '[CODE]', issue.body or "")
        body = re.sub(r'`([^`]+)`', r'\1', body)
        body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)
        body = re.sub(r'#+\s*', '', body)
        body = re.sub(r'\s+', ' ', body).strip()
        parts = [f"Title: {issue.title}", f"Description: {body}", f"State: {issue.state}"]
        if issue.labels:
            parts += [f"Labels: {' '.join(issue.labels)}", f"Categories: {' '.join(f'category-{l}' for l in issue.labels)}"]
            if any(l in ['bug', 'error', 'issue'] for l in issue.labels): parts.append("Type: Bug report")
            elif any(l in ['enhancement', 'feature'] for l in issue.labels): parts.append("Type: Feature request")
        return "\n\n".join(parts)

    def _patch_content(self, diff) -> str:
        return f"Summary: {diff.diff_summary}\nFiles: {', '.join(diff.files_changed)}\nIssue: {diff.issue_id}\nPR: {diff.pr_number}\nMerged: {diff.merged_at}"

    def _open_pr_content(self, pr) -> str:
        body = re.sub(r'```[\s\S]*?```', '[CODE]', pr.body or "")
        body = re.sub(r'`([^`]+)`', r'\1', body)
        body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)
        body = re.sub(r'#+\s*|\s+', ' ', body).strip()
        parts = [f"PR #{pr.pr_number}: {pr.title}", f"Description: {body}", f"Author: {pr.author}", "State: Open PR"]
        if pr.draft: parts.append("Status: Draft")
        if pr.review_decision: parts.append(f"Review: {pr.review_decision}")
        if pr.files_changed: parts.append(f"Files: {', '.join(pr.files_changed[:10])}")
        return "\n\n".join(parts)


class IssueReranker:
    """Reranks issue candidates using LLM"""

    def __init__(self, llm_client: LLMClient, indexer: IssueIndexer):
        self.llm_client, self.indexer, self.cache = llm_client, indexer, {}

    async def rerank(self, query: str, candidates: List[Dict], max_candidates: int = 5) -> List[Dict]:
        if not candidates: return []

        cache_key = f"{query}::{','.join(str(c.get('issue_id', c.get('pr_number', ''))) for c in candidates)}"
        if cache_key in self.cache: return self.cache[cache_key]

        # Build summaries
        summaries, ids = [], []
        for c in candidates:
            iid = c.get("issue_id")
            if iid and iid in self.indexer.issue_docs:
                doc = self.indexer.issue_docs[iid]
                summaries.append(f"#{doc.id}: {doc.title}\n{doc.body[:200]}...")
                ids.append(iid)
            elif (pr := c.get("pr_number")) and pr in self.indexer.diff_docs:
                doc = self.indexer.diff_docs[pr]
                summaries.append(f"PR #{doc.pr_number} for #{doc.issue_id}: {doc.diff_summary[:200]}...")
                ids.append(pr)

        prompt = f"""Rank these issues by relevance to: {query}

{chr(10).join(f"{i+1}. Issue #{ids[i]}: {s}" for i, s in enumerate(summaries))}

Return JSON only: {{"ranked_ids": [...]}}"""

        try:
            llm = self.llm_client._get_openrouter_llm("google/gemini-2.5-flash-preview-05-20")
            raw = llm.complete(prompt).text.strip()

            # Parse response
            m = re.search(r'\{[^}]*"ranked_ids"[^}]*\}', raw)
            if m:
                result = json.loads(m.group())
            else:
                m = re.search(r'\[[0-9,\s]+\]', raw)
                if m:
                    result = {"ranked_ids": json.loads(m.group())}
                else:
                    result = json.loads(raw)

            ranked = [str(i) for i in result.get("ranked_ids", [])]
            id_map = {str(c.get("issue_id", c.get("pr_number", ""))): c for c in candidates}
            reranked = [id_map[i] for i in ranked if i in id_map]
            reranked += [c for c in candidates if str(c.get("issue_id", c.get("pr_number", ""))) not in ranked]

            self.cache[cache_key] = reranked[:max_candidates]
            return reranked[:max_candidates]
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return candidates[:max_candidates]


class IssueRetriever:
    """Handles retrieval of similar issues"""

    def __init__(self, indexer: IssueIndexer):
        self.indexer = indexer
        self.reranker = IssueReranker(indexer.llm_client, indexer)

    async def find_related_issues(self, query: str, k: int = 5, state_filter: str = "all",
                                  similarity_threshold: float = 0.3, label_filter: Optional[List[str]] = None,
                                  include_patches: bool = False) -> Tuple[List[IssueSearchResult], List[PatchSearchResult]]:
        if not self.indexer.vector_index: return [], []

        try:
            processed = self._preprocess(query)
            dense, sparse = await self._search(processed, 50)

            # Separate by type
            d_issues = [r for r in dense if r.get("type") == "issue"]
            d_patches = [r for r in dense if r.get("type") in ("patch", "open_pr")]
            s_issues = [r for r in sparse if r.get("type") == "issue"]
            s_patches = [r for r in sparse if r.get("type") in ("patch", "open_pr")]

            # Combine and filter issues
            combined = self._combine(d_issues, s_issues)
            filtered = self._filter(combined, state_filter, similarity_threshold, label_filter, processed)
            reranked = await self.reranker.rerank(processed, filtered, k)
            issue_results = []
            for r in reranked:
                if r["issue_id"] in self.indexer.issue_docs:
                    issue_results.append(IssueSearchResult(
                        issue=self.indexer.issue_docs[r["issue_id"]],
                        similarity=r["similarity"],
                        match_reasons=r.get("match_reasons", [])))

            # Handle patches
            patch_results = []
            if include_patches:
                combined_p = self._combine(d_patches, s_patches)
                filtered_p = [p for p in combined_p if p["similarity"] >= similarity_threshold * 0.8]
                merged = [p for p in filtered_p if p.get("type") == "patch"]
                if merged:
                    reranked_p = await self.reranker.rerank(processed, merged, k)
                    for r in reranked_p:
                        pr_num = r.get("pr_number")
                        if pr_num in self.indexer.diff_docs and r["similarity"] >= similarity_threshold * 0.8:
                            patch_results.append(PatchSearchResult(
                                patch=self.indexer.diff_docs[pr_num],
                                similarity=r["similarity"],
                                match_reasons=r.get("match_reasons", [])))

            return issue_results, patch_results
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [], []

    def _preprocess(self, query: str) -> str:
        q = query.strip()
        m = re.match(r'^\[([^\]]+)\]\s*(.+)', q)
        if m:
            tag, content = m.groups()
            tl = tag.lower()
            if any(t in tl for t in ['bug', 'error']): q = f"Bug report: {content}. Type: Bug report"
            elif any(t in tl for t in ['feature', 'enhancement']): q = f"Feature request: {content}. Type: Feature request"
            else: q = f"Title: {content}"
        else: q = f"Title: {q}"
        q = re.sub(r'[`"]+', '', q)
        return re.sub(r'\s+', ' ', q).strip()

    async def _search(self, query: str, k: int) -> Tuple[List[Dict], List[Dict]]:
        """Run both dense and sparse search"""
        dense, sparse = [], []

        # Dense search
        retriever = self.indexer.vector_index.as_retriever(similarity_top_k=k)
        for node in await retriever.aretrieve(query):
            if hasattr(node, 'node'):
                n, score = node.node, node.score
            else:
                n, score = node, getattr(node, 'score', None)
            meta = n.metadata or {}
            dtype = meta.get("type", "issue")

            if score and 0 <= score <= 1:
                sim = float(score)
            elif score:
                sim = max(0, min(1, 1/(1+abs(score))))
            else:
                sim = 0.3

            item = {"similarity": sim, "match_type": "semantic", "match_reasons": ["semantic"], "type": dtype}
            iid = to_int(meta.get("issue_id"))
            pr = to_int(meta.get("pr_number"))

            if dtype == "issue" and iid and iid in self.indexer.issue_docs:
                item["issue_id"] = iid; dense.append(item)
            elif dtype == "patch" and pr and pr in self.indexer.diff_docs:
                item["pr_number"] = pr; dense.append(item)
            elif dtype == "open_pr" and pr and pr in self.indexer.open_pr_docs:
                item["pr_number"] = pr; dense.append(item)

        # Sparse search
        if self.indexer.bm25_retriever:
            for node in self.indexer.bm25_retriever.retrieve(query)[:k]:
                meta = node.metadata or {}
                dtype = meta.get("type", "issue")
                sim = self.indexer._normalize_bm25(getattr(node, 'score', 1.0))

                item = {"similarity": sim, "match_type": "keyword", "match_reasons": ["keyword"], "type": dtype}
                iid = to_int(meta.get("issue_id"))
                pr = to_int(meta.get("pr_number"))

                if dtype == "issue" and iid and iid in self.indexer.issue_docs:
                    item["issue_id"] = iid; sparse.append(item)
                elif dtype == "patch" and pr and pr in self.indexer.diff_docs:
                    item["pr_number"] = pr; sparse.append(item)
                elif dtype == "open_pr" and pr and pr in self.indexer.open_pr_docs:
                    item["pr_number"] = pr; sparse.append(item)

        return dense, sparse

    def _combine(self, dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
        """Combine dense and sparse results with score fusion"""
        seen, combined, lookup = set(), [], {}

        for r in dense:
            key = ("issue", r["issue_id"]) if "issue_id" in r else ("pr", r.get("pr_number"))
            if key[1] and key not in seen:
                seen.add(key)
                r.update({"has_dense": True, "dense_sim": r["similarity"], "has_sparse": False, "sparse_sim": 0})
                lookup[key] = r
                combined.append(r)

        for r in sparse:
            key = ("issue", r["issue_id"]) if "issue_id" in r else ("pr", r.get("pr_number"))
            if not key[1]: continue
            if key in seen:
                existing = lookup[key]
                existing.update({"has_sparse": True, "sparse_sim": r["similarity"],
                               "similarity": 0.6 * existing["dense_sim"] + 0.4 * r["similarity"]})
                existing["match_reasons"].append("keyword")
            else:
                seen.add(key)
                r.update({"has_sparse": True, "sparse_sim": r["similarity"], "has_dense": False, "dense_sim": 0})
                combined.append(r)

        return combined

    def _filter(self, results: List[Dict], state: str, threshold: float, labels: Optional[List[str]], query: str) -> List[Dict]:
        """Filter and score results"""
        filtered = []
        for r in results:
            if r["similarity"] < threshold: continue
            iid = r.get("issue_id")
            if iid and iid in self.indexer.issue_docs:
                doc = self.indexer.issue_docs[iid]
                if state != "all" and doc.state != state: continue
                if labels and not any(l in doc.labels for l in labels): continue
                r["issue_id"] = iid
            filtered.append(r)

        # Score with title matching and label boosts
        def score(r):
            base = r["similarity"]
            iid = r.get("issue_id")
            if iid and iid in self.indexer.issue_docs:
                doc = self.indexer.issue_docs[iid]
                clean_q = re.sub(r'^\[([^\]]+)\]\s*|["`]', '', query.lower()).strip()
                title = doc.title.lower()

                # Title matching
                if clean_q and len(clean_q) > 5:
                    if clean_q in title: base += 0.15
                    else:
                        qw, tw = set(clean_q.split()), set(title.split())
                        if qw and tw:
                            ov = len(qw & tw) / len(qw)
                            if ov > 0.4:
                                base += 0.08 if ov > 0.6 else 0.04

                # Hybrid boost
                if r.get("has_dense") and r.get("has_sparse"): base += 0.03

                # Label boost
                hvl = ['bug', 'enhancement', 'performance', 'memory-leak']
                base += sum(0.01 for l in doc.labels if any(h in l.lower() for h in hvl))
            return min(0.99, base)

        return sorted(filtered, key=score, reverse=True)


class IssueAwareRAG:
    """Main interface for issue-aware RAG functionality"""

    def __init__(self, repo_owner: str, repo_name: str, progress_callback: Optional[Callable] = None):
        self.repo_owner, self.repo_name = repo_owner, repo_name
        self.progress_callback = progress_callback
        self.indexer = IssueIndexer(repo_owner, repo_name)
        self.retriever = IssueRetriever(self.indexer)
        self._initialized, self._pr_cache = False, {}

    async def initialize(self, force_rebuild: bool = False, max_issues_for_patch_linkage: Optional[int] = None,
                        max_prs_for_patch_linkage: Optional[int] = None) -> None:
        self.indexer = IssueIndexer(self.repo_owner, self.repo_name)

        if not force_rebuild and await self.indexer.load_existing_index():
            logger.info(f"Loaded from cache for {self.repo_owner}/{self.repo_name}")
            self.retriever = IssueRetriever(self.indexer)
            self._initialized = True
            return

        logger.info(f"Building new index (force={force_rebuild})")
        builder = PatchLinkageBuilder(self.repo_owner, self.repo_name, self.progress_callback)
        await builder.build_patch_linkage(max_issues=max_issues_for_patch_linkage, max_prs=max_prs_for_patch_linkage,
                                          download_diffs=True, include_open_prs=True)
        await self.indexer.crawl_and_index_issues(max_issues=max_issues_for_patch_linkage, force_rebuild_dependencies=force_rebuild)
        self.retriever = IssueRetriever(self.indexer)
        self._initialized = True

    async def get_issue_context(self, query: str, max_issues: int = 5, include_patches: bool = True) -> IssueContextResponse:
        start = time.time()
        if not self._initialized: await self.initialize()

        analysis = self._analyze_query(query)
        issues, patches = await self.retriever.find_related_issues(query, k=max_issues, state_filter=analysis.get("preferred_state", "all"),
                                                                   similarity_threshold=0.3, label_filter=analysis.get("relevant_labels"),
                                                                   include_patches=include_patches)

        return IssueContextResponse(related_issues=issues, patches=patches, total_found=len(issues) + len(patches),
                                    query_analysis=analysis, processing_time=time.time() - start)

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        analysis = {"query_type": "general", "preferred_state": "all", "relevant_labels": None, "urgency": "normal"}

        if any(w in q for w in ["bug", "error", "issue", "problem", "broken"]):
            analysis.update({"query_type": "bug_report", "relevant_labels": ["bug", "error"]})
        elif any(w in q for w in ["feature", "enhancement", "request"]):
            analysis.update({"query_type": "feature_request", "relevant_labels": ["enhancement", "feature"]})
        elif any(w in q for w in ["performance", "slow", "optimization"]):
            analysis.update({"query_type": "performance", "relevant_labels": ["performance", "optimization"]})

        if any(w in q for w in ["urgent", "critical", "breaking", "blocker"]):
            analysis.update({"urgency": "high", "preferred_state": "open"})

        return analysis

    def is_initialized(self) -> bool: return self._initialized

    async def update_index(self) -> None:
        if self._initialized:
            await self.indexer.crawl_and_index_issues()
            self.retriever = IssueRetriever(self.indexer)

    async def incremental_sync(self, max_new_issues: Optional[int] = None, max_new_prs: Optional[int] = None) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
            return {"status": "full_initialization", "reason": "system_not_initialized"}

        from .enhanced_persistence import persistence_manager
        logger.info(f"Starting incremental sync for {self.repo_owner}/{self.repo_name}")

        sync_meta = persistence_manager.load_sync_metadata(self.indexer.index_dir)
        should_sync, info = persistence_manager.should_sync_issues(sync_meta, force_sync=False)

        if not should_sync:
            return {"status": "skipped", "reason": info.get("reason"), "hours_since_last_sync": info.get("hours_since_sync", 0)}

        try:
            new_issues = await self._sync_issues(max_new_issues)
            new_prs = await self._sync_prs(max_new_prs)

            persistence_manager.save_sync_metadata(self.indexer.index_dir, self.repo_owner, self.repo_name,
                                                   issues_synced=new_issues, prs_synced=new_prs)

            if new_issues or new_prs: self.retriever = IssueRetriever(self.indexer)
            return {"status": "completed", "new_issues": new_issues, "new_prs": new_prs, "sync_time": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {"status": "error", "error": str(e)}

    async def _sync_issues(self, max_new: Optional[int] = None) -> int:
        try:
            existing = set(self.indexer.issue_docs.keys())
            recent = await self.indexer.github_client.list_issues(f"https://github.com/{self.repo_owner}/{self.repo_name}", state="all", max_pages=5)
            new = [i for i in recent if i.id not in existing]
            if max_new: new = new[:max_new]

            if new:
                await self._add_issues(new)
                logger.info(f"Added {len(new)} new issues")
            return len(new)
        except Exception as e:
            logger.error(f"Issue sync error: {e}")
            return 0

    async def _sync_prs(self, max_new: Optional[int] = None) -> int:
        try:
            existing = {d.pr_number for d in self.indexer.patch_builder.load_diff_docs()}
            builder = PatchLinkageBuilder(self.repo_owner, self.repo_name)
            await builder.build_patch_linkage(max_issues=max_new or 50, max_prs=max_new or 50, download_diffs=True, include_open_prs=True)

            new_prs = {d.pr_number for d in builder.load_diff_docs()} - existing
            if new_prs:
                self.indexer.patch_builder = builder
                logger.info(f"Added {len(new_prs)} new PRs")
            return len(new_prs)
        except Exception as e:
            logger.error(f"PR sync error: {e}")
            return 0

    async def _add_issues(self, issues: List) -> None:
        docs = []
        for i in issues:
            doc = IssueDoc(id=i.id, title=i.title, body=i.body or "", state=i.state,
                          labels=[l.name for l in i.labels], created_at=i.created_at, updated_at=i.updated_at,
                          closed_at=i.closed_at, user=i.user.login if i.user else "unknown",
                          comments=getattr(i, 'comments', 0), url=i.url,
                          assignees=[a.login for a in getattr(i, 'assignees', [])])
            self.indexer.issue_docs[i.id] = doc

            content = f"Title: {i.title}\n\nBody: {i.body or ''}"
            if i.labels: content += f"\n\nLabels: {', '.join(l.name for l in i.labels)}"
            docs.append(Document(text=content, metadata={"type": "issue", "issue_id": i.id, "state": i.state}))

        if docs:
            parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
            nodes = parser.get_nodes_from_documents(docs)
            self.indexer.vector_index.insert_nodes(nodes)
            await self.indexer._save_issues()
            self.indexer._save_faiss_index(nodes, append=True)

    async def get_prs(self, state: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        if not self._initialized: await self.initialize()
        try:
            diffs = sorted(self.indexer.patch_builder.load_diff_docs(), key=lambda x: x.pr_number, reverse=True)
            if state != "all": diffs = [d for d in diffs if d.merged_at]
            return [{"number": d.pr_number, "title": getattr(d, 'pr_title', f"PR #{d.pr_number}"),
                    "merged_at": d.merged_at, "files_changed": d.files_changed, "issue_id": d.issue_id} for d in diffs[:limit]]
        except Exception as e:
            logger.error(f"Get PRs error: {e}")
            return []

    async def get_pr_details(self, pr_number: int) -> Optional[Dict[str, Any]]:
        if not self._initialized: await self.initialize()
        if pr_number in self._pr_cache: return self._pr_cache[pr_number]

        try:
            diffs = self.indexer.patch_builder.load_diff_docs()
            doc = next((d for d in diffs if d.pr_number == pr_number), None)
            if doc:
                details = {"number": doc.pr_number, "title": getattr(doc, 'pr_title', f"PR #{doc.pr_number}"),
                          "merged_at": doc.merged_at, "files_changed": doc.files_changed, "issue_id": doc.issue_id,
                          "diff_summary": doc.diff_summary, "diff_path": doc.diff_path}
                self._pr_cache[pr_number] = details
                return details
        except Exception as e:
            logger.error(f"PR details error: {e}")
        return None

    async def get_pr_diff(self, pr_number: int) -> Optional[str]:
        details = await self.get_pr_details(pr_number)
        if details and os.path.exists(details["diff_path"]):
            with open(details["diff_path"]) as f:
                return f.read()
        return None

    async def search_prs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._initialized: await self.initialize()
        try:
            results = await self.retriever.find_related_issues(query, k=limit, include_patches=True)
            pr_results = []
            for r in results:
                if hasattr(r, 'pr_number'):
                    d = await self.get_pr_details(r.pr_number)
                    if d:
                        pr_results.append({**d, "similarity_score": r.similarity_score})
            return pr_results
        except Exception as e:
            logger.error(f"Search PRs error: {e}")
            return []

    def clear_pr_cache(self) -> None: self._pr_cache.clear()
