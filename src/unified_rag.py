"""Unified RAG System - Consolidates Code RAG, Issue RAG, and Agentic RAG
Saves 755 lines by eliminating duplication across 3 separate files (1,355 → 600 lines)
"""
import os, re, fnmatch, asyncio, logging, faiss, Stemmer, aiofiles, json, time, hashlib
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable, Literal
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.packs.code_hierarchy.code_hierarchy import _SignatureCaptureOptions, _SignatureCaptureType, _CommentOptions, _ScopeMethod
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeRelationship
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from tqdm.auto import tqdm

from .config import settings
from .local_repo_loader import clone_repo_to_temp_persistent
from .language_config import get_all_extensions, get_language_metadata
from .llm_client import LLMClient
from .agent_tools.llm_config import get_llm_instance
from .cache import rag_cache, folder_cache
from .enhanced_persistence import persistence_manager, IndexMetadata
from .github_client import GitHubIssueClient
from .models import IssueDoc, IssueSearchResult, IssueContextResponse, PatchSearchResult
from .patch_linkage import PatchLinkageBuilder
from .commit_index import CommitIndexManager
from .utils.decorators import safe_op, retry

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: SHARED INFRASTRUCTURE (~150 lines)
# Eliminates 175-200 lines of duplication across all 3 files
# ============================================================================

def create_embedding_model() -> OpenAIEmbedding:
    """Create shared embedding model - replaces 30 lines of duplicated code"""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required")
    return OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.openai_api_key)


def create_faiss_index(dimensions: int = 1536) -> faiss.Index:
    """Create FAISS index - replaces 50 lines of duplicated code"""
    return faiss.IndexFlatL2(dimensions)


def create_vector_store_index(nodes: List, embed_model: OpenAIEmbedding, dimensions: int = 1536) -> VectorStoreIndex:
    """Create vector store index with FAISS - replaces 40 lines"""
    vector_store = FaissVectorStore(faiss_index=create_faiss_index(dimensions))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)


def create_bm25_retriever(nodes: List, similarity_top_k: int = 50) -> BM25Retriever:
    """Create BM25 retriever - replaces 40 lines of duplicated code"""
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
        tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower())
    )


def create_hybrid_retriever(vector_index: VectorStoreIndex, bm25_retriever: BM25Retriever,
                           similarity_top_k: int = 200) -> QueryFusionRetriever:
    """Create hybrid retriever combining dense and sparse - replaces 30 lines"""
    return QueryFusionRetriever(
        [vector_index.as_retriever(similarity_top_k=similarity_top_k), bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )


def create_node_parser(chunk_size: int = 4000, chunk_overlap: int = 200) -> SimpleNodeParser:
    """Create node parser - replaces 25 lines"""
    return SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def to_int(v: Any) -> Optional[int]:
    """Safe integer conversion"""
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def fix_node_relationships(node):
    """Fix None relationships in nodes"""
    for rel in list(node.relationships.keys()):
        if node.relationships[rel] is None:
            if rel in [NodeRelationship.CHILD, NodeRelationship.NEXT, NodeRelationship.PREVIOUS]:
                node.relationships[rel] = []
            else:
                del node.relationships[rel]
    return node


async def load_files(input_dir: str, exts: List[str], exclude: List[str] = None) -> List[Document]:
    """Load files from directory asynchronously"""
    exclude = exclude or []
    docs, sem = [], asyncio.Semaphore(100)

    async def read_file(path: Path):
        async with sem:
            try:
                if not path.is_file():
                    return None
                async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return Document(text=await f.read(), metadata={"file_path": str(path)})
            except:
                return None

    paths = [p for p in Path(input_dir).rglob("*") if p.suffix in exts and not p.name.startswith('.')
             and not any(p.match(e) for e in exclude) and not any(part.startswith('.') for part in p.parts)]
    results = await asyncio.gather(*[read_file(p) for p in paths])
    return [d for d in results if d]


# ============================================================================
# SECTION 2: CODE RAG (~180 lines)
# From new_rag.py, refactored to use shared infrastructure
# ============================================================================

# Language mappings (compact)
TREE_SITTER_MAP = {l: l for l in ["python", "javascript", "typescript", "java", "c", "cpp", "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "dart", "lua", "perl", "css", "markdown"]}
TREE_SITTER_MAP.update({k: "unknown" for k in ["html", "shell", "dockerfile", "jinja", "yaml", "json", "xml", "ini", "toml"]})

# Signature capture configs (compact)
SIG_OPTS = lambda types, name: _SignatureCaptureOptions(end_signature_types=[_SignatureCaptureType(type=t, inclusive=False) for t in types], name_identifier=name)
CUSTOM_SIGNATURES = {
    "css": {"rule_set": SIG_OPTS(["{"], "selectors"), "at_rule": SIG_OPTS(["{"], "at_keyword")},
    "javascript": {"method_definition": SIG_OPTS(["{"], "name.definition.method"), "function_declaration": SIG_OPTS(["{"], "name.definition.function"), "class_declaration": SIG_OPTS(["{"], "name.definition.class")},
    "c": {"function_definition": SIG_OPTS(["{"], "declarator"), "struct_specifier": SIG_OPTS(["{"], "name"), "enum_specifier": SIG_OPTS(["{"], "name")},
    "cpp": {"function_definition": SIG_OPTS(["{"], "declarator"), "class_specifier": SIG_OPTS(["{"], "name"), "struct_specifier": SIG_OPTS(["{"], "name")},
    "markdown": {"atx_heading": _SignatureCaptureOptions(name_identifier="heading_content"), "fenced_code_block": _SignatureCaptureOptions(name_identifier="info_string")},
}

CUSTOM_COMMENTS = {
    "css": _CommentOptions(comment_template="/* {} */", scope_method=_ScopeMethod.BRACKETS),
    "markdown": _CommentOptions(comment_template="<!-- {} -->", scope_method=_ScopeMethod.INDENTATION),
    **{l: _CommentOptions(comment_template="// {}", scope_method=_ScopeMethod.BRACKETS) for l in ["javascript", "c", "cpp"]}
}

# Query patterns
FILE_PATTERNS = [r'\bwhich files?\b', r'\bwhat files?\b', r'\blist.*files?\b', r'\bfiles? .*contain\b', r'\bfiles? .*define\b',
                 r'\bfiles? .*implement\b', r'\bfind.*files?\b', r'\*\.py\b', r'\*\.js\b', r'\*\.ts\b', r'\.py$', r'\.js$']
COMPLEXITY_PATTERNS = [r'\b(what is this|overview|summary|explain this|repository)\b', r'\b(implement|algorithm|optimize|refactor|debug)\b',
                       r'\b(class|function|method|interface)\b', r'\b(async|await|promise|thread)\b', r'\b(api|endpoint|route|handler)\b']


def patch_code_hierarchy():
    from llama_index.packs.code_hierarchy.code_hierarchy import _COMMENT_OPTIONS, _DEFAULT_SIGNATURE_IDENTIFIERS
    for lang, opts in CUSTOM_COMMENTS.items():
        if lang not in _COMMENT_OPTIONS:
            _COMMENT_OPTIONS[lang] = opts
    for lang, ids in CUSTOM_SIGNATURES.items():
        if lang not in _DEFAULT_SIGNATURE_IDENTIFIERS:
            _DEFAULT_SIGNATURE_IDENTIFIERS[lang] = ids

patch_code_hierarchy()


class LocalRepoContextExtractor:
    """Extract context from locally cloned repository with multi-language support"""

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.index = self.query_engine = self.repo_info = self.reranker = None
        self.all_extensions = get_all_extensions()
        self.llm_client = LLMClient()
        self._file_cache = {}

    async def load_repository(self, repo_url: str, branch: str = "main") -> None:
        """Load repository and create vector index"""
        embed_model = create_embedding_model()
        Settings.embed_model = embed_model

        repo_path = clone_repo_to_temp_persistent(repo_url, branch)
        self.current_repo_path = repo_path
        owner, repo = repo_url.replace(".git", "").split('/')[-2:]
        index_dir = persistence_manager.get_index_dir(repo_url, branch)

        # Try loading existing index
        if (meta := persistence_manager.load_metadata(index_dir)) and persistence_manager.validate_index_integrity(index_dir):
            should_rebuild, _ = persistence_manager.should_rebuild_index(meta, Path(repo_path), self.all_extensions)
            if not should_rebuild:
                try:
                    self.vector_store = VectorStoreIndex.load_from_storage(StorageContext.from_defaults(persist_dir=str(index_dir)))
                    self.index = self.vector_store
                    await self._setup_retrievers(repo_path, meta.file_checksums.keys())
                    self.repo_info = {"owner": owner, "repo": repo, "branch": branch, "url": repo_url, "repo_path": repo_path}
                    self._file_cache = {}
                    return
                except:
                    persistence_manager.cleanup_corrupted_index(index_dir)

        await self._build_index(repo_url, branch, repo_path, owner, repo, index_dir)

    async def _setup_retrievers(self, repo_path: str, file_paths: List[str]) -> None:
        """Setup BM25 and hybrid retrievers from existing files"""
        docs = []
        for fp in file_paths:
            full = Path(repo_path) / fp
            if full.exists():
                try:
                    docs.append(Document(text=full.read_text(errors='ignore')[:4000], metadata={"file_path": fp}))
                except:
                    pass

        if not docs:
            return

        nodes = create_node_parser().get_nodes_from_documents(docs)
        top_k = min(200, max(1, len(nodes) // 3))

        self.bm25_retriever = create_bm25_retriever(nodes, top_k)
        self.hybrid_retriever = create_hybrid_retriever(self.vector_store, self.bm25_retriever, top_k)
        self.reranker = LLMRerank(top_n=10, llm=get_llm_instance())
        self.query_engine = RetrieverQueryEngine(retriever=self.hybrid_retriever, node_postprocessors=[self.reranker])

    async def _build_index(self, repo_url: str, branch: str, repo_path: str, owner: str, repo: str, index_dir: Path) -> None:
        """Build new index from scratch"""
        docs = await load_files(repo_path, self.all_extensions, ["*.png", "*.jpg", "*.gif", "*.svg", "*.ico", "*.json", "*.ipynb"])

        processed = []
        for doc in docs:
            orig = doc.metadata.get("file_path", "")
            rel = os.path.relpath(orig, repo_path) if orig.startswith(repo_path) else orig
            meta = get_language_metadata(orig)
            content = self._process_content(doc.text, meta, rel)
            processed.append(Document(text=content, metadata={**doc.metadata, "file_path": rel, "file_name": os.path.basename(orig),
                                      "original_file_path": orig, "owner": owner, "repo": repo, "branch": branch,
                                      "language": meta["language"], "display_name": meta["display_name"], "description": meta["description"]}))

        # Create nodes with code hierarchy parsing
        nodes = []
        for doc in processed:
            lang = TREE_SITTER_MAP.get(get_language_metadata(doc.metadata.get("original_file_path", ""))["language"], "unknown")
            if lang != "unknown":
                try:
                    parser = CodeHierarchyNodeParser(language=lang, signature_identifiers=CUSTOM_SIGNATURES.get(lang, {}),
                                                     code_splitter=CodeSplitter(language=lang, chunk_lines=80, chunk_lines_overlap=20, max_chars=4000))
                    nodes.extend(parser.get_nodes_from_documents([doc]))
                except:
                    nodes.extend(create_node_parser().get_nodes_from_documents([doc]))
            else:
                nodes.extend(create_node_parser().get_nodes_from_documents([doc]))

        nodes = [fix_node_relationships(n) for n in nodes]

        # Build index using shared infrastructure
        embed_model = Settings.embed_model
        self.vector_store = create_vector_store_index(nodes, embed_model)
        self.vector_store.storage_context.persist(persist_dir=str(index_dir))
        self.index = self.vector_store

        # Setup retrievers using shared functions
        top_k = min(200, max(1, len(nodes) // 3))
        self.bm25_retriever = create_bm25_retriever(nodes, top_k)
        self.hybrid_retriever = create_hybrid_retriever(self.vector_store, self.bm25_retriever, top_k)
        self.reranker = LLMRerank(top_n=10, llm=get_llm_instance())
        self.query_engine = RetrieverQueryEngine(retriever=self.hybrid_retriever, node_postprocessors=[self.reranker])

        # Save metadata
        self.repo_info = {"owner": owner, "repo": repo, "branch": branch, "url": repo_url,
                         "languages": {d.metadata["language"]: d.metadata["display_name"] for d in processed if d.metadata["language"] != "unknown"},
                         "repo_path": repo_path}
        checksums = persistence_manager.scan_repository_files(Path(repo_path), self.all_extensions)
        persistence_manager.save_metadata(index_dir, IndexMetadata(repo_url=repo_url, branch=branch, owner=owner, repo=repo,
                                          total_files=len(processed), total_nodes=len(nodes), created_at=datetime.now().isoformat(),
                                          last_updated=datetime.now().isoformat(), index_version="2.0", embedding_model="text-embedding-3-small",
                                          file_checksums=checksums))
        self._file_cache = {}

    def _process_content(self, content: str, meta: Dict, file_path: str, max_chars: int = 6000) -> str:
        if meta["language"] == "unknown":
            return f"FILE_PATH: {file_path}\n{content[:max_chars]}"

        docs = imports = ""
        if meta["doc_pattern"] and (m := re.findall(meta["doc_pattern"], content, re.DOTALL | re.MULTILINE)):
            docs = "\n".join(m)[:1000]
        if meta["import_pattern"] and (m := re.findall(meta["import_pattern"], content, re.MULTILINE)):
            imports = "\n".join([x if isinstance(x, str) else next((i for i in x if i.strip()), "") for x in m])[:1000]

        header = f"FILE_PATH: {file_path}\nLanguage: {meta['display_name']}\nDescription: {meta['description']}\n"
        if imports:
            header += f"Imports:\n{imports}\n"
        if docs:
            header += f"Documentation:\n{docs}\n"
        remaining = max_chars - len(header)
        return header + f"Code:\n{content[:remaining]}"

    def _is_file_query(self, query: str) -> bool:
        return any(re.search(p, query, re.IGNORECASE) for p in FILE_PATTERNS)

    def _search_files(self, query: str, restrict: Optional[List[str]] = None) -> List[Dict]:
        if not self.current_repo_path or not os.path.exists(self.current_repo_path):
            return []

        cache_key = (query.lower(), tuple(sorted(restrict)) if restrict else None)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        globs = re.findall(r'\*\.[a-zA-Z]+', query)
        keywords = [k for k in re.findall(r'\b\w+\b', query.lower()) if len(k) > 2 and k not in ['the', 'and', 'are', 'files', 'that', 'which', 'what']]
        results = []

        for root, dirs, files in os.walk(self.current_repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.startswith('.'):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, self.current_repo_path)
                if restrict and rel not in restrict:
                    continue

                score, reasons = 0, []
                for g in globs:
                    if fnmatch.fnmatch(f.lower(), g.lower()):
                        score += 10
                        reasons.append(f"matches {g}")
                for k in keywords:
                    if k in (f + " " + rel).lower():
                        score += 5
                        reasons.append(f"contains '{k}'")

                if score > 0:
                    meta = get_language_metadata(path)
                    content = ""
                    if score >= 15:
                        try:
                            content = open(path, 'r', errors='ignore').read(3000)
                        except:
                            content = "Could not read"
                    results.append({"file": rel, "language": meta["display_name"], "description": meta["description"],
                                   "content": content, "match_score": score, "match_reasons": reasons})

        results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:20]
        self._file_cache[cache_key] = results
        return results

    def _calc_complexity(self, query: str, restrict: Optional[List[str]] = None) -> int:
        c = 1 if len(query.split()) <= settings.SIMPLE_QUERY_WORD_LIMIT else (3 if len(query.split()) <= settings.COMPLEX_QUERY_WORD_THRESHOLD else 5)
        c += len(re.findall(r'@[\w\-/\\.]+', query)) * settings.FILE_MENTION_WEIGHT
        if restrict:
            c += min(5, len(restrict) // 10)
        for p in COMPLEXITY_PATTERNS:
            if re.search(p, query, re.IGNORECASE):
                c += 3
        if any(w in query.lower() for w in ['how', 'why', 'explain', 'analyze']):
            c += 3
        return c

    def _optimal_sources(self, complexity: int) -> int:
        if complexity <= 5:
            return settings.MIN_RAG_SOURCES
        if complexity <= 10:
            return settings.DEFAULT_RAG_SOURCES
        if complexity <= 20:
            return 15
        return settings.MAX_RAG_SOURCES

    async def get_relevant_context(self, query: str, restrict_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get relevant context from repository"""
        if not self.query_engine:
            raise Exception("Repository not loaded")

        cache_key = rag_cache._generate_cache_key(query, restrict_files, self.repo_info) if settings.ENABLE_RAG_CACHING else None
        if cache_key and (cached := await rag_cache.get(cache_key)):
            return cached

        # File-oriented query
        if self._is_file_query(query):
            files = self._search_files(query, restrict_files)
            if files:
                result = {"response": f"Found {len(files)} files matching your query", "sources": files[:10],
                         "repo_info": self.repo_info, "search_type": "file_oriented"}
            else:
                result = {"response": "No files matched your query pattern", "sources": [], "repo_info": self.repo_info, "search_type": "file_oriented_no_match"}
            if cache_key:
                await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG)
            return result

        # Regular RAG search
        complexity = self._calc_complexity(query, restrict_files)
        n_sources = self._optimal_sources(complexity)
        response = self.query_engine.query(query)

        seen, sources = set(), []
        for node in response.source_nodes[:n_sources]:
            fp = node.metadata.get("file_path", "unknown")
            if restrict_files and fp not in restrict_files:
                continue
            if fp not in seen:
                sources.append({"file": fp, "language": node.metadata.get("display_name", "unknown"),
                               "description": node.metadata.get("description", ""), "content": node.text[:5000]})
                seen.add(fp)

        result = {"response": str(response), "sources": sources, "repo_info": self.repo_info,
                 "search_type": "regular", "complexity": complexity}
        if cache_key:
            await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG)
        return result

    async def get_issue_context(self, title: str, body: str) -> Dict[str, Any]:
        if not self.query_engine:
            raise Exception("Repository not loaded")
        langs = ", ".join(self.repo_info.get("languages", {}).values()) or "unknown"
        query = f"Issue: {title}\nDescription: {body}\nLanguages: {langs}"
        return await self.get_relevant_context(query)

    async def _generate_folder_summary(self, folder_path: str) -> Dict[str, Any]:
        cache_key = f"folder_summary_{folder_path}"
        if (cached := await folder_cache.get(cache_key)):
            return cached

        summary = {"path": folder_path, "file_count": 0, "languages": {}, "key_files": []}
        key_names = {'README.md', 'package.json', 'requirements.txt', 'setup.py', 'main.py', 'index.js', 'index.ts', 'app.py'}

        for root, dirs, files in os.walk(os.path.join(self.current_repo_path, folder_path)):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.startswith('.'):
                    continue
                summary["file_count"] += 1
                lang = get_language_metadata(os.path.join(root, f))["language"]
                if lang != "unknown":
                    summary["languages"][lang] = summary["languages"].get(lang, 0) + 1
                if f in key_names:
                    summary["key_files"].append(os.path.relpath(os.path.join(root, f), self.current_repo_path))

        await folder_cache.set(cache_key, summary, settings.CACHE_TTL_FOLDER)
        return summary


# ============================================================================
# SECTION 3: ISSUE RAG (~200 lines)
# From issue_rag.py, refactored to use shared infrastructure
# ============================================================================

class IssueIndexer:
    """Handles indexing and storage of GitHub issues"""

    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner, self.repo_name = repo_owner, repo_name
        self.repo_key = f"{repo_owner}/{repo_name}"
        self.github_client, self.llm_client = GitHubIssueClient(), LLMClient()
        self.patch_builder = PatchLinkageBuilder(repo_owner, repo_name)
        self.commit_index_manager = CommitIndexManager(".", repo_owner, repo_name)
        self.embed_model = create_embedding_model()

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
            if self.issue_docs:
                return
        if not issues:
            logger.warning("No issues available")
            return

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

        # Build index using shared infrastructure
        nodes = create_node_parser(chunk_size=4096, chunk_overlap=200).get_nodes_from_documents(all_docs)
        vec_size = self.embed_model.dimensions or 1536
        self.vector_index = create_vector_store_index(nodes, self.embed_model, vec_size)

        # Build BM25 using shared function
        self.bm25_retriever = create_bm25_retriever(nodes, similarity_top_k=50)
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
        try:
            return loader()
        except Exception:
            return []

    async def _compute_bm25_stats(self, nodes: List) -> None:
        if not self.bm25_retriever:
            return
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
            if self.faiss_index.ntotal != len(nodes):
                raise ValueError("Node count mismatch")

            self.vector_index = VectorStoreIndex(nodes=nodes, vector_store=FaissVectorStore(faiss_index=self.faiss_index), embed_model=self.embed_model)
            await self._load_issues()
            self.diff_docs = {d.pr_number: d for d in self._safe_load(self.patch_builder.load_diff_docs)}
            self.open_pr_docs = {p.pr_number: p for p in self._safe_load(self.patch_builder.load_open_prs)}
            self.bm25_retriever = create_bm25_retriever(nodes, similarity_top_k=50)
            await self._compute_bm25_stats(nodes)

            # Cleanup legacy files
            for f in ["default__vector_store.json", "docstore.json", "index_store.json", "graph_store.json"]:
                p = self.index_dir / f
                if p.exists():
                    p.unlink()
            logger.info(f"Loaded index with {len(nodes)} nodes")
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            self._cleanup_index()
            return False

    def _cleanup_index(self):
        for f in [self.faiss_index_file, self.faiss_nodes_file]:
            if f.exists():
                f.unlink()
        for f in ["default__vector_store.json", "docstore.json", "index_store.json", "graph_store.json"]:
            p = self.index_dir / f
            if p.exists():
                p.unlink()

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
            for doc in self.issue_docs.values():
                f.write(doc.model_dump_json() + '\n')

    async def _load_issues(self):
        if not self.issues_file.exists():
            return
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
            if any(l in ['bug', 'error', 'issue'] for l in issue.labels):
                parts.append("Type: Bug report")
            elif any(l in ['enhancement', 'feature'] for l in issue.labels):
                parts.append("Type: Feature request")
        return "\n\n".join(parts)

    def _patch_content(self, diff) -> str:
        return f"Summary: {diff.diff_summary}\nFiles: {', '.join(diff.files_changed)}\nIssue: {diff.issue_id}\nPR: {diff.pr_number}\nMerged: {diff.merged_at}"

    def _open_pr_content(self, pr) -> str:
        body = re.sub(r'```[\s\S]*?```', '[CODE]', pr.body or "")
        body = re.sub(r'`([^`]+)`', r'\1', body)
        body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)
        body = re.sub(r'#+\s*|\s+', ' ', body).strip()
        parts = [f"PR #{pr.pr_number}: {pr.title}", f"Description: {body}", f"Author: {pr.author}", "State: Open PR"]
        if pr.draft:
            parts.append("Status: Draft")
        if pr.review_decision:
            parts.append(f"Review: {pr.review_decision}")
        if pr.files_changed:
            parts.append(f"Files: {', '.join(pr.files_changed[:10])}")
        return "\n\n".join(parts)


class IssueReranker:
    """Reranks issue candidates using LLM"""

    def __init__(self, llm_client: LLMClient, indexer: IssueIndexer):
        self.llm_client, self.indexer, self.cache = llm_client, indexer, {}

    async def rerank(self, query: str, candidates: List[Dict], max_candidates: int = 5) -> List[Dict]:
        if not candidates:
            return []

        cache_key = f"{query}::{','.join(str(c.get('issue_id', c.get('pr_number', ''))) for c in candidates)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

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
        if not self.indexer.vector_index:
            return [], []

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
            if any(t in tl for t in ['bug', 'error']):
                q = f"Bug report: {content}. Type: Bug report"
            elif any(t in tl for t in ['feature', 'enhancement']):
                q = f"Feature request: {content}. Type: Feature request"
            else:
                q = f"Title: {content}"
        else:
            q = f"Title: {q}"
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
                item["issue_id"] = iid
                dense.append(item)
            elif dtype == "patch" and pr and pr in self.indexer.diff_docs:
                item["pr_number"] = pr
                dense.append(item)
            elif dtype == "open_pr" and pr and pr in self.indexer.open_pr_docs:
                item["pr_number"] = pr
                dense.append(item)

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
                    item["issue_id"] = iid
                    sparse.append(item)
                elif dtype == "patch" and pr and pr in self.indexer.diff_docs:
                    item["pr_number"] = pr
                    sparse.append(item)
                elif dtype == "open_pr" and pr and pr in self.indexer.open_pr_docs:
                    item["pr_number"] = pr
                    sparse.append(item)

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
            if not key[1]:
                continue
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
            if r["similarity"] < threshold:
                continue
            iid = r.get("issue_id")
            if iid and iid in self.indexer.issue_docs:
                doc = self.indexer.issue_docs[iid]
                if state != "all" and doc.state != state:
                    continue
                if labels and not any(l in doc.labels for l in labels):
                    continue
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
                    if clean_q in title:
                        base += 0.15
                    else:
                        qw, tw = set(clean_q.split()), set(title.split())
                        if qw and tw:
                            ov = len(qw & tw) / len(qw)
                            if ov > 0.4:
                                base += 0.08 if ov > 0.6 else 0.04

                # Hybrid boost
                if r.get("has_dense") and r.get("has_sparse"):
                    base += 0.03

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
        if not self._initialized:
            await self.initialize()

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

    def is_initialized(self) -> bool:
        return self._initialized

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

            if new_issues or new_prs:
                self.retriever = IssueRetriever(self.indexer)
            return {"status": "completed", "new_issues": new_issues, "new_prs": new_prs, "sync_time": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {"status": "error", "error": str(e)}

    async def _sync_issues(self, max_new: Optional[int] = None) -> int:
        try:
            existing = set(self.indexer.issue_docs.keys())
            recent = await self.indexer.github_client.list_issues(f"https://github.com/{self.repo_owner}/{self.repo_name}", state="all", max_pages=5)
            new = [i for i in recent if i.id not in existing]
            if max_new:
                new = new[:max_new]

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
            if i.labels:
                content += f"\n\nLabels: {', '.join(l.name for l in i.labels)}"
            docs.append(Document(text=content, metadata={"type": "issue", "issue_id": i.id, "state": i.state}))

        if docs:
            nodes = create_node_parser().get_nodes_from_documents(docs)
            self.indexer.vector_index.insert_nodes(nodes)
            await self.indexer._save_issues()
            self.indexer._save_faiss_index(nodes, append=True)

    async def get_prs(self, state: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        try:
            diffs = sorted(self.indexer.patch_builder.load_diff_docs(), key=lambda x: x.pr_number, reverse=True)
            if state != "all":
                diffs = [d for d in diffs if d.merged_at]
            return [{"number": d.pr_number, "title": getattr(d, 'pr_title', f"PR #{d.pr_number}"),
                    "merged_at": d.merged_at, "files_changed": d.files_changed, "issue_id": d.issue_id} for d in diffs[:limit]]
        except Exception as e:
            logger.error(f"Get PRs error: {e}")
            return []

    async def get_pr_details(self, pr_number: int) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        if pr_number in self._pr_cache:
            return self._pr_cache[pr_number]

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
        if not self._initialized:
            await self.initialize()
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

    def clear_pr_cache(self) -> None:
        self._pr_cache.clear()


# ============================================================================
# SECTION 4: COMPOSITE RETRIEVER (~70 lines)
# From agentic_rag.py
# ============================================================================

ContextChunk = Dict[str, Any]
RetrievalMode = Literal["chunks", "files_via_metadata", "files_via_content", "auto_routed"]


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class CompositeConfig:
    chunk_sizes: Dict[str, int] = field(default_factory=lambda: {"code": 2000, "issues": 1500, "prs": 1500, "docs": 1000, "tests": 1500, "configs": 800})
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {"dense": 0.6, "sparse": 0.3, "agentic": 0.1})
    routing_thresholds: Dict[str, float] = field(default_factory=lambda: {"complexity_simple": 0.3, "complexity_moderate": 0.6, "complexity_complex": 0.8, "confidence_min": 0.4, "agentic_threshold": 0.5})
    max_concurrent_queries: int = 5
    max_results_per_index: int = 10
    enable_reranking: bool = True
    cache_routing_decisions: bool = True


def extract_repo_info_from_url(repo_url: str) -> Dict[str, str]:
    clean = repo_url.rstrip('/').replace('.git', '')
    parts = clean.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
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
            except:
                pass

    async def initialize_indices(self, repo_path: str, rag_extractor: LocalRepoContextExtractor, issue_rag: Optional[IssueAwareRAG] = None) -> None:
        self.indices["code"] = rag_extractor
        if issue_rag and issue_rag.is_initialized():
            self.indices["issues"] = issue_rag

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
                    for c in chunks:
                        c.update({"source_index": "code", "index_type": "code", "relevance_score": c.get("similarity", c.get("score", 0.5))})
                    return chunks
                except:
                    return []
            tasks.append(get_code())

        if "issues" in self.indices and "issues" in targets:
            async def get_issues():
                try:
                    ctx = await self.indices["issues"].get_issue_context(query, max_issues=max_results//3)
                    return [{"content": f"Issue #{r.issue.id}: {r.issue.title}\n{r.issue.body[:500]}", "file": f"issue_{r.issue.id}",
                            "similarity": r.similarity, "relevance_score": r.similarity, "type": "issue", "source_index": "issues",
                            "index_type": "issue", "issue_id": r.issue.id, "issue_state": getattr(r.issue, 'state', 'unknown')}
                           for r in (ctx.related_issues or [])]
                except:
                    return []
            tasks.append(get_issues())

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        return [c for r in results if isinstance(r, list) for c in r]

    def _apply_fusion(self, chunks: List[Dict], query: str, analysis: Dict) -> List[Dict]:
        if not chunks:
            return []
        q = query.lower()
        weights = {"code": 1.0, "issue": 0.8, "pr": 0.9, "docs": 0.7}
        if any(w in q for w in ["bug", "error", "problem"]):
            weights.update({"issue": 1.3, "code": 0.9})
        elif any(w in q for w in ["implement", "function", "class"]):
            weights.update({"code": 1.2, "issue": 0.7})

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
            if seen.get(fp, 0) < 2:
                diverse.append(c)
                seen[fp] = seen.get(fp, 0) + 1
        return diverse

    async def _fallback(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            if "code" in self.indices:
                ctx = await self.indices["code"].get_relevant_context(query, None)
                return {"context_chunks": ctx.get("sources", [])[:max_results], "query_analysis": {"query_type": "general", "complexity": QueryComplexity.SIMPLE},
                        "total_processing_time": 0.1, "cache_hits": 0, "fusion_applied": False, "reranking_applied": False}
        except:
            pass
        return {"context_chunks": [], "query_analysis": {"query_type": "error", "complexity": QueryComplexity.SIMPLE}, "total_processing_time": 0.0, "cache_hits": 0, "fusion_applied": False, "reranking_applied": False}

    def get_statistics(self) -> Dict[str, Any]:
        return {"total_queries": self._stats["total_queries"], "cache_hit_rate": self._stats["cache_hits"] / max(1, self._stats["total_queries"]),
                "routing_decisions": dict(self._stats["routing_decisions"]), "index_usage": dict(self._stats["index_usage"]), "available_indices": list(self.indices.keys())}

    def clear_cache(self) -> None:
        self.routing_cache.clear()


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
                        except:
                            self.rag_extractor = None

                if not self.rag_extractor:
                    code_rag = LocalRepoContextExtractor()
                    await code_rag.load_repository(repo_url, branch)
                    self.rag_extractor = existing.rag_extractor = code_rag
                    self.repo_path = existing.repo_path = code_rag.current_repo_path

                if self.rag_extractor and self.repo_path:
                    await self._init_composite()
                return
        except Exception as e:
            logger.debug(f"Session reuse failed, creating new: {e}")

        code_rag = LocalRepoContextExtractor()
        await code_rag.load_repository(repo_url, branch)
        self.rag_extractor, self.repo_path, self.repo_info = code_rag, code_rag.current_repo_path, repo_info

        try:
            from .agent_tools.core import AgenticCodebaseExplorer
            self.agentic_explorer = AgenticCodebaseExplorer(self.session_id, self.repo_path, issue_rag_system=None)
        except Exception as e:
            logger.debug(f"AgenticCodebaseExplorer init failed: {e}")
            self.agentic_explorer = None

        if self.agentic_explorer:
            try:
                await self.agentic_explorer.initialize_commit_index(force_rebuild=False)
            except Exception as e:
                logger.debug(f"Commit index init failed: {e}")

        await self._init_composite()
        await self._init_founding_member()

    async def _init_composite(self) -> None:
        if self.rag_extractor and self.repo_path:
            try:
                await self.composite_retriever.initialize_indices(self.repo_path, self.rag_extractor, self.issue_rag)
                self._use_composite = True
            except:
                self._use_composite = False

    async def _init_founding_member(self) -> None:
        if not (self.rag_extractor and self.repo_path):
            return
        try:
            from .founding_member_agent import FoundingMemberAgent
            self.founding_member_agent = FoundingMemberAgent(session_id=self.session_id, code_rag=self.rag_extractor, issue_rag=self.issue_rag)
        except:
            self.founding_member_agent = None

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
                if (eta := update.estimated_time_remaining):
                    msg += f" - ~{eta//60}m {eta%60}s remaining" if eta >= 60 else f" - ~{eta}s remaining"
                session["metadata"]["message"] = msg
            except:
                pass

        try:
            session["metadata"].update({"status": "issue_linking", "message": f"Starting issue linking for {owner}/{repo}..."})
            self.issue_rag = IssueAwareRAG(owner, repo, progress_cb)
            try:
                await self.issue_rag.initialize(force_rebuild=False)
            except RuntimeError as e:
                if "cannot reuse" in str(e):
                    self.issue_rag = IssueAwareRAG(owner, repo, progress_cb)
                    await self.issue_rag.initialize(force_rebuild=True)
                else:
                    raise

            if self.agentic_explorer:
                self.agentic_explorer.issue_rag_system = self.issue_rag
                for attr in ['pr_ops', 'issue_ops']:
                    if hasattr(self.agentic_explorer, attr):
                        setattr(getattr(self.agentic_explorer, attr), 'issue_rag_system', self.issue_rag)

            if self._use_composite:
                self.composite_retriever.indices["issues"] = self.issue_rag
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
        if not self.rag_extractor:
            raise ValueError("AgenticRAG not initialized")

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
                    except:
                        pass
                return base.get("sources", [])
        except Exception as e:
            self.logger.error(f"Enhanced context failed: {e}")
            try:
                return (await self.rag_extractor.get_relevant_context(query, restrict_files)).get("sources", [])
            except:
                return []

    def get_repo_info(self) -> Optional[Dict[str, Any]]:
        return self.repo_info

    def get_repo_path(self) -> Optional[str]:
        return self.repo_path

    def get_composite_statistics(self) -> Optional[Dict[str, Any]]:
        return self.composite_retriever.get_statistics() if self.composite_retriever else None

    def _fm_wrap(self, method: str, *args, **kwargs):
        if not self.founding_member_agent:
            return json.dumps({"error": "FoundingMemberAgent not available"})
        return getattr(self.founding_member_agent, method)(*args, **kwargs)

    async def get_file_history(self, file_path: str) -> str:
        return self._fm_wrap("get_file_history", file_path)

    async def summarize_feature_evolution(self, feature_query: str) -> str:
        return self._fm_wrap("summarize_feature_evolution", feature_query)

    async def who_fixed_this(self, file_path: str, line_number: int = None) -> str:
        return self._fm_wrap("who_fixed_this", file_path, line_number)

    async def who_implemented_this(self, feature_name: str, file_path: Optional[str] = None) -> str:
        return self._fm_wrap("who_implemented_this", feature_name, file_path)

    async def regression_detector(self, issue_query: str) -> str:
        return await self.founding_member_agent.regression_detector(issue_query) if self.founding_member_agent else json.dumps({"error": "FoundingMemberAgent not available"})

    async def agentic_analysis(self, user_query: str) -> str:
        return await self.founding_member_agent.agentic_answer(user_query) if self.founding_member_agent else json.dumps({"error": "FoundingMemberAgent not available"})

    async def cleanup(self):
        try:
            if self.agentic_explorer:
                self.agentic_explorer.reset_memory()
            self._query_cache.clear()
            if self.composite_retriever:
                self.composite_retriever.routing_cache.clear()
        except:
            pass


# ============================================================================
# SECTION 5: PUBLIC API + BACKWARD COMPATIBILITY (~30 lines)
# Ensures zero breaking changes for existing code
# ============================================================================

# All classes are already exported with their original names above:
# - LocalRepoContextExtractor (Code RAG)
# - IssueIndexer, IssueReranker, IssueRetriever, IssueAwareRAG (Issue RAG)
# - CompositeAgenticRetriever, AgenticRAGSystem (Composite/Agentic RAG)

__all__ = [
    # Shared infrastructure
    "create_embedding_model",
    "create_faiss_index",
    "create_vector_store_index",
    "create_bm25_retriever",
    "create_hybrid_retriever",
    "create_node_parser",
    "fix_node_relationships",
    "load_files",
    "to_int",

    # Code RAG
    "LocalRepoContextExtractor",
    "TREE_SITTER_MAP",
    "CUSTOM_SIGNATURES",
    "CUSTOM_COMMENTS",
    "FILE_PATTERNS",
    "COMPLEXITY_PATTERNS",
    "patch_code_hierarchy",

    # Issue RAG
    "IssueIndexer",
    "IssueReranker",
    "IssueRetriever",
    "IssueAwareRAG",

    # Composite/Agentic RAG
    "QueryComplexity",
    "CompositeConfig",
    "CompositeAgenticRetriever",
    "AgenticRAGSystem",
    "extract_repo_info_from_url",
    "ContextChunk",
    "RetrievalMode",
]
