"""Local Repository RAG System - tinygrad-style rewrite (1,277→~450 lines)"""
import os, re, fnmatch, asyncio, logging, faiss, Stemmer, aiofiles
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.packs.code_hierarchy.code_hierarchy import _SignatureCaptureOptions, _SignatureCaptureType, _CommentOptions, _ScopeMethod
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document, NodeRelationship
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from .config import settings
from .local_repo_loader import clone_repo_to_temp_persistent
from .language_config import get_all_extensions, get_language_metadata
from .llm_client import LLMClient
from .agent_tools.llm_config import get_llm_instance
from .cache import rag_cache, folder_cache
from .enhanced_persistence import persistence_manager, IndexMetadata

logger = logging.getLogger(__name__)

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
        if lang not in _COMMENT_OPTIONS: _COMMENT_OPTIONS[lang] = opts
    for lang, ids in CUSTOM_SIGNATURES.items():
        if lang not in _DEFAULT_SIGNATURE_IDENTIFIERS: _DEFAULT_SIGNATURE_IDENTIFIERS[lang] = ids

patch_code_hierarchy()


def fix_node_relationships(node):
    for rel in list(node.relationships.keys()):
        if node.relationships[rel] is None:
            if rel in [NodeRelationship.CHILD, NodeRelationship.NEXT, NodeRelationship.PREVIOUS]:
                node.relationships[rel] = []
            else: del node.relationships[rel]
    return node


async def load_files(input_dir: str, exts: List[str], exclude: List[str] = None) -> List[Document]:
    """Load files from directory asynchronously"""
    exclude = exclude or []
    docs, sem = [], asyncio.Semaphore(100)

    async def read_file(path: Path):
        async with sem:
            try:
                if not path.is_file(): return None
                async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return Document(text=await f.read(), metadata={"file_path": str(path)})
            except: return None

    paths = [p for p in Path(input_dir).rglob("*") if p.suffix in exts and not p.name.startswith('.')
             and not any(p.match(e) for e in exclude) and not any(part.startswith('.') for part in p.parts)]
    results = await asyncio.gather(*[read_file(p) for p in paths])
    return [d for d in results if d]


class LocalRepoContextExtractor:
    """Extract context from locally cloned repository with multi-language support"""

    def __init__(self):
        if not settings.openai_api_key: raise ValueError("OPENAI_API_KEY required")
        self.index = self.query_engine = self.repo_info = self.reranker = None
        self.all_extensions = get_all_extensions()
        self.llm_client = LLMClient()
        self._file_cache = {}

    async def load_repository(self, repo_url: str, branch: str = "main") -> None:
        """Load repository and create vector index"""
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.openai_api_key)
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
                except: persistence_manager.cleanup_corrupted_index(index_dir)

        await self._build_index(repo_url, branch, repo_path, owner, repo, index_dir)

    async def _setup_retrievers(self, repo_path: str, file_paths: List[str]) -> None:
        """Setup BM25 and hybrid retrievers from existing files"""
        docs = []
        for fp in file_paths:
            full = Path(repo_path) / fp
            if full.exists():
                try:
                    docs.append(Document(text=full.read_text(errors='ignore')[:4000], metadata={"file_path": fp}))
                except: pass

        if not docs: return
        parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
        nodes = parser.get_nodes_from_documents(docs)
        top_k = min(200, max(1, len(nodes) // 3))

        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k, stemmer=Stemmer.Stemmer("english"),
                                                          language="english", tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower()))
        self.hybrid_retriever = QueryFusionRetriever([self.vector_store.as_retriever(similarity_top_k=top_k), self.bm25_retriever],
                                                      similarity_top_k=200, num_queries=1, mode="reciprocal_rerank", use_async=True)
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
                    nodes.extend(SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200).get_nodes_from_documents([doc]))
            else:
                nodes.extend(SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200).get_nodes_from_documents([doc]))

        nodes = [fix_node_relationships(n) for n in nodes]

        # Build FAISS index
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        self.vector_store = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=vector_store))
        self.vector_store.storage_context.persist(persist_dir=str(index_dir))
        self.index = self.vector_store

        # Setup retrievers
        top_k = min(200, max(1, len(nodes) // 3))
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k, stemmer=Stemmer.Stemmer("english"),
                                                          language="english", tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower()))
        self.hybrid_retriever = QueryFusionRetriever([self.vector_store.as_retriever(similarity_top_k=top_k), self.bm25_retriever],
                                                      similarity_top_k=200, num_queries=1, mode="reciprocal_rerank", use_async=True)
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
        if imports: header += f"Imports:\n{imports}\n"
        if docs: header += f"Documentation:\n{docs}\n"
        remaining = max_chars - len(header)
        return header + f"Code:\n{content[:remaining]}"

    def _is_file_query(self, query: str) -> bool:
        return any(re.search(p, query, re.IGNORECASE) for p in FILE_PATTERNS)

    def _search_files(self, query: str, restrict: Optional[List[str]] = None) -> List[Dict]:
        if not self.current_repo_path or not os.path.exists(self.current_repo_path): return []

        cache_key = (query.lower(), tuple(sorted(restrict)) if restrict else None)
        if cache_key in self._file_cache: return self._file_cache[cache_key]

        globs = re.findall(r'\*\.[a-zA-Z]+', query)
        keywords = [k for k in re.findall(r'\b\w+\b', query.lower()) if len(k) > 2 and k not in ['the', 'and', 'are', 'files', 'that', 'which', 'what']]
        results = []

        for root, dirs, files in os.walk(self.current_repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.startswith('.'): continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, self.current_repo_path)
                if restrict and rel not in restrict: continue

                score, reasons = 0, []
                for g in globs:
                    if fnmatch.fnmatch(f.lower(), g.lower()): score += 10; reasons.append(f"matches {g}")
                for k in keywords:
                    if k in (f + " " + rel).lower(): score += 5; reasons.append(f"contains '{k}'")

                if score > 0:
                    meta = get_language_metadata(path)
                    content = ""
                    if score >= 15:
                        try: content = open(path, 'r', errors='ignore').read(3000)
                        except: content = "Could not read"
                    results.append({"file": rel, "language": meta["display_name"], "description": meta["description"],
                                   "content": content, "match_score": score, "match_reasons": reasons})

        results = sorted(results, key=lambda x: x["match_score"], reverse=True)[:20]
        self._file_cache[cache_key] = results
        return results

    def _calc_complexity(self, query: str, restrict: Optional[List[str]] = None) -> int:
        c = 1 if len(query.split()) <= settings.SIMPLE_QUERY_WORD_LIMIT else (3 if len(query.split()) <= settings.COMPLEX_QUERY_WORD_THRESHOLD else 5)
        c += len(re.findall(r'@[\w\-/\\.]+', query)) * settings.FILE_MENTION_WEIGHT
        if restrict: c += min(5, len(restrict) // 10)
        for p in COMPLEXITY_PATTERNS:
            if re.search(p, query, re.IGNORECASE): c += 3
        if any(w in query.lower() for w in ['how', 'why', 'explain', 'analyze']): c += 3
        return c

    def _optimal_sources(self, complexity: int) -> int:
        if complexity <= 5: return settings.MIN_RAG_SOURCES
        if complexity <= 10: return settings.DEFAULT_RAG_SOURCES
        if complexity <= 20: return 15
        return settings.MAX_RAG_SOURCES

    async def get_relevant_context(self, query: str, restrict_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get relevant context from repository"""
        if not self.query_engine: raise Exception("Repository not loaded")

        cache_key = rag_cache._generate_cache_key(query, restrict_files, self.repo_info) if settings.ENABLE_RAG_CACHING else None
        if cache_key and (cached := await rag_cache.get(cache_key)): return cached

        # File-oriented query
        if self._is_file_query(query):
            files = self._search_files(query, restrict_files)
            if files:
                result = {"response": f"Found {len(files)} files matching your query", "sources": files[:10],
                         "repo_info": self.repo_info, "search_type": "file_oriented"}
            else:
                result = {"response": "No files matched your query pattern", "sources": [], "repo_info": self.repo_info, "search_type": "file_oriented_no_match"}
            if cache_key: await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG)
            return result

        # Regular RAG search
        complexity = self._calc_complexity(query, restrict_files)
        n_sources = self._optimal_sources(complexity)
        response = self.query_engine.query(query)

        seen, sources = set(), []
        for node in response.source_nodes[:n_sources]:
            fp = node.metadata.get("file_path", "unknown")
            if restrict_files and fp not in restrict_files: continue
            if fp not in seen:
                sources.append({"file": fp, "language": node.metadata.get("display_name", "unknown"),
                               "description": node.metadata.get("description", ""), "content": node.text[:5000]})
                seen.add(fp)

        result = {"response": str(response), "sources": sources, "repo_info": self.repo_info,
                 "search_type": "regular", "complexity": complexity}
        if cache_key: await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG)
        return result

    async def get_issue_context(self, title: str, body: str) -> Dict[str, Any]:
        if not self.query_engine: raise Exception("Repository not loaded")
        langs = ", ".join(self.repo_info.get("languages", {}).values()) or "unknown"
        query = f"Issue: {title}\nDescription: {body}\nLanguages: {langs}"
        return await self.get_relevant_context(query)

    async def _generate_folder_summary(self, folder_path: str) -> Dict[str, Any]:
        cache_key = f"folder_summary_{folder_path}"
        if (cached := await folder_cache.get(cache_key)): return cached

        summary = {"path": folder_path, "file_count": 0, "languages": {}, "key_files": []}
        key_names = {'README.md', 'package.json', 'requirements.txt', 'setup.py', 'main.py', 'index.js', 'index.ts', 'app.py'}

        for root, dirs, files in os.walk(os.path.join(self.current_repo_path, folder_path)):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.startswith('.'): continue
                summary["file_count"] += 1
                lang = get_language_metadata(os.path.join(root, f))["language"]
                if lang != "unknown": summary["languages"][lang] = summary["languages"].get(lang, 0) + 1
                if f in key_names: summary["key_files"].append(os.path.relpath(os.path.join(root, f), self.current_repo_path))

        await folder_cache.set(cache_key, summary, settings.CACHE_TTL_FOLDER)
        return summary
