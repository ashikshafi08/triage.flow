"""Commit Metadata Index - tinygrad-style rewrite (1,207→~400 lines)"""
import os, json, asyncio, logging, subprocess, re, shutil, hashlib, faiss, Stemmer
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from tqdm.auto import tqdm

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CommitMeta:
    sha: str; author_name: str; author_email: str; committer_name: str; committer_email: str
    commit_date: str; author_date: str; subject: str; body: str
    files_changed: List[str]; files_added: List[str]; files_modified: List[str]; files_deleted: List[str]
    insertions: int; deletions: int; is_merge: bool; parent_shas: List[str]
    branch_info: Optional[str] = None; pr_number: Optional[int] = None

    def to_dict(self) -> dict: return asdict(self)


@dataclass
class CommitSearchResult:
    commit: CommitMeta; similarity: float; match_reasons: List[str]; file_relevance: float = 0.0

    def to_dict(self) -> dict: return {"commit": self.commit.to_dict(), "similarity": self.similarity, "match_reasons": self.match_reasons, "file_relevance": self.file_relevance}


def git_cmd(repo_path: Path, args: List[str], timeout: int = 60) -> Optional[str]:
    """Run git command and return stdout or None on failure"""
    try:
        r = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=repo_path, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception as e:
        logger.warning(f"Git command failed: {e}")
        return None


class CommitIndexer:
    """Handles indexing and storage of Git commit metadata"""

    def __init__(self, repo_path: str, repo_owner: str = None, repo_name: str = None):
        self.repo_path = Path(repo_path)
        self.repo_owner, self.repo_name = repo_owner, repo_name
        self.repo_key = f"{repo_owner}/{repo_name}" if repo_owner and repo_name else "unknown"
        repo_hash = hashlib.md5(str(self.repo_path.resolve()).encode()).hexdigest()[:8]
        self.unique_repo_key = f"{self.repo_key}_{repo_hash}"

        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.openai_api_key)

        self.index_dir = Path(".") / ".index_cache" / "commit_indexes" / self.unique_repo_key.replace('/', '_')
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.commits_file = self.index_dir / "commits.jsonl"
        self.metadata_file = self.index_dir / "metadata.json"
        self.file_stats_file = self.index_dir / "file_stats.json"

        self.vector_index = self.bm25_retriever = None
        self.commit_metas, self.file_touch_stats = {}, {}

    async def build_commit_index(self, max_commits: Optional[int] = None, since_date: Optional[str] = None,
                                  until_date: Optional[str] = None, force_rebuild: bool = False) -> None:
        max_commits = max_commits or getattr(settings, 'MAX_COMMITS_TO_PROCESS', 5000)
        logger.info(f"Building commit index for {self.repo_path} (max={max_commits})")

        if force_rebuild:
            if self.index_dir.exists(): shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(exist_ok=True)
        elif await self.load_existing_index():
            return

        from .local_repo_loader import unshallow_repository
        unshallow_repository(str(self.repo_path))

        commits = await self._extract_commits(max_commits, since_date, until_date)
        if not commits: logger.warning("No commits found"); return

        self._build_file_stats(commits)
        docs = self._create_docs(commits)
        await self._build_faiss(docs)
        await self._build_bm25(docs)
        await self._save_all(commits)
        logger.info(f"Built index with {len(commits)} commits")

    async def _extract_commits(self, max_commits: int, since: Optional[str], until: Optional[str]) -> List[CommitMeta]:
        cmd = ["rev-list", "--all", f"-{max_commits}"]
        if since: cmd += ["--since", since]
        if until: cmd += ["--until", until]

        output = git_cmd(self.repo_path, cmd)
        if not output: return []
        shas = [s for s in output.strip().split('\n') if s.strip()]
        logger.info(f"Found {len(shas)} commits to process")

        commits = []
        for i in tqdm(range(0, len(shas), 50), desc="Processing commits"):
            for sha in shas[i:i+50]:
                if (c := self._parse_commit(sha)): commits.append(c); self.commit_metas[sha] = c
        return commits

    def _parse_commit(self, sha: str) -> Optional[CommitMeta]:
        output = git_cmd(self.repo_path, ["show", "--pretty=format:%H%n%an%n%ae%n%cn%n%ce%n%ci%n%ai%n%P%n%s%n%b%nEND_MESSAGE", "--name-status", sha], timeout=5)
        if not output: return None

        lines = output.split('\n')
        if len(lines) < 9: return None

        try:
            sha, author, email, cname, cemail, cdate, adate = lines[:7]
            parents = lines[7].strip().split() if lines[7].strip() else []
            subject = lines[8].strip()

            body = []
            i = 9
            while i < len(lines) and lines[i] != "END_MESSAGE": body.append(lines[i]); i += 1

            files, added, modified, deleted = [], [], [], []
            i += 1
            while i < len(lines) and not lines[i].strip(): i += 1
            while i < len(lines):
                if '\t' in lines[i]:
                    status, path = lines[i].strip().split('\t', 1)
                    if path.strip():
                        files.append(path.strip())
                        if status.startswith('A'): added.append(path)
                        elif status.startswith('M'): modified.append(path)
                        elif status.startswith('D'): deleted.append(path)
                        elif status.startswith('R') and ' -> ' in path:
                            old, new = path.split(' -> ', 1)
                            files[-1] = new; deleted.append(old); added.append(new)
                i += 1

            ins, dels = 0, 0
            if (stats := git_cmd(self.repo_path, ["show", "--stat", "--format=", sha], timeout=2)):
                for line in stats.split('\n'):
                    if (m := re.search(r'(\d+) insertion', line)): ins = int(m.group(1))
                    if (m := re.search(r'(\d+) deletion', line)): dels = int(m.group(1))

            return CommitMeta(sha=sha, author_name=author, author_email=email, committer_name=cname, committer_email=cemail,
                             commit_date=cdate, author_date=adate, subject=subject, body='\n'.join(body).strip(),
                             files_changed=files, files_added=added, files_modified=modified, files_deleted=deleted,
                             insertions=ins, deletions=dels, is_merge=len(parents) > 1, parent_shas=parents,
                             pr_number=self._extract_pr(subject + " " + '\n'.join(body)))
        except Exception as e:
            logger.warning(f"Failed to parse commit: {e}")
            return None

    def _extract_pr(self, text: str) -> Optional[int]:
        for p in [r'#(\d+)', r'pull request #(\d+)', r'PR #(\d+)', r'\(#(\d+)\)']:
            if (m := re.search(p, text, re.IGNORECASE)):
                try: return int(m.group(1))
                except: pass
        return None

    def _build_file_stats(self, commits: List[CommitMeta]) -> None:
        self.file_touch_stats = {}
        for c in commits:
            for fp in c.files_changed:
                if not fp.strip(): continue
                if fp not in self.file_touch_stats:
                    self.file_touch_stats[fp] = {"touch_count": 0, "authors": set(), "commits": [],
                                                 "first_seen": c.commit_date, "last_seen": c.commit_date, "additions": 0, "deletions": 0}
                s = self.file_touch_stats[fp]
                s["touch_count"] += 1; s["authors"].add(c.author_email)
                s["commits"].append({"sha": c.sha, "date": c.commit_date, "author": c.author_name, "subject": c.subject})
                s["first_seen"] = min(s["first_seen"], c.commit_date); s["last_seen"] = max(s["last_seen"], c.commit_date)
                s["additions"] += c.insertions; s["deletions"] += c.deletions
        for fp in self.file_touch_stats: self.file_touch_stats[fp]["authors"] = list(self.file_touch_stats[fp]["authors"])
        logger.info(f"Built file statistics for {len(self.file_touch_stats)} files")

    def _create_docs(self, commits: List[CommitMeta]) -> List[Document]:
        docs = []
        for c in commits:
            parts = [f"Commit: {c.sha[:12]}", f"Author: {c.author_name} <{c.author_email}>", f"Date: {c.commit_date}", f"Subject: {c.subject}"]
            if c.body.strip(): parts.append(f"Body: {c.body}")
            if c.files_changed: parts.append(f"Files: {', '.join(c.files_changed[:10])}" + (f" (+{len(c.files_changed)-10} more)" if len(c.files_changed) > 10 else ""))
            if c.pr_number: parts.append(f"PR: #{c.pr_number}")
            if c.is_merge: parts.append("Type: Merge")
            parts.append(f"Changes: +{c.insertions} -{c.deletions}")
            docs.append(Document(text="\n".join(parts), metadata={"commit_sha": c.sha, "author": c.author_name, "date": c.commit_date, "type": "commit", "pr_number": c.pr_number, "file_count": len(c.files_changed)}))
        return docs

    async def _build_faiss(self, docs: List[Document]) -> None:
        if not docs: return
        parser = SimpleNodeParser.from_defaults(chunk_size=2048, chunk_overlap=100)
        nodes = parser.get_nodes_from_documents(docs)
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        self.vector_index = VectorStoreIndex(nodes=nodes, storage_context=StorageContext.from_defaults(vector_store=vector_store), embed_model=self.embed_model)
        self.vector_index.storage_context.persist(str(self.index_dir))

    async def _build_bm25(self, docs: List[Document]) -> None:
        if not docs: return
        nodes = SimpleNodeParser.from_defaults(chunk_size=2048).get_nodes_from_documents(docs)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=50, stemmer=Stemmer.Stemmer("english"), language="english")

    async def load_existing_index(self) -> bool:
        files = [self.commits_file, self.metadata_file, self.file_stats_file]
        if not all(f.exists() for f in files): return False

        try:
            meta = json.load(open(self.metadata_file))
            if meta.get("total_commits", 0) < 5: return False
            if meta.get("repo_path", "") and meta["repo_path"] != str(self.repo_path.resolve()): return False

            self.commit_metas = {}
            for line in open(self.commits_file, 'r', encoding='utf-8'):
                data = json.loads(line.strip())
                c = CommitMeta(**data)
                self.commit_metas[c.sha] = c
            self.file_touch_stats = json.load(open(self.file_stats_file, 'r', encoding='utf-8'))

            if not self.commit_metas: return False

            # Try loading vector store
            vs_files = ["default__vector_store.json", "docstore.json", "index_store.json"]
            if all((self.index_dir / f).exists() for f in vs_files):
                try:
                    for f in vs_files: json.load(open(self.index_dir / f))  # Validate JSON
                    Settings.embed_model = self.embed_model
                    self.vector_index = VectorStoreIndex.from_storage(StorageContext.from_defaults(persist_dir=str(self.index_dir)))
                except:
                    for f in vs_files + ["graph_store.json", "image__vector_store.json"]:
                        if (p := self.index_dir / f).exists(): p.unlink()
                    self.vector_index = None

            # Rebuild vector store if needed
            if self.vector_index is None and self.commit_metas:
                docs = self._create_docs(list(self.commit_metas.values()))
                await self._build_faiss(docs)

            # Always rebuild BM25
            if self.commit_metas:
                await self._build_bm25(self._create_docs(list(self.commit_metas.values())))

            logger.info(f"Loaded index with {len(self.commit_metas)} commits")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    async def _save_all(self, commits: List[CommitMeta]) -> None:
        with open(self.commits_file, 'w', encoding='utf-8', errors='replace') as f:
            for c in commits:
                d = {k: v.encode('utf-8', 'replace').decode('utf-8') if isinstance(v, str) else v for k, v in asdict(c).items()}
                f.write(json.dumps(d, ensure_ascii=True) + '\n')
        json.dump(self.file_touch_stats, open(self.file_stats_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=True)
        json.dump({"total_commits": len(commits), "created_at": datetime.now().isoformat(), "repo_path": str(self.repo_path.resolve()), "repo_key": self.repo_key}, open(self.metadata_file, 'w'), indent=2)


class CommitRetriever:
    """Handles commit search and retrieval"""

    def __init__(self, indexer: CommitIndexer):
        self.indexer = indexer

    async def search_commits(self, query: str, k: int = 10, author_filter: Optional[str] = None, date_range: Optional[Tuple[str, str]] = None,
                            file_filter: Optional[List[str]] = None, pr_filter: Optional[int] = None, include_merges: bool = True) -> List[CommitSearchResult]:
        if not self.indexer.bm25_retriever: return []

        dense = await self._search(query, k * 2, "vector_index", "dense") if self.indexer.vector_index else []
        sparse = await self._search(query, k * 2, "bm25_retriever", "sparse")

        combined = {}
        for src, weight in [(dense, 0.7), (sparse, 0.5)]:
            for r in src:
                sha = r["commit"].sha
                if sha not in combined:
                    combined[sha] = {**r, "combined_score": r["score"] * weight}
                else:
                    combined[sha]["combined_score"] += r["score"] * 0.3

        results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        filtered = [r for r in results if self._passes_filter(r["commit"], author_filter, date_range, file_filter, pr_filter, include_merges)]
        return [CommitSearchResult(commit=r["commit"], similarity=r["combined_score"], match_reasons=["hybrid" if r.get("source") else "search"]) for r in filtered[:k]]

    async def _search(self, query: str, k: int, index_attr: str, source: str) -> List[Dict]:
        try:
            retriever = getattr(self.indexer, index_attr)
            if index_attr == "vector_index": retriever = retriever.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)
            return [{"commit": self.indexer.commit_metas[n.metadata.get("commit_sha")], "score": getattr(n, 'score', 0.0), "source": source}
                    for n in nodes if n.metadata.get("commit_sha") in self.indexer.commit_metas]
        except Exception as e:
            logger.error(f"{source} search failed: {e}")
            return []

    def _passes_filter(self, c: CommitMeta, author: Optional[str], dates: Optional[Tuple[str, str]], files: Optional[List[str]], pr: Optional[int], merges: bool) -> bool:
        if author and author.lower() not in c.author_name.lower(): return False
        if dates:
            try:
                cd = datetime.fromisoformat(c.commit_date.replace('Z', '+00:00'))
                if dates[0] and cd < datetime.fromisoformat(dates[0]): return False
                if dates[1] and cd > datetime.fromisoformat(dates[1]): return False
            except: return False
        if files and not any(f in c.files_changed for f in files): return False
        if pr and c.pr_number != pr: return False
        if not merges and c.is_merge: return False
        return True

    def get_file_timeline(self, file_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        timeline = []
        for c in self.indexer.commit_metas.values():
            if file_path in c.files_changed:
                change = "added" if file_path in c.files_added else ("modified" if file_path in c.files_modified else ("deleted" if file_path in c.files_deleted else "unknown"))
                timeline.append({"sha": c.sha, "author": c.author_name, "date": c.commit_date, "subject": c.subject, "change_type": change, "insertions": c.insertions, "deletions": c.deletions, "pr_number": c.pr_number})
        return sorted(timeline, key=lambda x: x["date"], reverse=True)[:limit]

    def get_file_statistics(self, file_path: str) -> Optional[Dict[str, Any]]: return self.indexer.file_touch_stats.get(file_path)
    def get_commit_by_sha(self, sha: str) -> Optional[CommitMeta]: return self.indexer.commit_metas.get(sha)


class CommitIndexManager:
    """High-level manager for commit indexing"""

    def __init__(self, repo_path: str, repo_owner: str = None, repo_name: str = None):
        self.indexer = CommitIndexer(repo_path, repo_owner, repo_name)
        self.retriever = CommitRetriever(self.indexer)
        self._initialized = False

    async def initialize(self, max_commits: Optional[int] = None, force_rebuild: bool = False, since_date: Optional[str] = None) -> None:
        await self.indexer.build_commit_index(max_commits=max_commits, force_rebuild=force_rebuild, since_date=since_date)
        self._initialized = True

    def is_initialized(self) -> bool: return self._initialized and bool(self.indexer.commit_metas)

    async def search_commits(self, query: str, **kwargs) -> List[CommitSearchResult]:
        return await self.retriever.search_commits(query, **kwargs) if self.is_initialized() else []

    def get_file_timeline(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        return self.retriever.get_file_timeline(file_path, **kwargs) if self.is_initialized() else []

    def get_file_statistics(self, file_path: str) -> Optional[Dict[str, Any]]:
        return self.retriever.get_file_statistics(file_path) if self.is_initialized() else None

    def get_commit_by_sha(self, sha: str) -> Optional[CommitMeta]:
        return self.retriever.get_commit_by_sha(sha) if self.is_initialized() else None

    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_initialized(): return {"initialized": False}
        commits = self.indexer.commit_metas.values()
        return {"initialized": True, "total_commits": len(commits), "total_files_touched": len(self.indexer.file_touch_stats),
                "total_authors": len({c.author_email for c in commits}), "merge_commits": sum(1 for c in commits if c.is_merge),
                "total_insertions": sum(c.insertions for c in commits), "total_deletions": sum(c.deletions for c in commits),
                "repo_path": str(self.indexer.repo_path), "index_path": str(self.indexer.index_dir)}
