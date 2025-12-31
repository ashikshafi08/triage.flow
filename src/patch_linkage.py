"""Patch→Issue Linkage Module - tinygrad-style rewrite (1,349→~500 lines)"""
import os, json, asyncio, logging, time, re, uuid, aiohttp
from typing import Dict, Any, Optional, List, Set, Callable, NamedTuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from tqdm.auto import tqdm
from src.config import settings
from .utils.decorators import safe_op, retry

logger = logging.getLogger(__name__)
DIFF_TRUNCATION_SENTINEL = "... [diff truncated for embedding] ..."

# GraphQL Queries as module constants
GQL_ISSUES_WITH_PRS = """
query($owner: String!, $name: String!, $after: String, $first: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $first, after: $after, states: [CLOSED], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number title state closedAt
        timelineItems(first: 10, itemTypes: [CLOSED_EVENT, REFERENCED_EVENT, CROSS_REFERENCED_EVENT]) {
          nodes { __typename
            ... on ClosedEvent { closer { __typename
              ... on PullRequest { number title state mergedAt url files(first: 100) { nodes { path } } }
              ... on Commit { oid associatedPullRequests(first: 1) { nodes { number title state mergedAt url files(first: 100) { nodes { path } } } } } } }
            ... on ReferencedEvent { commit { oid associatedPullRequests(first: 1) { nodes { number title state mergedAt url files(first: 100) { nodes { path } } } } } }
            ... on CrossReferencedEvent { source { __typename ... on PullRequest { number title state mergedAt url files(first: 100) { nodes { path } } } } } } } }
      pageInfo { hasNextPage endCursor } } } }"""

GQL_MERGED_PRS = """
query($owner: String!, $name: String!, $after: String, $first: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $first, after: $after, states: [MERGED], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number title state mergedAt url changedFiles files(first: 100) { nodes { path } } }
      pageInfo { hasNextPage endCursor } } } }"""

GQL_OPEN_PRS = """
query($owner: String!, $name: String!, $after: String, $first: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $first, after: $after, states: [OPEN], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number title body createdAt updatedAt url isDraft author { login } reviewDecision mergeable
        reviews(last: 10) { nodes { author { login } state submittedAt body } }
        commits(last: 1) { nodes { commit { statusCheckRollup { state contexts(first: 5) {
          nodes { __typename ... on StatusContext { context state } ... on CheckRun { name conclusion } } } } } } }
        files(first: 50) { nodes { path } } }
      pageInfo { hasNextPage endCursor } } } }"""


class ProgressUpdate(NamedTuple):
    stage: str; step: str; pct: float; done: int; total: int; item: str = ""; eta: int = 0


@dataclass
class PatchLink:
    issue_id: int; pr_number: int; merged_at: Optional[str]; pr_title: str; pr_url: str; pr_diff_url: str; files_changed: List[str]


@dataclass
class DiffDoc:
    pr_number: int; issue_id: int; files_changed: List[str]; diff_path: str; diff_text: str; diff_summary: str; merged_at: Optional[str] = None


@dataclass
class OpenPRDoc:
    pr_number: int; title: str; body: str; author: str; created_at: str; updated_at: str; files_changed: List[str]
    review_decision: Optional[str] = None; reviews_summary: str = ""; status_summary: str = ""
    draft: bool = False; mergeable: Optional[str] = None; url: Optional[str] = None


class PatchLinkageBuilder:
    """Builds and persists the issue→PR mapping for a repository"""

    def __init__(self, repo_owner: str, repo_name: str, progress_callback: Optional[Callable] = None):
        self.repo_owner, self.repo_name = repo_owner, repo_name
        self.repo_key = f"{repo_owner}/{repo_name}"
        self.progress_callback = progress_callback
        self.start_time = None
        self.instance_id = str(uuid.uuid4())[:8]

        token = os.getenv("GITHUB_TOKEN")
        if not token: raise ValueError("GITHUB_TOKEN required")
        self.headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

        self.index_dir = Path(f"index_{repo_owner}_{repo_name}")
        self.index_dir.mkdir(exist_ok=True)
        self.patch_links_file = self.index_dir / "patch_links.jsonl"
        self.open_prs_file = self.index_dir / "open_prs.jsonl"
        self.diffs_dir = self.index_dir / "diffs"
        self.diffs_dir.mkdir(exist_ok=True)
        self.pr_cache = {}

    def _progress(self, stage: str, step: str, pct: float, done: int, total: int, item: str = ""):
        if not self.progress_callback: return
        eta = int((time.time() - self.start_time) / max(pct, 1) * 100 - (time.time() - self.start_time)) if self.start_time and pct > 0 else 0
        try: self.progress_callback(ProgressUpdate(stage, step, pct, done, total, item, eta))
        except Exception as e: logger.error(f"Progress callback error: {e}")

    @retry(attempts=3, delay=1.0, backoff=2.0)
    async def _get(self, session: aiohttp.ClientSession, url: str) -> aiohttp.ClientResponse:
        resp = await session.get(url, headers=self.headers)
        if resp.status == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            await asyncio.sleep(max(0, reset - time.time()) + 1)
            resp.close()
            return await session.get(url, headers=self.headers)
        return resp

    async def _gql(self, session: aiohttp.ClientSession, query: str, variables: dict) -> Optional[dict]:
        try:
            async with session.post("https://api.github.com/graphql", json={"query": query, "variables": variables}, headers=self.headers) as r:
                if r.status != 200: return None
                data = await r.json()
                return None if "errors" in data else data
        except Exception as e:
            logger.error(f"GraphQL error: {e}")
            return None

    async def build_patch_linkage(self, max_issues: Optional[int] = None, max_prs: Optional[int] = None,
                                  download_diffs: bool = True, include_open_prs: bool = True) -> None:
        self.start_time = time.time()
        max_issues = max_issues or settings.MAX_ISSUES_TO_PROCESS
        max_prs = max_prs or settings.MAX_PR_TO_PROCESS
        logger.info(f"Building patch linkage for {self.repo_key}")

        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300, force_close=True)
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60)) as session:
            # Verify connectivity
            self._progress("connectivity", "Verifying API access", 5, 0, 1, self.repo_key)
            resp = await self._get(session, f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}")
            if resp.status != 200: raise RuntimeError(f"Cannot access {self.repo_key}")

            # Fetch issues with linked PRs
            self._progress("issues_and_prs", "Fetching issues", 10, 0, max_issues)
            patch_links = await self._fetch_issues_prs(session, max_issues)
            self._save_jsonl(self.patch_links_file, [self._link_to_dict(l) for l in patch_links])

            # Fetch merged PRs
            self._progress("merged_prs", "Fetching merged PRs", 40, 0, max_prs)
            merged_prs = await self._fetch_merged_prs(session, max_prs)

            # Fetch open PRs
            open_prs = []
            if include_open_prs:
                self._progress("open_prs", "Fetching open PRs", 50, 0, max_prs // 2)
                open_prs = await self._fetch_open_prs(session, max_prs // 2)
                self._save_jsonl(self.open_prs_file, [self._open_pr_to_dict(p) for p in open_prs])

            # Download diffs
            if download_diffs and (patch_links or merged_prs):
                self._progress("downloading_diffs", "Downloading diffs", 60, 0, len(patch_links) + len(merged_prs))
                diff_docs = await self._download_diffs(session, patch_links, merged_prs)
                self._save_jsonl(self.index_dir / "diff_docs.jsonl", [self._diff_to_dict(d) for d in diff_docs])

            self._progress("finalizing", "Complete", 100, 1, 1, "Done!")
            logger.info(f"Patch linkage complete: {len(patch_links)} links, {len(open_prs)} open PRs")

    async def _fetch_issues_prs(self, session: aiohttp.ClientSession, max_issues: int) -> List[PatchLink]:
        links, seen = [], set()
        cursor, processed = None, 0

        while processed < max_issues:
            data = await self._gql(session, GQL_ISSUES_WITH_PRS, {"owner": self.repo_owner, "name": self.repo_name, "first": min(100, max_issues - processed), "after": cursor})
            if not data: break

            issues = data["data"]["repository"]["issues"]
            for issue in issues["nodes"]:
                processed += 1
                for item in issue["timelineItems"]["nodes"]:
                    pr = self._extract_pr(item)
                    if pr and pr.get("mergedAt") and (pair := (issue["number"], pr["number"])) not in seen:
                        seen.add(pair)
                        links.append(PatchLink(
                            issue_id=issue["number"], pr_number=pr["number"], merged_at=pr["mergedAt"],
                            pr_title=pr["title"], pr_url=pr["url"],
                            pr_diff_url=f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr['number']}.diff",
                            files_changed=[f["path"] for f in pr.get("files", {}).get("nodes", [])]))

            if not issues["pageInfo"]["hasNextPage"]: break
            cursor = issues["pageInfo"]["endCursor"]
            self._progress("issues_and_prs", "Processing issues", 10 + (processed / max_issues) * 30, processed, max_issues)

        return links

    def _extract_pr(self, item: dict) -> Optional[dict]:
        if item["__typename"] == "ClosedEvent" and (c := item.get("closer")):
            if c["__typename"] == "PullRequest": return c
            if c["__typename"] == "Commit" and (prs := c.get("associatedPullRequests", {}).get("nodes")): return prs[0]
        if item["__typename"] == "ReferencedEvent" and (commit := item.get("commit")):
            if prs := commit.get("associatedPullRequests", {}).get("nodes"): return prs[0]
        if item["__typename"] == "CrossReferencedEvent" and (src := item.get("source")):
            if src["__typename"] == "PullRequest": return src
        return None

    async def _fetch_merged_prs(self, session: aiohttp.ClientSession, max_prs: int) -> List[dict]:
        prs, cursor = [], None
        while len(prs) < max_prs:
            data = await self._gql(session, GQL_MERGED_PRS, {"owner": self.repo_owner, "name": self.repo_name, "first": min(100, max_prs - len(prs)), "after": cursor})
            if not data: break
            pr_data = data["data"]["repository"]["pullRequests"]
            for pr in pr_data["nodes"]:
                if len(prs) >= max_prs: break
                prs.append({"number": pr["number"], "title": pr["title"], "merged_at": pr["mergedAt"], "url": pr["url"],
                           "diff_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr['number']}.diff",
                           "files_changed": [f["path"] for f in pr.get("files", {}).get("nodes", [])]})
            if not pr_data["pageInfo"]["hasNextPage"]: break
            cursor = pr_data["pageInfo"]["endCursor"]
        return prs

    async def _fetch_open_prs(self, session: aiohttp.ClientSession, max_prs: int) -> List[OpenPRDoc]:
        prs, cursor = [], None
        while len(prs) < max_prs:
            data = await self._gql(session, GQL_OPEN_PRS, {"owner": self.repo_owner, "name": self.repo_name, "first": min(50, max_prs - len(prs)), "after": cursor})
            if not data: break
            pr_data = data["data"]["repository"]["pullRequests"]
            for pr in pr_data["nodes"]:
                prs.append(self._create_open_pr(pr))
            if not pr_data["pageInfo"]["hasNextPage"]: break
            cursor = pr_data["pageInfo"]["endCursor"]
        return prs

    def _create_open_pr(self, pr: dict) -> OpenPRDoc:
        reviews = pr.get("reviews", {}).get("nodes", [])
        rev_counts = {"APPROVED": 0, "CHANGES_REQUESTED": 0, "COMMENTED": 0}
        for r in reviews: rev_counts[r["state"]] = rev_counts.get(r["state"], 0) + 1
        reviews_summary = f"{len(reviews)} reviews: {rev_counts['APPROVED']} approved, {rev_counts['CHANGES_REQUESTED']} changes"

        status_parts = []
        if commits := pr.get("commits", {}).get("nodes"):
            if rollup := commits[0].get("commit", {}).get("statusCheckRollup"):
                status_parts.append(f"CI: {rollup.get('state', 'UNKNOWN')}")

        return OpenPRDoc(
            pr_number=pr["number"], title=pr["title"], body=pr.get("body", ""),
            author=pr.get("author", {}).get("login", "Unknown"),
            created_at=pr["createdAt"], updated_at=pr["updatedAt"],
            files_changed=[f["path"] for f in pr.get("files", {}).get("nodes", [])],
            review_decision=pr.get("reviewDecision"), reviews_summary=reviews_summary,
            status_summary=". ".join(status_parts), draft=pr.get("isDraft", False),
            mergeable=pr.get("mergeable"), url=pr.get("url"))

    async def _download_diffs(self, session: aiohttp.ClientSession, links: List[PatchLink], merged_prs: List[dict]) -> List[DiffDoc]:
        # Prepare all tasks
        tasks = [(l, l) for l in links]
        linked_prs = {l.pr_number for l in links}
        for pr in merged_prs:
            if pr["number"] not in linked_prs:
                link = PatchLink(issue_id=None, pr_number=pr["number"], merged_at=pr.get("merged_at"),
                               pr_title=pr["title"], pr_url=pr["url"], pr_diff_url=pr["diff_url"],
                               files_changed=pr.get("files_changed", []))
                tasks.append((link, link))

        docs, total = [], len(tasks)
        for i in range(0, total, 10):
            batch = [self._download_diff(session, t[0]) for t in tasks[i:i+10]]
            results = await asyncio.gather(*batch, return_exceptions=True)
            docs.extend([r for r in results if r and not isinstance(r, Exception)])
            self._progress("downloading_diffs", "Downloading", 60 + (min(i + 10, total) / total) * 30, min(i + 10, total), total)
            if i + 10 < total: await asyncio.sleep(0.5)
        return docs

    async def _download_diff(self, session: aiohttp.ClientSession, link: PatchLink) -> Optional[DiffDoc]:
        try:
            resp = await self._get(session, link.pr_diff_url)
            if resp.status != 200: return None
            diff_text = await resp.text()
            diff_path = self.diffs_dir / f"pr_{link.pr_number}.diff"
            diff_path.write_text(diff_text)
            return DiffDoc(pr_number=link.pr_number, issue_id=link.issue_id, files_changed=link.files_changed,
                          diff_path=str(diff_path), diff_text=diff_text, diff_summary=self._extract_hunks(diff_text),
                          merged_at=link.merged_at)
        except Exception as e:
            logger.warning(f"Diff download failed PR#{link.pr_number}: {e}")
            return None

    def _extract_hunks(self, diff: str, max_chars: int = 4000) -> str:
        if not diff.strip(): return "No diff content"
        lines, hunks, files = diff.split('\n'), [], set()
        for line in lines:
            if line.startswith('diff --git') and (p := line.split()) and len(p) >= 4:
                files.add(p[3].split('/')[-1])
            elif line.startswith('+++ b/'): hunks.append(f"\n--- {line[6:]} ---")
            elif line.startswith('@@') or (line.startswith(('+', '-', ' ')) and line.strip()):
                hunks.append(line)
            if len('\n'.join(hunks)) > max_chars:
                hunks.append(DIFF_TRUNCATION_SENTINEL); break
        header = f"Files: {', '.join(files) or 'unknown'}\n--- Changes ---\n"
        return header + '\n'.join(hunks)

    # Save/Load helpers
    def _save_jsonl(self, path: Path, items: List[dict]):
        with open(path, 'w') as f:
            for item in items: f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _load_jsonl(self, path: Path) -> List[dict]:
        if not path.exists(): return []
        items = []
        with open(path) as f:
            for line in f:
                try: items.append(json.loads(line.strip()))
                except: pass
        return items

    def _link_to_dict(self, l: PatchLink) -> dict:
        return {"issue_id": l.issue_id, "pr_number": l.pr_number, "merged_at": l.merged_at, "pr_title": l.pr_title,
                "pr_url": l.pr_url, "pr_diff_url": l.pr_diff_url, "files_changed": l.files_changed, "created_at": datetime.now().isoformat()}

    def _diff_to_dict(self, d: DiffDoc) -> dict:
        return {"pr_number": d.pr_number, "issue_id": d.issue_id, "files_changed": d.files_changed,
                "diff_path": d.diff_path, "diff_summary": d.diff_summary, "merged_at": d.merged_at, "created_at": datetime.now().isoformat()}

    def _open_pr_to_dict(self, p: OpenPRDoc) -> dict:
        return {"pr_number": p.pr_number, "title": p.title, "body": p.body, "author": p.author,
                "created_at": p.created_at, "updated_at": p.updated_at, "files_changed": p.files_changed,
                "review_decision": p.review_decision, "reviews_summary": p.reviews_summary,
                "status_summary": p.status_summary, "draft": p.draft, "mergeable": p.mergeable, "url": p.url}

    def load_patch_links(self) -> Dict[int, List[PatchLink]]:
        links_by_issue = {}
        for d in self._load_jsonl(self.patch_links_file):
            link = PatchLink(issue_id=d["issue_id"], pr_number=d["pr_number"], merged_at=d.get("merged_at"),
                           pr_title=d["pr_title"], pr_url=d["pr_url"], pr_diff_url=d["pr_diff_url"],
                           files_changed=d.get("files_changed", []))
            links_by_issue.setdefault(link.issue_id, []).append(link)
        return links_by_issue

    def get_patch_url_for_issue(self, issue_id: int) -> Optional[str]:
        links = self.load_patch_links()
        return links[issue_id][0].pr_diff_url if issue_id in links else None

    def load_diff_docs(self) -> List[DiffDoc]:
        return [DiffDoc(pr_number=d["pr_number"], issue_id=d["issue_id"], files_changed=d.get("files_changed", []),
                       diff_path=d["diff_path"], diff_text="", diff_summary=d["diff_summary"], merged_at=d.get("merged_at"))
                for d in self._load_jsonl(self.index_dir / "diff_docs.jsonl")]

    def load_open_prs(self) -> List[OpenPRDoc]:
        return [OpenPRDoc(pr_number=d["pr_number"], title=d["title"], body=d.get("body", ""), author=d["author"],
                         created_at=d["created_at"], updated_at=d["updated_at"], files_changed=d.get("files_changed", []),
                         review_decision=d.get("review_decision"), reviews_summary=d.get("reviews_summary", ""),
                         status_summary=d.get("status_summary", ""), draft=d.get("draft", False),
                         mergeable=d.get("mergeable"), url=d.get("url"))
                for d in self._load_jsonl(self.open_prs_file)]


async def build_repository_patch_linkage(repo_owner: str, repo_name: str, max_issues: Optional[int] = None, max_prs: Optional[int] = None) -> None:
    builder = PatchLinkageBuilder(repo_owner, repo_name)
    await builder.build_patch_linkage(max_issues or settings.MAX_ISSUES_TO_PROCESS, max_prs or settings.MAX_PR_TO_PROCESS)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3: print("Usage: python -m src.patch_linkage <owner> <repo>"); sys.exit(1)
    asyncio.run(build_repository_patch_linkage(sys.argv[1], sys.argv[2], 500, 500))
