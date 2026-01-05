"""GitHub API Client - tinygrad-style rewrite (937→~300 lines)"""
import re, asyncio, time, logging, aiohttp
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from .config import settings
from .models import Issue, IssueResponse, IssueComment, PullRequestInfo, PullRequestUser, EnhancedPullRequestInfo, PullRequestReview, PullRequestReviewer, PullRequestStatusCheck
from .utils.decorators import cached, retry, safe_op

logger = logging.getLogger(__name__)

# GraphQL Queries as module constants
GQL_OPEN_PRS = """query($owner: String!, $repo: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number title url createdAt updatedAt body isDraft author { login } reviewDecision mergeable additions deletions changedFiles
        reviewRequests(first: 10) { nodes { requestedReviewer { __typename ... on User { login } ... on Team { name } } } }
        reviews(last: 20) { nodes { author { login } state submittedAt body } }
        commits(last: 1) { nodes { commit { statusCheckRollup { state contexts(first: 10) { nodes { __typename
          ... on StatusContext { context description state } ... on CheckRun { name conclusion status } } } } } } }
        files(first: 100) { nodes { path } } }
      pageInfo { hasNextPage endCursor } } } }"""

GQL_PR_REVIEWS = """query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) { reviews(first: 100) { nodes { author { login } state submittedAt body } } } } }"""

GQL_PR_DETAILS = """query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) { number title url state createdAt updatedAt mergedAt body isDraft author { login } reviewDecision mergeable additions deletions changedFiles
      reviewRequests(first: 10) { nodes { requestedReviewer { __typename ... on User { login } ... on Team { name } } } }
      reviews(last: 50) { nodes { author { login } state submittedAt body } }
      commits(last: 1) { nodes { commit { statusCheckRollup { state contexts(first: 20) { nodes { __typename
        ... on StatusContext { context description state } ... on CheckRun { name conclusion status } } } } } } }
      files(first: 200) { nodes { path } } } } }"""


class GitHubIssueClient:
    def __init__(self):
        if not settings.github_token: raise ValueError("GITHUB_TOKEN required")
        self.token = settings.github_token
        self.headers = {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}

    def _parse_url(self, url: str) -> Optional[Tuple[str, str, int]]:
        if (m := re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)):
            return m.group(1), m.group(2), int(m.group(3))
        return None

    @retry(attempts=3, delay=1.0, backoff=2.0)
    async def _api_get(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        async with session.get(url, headers=self.headers) as r:
            if r.status == 404: return None
            if r.status == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                await asyncio.sleep(max(0, reset - time.time()) + 1)
                return await self._api_get(session, url)  # Retry after rate limit
            if r.status != 200: return None
            return await r.json()

    async def _api_post(self, session: aiohttp.ClientSession, url: str, payload: Dict) -> Optional[Dict]:
        async with session.post(url, headers=self.headers, json=payload) as r:
            if r.status in [200, 201]: return await r.json()
            return None

    @retry(attempts=3, delay=1.0, backoff=2.0)
    async def _gql(self, session: aiohttp.ClientSession, query: str, variables: Dict) -> Optional[Dict]:
        async with session.post("https://api.github.com/graphql", json={"query": query, "variables": variables}, headers=self.headers) as r:
            if r.status != 200: return None
            data = await r.json()
            return None if "errors" in data else data

    async def _fetch_comments(self, session: aiohttp.ClientSession, owner: str, repo: str, num: int) -> List[IssueComment]:
        data = await self._api_get(session, f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments")
        if not data: return []
        return [IssueComment(body=c["body"], user=c.get("user", {}).get("login", ""),
                            created_at=datetime.fromisoformat(c["created_at"].replace('Z', '+00:00'))) for c in data]

    async def get_issue(self, issue_url: str) -> IssueResponse:
        info = self._parse_url(issue_url)
        if not info: return IssueResponse(status="error", error="Invalid GitHub issue URL")
        owner, repo, num = info

        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = await self._api_get(session, f"https://api.github.com/repos/{owner}/{repo}/issues/{num}")
            if not data: return IssueResponse(status="error", error=f"Issue #{num} not found")

            comments = await self._fetch_comments(session, owner, repo, num)
            issue = Issue(number=num, title=data['title'], body=data.get('body') or "", state=data['state'],
                         created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                         closed_at=datetime.fromisoformat(data['closed_at'].replace('Z', '+00:00')) if data.get('closed_at') else None,
                         url=issue_url, labels=[l['name'] for l in data.get('labels', [])],
                         assignees=[a['login'] for a in data.get('assignees', [])], comments=comments)

            return IssueResponse(status="success", data=issue)

    def get_issue_data(self, issue_url: str) -> Dict[str, Any]:
        try:
            loop = None
            try: loop = asyncio.get_running_loop()
            except: pass

            if loop:
                with ThreadPoolExecutor() as ex:
                    response = ex.submit(asyncio.run, self.get_issue(issue_url)).result(timeout=30)
            else:
                response = asyncio.run(self.get_issue(issue_url))

            if response.status != "success" or not response.data:
                raise Exception(response.error or "Unknown error")

            data = response.data.model_dump()
            if (info := self._parse_url(issue_url)):
                owner, repo, _ = info
                data["repository"] = {"owner": {"login": owner}, "name": repo, "clone_url": f"https://github.com/{owner}/{repo}.git", "default_branch": "main"}
            return data
        except Exception as e:
            raise Exception(f"Failed to fetch issue: {e}")

    async def list_issues(self, repo_url: str, state: str = "open", per_page: int = 30, max_pages: int = 5) -> List[Issue]:
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        issues, page = [], 1

        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while page <= max_pages:
                data = await self._api_get(session, f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}&per_page={per_page}&page={page}")
                if not data: break

                for d in data:
                    if 'pull_request' in d: continue
                    try:
                        issues.append(Issue(number=d['number'], title=d['title'], body=d.get('body') or "", state=d['state'],
                                           created_at=datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')),
                                           closed_at=datetime.fromisoformat(d['closed_at'].replace('Z', '+00:00')) if d.get('closed_at') else None,
                                           url=d['html_url'], labels=[l['name'] for l in d.get('labels', [])],
                                           assignees=[a['login'] for a in d.get('assignees', [])], comments=[]))
                    except: continue

                if len(data) < per_page: break
                page += 1
        return issues

    async def create_issue(self, owner: str, repo: str, title: str, body: str, labels: List[str] = None) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            payload = {"title": title, "body": body}
            if labels: payload["labels"] = labels
            result = await self._api_post(session, f"https://api.github.com/repos/{owner}/{repo}/issues", payload)
            if not result: raise Exception("Failed to create issue")
            return result

    async def post_issue_comment(self, issue_url: str, comment_body: str) -> Dict[str, Any]:
        info = self._parse_url(issue_url)
        if not info: raise ValueError("Invalid GitHub issue URL")
        owner, repo, num = info

        async with aiohttp.ClientSession() as session:
            result = await self._api_post(session, f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments", {"body": comment_body})
            if not result: raise Exception(f"Failed to post comment to issue #{num}")
            return {"id": result["id"], "url": result["html_url"], "body": result["body"], "created_at": result["created_at"], "user": result["user"]["login"]}

    async def list_pull_requests(self, repo_url: str, state: str = "merged", per_page: int = 30, max_pages: int = 5) -> List[PullRequestInfo]:
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        prs, page = [], 1
        api_state = "closed" if state == "merged" else state

        async with aiohttp.ClientSession() as session:
            while page <= max_pages:
                data = await self._api_get(session, f"https://api.github.com/repos/{owner}/{repo}/pulls?state={api_state}&per_page={per_page}&page={page}&sort=updated&direction=desc")
                if not data: break

                for d in data:
                    if state == "merged" and not d.get("merged_at"): continue
                    user = PullRequestUser(login=d["user"]["login"]) if d.get("user") else None
                    prs.append(PullRequestInfo(number=d['number'], title=d['title'], merged_at=d.get('merged_at'),
                                              files_changed=[], issue_id=None, url=d.get('html_url'), user=user, body=d.get('body')))
                if len(data) < per_page: break
                page += 1
        return prs

    def _parse_pr(self, d: Dict) -> EnhancedPullRequestInfo:
        reviews = [PullRequestReview(author=r["author"]["login"], state=r["state"], submitted_at=r["submittedAt"], body=r.get("body"))
                  for r in d.get("reviews", {}).get("nodes", []) if r.get("author")]

        review_requests = [PullRequestReviewer(login=rr.get("login"), name=rr.get("name"), type=rr.get("__typename", "User"))
                         for req in d.get("reviewRequests", {}).get("nodes", []) if (rr := req.get("requestedReviewer"))]

        status_checks = []
        if (commits := d.get("commits", {}).get("nodes")) and (rollup := commits[0].get("commit", {}).get("statusCheckRollup")):
            status_checks.append(PullRequestStatusCheck(state=rollup.get("state", "UNKNOWN"), context="overall", description="Overall status"))
            for ctx in rollup.get("contexts", {}).get("nodes", []):
                if ctx.get("__typename") == "StatusContext":
                    status_checks.append(PullRequestStatusCheck(state=ctx.get("state", "UNKNOWN"), context=ctx.get("context"), description=ctx.get("description")))
                elif ctx.get("__typename") == "CheckRun":
                    state_map = {"SUCCESS": "SUCCESS", "FAILURE": "FAILURE", "NEUTRAL": "PENDING", "CANCELLED": "FAILURE", "TIMED_OUT": "FAILURE"}
                    status_checks.append(PullRequestStatusCheck(state=state_map.get(ctx.get("conclusion", "NEUTRAL"), "UNKNOWN"),
                                                               context=ctx.get("name"), description=f"Check: {ctx.get('status', 'unknown')}"))

        user = PullRequestUser(login=d["author"]["login"]) if d.get("author") else None
        return EnhancedPullRequestInfo(number=d["number"], title=d["title"], state=d.get("state", "open"),
                                       created_at=d.get("createdAt"), updated_at=d.get("updatedAt"), url=d.get("url"),
                                       body=d.get("body"), user=user, files_changed=[f["path"] for f in d.get("files", {}).get("nodes", [])],
                                       review_decision=d.get("reviewDecision"), reviews=reviews, review_requests=review_requests,
                                       mergeable=d.get("mergeable"), status_checks=status_checks, draft=d.get("isDraft", False),
                                       commits_count=d.get("changedFiles"), additions=d.get("additions"), deletions=d.get("deletions"))

    async def list_open_pull_requests_with_reviews(self, repo_url: str, per_page: int = 50, max_pages: int = 10) -> List[EnhancedPullRequestInfo]:
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        prs, cursor, page = [], None, 0

        async with aiohttp.ClientSession() as session:
            while page < max_pages:
                data = await self._gql(session, GQL_OPEN_PRS, {"owner": owner, "repo": repo, "first": per_page, "after": cursor})
                if not data: break

                pr_data = data["data"]["repository"]["pullRequests"]
                prs.extend(self._parse_pr(p) for p in pr_data["nodes"])

                if not pr_data["pageInfo"]["hasNextPage"]: break
                cursor = pr_data["pageInfo"]["endCursor"]
                page += 1
        return prs

    async def get_pr_reviews(self, repo_url: str, pr_number: int) -> List[PullRequestReview]:
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)

        async with aiohttp.ClientSession() as session:
            data = await self._gql(session, GQL_PR_REVIEWS, {"owner": owner, "repo": repo, "number": pr_number})
            if not data: return []
            return [PullRequestReview(author=r["author"]["login"], state=r["state"], submitted_at=r["submittedAt"], body=r.get("body"))
                   for r in data["data"]["repository"]["pullRequest"]["reviews"]["nodes"] if r.get("author")]

    async def get_pr_detailed_info(self, repo_url: str, pr_number: int) -> Optional[EnhancedPullRequestInfo]:
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)

        async with aiohttp.ClientSession() as session:
            data = await self._gql(session, GQL_PR_DETAILS, {"owner": owner, "repo": repo, "number": pr_number})
            if not data or not data.get("data", {}).get("repository", {}).get("pullRequest"): return None
            return self._parse_pr(data["data"]["repository"]["pullRequest"])

    async def create_gist(self, gist_data: Dict) -> Optional[Dict]:
        async with aiohttp.ClientSession() as session:
            return await self._api_post(session, "https://api.github.com/gists", gist_data)
