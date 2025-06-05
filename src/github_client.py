from typing import Optional, Dict, Any, Tuple
import re
import asyncio
from datetime import datetime, timedelta
import aiohttp
from .config import settings
from .models import Issue, IssueResponse, IssueComment

class GitHubIssueClient:
    def __init__(self):
        if not settings.github_token:
            raise ValueError("GitHub token is required. Please set GITHUB_TOKEN in your .env file.")
        self.token = settings.github_token
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = timedelta(minutes=30)  # Default 30 minutes cache
        self.max_retries = 3  # Default 3 retries
        self.backoff_factor = 2  # Default exponential backoff factor
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def _extract_issue_info(self, url: str) -> Optional[Tuple[str, str, int]]:
        """Extract owner, repo, and issue number from URL with validation."""
        pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        match = re.search(pattern, url)
        if match:
            owner, repo, issue_number = match.groups()
            if owner and repo and issue_number.isdigit():
                return owner, repo, int(issue_number)
        return None

    def _get_cached_issue(self, issue_url: str) -> Optional[Dict[str, Any]]:
        """Get cached issue if it exists and is not expired."""
        if issue_url in self.cache:
            cached_data = self.cache[issue_url]
            if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
        return None

    def _cache_issue(self, issue_url: str, data: Dict[str, Any]):
        """Cache an issue with timestamp."""
        self.cache[issue_url] = {
            'data': data,
            'timestamp': datetime.now()
        }

    async def _fetch_issue_comments(self, owner: str, repo: str, issue_number: int) -> list:
        """Fetch comments for a specific issue using GitHub API."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        comments = await response.json()
                        return [
                            IssueComment(
                                body=c["body"],
                                user=c["user"]["login"] if "user" in c and c["user"] else "",
                                created_at=datetime.fromisoformat(c["created_at"].replace('Z', '+00:00'))
                            )
                            for c in comments
                        ]
                    else:
                        return []
        except Exception as e:
            return []

    async def get_issue(self, issue_url: str) -> IssueResponse:
        """
        Fetch a GitHub issue with retry logic and caching.
        
        Args:
            issue_url: The full URL of the GitHub issue
            
        Returns:
            IssueResponse containing issue data or error information
        """
        # Check cache first
        cached_issue = self._get_cached_issue(issue_url)
        if cached_issue:
            return IssueResponse(status="success", data=Issue(**cached_issue))

        # Extract issue information
        issue_info = self._extract_issue_info(issue_url)
        if not issue_info:
            return IssueResponse(status="error", error="Invalid GitHub issue URL")

        owner, repo, issue_number = issue_info
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Get issue details
                    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 404:
                            return IssueResponse(
                                status="error",
                                error=f"Issue #{issue_number} not found"
                            )
                        response.raise_for_status()
                        issue_data = await response.json()
                        
                        # Fetch comments as IssueComment objects
                        comments = await self._fetch_issue_comments(owner, repo, issue_number)
                        
                        # Create Issue object directly
                        issue = Issue(
                            number=issue_number,
                            title=issue_data['title'],
                            body=issue_data['body'] or "",  # Ensure body is never None
                            state=issue_data['state'],
                            created_at=datetime.fromisoformat(issue_data['created_at'].replace('Z', '+00:00')),
                            url=issue_url,
                            labels=[label['name'] for label in issue_data.get('labels', [])],
                            assignees=[assignee['login'] for assignee in issue_data.get('assignees', [])],
                            comments=comments
                        )
                        
                        # Cache the successful result (as dict for caching)
                        self._cache_issue(issue_url, issue.model_dump())
                        return IssueResponse(status="success", data=issue)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return IssueResponse(
                        status="error",
                        error=f"Failed to fetch issue after {self.max_retries} attempts: {str(e)}"
                    )
                # Exponential backoff
                await asyncio.sleep(self.backoff_factor ** attempt)
        
        return IssueResponse(status="error", error="Unknown error occurred")
    
    def get_issue_data(self, issue_url: str) -> Dict[str, Any]:
        """
        Synchronous wrapper to get issue data as a dictionary.
        Used by session_manager for backward compatibility.
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.get_issue(issue_url))
            if response.status == "success" and response.data:
                # Convert Issue model to dict and add repository info
                issue_dict = response.data.model_dump()
                
                # Extract repository info from URL
                issue_info = self._extract_issue_info(issue_url)
                if issue_info:
                    owner, repo, _ = issue_info
                    issue_dict["repository"] = {
                        "owner": {"login": owner},
                        "name": repo,
                        "clone_url": f"https://github.com/{owner}/{repo}.git",
                        "default_branch": "main"  # Default assumption
                    }
                
                return issue_dict
            else:
                raise Exception(f"Failed to fetch issue: {response.error}")
        finally:
            loop.close()

    async def list_issues(self, repo_url: str, state: str = "open", per_page: int = 30, max_pages: int = 5) -> list:
        """
        List issues for a given repository URL and state (open/closed/all).
        Args:
            repo_url: The GitHub repository URL (e.g., https://github.com/owner/repo)
            state: 'open', 'closed', or 'all'
            per_page: Number of issues per page (max 100)
            max_pages: Maximum number of pages to fetch (to avoid huge requests)
        Returns:
            List of Issue objects
        """
        # Extract owner and repo
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        issues = []
        page = 1
        fetched = 0
        while page <= max_pages:
            url = f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}&per_page={per_page}&page={page}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        break
                    data = await response.json()
                    if not data:
                        break
                    for issue_data in data:
                        # Exclude pull requests (GitHub returns PRs in issues API)
                        if 'pull_request' in issue_data:
                            continue
                        issue = Issue(
                            number=issue_data['number'],
                            title=issue_data['title'],
                            body=issue_data.get('body', ""),
                            state=issue_data['state'],
                            created_at=datetime.fromisoformat(issue_data['created_at'].replace('Z', '+00:00')),
                            url=issue_data['html_url'],
                            labels=[label['name'] for label in issue_data.get('labels', [])],
                            assignees=[assignee['login'] for assignee in issue_data.get('assignees', [])],
                            comments=[]  # Comments can be fetched separately if needed
                        )
                        issues.append(issue)
                        fetched += 1
                    if len(data) < per_page:
                        break  # No more pages
            page += 1
        return issues 