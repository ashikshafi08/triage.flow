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
        self.cache_duration = timedelta(minutes=settings.cache_duration_minutes)
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
                            {
                                "body": c["body"],
                                "user": c["user"]["login"] if "user" in c and c["user"] else "",
                                "created_at": c["created_at"]
                            }
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
        
        for attempt in range(settings.max_retries):
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
                        
                        # Fetch comments
                        comments = await self._fetch_issue_comments(owner, repo, issue_number)
                        
                        result = {
                            "number": issue_number,
                            "title": issue_data['title'],
                            "body": issue_data['body'],
                            "state": issue_data['state'],
                            "created_at": issue_data['created_at'],
                            "url": issue_url,
                            "labels": [label['name'] for label in issue_data.get('labels', [])],
                            "assignees": [assignee['login'] for assignee in issue_data.get('assignees', [])],
                            "comments": comments
                        }
                        
                        # Cache the successful result
                        self._cache_issue(issue_url, result)
                        return IssueResponse(status="success", data=Issue(**result))
                
            except Exception as e:
                if attempt == settings.max_retries - 1:
                    return IssueResponse(
                        status="error",
                        error=f"Failed to fetch issue after {settings.max_retries} attempts: {str(e)}"
                    )
                # Exponential backoff
                await asyncio.sleep(settings.backoff_factor ** attempt)
        
        return IssueResponse(status="error", error="Unknown error occurred") 