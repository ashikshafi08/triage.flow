from typing import Optional, Dict, Any, Tuple
import re
import asyncio
from datetime import datetime, timedelta
from llama_index.readers.github import GitHubIssuesClient
from .config import settings
from .models import Issue, IssueResponse

class GitHubIssueClient:
    def __init__(self):
        if not settings.github_token:
            raise ValueError("GitHub token is required. Please set GITHUB_TOKEN in your .env file.")
        self.client = GitHubIssuesClient(github_token=settings.github_token)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = timedelta(minutes=settings.cache_duration_minutes)

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
                # Get all issues and filter for the specific one
                issues = await self.client.get_issues(owner=owner, repo=repo)
                specific_issue = next((issue for issue in issues if issue["number"] == issue_number), None)
                
                if not specific_issue:
                    return IssueResponse(
                        status="error",
                        error=f"Issue #{issue_number} not found"
                    )
                    
                result = {
                    "number": issue_number,
                    "title": specific_issue['title'],
                    "body": specific_issue['body'],
                    "state": specific_issue['state'],
                    "created_at": specific_issue['created_at'],
                    "url": issue_url,
                    "labels": [label['name'] for label in specific_issue.get('labels', [])],
                    "assignees": [assignee['login'] for assignee in specific_issue.get('assignees', [])],
                    "comments": []  # Would need additional API call to get comments
                }
                
                # Cache the successful result
                self._cache_issue(issue_url, result)
                return IssueResponse(status="success", data=Issue(**result))
                
            except Exception as e:
                print(f"Error fetching issue: {str(e)}")  # Add debug print
                if attempt == settings.max_retries - 1:
                    return IssueResponse(
                        status="error",
                        error=f"Failed to fetch issue after {settings.max_retries} attempts: {str(e)}"
                    )
                # Exponential backoff
                await asyncio.sleep(settings.backoff_factor ** attempt)
        
        return IssueResponse(status="error", error="Unknown error occurred") 