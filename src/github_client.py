from typing import Optional, Dict, Any, Tuple, List
import re
import asyncio
from datetime import datetime, timedelta
import aiohttp
from .config import settings
from .models import Issue, IssueResponse, IssueComment, PullRequestInfo, PullRequestUser, EnhancedPullRequestInfo, PullRequestReview, PullRequestReviewer, PullRequestStatusCheck

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
                            closed_at=datetime.fromisoformat(issue_data['closed_at'].replace('Z', '+00:00')) if issue_data.get('closed_at') else None,
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
        # Handle async method properly
        try:
            # Check if we're already in an event loop
            current_loop = None
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            
            if current_loop:
                # We're already in an async context, use concurrent.futures
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.get_issue(issue_url)
                    )
                    response = future.result(timeout=30)  # 30 second timeout
            else:
                # No event loop running, safe to use asyncio.run
                response = asyncio.run(self.get_issue(issue_url))
            
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
        except Exception as e:
            raise Exception(f"Failed to fetch issue: {str(e)}")

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
                            body=issue_data.get('body') or "",  # Ensure body is an empty string if None
                            state=issue_data['state'],
                            created_at=datetime.fromisoformat(issue_data['created_at'].replace('Z', '+00:00')),
                            closed_at=datetime.fromisoformat(issue_data['closed_at'].replace('Z', '+00:00')) if issue_data.get('closed_at') else None,
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

    async def create_issue(self, owner: str, repo: str, title: str, body: str, labels: list = None) -> Dict[str, Any]:
        """
        Create a new GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: List of label names
            
        Returns:
            Dict containing the created issue data
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/repos/{owner}/{repo}/issues"
                
                issue_data = {
                    "title": title,
                    "body": body
                }
                
                if labels:
                    issue_data["labels"] = labels
                
                async with session.post(url, headers=self.headers, json=issue_data) as response:
                    if response.status == 201:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to create issue: HTTP {response.status} - {error_text}")
                        
        except Exception as e:
            raise Exception(f"Error creating issue: {str(e)}")

    async def list_pull_requests(self, repo_url: str, state: str = "merged", per_page: int = 30, max_pages: int = 5) -> list:
        """
        List pull requests for a given repository URL and state.
        Args:
            repo_url: The GitHub repository URL (e.g., https://github.com/owner/repo)
            state: 'open', 'closed', 'merged', or 'all'. Note: GitHub API uses 'closed' for merged PRs if not specifying 'merged'.
                   If 'merged' is passed, we will fetch 'closed' PRs and then filter by `merged_at` if necessary,
                   or use a specific search query if more robust. For now, we'll assume 'closed' covers 'merged'.
            per_page: Number of PRs per page (max 100)
            max_pages: Maximum number of pages to fetch
        Returns:
            List of PullRequestInfo (as dicts, to be Pydantic models later)
        """
        from .local_repo_loader import get_repo_info
        # It's better to import Pydantic models at the top of the file,
        # but if there's a circular dependency risk with models.py importing github_client,
        # a local import like this is a common workaround.
        # However, given the current structure, models.py does not import github_client.
        # So, PullRequestInfo and PullRequestUser should ideally be imported at the top.
        # For this insertion, we'll keep it local to minimize changes if that was the original intent.
        from .models import PullRequestInfo, PullRequestUser

        owner, repo = get_repo_info(repo_url)
        pull_requests_data = []
        page = 1
        
        api_state = "closed" if state == "merged" else state

        while page <= max_pages:
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state={api_state}&per_page={per_page}&page={page}&sort=updated&direction=desc"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Consider logging this error instead of just printing
                        print(f"Error fetching PRs: {response.status} - {error_text}")
                        break 
                    data = await response.json()
                    if not data:
                        break 
                    
                    for pr_data_item in data: # Renamed pr_data to pr_data_item to avoid conflict
                        if state == "merged" and not pr_data_item.get("merged_at"):
                            continue

                        files_changed = [] # Placeholder for now

                        user_data_item = pr_data_item.get("user") # Renamed user_data
                        pr_user = PullRequestUser(login=user_data_item["login"]) if user_data_item else None

                        pull_request = PullRequestInfo(
                            number=pr_data_item['number'],
                            title=pr_data_item['title'],
                            merged_at=pr_data_item.get('merged_at'),
                            files_changed=files_changed,
                            issue_id=None, 
                            url=pr_data_item.get('html_url'),
                            user=pr_user,
                            body=pr_data_item.get('body')
                        )
                        pull_requests_data.append(pull_request)

                    if len(data) < per_page:
                        break 
            page += 1
        return pull_requests_data

    async def list_open_pull_requests_with_reviews(
        self, 
        repo_url: str, 
        per_page: int = 50, 
        max_pages: int = 10
    ) -> List[EnhancedPullRequestInfo]:
        """
        Fetch open pull requests with review information using GraphQL.
        This provides comprehensive PR data including reviews, status checks, and review requests.
        """
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        
        # GraphQL query to get open PRs with review information
        query = """
        query OpenPullRequests($owner: String!, $repo: String!, $first: Int!, $after: String) {
            repository(owner: $owner, name: $repo) {
                pullRequests(
                    states: OPEN, 
                    first: $first, 
                    after: $after,
                    orderBy: {field: UPDATED_AT, direction: DESC}
                ) {
                    nodes {
                        number
                        title
                        url
                        createdAt
                        updatedAt
                        body
                        isDraft
                        author { 
                            login 
                        }
                        
                        # Review information
                        reviewDecision
                        
                        # Who still needs to review?
                        reviewRequests(first: 10) {
                            nodes {
                                requestedReviewer {
                                    __typename
                                    ... on User { 
                                        login 
                                    }
                                    ... on Team { 
                                        name 
                                    }
                                }
                            }
                        }
                        
                        # Actual reviews
                        reviews(last: 20) {
                            nodes {
                                author { 
                                    login 
                                }
                                state
                                submittedAt
                                body
                            }
                        }
                        
                        # Mergeability and status
                        mergeable
                        
                        # CI status from latest commit
                        commits(last: 1) {
                            nodes {
                                commit {
                                    statusCheckRollup {
                                        state
                                        contexts(first: 10) {
                                            nodes {
                                                __typename
                                                ... on StatusContext {
                                                    context
                                                    description
                                                    state
                                                }
                                                ... on CheckRun {
                                                    name
                                                    conclusion
                                                    status
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        # Files changed
                        files(first: 100) {
                            nodes {
                                path
                            }
                        }
                        
                        # Additional stats
                        additions
                        deletions
                        changedFiles
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
        """
        
        all_prs = []
        has_next_page = True
        after_cursor = None
        page = 0
        
        async with aiohttp.ClientSession() as session:
            while has_next_page and page < max_pages:
                variables = {
                    "owner": owner,
                    "repo": repo,
                    "first": per_page,
                    "after": after_cursor
                }
                
                try:
                    async with session.post(
                        "https://api.github.com/graphql",
                        json={"query": query, "variables": variables},
                        headers=self.headers
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"GraphQL request failed: HTTP {response.status} - {error_text}")
                        
                        data = await response.json()
                        
                        if "errors" in data:
                            raise Exception(f"GraphQL errors: {data['errors']}")
                        
                        prs_data = data["data"]["repository"]["pullRequests"]
                        prs = prs_data["nodes"]
                        page_info = prs_data["pageInfo"]
                        
                        # Process each PR
                        for pr_data in prs:
                            enhanced_pr = self._parse_enhanced_pr_data(pr_data)
                            all_prs.append(enhanced_pr)
                        
                        # Update pagination
                        has_next_page = page_info["hasNextPage"]
                        after_cursor = page_info["endCursor"]
                        page += 1
                        
                except Exception as e:
                    print(f"Error fetching open PRs: {e}")
                    break
        
        return all_prs
    
    def _parse_enhanced_pr_data(self, pr_data: Dict[str, Any]) -> EnhancedPullRequestInfo:
        """Parse GraphQL PR data into EnhancedPullRequestInfo"""
        
        # Parse reviews
        reviews = []
        for review_data in pr_data.get("reviews", {}).get("nodes", []):
            if review_data.get("author"):
                review = PullRequestReview(
                    author=review_data["author"]["login"],
                    state=review_data["state"],
                    submitted_at=review_data["submittedAt"],
                    body=review_data.get("body")
                )
                reviews.append(review)
        
        # Parse review requests
        review_requests = []
        for request_data in pr_data.get("reviewRequests", {}).get("nodes", []):
            reviewer_data = request_data.get("requestedReviewer", {})
            if reviewer_data:
                reviewer = PullRequestReviewer(
                    login=reviewer_data.get("login"),
                    name=reviewer_data.get("name"),
                    type=reviewer_data.get("__typename", "User")
                )
                review_requests.append(reviewer)
        
        # Parse status checks
        status_checks = []
        commits = pr_data.get("commits", {}).get("nodes", [])
        if commits:
            latest_commit = commits[0]
            rollup = latest_commit.get("commit", {}).get("statusCheckRollup")
            if rollup:
                # Overall status
                overall_status = PullRequestStatusCheck(
                    state=rollup.get("state", "UNKNOWN"),
                    context="overall",
                    description="Overall status check rollup"
                )
                status_checks.append(overall_status)
                
                # Individual contexts
                for context_data in rollup.get("contexts", {}).get("nodes", []):
                    if context_data.get("__typename") == "StatusContext":
                        status_check = PullRequestStatusCheck(
                            state=context_data.get("state", "UNKNOWN"),
                            context=context_data.get("context"),
                            description=context_data.get("description")
                        )
                        status_checks.append(status_check)
                    elif context_data.get("__typename") == "CheckRun":
                        # Map CheckRun conclusion to status state
                        conclusion = context_data.get("conclusion", "NEUTRAL")
                        state_mapping = {
                            "SUCCESS": "SUCCESS",
                            "FAILURE": "FAILURE", 
                            "NEUTRAL": "PENDING",
                            "CANCELLED": "FAILURE",
                            "TIMED_OUT": "FAILURE",
                            "ACTION_REQUIRED": "FAILURE"
                        }
                        
                        status_check = PullRequestStatusCheck(
                            state=state_mapping.get(conclusion, "UNKNOWN"),
                            context=context_data.get("name"),
                            description=f"Check run: {context_data.get('status', 'unknown')}"
                        )
                        status_checks.append(status_check)
        
        # Parse files changed
        files_changed = []
        for file_data in pr_data.get("files", {}).get("nodes", []):
            files_changed.append(file_data["path"])
        
        # Parse user
        user = None
        if pr_data.get("author"):
            user = PullRequestUser(login=pr_data["author"]["login"])
        
        return EnhancedPullRequestInfo(
            number=pr_data["number"],
            title=pr_data["title"],
            state="open",
            created_at=pr_data.get("createdAt"),
            updated_at=pr_data.get("updatedAt"),
            url=pr_data.get("url"),
            body=pr_data.get("body"),
            user=user,
            files_changed=files_changed,
            review_decision=pr_data.get("reviewDecision"),
            reviews=reviews,
            review_requests=review_requests,
            mergeable=pr_data.get("mergeable"),
            status_checks=status_checks,
            draft=pr_data.get("isDraft", False),
            commits_count=pr_data.get("changedFiles"),
            additions=pr_data.get("additions"),
            deletions=pr_data.get("deletions")
        )

    async def get_pr_reviews(
        self,
        repo_url: str,
        pr_number: int
    ) -> List[PullRequestReview]:
        """
        Get detailed review information for a specific PR.
        """
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        
        query = """
        query PRReviews($owner: String!, $repo: String!, $number: Int!) {
            repository(owner: $owner, name: $repo) {
                pullRequest(number: $number) {
                    reviews(first: 100) {
                        nodes {
                            author {
                                login
                            }
                            state
                            submittedAt
                            body
                        }
                    }
                }
            }
        }
        """
        
        variables = {
            "owner": owner,
            "repo": repo,
            "number": pr_number
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    if "errors" in data:
                        return []
                    
                    reviews_data = data["data"]["repository"]["pullRequest"]["reviews"]["nodes"]
                    
                    reviews = []
                    for review_data in reviews_data:
                        if review_data.get("author"):
                            review = PullRequestReview(
                                author=review_data["author"]["login"],
                                state=review_data["state"],
                                submitted_at=review_data["submittedAt"],
                                body=review_data.get("body")
                            )
                            reviews.append(review)
                    
                    return reviews
                    
        except Exception as e:
            print(f"Error fetching PR reviews: {e}")
            return []

    async def get_pr_detailed_info(
        self,
        repo_url: str,
        pr_number: int
    ) -> Optional[EnhancedPullRequestInfo]:
        """
        Get detailed information for a specific PR including reviews and status.
        """
        from .local_repo_loader import get_repo_info
        owner, repo = get_repo_info(repo_url)
        
        query = """
        query PRDetails($owner: String!, $repo: String!, $number: Int!) {
            repository(owner: $owner, name: $repo) {
                pullRequest(number: $number) {
                    number
                    title
                    url
                    state
                    createdAt
                    updatedAt
                    mergedAt
                    body
                    isDraft
                    author { 
                        login 
                    }
                    
                    # Review information
                    reviewDecision
                    
                    reviewRequests(first: 10) {
                        nodes {
                            requestedReviewer {
                                __typename
                                ... on User { login }
                                ... on Team { name }
                            }
                        }
                    }
                    
                    reviews(last: 50) {
                        nodes {
                            author { login }
                            state
                            submittedAt
                            body
                        }
                    }
                    
                    # Status and mergeability
                    mergeable
                    
                    commits(last: 1) {
                        nodes {
                            commit {
                                statusCheckRollup {
                                    state
                                    contexts(first: 20) {
                                        nodes {
                                            __typename
                                            ... on StatusContext {
                                                context
                                                description
                                                state
                                            }
                                            ... on CheckRun {
                                                name
                                                conclusion
                                                status
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    # Files and stats
                    files(first: 200) {
                        nodes {
                            path
                        }
                    }
                    
                    additions
                    deletions
                    changedFiles
                }
            }
        }
        """
        
        variables = {
            "owner": owner,
            "repo": repo,
            "number": pr_number
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if "errors" in data or not data.get("data", {}).get("repository", {}).get("pullRequest"):
                        return None
                    
                    pr_data = data["data"]["repository"]["pullRequest"]
                    return self._parse_enhanced_pr_data(pr_data)
                    
        except Exception as e:
            print(f"Error fetching PR details: {e}")
            return None
