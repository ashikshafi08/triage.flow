"""
Patch→Issue Linkage Module (Optimized for Speed)
Builds mapping from closed issues to their linked pull requests for enhanced context
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from pathlib import Path
from datetime import datetime
import aiohttp
from dataclasses import dataclass
from tqdm.auto import tqdm
from collections import defaultdict
import re

from .config import settings

logger = logging.getLogger(__name__)

# Add sentinel constant at the top of the file
DIFF_TRUNCATION_SENTINEL = "... [diff truncated for embedding] ..."

@dataclass
class ProgressUpdate:
    """Represents a progress update with detailed information"""
    stage: str
    current_step: str
    progress_percentage: float
    items_processed: int
    total_items: int
    current_item: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class PatchLink:
    """Represents a link between an issue and its fixing pull request"""
    issue_id: int
    pr_number: int
    merged_at: Optional[str]
    pr_title: str
    pr_url: str
    pr_diff_url: str
    files_changed: List[str]

@dataclass
class DiffDoc:
    """Represents a downloaded diff for embedding and retrieval"""
    pr_number: int
    issue_id: int
    files_changed: List[str]
    diff_path: str
    diff_text: str  # The actual diff content
    diff_summary: str  # Cleaned summary for embedding

class PatchLinkageBuilder:
    """Builds and persists the issue→PR mapping for a repository"""
    
    def __init__(self, repo_owner: str, repo_name: str, progress_callback: Optional[Callable[[ProgressUpdate], None]] = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_key = f"{repo_owner}/{repo_name}"
        self.progress_callback = progress_callback
        
        # Progress tracking
        self.start_time = None
        self.stage_start_times = {}
        self.stage_estimates = {
            "connectivity": 2,  # seconds
            "issues_and_prs": 30,  # seconds
            "merged_prs": 20,  # seconds
            "downloading_diffs": 120,  # seconds
            "processing_diffs": 60,  # seconds
            "finalizing": 5  # seconds
        }
        
        # GitHub API configuration
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logger.error("GITHUB_TOKEN environment variable not found!")
            logger.info(f"Environment variables: {list(os.environ.keys())}")
            raise ValueError("GITHUB_TOKEN environment variable is required")
        
        # Log token presence (but not the actual token)
        logger.debug(f"GitHub token loaded: {'*' * 10}{github_token[-4:]}")
        
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # File paths
        self.index_dir = Path(f"index_{self.repo_key.replace('/', '_')}")
        self.index_dir.mkdir(exist_ok=True)
        self.patch_links_file = self.index_dir / "patch_links.jsonl"
        self.diffs_dir = self.index_dir / "diffs"
        self.diffs_dir.mkdir(exist_ok=True)
        
        # Cache for PR data to avoid duplicate fetches
        self.pr_cache = {}
        
        logger.info(f"Initialized PatchLinkageBuilder for {self.repo_key}")
        logger.info(f"Index directory: {self.index_dir}")
        logger.info(f"Diffs directory: {self.diffs_dir}")
        
    def _report_progress(self, stage: str, current_step: str, progress_percentage: float, 
                        items_processed: int, total_items: int, current_item: str = None,
                        details: Dict[str, Any] = None):
        """Report progress update with detailed information"""
        if not self.progress_callback:
            return
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        if self.start_time and progress_percentage > 0:
            elapsed = time.time() - self.start_time
            if progress_percentage < 100:
                total_estimated = elapsed / (progress_percentage / 100)
                estimated_time_remaining = int(total_estimated - elapsed)
        
        update = ProgressUpdate(
            stage=stage,
            current_step=current_step,
            progress_percentage=progress_percentage,
            items_processed=items_processed,
            total_items=total_items,
            current_item=current_item,
            estimated_time_remaining=estimated_time_remaining,
            details=details or {}
        )
        
        try:
            self.progress_callback(update)
        except Exception as e:
            logger.error(f"Error in progress callback: {e}")

    async def rate_limited_get(self, session: aiohttp.ClientSession, url: str, headers: Optional[Dict] = None, **kwargs) -> aiohttp.ClientResponse:
        """Make rate-limited GET request with proper backoff and retry logic"""
        request_headers = headers or self.headers
        max_retries = 3
        backoff = 1
        
        for attempt in range(max_retries):
            try:
                response = await session.get(url, headers=request_headers, **kwargs)
                
                if response.status == 403:
                    # Check if we hit rate limit
                    remaining = response.headers.get("X-RateLimit-Remaining", "1")
                    if remaining == "0":
                        reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                        sleep_duration = max(0, reset_time - time.time()) + 1
                        logger.warning(f"Rate limit hit, sleeping for {sleep_duration} seconds")
                        await asyncio.sleep(sleep_duration)
                        response.close()
                        continue
                
                return response
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {url}: {e}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 32)
                    continue
                else:
                    logger.error(f"Final network error for {url}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {e}")
                raise

    async def build_patch_linkage(self, max_issues: Optional[int] = None, max_prs: Optional[int] = None, download_diffs: bool = True) -> None:
        """
        Build complete patch linkage using optimized GraphQL queries with detailed progress tracking
        """
        self.start_time = time.time()
        
        # Use settings.MAX_ISSUES_TO_PROCESS if max_issues is None
        max_issues = max_issues or settings.MAX_ISSUES_TO_PROCESS
        max_prs = max_prs or settings.MAX_PR_TO_PROCESS
        
        logger.info(f"Building patch linkage for {self.repo_key} (max_issues={max_issues}, max_prs={max_prs}, download_diffs={download_diffs})")
        
        self._report_progress(
            stage="initialization",
            current_step="Setting up connections",
            progress_percentage=0,
            items_processed=0,
            total_items=1,
            current_item=f"Initializing {self.repo_key}",
            details={"max_issues": max_issues, "max_prs": max_prs, "download_diffs": download_diffs}
        )
        
        # Increase connection pool size for better parallelism
        connector = aiohttp.TCPConnector(
            limit=50,  # Increased from 10
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=True,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60, connect=10)
        ) as session:
            # Step 0: Smoke test - verify we can reach the repository
            self._report_progress(
                stage="connectivity",
                current_step="Verifying GitHub API access",
                progress_percentage=5,
                items_processed=0,
                total_items=1,
                current_item=f"Connecting to {self.repo_key}"
            )
            logger.info("Step 0: Verifying connectivity to GitHub API...")
            await self._verify_connectivity(session)
            
            # Step 1: Use GraphQL to fetch issues AND their linked PRs in ONE query
            self._report_progress(
                stage="issues_and_prs",
                current_step="Fetching issues with linked PRs",
                progress_percentage=10,
                items_processed=0,
                total_items=max_issues,
                current_item="Starting GraphQL query for issues"
            )
            logger.info("Step 1: Fetching issues with linked PRs using GraphQL...")
            all_patch_links = await self._fetch_issues_and_prs_graphql(session, max_issues)
            
            logger.info(f"Found {len(all_patch_links)} patch links")
            
            # Step 2: Save patch links
            self._report_progress(
                stage="processing",
                current_step="Saving patch links",
                progress_percentage=40,
                items_processed=len(all_patch_links),
                total_items=len(all_patch_links),
                current_item=f"Saving {len(all_patch_links)} patch links"
            )
            await self._save_patch_links(all_patch_links)
            
            # Step 3: Fetch all merged PRs using GraphQL
            self._report_progress(
                stage="merged_prs",
                current_step="Fetching merged pull requests",
                progress_percentage=50,
                items_processed=0,
                total_items=max_prs,
                current_item="Starting GraphQL query for PRs"
            )
            logger.info("Step 3: Fetching all merged PRs...")
            merged_prs = await self._fetch_merged_prs_graphql_optimized(session, max_prs)
            
            # Step 4: Download diffs if requested (with parallel downloads)
            if download_diffs and (all_patch_links or merged_prs):
                self._report_progress(
                    stage="downloading_diffs",
                    current_step="Preparing diff downloads",
                    progress_percentage=60,
                    items_processed=0,
                    total_items=len(all_patch_links) + len(merged_prs),
                    current_item="Preparing download tasks"
                )
                logger.info("Step 4: Downloading diffs in parallel...")
                
                # Prepare all diff download tasks
                diff_tasks = []
                
                # Add tasks for issue-linked PRs
                for link in all_patch_links:
                    task = self._download_single_diff(session, link)
                    diff_tasks.append(task)
                
                # Add tasks for standalone merged PRs (avoid duplicates)
                linked_pr_numbers = {link.pr_number for link in all_patch_links}
                for pr in merged_prs:
                    if pr["number"] not in linked_pr_numbers:
                        # Create a temporary PatchLink for the PR
                        link = PatchLink(
                            issue_id=None,
                            pr_number=pr["number"],
                            merged_at=pr.get("merged_at"),
                            pr_title=pr["title"],
                            pr_url=pr["url"],
                            pr_diff_url=pr["diff_url"],
                            files_changed=pr.get("files_changed", [])
                        )
                        task = self._download_single_diff(session, link)
                        diff_tasks.append(task)
                
                # Download diffs in batches to avoid overwhelming the API
                batch_size = 10  # Reduced from 20 to be more conservative
                all_diff_docs = []
                failed_downloads = []
                
                total_diffs = len(diff_tasks)
                processed_diffs = 0
                
                for i in range(0, len(diff_tasks), batch_size):
                    batch = diff_tasks[i:i + batch_size]
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    
                    rate_limited = False
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to download diff: {result}")
                            failed_downloads.append(i + j)
                        elif result is None:
                            # This was rate limited
                            rate_limited = True
                            failed_downloads.append(i + j)
                        else:
                            all_diff_docs.append(result)
                        processed_diffs += 1
                        progress_percentage = (processed_diffs / total_diffs) * 100
                        self._report_progress(
                            stage="downloading_diffs",
                            current_step="Downloading diffs",
                            progress_percentage=progress_percentage,
                            items_processed=processed_diffs,
                            total_items=total_diffs,
                            current_item=f"Downloading diff {processed_diffs}/{total_diffs}"
                        )
                    
                    # If we hit rate limit, wait longer before next batch
                    if rate_limited:
                        logger.info("Hit rate limit, waiting 60 seconds before continuing...")
                        await asyncio.sleep(60)
                    elif i + batch_size < len(diff_tasks):
                        # Normal delay between batches
                        await asyncio.sleep(1)
                
                # Retry failed downloads with larger delays
                if failed_downloads:
                    self._report_progress(
                        stage="downloading_diffs",
                        current_step="Retrying failed downloads",
                        progress_percentage=90,
                        items_processed=len(all_diff_docs),
                        total_items=total_diffs,
                        current_item=f"Retrying {len(failed_downloads)} failed downloads"
                    )
                    logger.info(f"Retrying {len(failed_downloads)} failed downloads...")
                    for idx_num, idx in enumerate(failed_downloads):
                        if idx < len(diff_tasks):
                            await asyncio.sleep(2)  # Wait 2 seconds between retries
                            self._report_progress(
                                stage="downloading_diffs",
                                current_step="Retrying failed downloads",
                                progress_percentage=90 + (idx_num / len(failed_downloads)) * 5,
                                items_processed=len(all_diff_docs),
                                total_items=total_diffs,
                                current_item=f"Retry {idx_num + 1}/{len(failed_downloads)}"
                            )
                            result = await diff_tasks[idx]
                            if result and not isinstance(result, Exception):
                                all_diff_docs.append(result)
                
                self._report_progress(
                    stage="processing_diffs",
                    current_step="Saving processed diffs",
                    progress_percentage=95,
                    items_processed=len(all_diff_docs),
                    total_items=len(all_diff_docs),
                    current_item=f"Saving {len(all_diff_docs)} processed diffs"
                )
                await self._save_diff_docs(all_diff_docs)
                logger.info(f"Downloaded and processed {len(all_diff_docs)} diffs total")
            
        self._report_progress(
            stage="finalizing",
            current_step="Completing patch linkage",
            progress_percentage=100,
            items_processed=1,
            total_items=1,
            current_item="Patch linkage build complete!"
        )
        logger.info(f"Patch linkage build complete! Files saved to {self.index_dir}")

    async def _verify_connectivity(self, session: aiohttp.ClientSession) -> None:
        """Verify we can connect to GitHub API and access the repository"""
        try:
            # Test basic GitHub API connectivity
            test_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            response = await self.rate_limited_get(session, test_url)
            
            if response.status == 404:
                raise RuntimeError(f"Repository {self.repo_key} not found or not accessible. Check repository name and token permissions.")
            elif response.status != 200:
                raise RuntimeError(f"Cannot access repository metadata: HTTP {response.status}")
            
            repo_data = await response.json()
            logger.info(f"✅ Connected to {repo_data['full_name']} (⭐ {repo_data['stargazers_count']}, 🍴 {repo_data['forks_count']})")
            
        except Exception as e:
            logger.error(f"❌ Connectivity check failed: {e}")
            raise

    async def _fetch_issues_and_prs_graphql(self, session: aiohttp.ClientSession, max_issues: int) -> List[PatchLink]:
        """
        Fetch issues and their linked PRs in a single GraphQL query for maximum efficiency
        """
        logger.info(f"Fetching up to {max_issues} issues with linked PRs using GraphQL...")
        
        # GraphQL query that fetches issues and their linked PRs in one go
        query = """
        query($owner: String!, $name: String!, $after: String, $first: Int!) {
            repository(owner: $owner, name: $name) {
                issues(
                    first: $first,
                    after: $after,
                    states: [CLOSED],
                    orderBy: {field: UPDATED_AT, direction: DESC}
                ) {
                    nodes {
                        number
                        title
                        state
                        closedAt
                        timelineItems(first: 10, itemTypes: [CLOSED_EVENT, REFERENCED_EVENT, CONNECTED_EVENT, CROSS_REFERENCED_EVENT]) {
                            nodes {
                                __typename
                                ... on ClosedEvent {
                                    closer {
                                        __typename
                                        ... on PullRequest {
                                            number
                                            title
                                            state
                                            mergedAt
                                            url
                                            files(first: 10) {
                                                nodes {
                                                    path
                                                }
                                            }
                                        }
                                        ... on Commit {
                                            oid
                                            message
                                            associatedPullRequests(first: 1) {
                                                nodes {
                                                    number
                                                    title
                                                    state
                                                    mergedAt
                                                    url
                                                    files(first: 100) {
                                                        nodes {
                                                            path
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                ... on ReferencedEvent {
                                    commit {
                                        oid
                                        message
                                        associatedPullRequests(first: 1) {
                                            nodes {
                                                number
                                                title
                                                state
                                                mergedAt
                                                url
                                                files(first: 100) {
                                                    nodes {
                                                        path
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                ... on CrossReferencedEvent {
                                    source {
                                        __typename
                                        ... on PullRequest {
                                            number
                                            title
                                            state
                                            mergedAt
                                            url
                                            files(first: 100) {
                                                nodes {
                                                    path
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
        """
        
        all_patch_links = []
        seen_pr_issue_pairs = set()  # Track (issue_id, pr_number) to avoid duplicates
        has_next_page = True
        after_cursor = None
        
        # Process in larger batches for efficiency
        batch_size = min(100, max_issues)  # GraphQL max is 100
        issues_processed = 0
        
        while has_next_page and issues_processed < max_issues:
            variables = {
                "owner": self.repo_owner,
                "name": self.repo_name,
                "first": batch_size,
                "after": after_cursor
            }
            
            # Report progress for this batch
            progress_percentage = 10 + (issues_processed / max_issues) * 30  # 10-40% for this stage
            self._report_progress(
                stage="issues_and_prs",
                current_step="Fetching issues with linked PRs",
                progress_percentage=progress_percentage,
                items_processed=issues_processed,
                total_items=max_issues,
                current_item=f"Processing batch {issues_processed//batch_size + 1}"
            )
            
            try:
                async with session.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"GraphQL request failed with status {response.status}")
                        break
                    
                    data = await response.json()
                    
                    if "errors" in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        break
                    
                    issues_data = data["data"]["repository"]["issues"]
                    issues = issues_data["nodes"]
                    page_info = issues_data["pageInfo"]
                    
                    # Process issues and extract PR links
                    for issue in issues:
                        issue_number = issue["number"]
                        issues_processed += 1
                        
                        # Report progress for individual issues periodically
                        if issues_processed % 10 == 0:
                            progress_percentage = 10 + (issues_processed / max_issues) * 30
                            self._report_progress(
                                stage="issues_and_prs",
                                current_step="Processing issues",
                                progress_percentage=progress_percentage,
                                items_processed=issues_processed,
                                total_items=max_issues,
                                current_item=f"Processing issue #{issue_number}"
                            )
                        
                        # Look through timeline items for linked PRs
                        for item in issue["timelineItems"]["nodes"]:
                            pr_data = None
                            
                            if item["__typename"] == "ClosedEvent" and item.get("closer"):
                                closer = item["closer"]
                                if closer["__typename"] == "PullRequest":
                                    pr_data = closer
                                elif closer["__typename"] == "Commit" and closer.get("associatedPullRequests"):
                                    prs = closer["associatedPullRequests"]["nodes"]
                                    if prs:
                                        pr_data = prs[0]
                            
                            elif item["__typename"] == "ReferencedEvent" and item.get("commit"):
                                commit = item["commit"]
                                if commit.get("associatedPullRequests"):
                                    prs = commit["associatedPullRequests"]["nodes"]
                                    if prs:
                                        pr_data = prs[0]
                            
                            elif item["__typename"] == "CrossReferencedEvent" and item.get("source"):
                                source = item["source"]
                                if source["__typename"] == "PullRequest":
                                    pr_data = source
                            
                            # Create patch link if we found a PR
                            if pr_data and pr_data.get("mergedAt"):
                                pair = (issue_number, pr_data["number"])
                                if pair not in seen_pr_issue_pairs:
                                    seen_pr_issue_pairs.add(pair)
                                    
                                    files_changed = [f["path"] for f in pr_data.get("files", {}).get("nodes", [])]
                                    
                                    patch_link = PatchLink(
                                        issue_id=issue_number,
                                        pr_number=pr_data["number"],
                                        merged_at=pr_data["mergedAt"],
                                        pr_title=pr_data["title"],
                                        pr_url=pr_data["url"],
                                        pr_diff_url=f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr_data['number']}.diff",
                                        files_changed=files_changed
                                    )
                                    all_patch_links.append(patch_link)
                    
                    # Update pagination
                    has_next_page = page_info["hasNextPage"]
                    after_cursor = page_info["endCursor"]
                    
                    if len(issues) < batch_size:
                        has_next_page = False
                        
            except Exception as e:
                logger.error(f"Error in GraphQL query: {e}")
                break
        
        logger.info(f"Found {len(all_patch_links)} patch links from GraphQL query")
        return all_patch_links

    async def _fetch_merged_prs_graphql_optimized(self, session: aiohttp.ClientSession, max_prs: int) -> List[Dict[str, Any]]:
        """
        Fetch merged PRs using optimized GraphQL query
        """
        logger.info(f"Fetching up to {max_prs} merged PRs using optimized GraphQL...")
        
        # Simpler query focused on just what we need
        query = """
        query($owner: String!, $name: String!, $after: String, $first: Int!) {
            repository(owner: $owner, name: $name) {
                pullRequests(
                    first: $first,
                    after: $after,
                    states: [MERGED],
                    orderBy: {field: UPDATED_AT, direction: DESC}
                ) {
                    nodes {
                        number
                        title
                        state
                        mergedAt
                        url
                        changedFiles
                        files(first: 100) {
                            nodes {
                                path
                            }
                        }
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
        batch_size = 100  # Maximum allowed by GitHub
        
        while has_next_page and len(all_prs) < max_prs:
            variables = {
                "owner": self.repo_owner,
                "name": self.repo_name,
                "first": min(batch_size, max_prs - len(all_prs)),
                "after": after_cursor
            }
            
            # Report progress for this batch
            progress_percentage = 50 + (len(all_prs) / max_prs) * 10  # 50-60% for this stage
            self._report_progress(
                stage="merged_prs",
                current_step="Fetching merged pull requests",
                progress_percentage=progress_percentage,
                items_processed=len(all_prs),
                total_items=max_prs,
                current_item=f"Processing batch {len(all_prs)//batch_size + 1}"
            )
            
            try:
                async with session.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"GraphQL request failed with status {response.status}")
                        break
                        
                    data = await response.json()
                    
                    if "errors" in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        break
                    
                    prs_data = data["data"]["repository"]["pullRequests"]
                    prs = prs_data["nodes"]
                    page_info = prs_data["pageInfo"]
                    
                    # Process PRs
                    for pr in prs:
                        if len(all_prs) >= max_prs:
                            break
                            
                        pr_processed = {
                            "number": pr["number"],
                            "title": pr["title"],
                            "state": pr["state"],
                            "merged_at": pr["mergedAt"],
                            "url": pr["url"],
                            "diff_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr['number']}.diff",
                            "files_changed": [f["path"] for f in pr.get("files", {}).get("nodes", [])]
                        }
                        all_prs.append(pr_processed)
                        
                        # Report progress every 10 PRs
                        if len(all_prs) % 10 == 0:
                            progress_percentage = 50 + (len(all_prs) / max_prs) * 10
                            self._report_progress(
                                stage="merged_prs",
                                current_step="Processing merged PRs",
                                progress_percentage=progress_percentage,
                                items_processed=len(all_prs),
                                total_items=max_prs,
                                current_item=f"Processed PR #{pr['number']}"
                            )
                    
                    # Update pagination
                    has_next_page = page_info["hasNextPage"]
                    after_cursor = page_info["endCursor"]
                    
                    if len(prs) < batch_size:
                        has_next_page = False
                
            except Exception as e:
                logger.error(f"Error in GraphQL query: {e}")
                break
        
        logger.info(f"Fetched {len(all_prs)} merged PRs")
        return all_prs
    
    async def _save_patch_links(self, patch_links: List[PatchLink]) -> None:
        """Save patch links to JSONL file"""
        logger.info(f"Saving {len(patch_links)} patch links to {self.patch_links_file}")
        
        with open(self.patch_links_file, 'w', encoding='utf-8') as f:
            for link in patch_links:
                link_dict = {
                    "issue_id": link.issue_id,
                    "pr_number": link.pr_number,
                    "merged_at": link.merged_at,
                    "pr_title": link.pr_title,
                    "pr_url": link.pr_url,
                    "pr_diff_url": link.pr_diff_url,
                    "files_changed": link.files_changed,
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(link_dict, ensure_ascii=False) + '\n')
    
    def load_patch_links(self) -> Dict[int, List[PatchLink]]:
        """Load existing patch links from file"""
        if not self.patch_links_file.exists():
            return {}
        
        links_by_issue = {}
        
        try:
            with open(self.patch_links_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        link_data = json.loads(line.strip())
                        
                        patch_link = PatchLink(
                            issue_id=link_data["issue_id"],
                            pr_number=link_data["pr_number"],
                            merged_at=link_data.get("merged_at"),
                            pr_title=link_data["pr_title"],
                            pr_url=link_data["pr_url"],
                            pr_diff_url=link_data["pr_diff_url"],
                            files_changed=link_data.get("files_changed", [])
                        )
                        
                        if patch_link.issue_id not in links_by_issue:
                            links_by_issue[patch_link.issue_id] = []
                        links_by_issue[patch_link.issue_id].append(patch_link)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num} in patch links file: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading patch links: {e}")
            return {}
        
        logger.info(f"Loaded patch links for {len(links_by_issue)} issues")
        return links_by_issue
    
    def get_patch_url_for_issue(self, issue_id: int) -> Optional[str]:
        """Get the patch URL for a specific issue"""
        links_by_issue = self.load_patch_links()
        
        if issue_id in links_by_issue:
            # Return the first (and usually only) patch URL
            return links_by_issue[issue_id][0].pr_diff_url
        
        return None

    async def _download_single_diff(self, session: aiohttp.ClientSession, link: PatchLink) -> Optional[DiffDoc]:
        """Download and process a single diff file"""
        try:
            # Download the diff
            response = await self.rate_limited_get(session, link.pr_diff_url)
            
            if response.status == 429:
                # Rate limited - return None so it can be retried later
                logger.warning(f"Rate limited on PR #{link.pr_number}, will retry later")
                return None
            
            if response.status != 200:
                logger.warning(f"Failed to download diff for PR #{link.pr_number}: HTTP {response.status}")
                return None
            
            diff_text = await response.text()
            
            # Process the diff
            diff_summary = self._extract_diff_hunks(diff_text)
            
            # Save to file
            diff_filename = f"pr_{link.pr_number}.diff"
            diff_path = self.diffs_dir / diff_filename
            
            with open(diff_path, 'w', encoding='utf-8') as f:
                f.write(diff_text)
            
            # Create DiffDoc
            diff_doc = DiffDoc(
                pr_number=link.pr_number,
                issue_id=link.issue_id,
                files_changed=link.files_changed,
                diff_path=str(diff_path),
                diff_text=diff_text,
                diff_summary=diff_summary
            )
            
            return diff_doc
            
        except Exception as e:
            logger.warning(f"Error downloading diff for PR #{link.pr_number}: {e}")
            return None

    def _extract_diff_hunks(self, diff_text: str, max_chars: int = 8000) -> str:
        """Extract and format diff hunks for embedding, with size limit"""
        if not diff_text.strip():
            return "No diff content available"
        
        lines = diff_text.split('\n')
        files_changed = set()
        hunk_lines = []
        current_file = None
        current_file_path = None
        current_length = 0
        
        # Extract file names and changes
        for line in lines:
            if line.startswith('diff --git'):
                # Extract filename
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3]  # Keep full path
                    files_changed.add(file_path.split('/')[-1])  # Just filename for the list
            elif line.startswith('--- a/') or line.startswith('+++ b/'):
                # Track current file being processed
                if line.startswith('+++ b/'):
                    file_path = line[6:]  # Remove '+++ b/'
                    current_file_path = file_path  # Keep full path
                    current_file = file_path.split('/')[-1]  # Just filename for display
            elif line.startswith('@@'):
                # New hunk header
                if current_file_path:
                    hunk_lines.append(f"\n--- {current_file_path} ---")  # Use full path
                hunk_lines.append(line)
                current_length = len('\n'.join(hunk_lines))
                if current_length > max_chars:
                    hunk_lines.append(DIFF_TRUNCATION_SENTINEL)
                    break
            elif line.startswith(('+', '-', ' ')) and len(line.strip()) > 0:
                # Actual diff line
                hunk_lines.append(line)
                current_length = len('\n'.join(hunk_lines))
                if current_length > max_chars:
                    hunk_lines.append(DIFF_TRUNCATION_SENTINEL)
                    break
        
        # Build summary
        files_list = list(files_changed) if files_changed else ["unknown"]
        metadata_lines = [
            f"Patch summary for PR (max {max_chars} chars):",
            f"Files changed: {', '.join(files_list)}",
            "--- Changes ---"
        ]
        
        summary = '\n'.join(hunk_lines) if hunk_lines else "No changes detected"
        final_summary = '\n'.join(metadata_lines) + '\n' + summary
        
        # Ensure we don't exceed limit while preserving sentinel
        if len(final_summary) > max_chars:
            # Reserve space for the sentinel if we need to truncate
            cut_point = max_chars - len(DIFF_TRUNCATION_SENTINEL)
            final_summary = final_summary[:cut_point] + DIFF_TRUNCATION_SENTINEL
        
        return final_summary

    async def _save_diff_docs(self, diff_docs: List[DiffDoc]) -> None:
        """Save diff docs metadata to JSON for indexing"""
        diff_docs_file = self.index_dir / "diff_docs.jsonl"
        
        logger.info(f"Saving {len(diff_docs)} diff docs to {diff_docs_file}")
        
        with open(diff_docs_file, 'w', encoding='utf-8') as f:
            for doc in diff_docs:
                doc_dict = {
                    "pr_number": doc.pr_number,
                    "issue_id": doc.issue_id,
                    "files_changed": doc.files_changed,
                    "diff_path": doc.diff_path,
                    "diff_summary": doc.diff_summary,
                    "created_at": datetime.now().isoformat()
                }
                f.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')

    def load_diff_docs(self) -> List[DiffDoc]:
        """Load diff docs from saved metadata"""
        diff_docs_file = self.index_dir / "diff_docs.jsonl"
        
        if not diff_docs_file.exists():
            return []
        
        diff_docs = []
        
        try:
            with open(diff_docs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        doc_data = json.loads(line.strip())
                        
                        diff_doc = DiffDoc(
                            pr_number=doc_data["pr_number"],
                            issue_id=doc_data["issue_id"],
                            files_changed=doc_data.get("files_changed", []),
                            diff_path=doc_data["diff_path"],
                            diff_text="",  # We'll load this on demand
                            diff_summary=doc_data["diff_summary"]
                        )
                        
                        diff_docs.append(diff_doc)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse diff doc line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading diff docs: {e}")
            return []
        
        logger.info(f"Loaded {len(diff_docs)} diff docs")
        return diff_docs

# Utility function to build patch linkage for a repository
async def build_repository_patch_linkage(repo_owner: str, repo_name: str, max_issues: Optional[int] = None, max_prs: Optional[int] = None) -> None:
    """Build patch linkage for a repository - convenience function"""
    # Use settings.MAX_ISSUES_TO_PROCESS if max_issues is None
    max_issues = max_issues or settings.MAX_ISSUES_TO_PROCESS
    max_prs = max_prs or settings.MAX_PR_TO_PROCESS
    builder = PatchLinkageBuilder(repo_owner, repo_name)
    await builder.build_patch_linkage(max_issues, max_prs)

# Command-line interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python -m src.patch_linkage <owner> <repo>")
        sys.exit(1)
    
    owner, repo = sys.argv[1], sys.argv[2]
    
    async def main():
        await build_repository_patch_linkage(owner, repo, max_issues=500, max_prs=500)  # Smaller limit for testing
    
    asyncio.run(main)