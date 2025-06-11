"""Utilities to determine whether an issue already has an associated PR.

Design choices
--------------
* Keeps logic separate from the heavy `GitHubIssueClient` so it can be
  used independently (e.g. in cron-like scanners) without pulling the
  whole client.
* Uses GitHub search REST API (v3) instead of GraphQL to keep it simple.
* Accepts **repository URL** and **issue number**.  If the caller has the
  full issue URL, helper ``from_issue_url`` extracts both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import aiohttp

from ..config import settings

_PRState = Literal["open", "merged", "closed", None]


@dataclass
class PRInfo:
    state: _PRState
    pr_number: Optional[int]
    pr_url: Optional[str]


class PRChecker:
    """Checks if an issue already has an open or merged PR."""

    api_search = "https://api.github.com/search/issues?q="

    def __init__(self, token: str | None = None):
        if token is None:
            token = settings.github_token
        if not token:
            raise ValueError("GitHub token missing for PRChecker")
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "triage-flow-bot",
        }

    @staticmethod
    def _extract(url: str) -> Tuple[str, str, int]:
        pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        m = re.search(pattern, url)
        if not m:
            raise ValueError("Invalid GitHub issue URL")
        owner, repo, num = m.groups()
        return owner, repo, int(num)

    @classmethod
    async def from_issue_url(cls, issue_url: str) -> PRInfo:
        owner, repo, num = cls._extract(issue_url)
        self = cls()
        return await self.check(owner, repo, num)

    async def check(self, owner: str, repo: str, issue_number: int) -> PRInfo:
        """Return PRInfo; ``state`` is ``None`` if no PR linked."""
        query = (
            f"repo:{owner}/{repo}+type:pr+is:open+linked:issue:{issue_number}"
        )
        async with aiohttp.ClientSession() as sess:
            async with sess.get(self.api_search + query, headers=self.headers) as resp:
                if resp.status not in (200, 422):
                    raise RuntimeError(f"GitHub search API failed {resp.status}")
                if resp.status == 422:
                    # Query validation error – treat as no result, fall back later
                    data = {"total_count": 0}
                else:
                    data = await resp.json()

        if data.get("total_count", 0) == 0:
            # maybe merged/closed – run second query
            merged_q = (
                f"repo:{owner}/{repo}+type:pr+is:merged+linked:issue:{issue_number}"
            )
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    self.api_search + merged_q, headers=self.headers
                ) as resp:
                    if resp.status not in (200, 422):
                        raise RuntimeError("GitHub search API failed on merged search")
                    if resp.status == 422:
                        merged = {"total_count": 0}
                    else:
                        merged = await resp.json()
            if merged["total_count"] == 0:
                return PRInfo(state=None, pr_number=None, pr_url=None)
            pr = merged["items"][0]
            return PRInfo(state="merged", pr_number=pr["number"], pr_url=pr["html_url"])

        pr = data["items"][0]
        return PRInfo(state="open", pr_number=pr["number"], pr_url=pr["html_url"])
