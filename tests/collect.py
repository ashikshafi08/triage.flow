#!/usr/bin/env python3
import asyncio
import aiohttp
import csv
import os
import sys
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# CONFIGURATION: change these before you run
# -----------------------------------------------------------------------------
OWNER = "huggingface"
REPO = "smolagents"
MAX_ISSUES = 50          # how many issues to pull
STATE = "closed"         # "open", "closed", or "all" - changed to closed for better evaluation
OUTPUT_CSV = "smolagents_eval_queries.csv"
# -----------------------------------------------------------------------------

# If you want to use a GITHUB_TOKEN (recommended to avoid rate‐limit), put it in .env
load_dotenv()


async def fetch_issues(session: aiohttp.ClientSession, page: int, per_page: int):
    """
    Fetch one page of issues from GitHub API. Filters out pull requests automatically.
    """
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
    params = {
        "state": STATE,
        "per_page": per_page,
        "page": page,
        "sort": "updated",      # you can change sort if you prefer
        "direction": "desc"
    }
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    async with session.get(url, params=params, headers=headers) as resp:
        if resp.status != 200:
            text = await resp.text()
            print(f"Error fetching page {page}: HTTP {resp.status} → {text}")
            return []
        data = await resp.json()
        # Filter out any "issues" that are actually pull requests
        return [i for i in data if "pull_request" not in i]


async def main():
    per_page = 50
    pages = (MAX_ISSUES + per_page - 1) // per_page
    all_issues = []

    async with aiohttp.ClientSession() as session:
        for page in range(1, pages + 1):
            batch = await fetch_issues(session, page, per_page)
            if not batch:
                break
            all_issues.extend(batch)
            if len(all_issues) >= MAX_ISSUES:
                break
            await asyncio.sleep(0.1)   # slight delay to be polite

    # Trim to exactly MAX_ISSUES
    all_issues = all_issues[:MAX_ISSUES]

    # Now write the CSV: use the issue title as query_text, issue_number as expected_issue_number
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["query_text", "expected_issue_number"])
        for issue in all_issues:
            title = issue.get("title", "").replace("\n", " ").strip()
            number = issue.get("number")
            # Surround the title in quotes, escape any internal quotes
            safe_title = '"' + title.replace('"', '""') + '"'
            writer.writerow([safe_title, number])

    print(f"Wrote {len(all_issues)} rows to {OUTPUT_CSV}.")


if __name__ == "__main__":
    asyncio.run(main())
