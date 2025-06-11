#!/usr/bin/env python
"""Run the issue→plan pipeline from the CLI.

Usage::
    python scripts/run_issue_analysis.py <github-issue-url>

Example::
    python scripts/run_issue_analysis.py https://github.com/pallets/flask/issues/5530

The script prints a compact JSON summary so you can quickly verify that the
pipeline works end-to-end without spinning up FastAPI.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH when running via `python scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.issue_analysis import analyse_issue  # noqa: E402  pylint: disable=wrong-import-position


async def _main(issue_url: str):  # noqa: D401
    """Async entry so we can await the pipeline."""
    result = await analyse_issue(issue_url)

    # Pretty-print but truncate long markdown for readability
    plan_preview = textwrap.shorten(result.get("plan_markdown", ""), width=280, placeholder=" …")
    result_copy = {**result, "plan_markdown": plan_preview}
    print(json.dumps(result_copy, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run issue analysis pipeline")
    parser.add_argument("issue_url", help="Full GitHub issue URL")
    args = parser.parse_args()
    asyncio.run(_main(args.issue_url))


if __name__ == "__main__":
    main()
