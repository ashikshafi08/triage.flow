"""Background task that runs the issue → plan pipeline."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any
import logging

from src.issue_analysis.analyzer import analyse_issue
from .store import jobs

logger = logging.getLogger(__name__)


def _update_progress(job_id: str, msg: str) -> None:
    if job_id in jobs:
        jobs[job_id]["progress_log"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": msg,
        })


async def process_issue_analysis(job_id: str, issue_url: str):
    try:
        _update_progress(job_id, "Running analysis pipeline…")
        result: Dict[str, Any] = await analyse_issue(issue_url)

        if result.get("status") == "error":
            jobs[job_id] = {
                "status": "error",
                "error": result.get("error", "unknown"),
                "progress_log": jobs[job_id]["progress_log"],
            }
            return

        jobs[job_id].update({
            "status": result["status"],
            "result": result,
            "progress_log": jobs[job_id]["progress_log"],
        })
    except Exception as exc:
        logger.exception("Issue analysis failed")
        jobs[job_id] = {
            "status": "error",
            "error": str(exc),
            "progress_log": jobs.get(job_id, {}).get("progress_log", []),
        }
