"""High-level facade for issue → plan pipeline.

This sub-package bundles helpers that analyse GitHub issues, classify them,
check for related PRs and draft remediation plans.  Importing
`issue_analysis` gives you ready-to-use async helpers:

Example:
    from issue_analysis import analyse_issue
    results = await analyse_issue(issue_url)
"""

from .pr_checker import PRChecker  # noqa: F401
from .classifier import IssueClassifier  # noqa: F401
from .plan_generator import PlanGenerator  # noqa: F401
from .analyzer import analyse_issue, analyse_issue_with_existing_rag  # noqa: F401

__all__ = [
    "PRChecker",
    "IssueClassifier",
    "PlanGenerator",
    "analyse_issue",
    "analyse_issue_with_existing_rag",
]
