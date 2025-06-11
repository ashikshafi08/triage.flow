"""IssueClassifier – lightweight LLM-based labeler.

Returns ``label`` and ``confidence``.  Few-shot examples are embedded; we
keep them minimal for token efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import aiohttp

from ..llm_client import LLMClient

_Label = Literal[
    "bug-code",
    "bug-test",
    "documentation",
    "build/CI",
    "dependency",
    "refactor",
    "question",
]


@dataclass
class Classification:
    label: _Label
    confidence: float  # 0-1 scale


class IssueClassifier:
    system_prompt = (
        "You are an AI that classifies GitHub issues.  Return JSON with keys "
        "label and confidence. Labels: bug-code, bug-test, documentation, "
        "build/CI, dependency, refactor, question. Give best guess confidence "
        "(0-1). Respond ONLY with JSON."
    )

    examples: Tuple[Tuple[str, str], ...] = (
        (
            "Fix typo in README",
            '{"label":"documentation","confidence":0.95}',
        ),
        (
            "ci: bump actions/setup-node to v4",
            '{"label":"dependency","confidence":0.9}',
        ),
        (
            "NullPointerException when saving user",
            '{"label":"bug-code","confidence":0.83}',
        ),
    )

    def __init__(self):
        self.llm = LLMClient()

    async def classify(self, title: str, body: str | None = None) -> Classification:
        prompt_parts = [
            "Examples:",
        ]
        for t, j in self.examples:
            prompt_parts.append(f"Issue: {t}\nAnswer: {j}")
        prompt_parts.append("Now classify:\nIssue:" + title + ("\nBody:" + body if body else ""))
        prompt = "\n\n".join(prompt_parts)

        resp = await self.llm.process_prompt(prompt, prompt_type="classification")
        if resp.status != "success":
            raise RuntimeError("LLM classification failed: " + (resp.error or ""))
        import json, re

        # sometimes model still responds with extra text; extract JSON block
        m = re.search(r"\{.*\}", resp.prompt, flags=re.S)
        if not m:
            raise ValueError("Could not parse classification JSON")
        data = json.loads(m.group(0))
        return Classification(label=data["label"], confidence=float(data["confidence"]))
