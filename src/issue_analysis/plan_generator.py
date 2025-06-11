"""Generate remediation plan for a GitHub issue.

Consumes:
    * Classification result
    * Context string from AgenticRAGSystem (already summarised)
    * Explorer analysis JSON (files, reasons)

Produces Markdown plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ..llm_client import LLMClient


@dataclass
class PlanInput:
    issue_url: str
    title: str
    body: str | None
    classification: str
    explorer_json: Dict[str, Any]
    rag_context: str


class PlanGenerator:
    def __init__(self):
        self.llm = LLMClient()

    async def generate(self, data: PlanInput) -> str:
        """Return Markdown string plan."""
        prompt = (
            "Create a comprehensive remediation plan for the GitHub issue below.\n"
            "Respond in GitHub-flavoured Markdown **only**.\n\n"
            
            "## REQUIREMENTS ##\n"
            "1. Include sections: Root Cause, Files Impacted, Step-by-Step Fix, Code Changes, Tests/Verification\n"
            "2. For code changes, provide EXACT unified diff format showing before/after\n"
            "3. Use ```diff blocks to show the precise changes needed\n"
            "4. Include line numbers and file paths in diff headers\n"
            "5. Make the solution immediately actionable\n\n"
            
            f"ISSUE URL: {data.issue_url}\n"
            f"TITLE: {data.title}\n"
            f"CLASSIFICATION: {data.classification}\n\n"
            "== ISSUE BODY ==\n" + (data.body or "(no body)") + "\n\n"
            "== RAG CONTEXT ==\n" + data.rag_context + "\n\n"
            "== EXPLORER ANALYSIS JSON ==\n" + str(data.explorer_json) + "\n\n"
            
            "## EXAMPLE DIFF FORMAT ##\n"
            "When showing code changes, use this exact format:\n\n"
            "```diff\n"
            "--- a/path/to/file.py\n"
            "+++ b/path/to/file.py\n"
            "@@ -49,1 +49,1 @@\n"
            "-old_code_line\n"
            "+new_code_line\n"
            "```\n\n"
            
            "Focus on providing concrete, copy-pasteable solutions with precise file locations and changes."
        )
        resp = await self.llm.process_prompt(prompt, prompt_type="plan")
        if resp.status != "success":
            raise RuntimeError("Plan generation failed: " + (resp.error or ""))
        return resp.prompt
