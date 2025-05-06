from typing import Dict, Optional
from .models import Issue, PromptRequest, PromptResponse
from .config import settings

class PromptGenerator:
    def __init__(self):
        self.templates = {
            "explain": self._generate_explain_prompt,
            "fix": self._generate_fix_prompt,
            "test": self._generate_test_prompt,
            "summarize": self._generate_summarize_prompt
        }

    def _format_context(self, context: Dict) -> str:
        if not context:
            return "No additional context provided."
        
        # Filter out repo_context which is handled separately
        filtered_context = {k: v for k, v in context.items() if k != "repo_context"}
        if not filtered_context:
            return "No additional context provided."
            
        return "\n".join(f"{key}: {value}" for key, value in filtered_context.items())

    def _format_repo_context(self, repo_context: Dict) -> str:
        if not repo_context or not repo_context.get('sources'):
            return "No repository context available."
        
        # Collect unique file names
        files = {source['file'] for source in repo_context['sources']}
        file_list = "\n".join(f"- {file}" for file in files)
        
        sources = "\n".join([
            f"File: {source['file']}\nContent: {source['content'][:500]}...\n"
            for source in repo_context.get('sources', [])
        ])
        
        return f"""
Relevant Files:
{file_list}

Repository Context:
{repo_context.get('response', 'No response')}

Relevant Code and Documentation:
{sources}
"""

    def _generate_explain_prompt(self, issue: Issue, context: Dict, repo_context: Dict) -> str:
        return f"""Please explain the following GitHub issue:

Title: {issue.title}
Description: {issue.body}

Repository Context:
{self._format_repo_context(repo_context)}

Additional Context:
{self._format_context(context)}

Please provide:
1. A clear explanation of what the issue is about
2. The root cause of the problem
3. Any relevant technical details from the codebase
4. Potential impact if not addressed"""

    def _generate_fix_prompt(self, issue: Issue, context: Dict, repo_context: Dict) -> str:
        return f"""Please help fix the following GitHub issue:

Title: {issue.title}
Description: {issue.body}

Repository Context:
{self._format_repo_context(repo_context)}

Additional Context:
{self._format_context(context)}

Please provide:
1. A detailed solution to the problem
2. Code changes needed, referencing relevant files
3. Any necessary tests
4. Potential edge cases to consider"""

    def _generate_test_prompt(self, issue: Issue, context: Dict, repo_context: Dict) -> str:
        return f"""Please create test cases for the following GitHub issue:

Title: {issue.title}
Description: {issue.body}

Repository Context:
{self._format_repo_context(repo_context)}

Additional Context:
{self._format_context(context)}

Please provide:
1. Test scenarios that verify the issue
2. Test cases that validate the fix
3. Edge cases to consider
4. Test implementation in the appropriate language"""

    def _generate_summarize_prompt(self, issue: Issue, context: Dict, repo_context: Dict) -> str:
        return f"""Please summarize the following GitHub issue:

Title: {issue.title}
Description: {issue.body}

Repository Context:
{self._format_repo_context(repo_context)}

Additional Context:
{self._format_context(context)}

Please provide:
1. A concise summary of the issue
2. Key points from the discussion
3. Current status and next steps
4. Any blockers or dependencies"""

    async def generate_prompt(self, request: PromptRequest, issue: Issue) -> PromptResponse:
        try:
            if request.prompt_type not in self.templates:
                return PromptResponse(
                    status="error",
                    error=f"Invalid prompt type: {request.prompt_type}"
                )

            # Get repository context from the request if provided
            repo_context = request.context.get("repo_context", {})
            
            # Generate prompt with repository context
            prompt = self.templates[request.prompt_type](issue, request.context, repo_context)
            return PromptResponse(status="success", prompt=prompt)
        except Exception as e:
            return PromptResponse(status="error", error=str(e)) 