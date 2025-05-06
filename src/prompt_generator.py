from typing import Dict, Optional
from .models import Issue, PromptRequest, PromptResponse
from .config import settings
from .repo_context import RepoContextExtractor

class PromptGenerator:
    def __init__(self):
        self.templates = {
            "explain": self._generate_explain_prompt,
            "fix": self._generate_fix_prompt,
            "test": self._generate_test_prompt,
            "summarize": self._generate_summarize_prompt
        }
        self.repo_context = RepoContextExtractor()

    def _format_context(self, context: Dict) -> str:
        if not context:
            return "No additional context provided."
        return "\n".join(f"{key}: {value}" for key, value in context.items())

    def _format_repo_context(self, repo_context: Dict) -> str:
        if not repo_context:
            return "No repository context available."
        
        sources = "\n".join([
            f"File: {source['file']}\nContent: {source['content']}\n"
            for source in repo_context.get('sources', [])
        ])
        
        return f"""
Repository Context:
{repo_context.get('response', 'No response')}

Relevant Code and Documentation:
{sources}
"""

    async def _get_repo_context(self, issue: Issue) -> Dict:
        """Get relevant context from the repository for the issue."""
        try:
            # Extract owner and repo from issue URL
            url_parts = issue.url.split('/')
            owner = url_parts[3]
            repo = url_parts[4]
            
            # Load repository data
            await self.repo_context.load_repository(owner, repo)
            
            # Get issue-specific context
            context = await self.repo_context.get_issue_context(issue.title, issue.body)
            if not context:
                print("Warning: No repository context found for the issue")
            return context
        except Exception as e:
            print(f"Warning: Failed to get repository context: {str(e)}")
            return {}

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

            # Get repository context
            repo_context = await self._get_repo_context(issue)
            
            # Generate prompt with repository context
            prompt = self.templates[request.prompt_type](issue, request.context, repo_context)
            return PromptResponse(status="success", prompt=prompt)
        except Exception as e:
            return PromptResponse(status="error", error=str(e)) 