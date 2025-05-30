import asyncio
from src.github_client import GitHubIssueClient
from src.local_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
from src.llm_client import LLMClient
from src.models import IssueResponse, PromptRequest, LLMConfig
from store import jobs
import re

async def process_issue(job_id: str, issue_url: str, prompt_type: str):
    try:
        # Initialize clients
        github_client = GitHubIssueClient()
        repo_extractor = LocalRepoContextExtractor()
        prompt_generator = PromptGenerator()
        llm_client = LLMClient()
        
        # Get GitHub issue
        issue_response = await github_client.get_issue(issue_url)
        if not isinstance(issue_response, IssueResponse) or issue_response.status != "success":
            jobs[job_id] = {
                "status": "error",
                "error": f"GitHub error: {issue_response.error}"
            }
            return
        
        # Extract repo info from URL
        url_parts = issue_url.split('/')
        owner = url_parts[3]
        repo = url_parts[4]
        repo_url = f"https://github.com/{owner}/{repo}.git"
        
        # Load repository context
        await repo_extractor.load_repository(repo_url)
        context = await repo_extractor.get_issue_context(
            issue_response.data.title,
            issue_response.data.body
        )
        
        # Configure LLM (using OpenRouter by default)
        llm_config = LLMConfig(
            provider="openrouter",
            name="google/gemini-2.5-flash-preview-05-20",
            additional_params={"max_tokens": 32000}
        )
        
        # Generate prompt
        request = PromptRequest(
            issue_url=issue_url,
            prompt_type=prompt_type,
            llm_config=llm_config,
            context={"repo_context": context}
        )
        prompt_response = await prompt_generator.generate_prompt(request, issue_response.data)
        
        if prompt_response.status == "error":
            jobs[job_id] = {
                "status": "error",
                "error": f"Prompt error: {prompt_response.error}"
            }
            return
        
        # Get LLM response
        llm_response = await llm_client.process_prompt(
            prompt=prompt_response.prompt,
            prompt_type=prompt_type,
            model=llm_config.name
        )
        
        # Update job status with results
        jobs[job_id] = {
            "status": "completed",
            "prompt": prompt_response.prompt,
            "result": llm_response.prompt,
            "tokens_used": getattr(llm_response, "tokens_used", 0)
        }
        
    except Exception as e:
        jobs[job_id] = {
            "status": "error",
            "error": f"Processing error: {str(e)}"
        }
