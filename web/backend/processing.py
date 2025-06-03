import asyncio
from src.github_client import GitHubIssueClient
from src.new_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
from src.llm_client import LLMClient
from src.models import IssueResponse, PromptRequest, LLMConfig
from store import jobs  # Import job store from store.py
import re
from datetime import datetime

def update_progress(job_id: str, message: str):
    """Helper function to update the progress log for a given job."""
    if job_id in jobs:
        jobs[job_id]["progress_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

async def process_issue(job_id: str, issue_url: str, prompt_type: str):
    try:
        update_progress(job_id, "Starting analysis...")

        # Initialize clients
        github_client = GitHubIssueClient()
        repo_extractor = LocalRepoContextExtractor()
        prompt_generator = PromptGenerator()
        llm_client = LLMClient()
        
        update_progress(job_id, "Fetching GitHub issue details...")
        # Get GitHub issue
        issue_response = await github_client.get_issue(issue_url)
        if not isinstance(issue_response, IssueResponse) or issue_response.status != "success":
            jobs[job_id] = {
                "status": "error",
                "error": f"GitHub error: {issue_response.error}",
                "progress_log": jobs[job_id]["progress_log"]
            }
            return
        
        # Extract repo info from URL
        url_parts = issue_url.split('/')
        owner = url_parts[3]
        repo = url_parts[4]
        repo_url = f"https://github.com/{owner}/{repo}.git"
        
        update_progress(job_id, f"Cloning and processing repository: {owner}/{repo}...")
        # Load repository context
        await repo_extractor.load_repository(repo_url)
        
        update_progress(job_id, "Extracting relevant context from code and documentation...")
        context = await repo_extractor.get_issue_context(
            issue_response.data.title,
            issue_response.data.body
        )
        
        # Configure LLM (using OpenRouter by default)
        llm_config = LLMConfig(
            provider="openrouter",
            name="google/gemini-2.5-flash-preview-05-20", 

        )
        
        update_progress(job_id, "Generating prompt for LLM...")
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
                "error": f"Prompt error: {prompt_response.error}",
                "progress_log": jobs[job_id]["progress_log"]
            }
            return
        
        update_progress(job_id, "Calling LLM for response...")
        # Get LLM response
        llm_response = await llm_client.process_prompt(
            prompt=prompt_response.prompt,
            prompt_type=prompt_type,
            model=llm_config.name
        )
        
        update_progress(job_id, "Finalizing results...")
        # Update job status with results
        jobs[job_id] = {
            "status": "completed",
            "prompt": prompt_response.prompt,
            "result": llm_response.prompt,
            "tokens_used": getattr(llm_response, "tokens_used", 0),
            "progress_log": jobs[job_id]["progress_log"]
        }
        
    except Exception as e:
        update_progress(job_id, f"Processing failed: {str(e)}")
        jobs[job_id] = {
            "status": "error",
            "error": f"Processing error: {str(e)}",
            "progress_log": jobs[job_id]["progress_log"]
        }
