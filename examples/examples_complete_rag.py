#!/usr/bin/env python3
"""
Complete example demonstrating GitHub Issue Prompt with Local RAG capabilities
Make sure to set environment variables before running:

export OPENAI_API_KEY=your_openai_api_key_here
export GITHUB_TOKEN=your_github_token_here
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from src.models import PromptRequest, IssueResponse
from src.github_client import GitHubIssueClient
from src.llm_client import LLMClient
from src.prompt_generator import PromptGenerator
from src.local_rag import LocalRepoContextExtractor  # Use local RAG instead
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN")
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set them using 'export VAR=value' or in a .env file")
    sys.exit(1)

# Initialize clients
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()

async def run_example(issue_url: str, prompt_type: str, model: str = "gpt-4o-mini"):
    """Run a comprehensive example with local RAG context and prompt generation."""
    print(f"\nRunning example for {prompt_type} prompt type with Local RAG...")
    print(f"Issue URL: {issue_url}")
    print(f"Model: {model}")
    
    # Get GitHub issue
    issue_response = await github_client.get_issue(issue_url)
    if not isinstance(issue_response, IssueResponse) or issue_response.status != "success":
        print(f"Error fetching issue: {issue_response.error if hasattr(issue_response, 'error') else 'Unknown error'}")
        return
    
    # Parse owner and repo from URL for repo URL
    # Format: https://github.com/owner/repo/issues/number
    url_parts = issue_url.split('/')
    owner = url_parts[3]
    repo = url_parts[4]
    repo_url = f"https://github.com/{owner}/{repo}.git"
    
    print(f"Extracting context from repository: {owner}/{repo} (via local clone)...")
    
    # Initialize and load repository context using local clone
    repo_extractor = LocalRepoContextExtractor()
    await repo_extractor.load_repository(repo_url)
    
    # Get relevant context for the issue
    issue_title = issue_response.data.title
    issue_body = issue_response.data.body
    context = await repo_extractor.get_issue_context(issue_title, issue_body)
    
    print(f"Found {len(context['sources'])} relevant files in the repository")
    
    # Create prompt request with repository context
    request = PromptRequest(
        issue_url=issue_url,
        prompt_type=prompt_type,
        model=model,
        context={"repo_context": context}
    )
    
    # Generate and process prompt
    prompt_response = await prompt_generator.generate_prompt(request, issue_response.data)
    if prompt_response.status == "error":
        print(f"Error generating prompt: {prompt_response.error}")
        return
    
    print("\nGenerated Prompt with RAG context:")
    print("=" * 80)
    print(prompt_response.prompt)
    print("=" * 80)
    
    # Process with LLM
    llm_response = await llm_client.process_prompt(
        prompt=prompt_response.prompt,
        prompt_type=prompt_type,
        model=model
    )
    print("\nLLM Response:")
    print("=" * 80)
    print(llm_response)
    print("=" * 80)

async def main():
    # Example GitHub issue
    issue = {
        "url": "https://github.com/huggingface/smolagents/issues/1232",
        "prompt_type": "explain",  # Options: explain, fix, test, summarize
        "model": "gpt-4o-mini"
    }
    
    await run_example(issue["url"], issue["prompt_type"], issue["model"])

if __name__ == "__main__":
    asyncio.run(main()) 