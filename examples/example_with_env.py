#!/usr/bin/env python3
"""
Example script showing how to use the GitHub Issue Prompt with RAG
Make sure to set environment variables before running:

export OPENAI_API_KEY=your_openai_api_key_here
export GITHUB_TOKEN=your_github_token_here

Or create a .env file with these variables.
"""

import os
import sys
from dotenv import load_dotenv
from src.repo_context import RepoContextExtractor
import asyncio

# Load environment variables from .env file if present
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

async def main():
    # Initialize the repo context extractor
    extractor = RepoContextExtractor()
    
    # Example repository to analyze
    owner = "huggingface"
    repo = "transformers"
    branch = "main"
    
    print(f"Loading repository {owner}/{repo}...")
    await extractor.load_repository(owner, repo, branch)
    
    # Example issue data
    issue_title = "Model fails to generate text with long context"
    issue_body = """
    I'm trying to use the T5 model for text generation with a long context (over 8k tokens),
    but it's failing to generate the expected output. The model seems to truncate the input
    or ignore parts of the context. What could be causing this issue?
    """
    
    print(f"\nGetting context for issue: {issue_title}")
    context = await extractor.get_issue_context(issue_title, issue_body)
    
    print("\nRelevant context found:")
    print(f"Response summary: {context['response'][:200]}...")
    print(f"\nFound {len(context['sources'])} relevant sources")
    
    # Print first few sources
    for i, source in enumerate(context['sources'][:3], 1):
        print(f"\nSource {i}: {source['file']}")
        print(f"Content snippet: {source['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main()) 