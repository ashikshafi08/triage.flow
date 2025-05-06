#!/usr/bin/env python3
"""
Simple example to test the LocalRepoContextExtractor
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from src.local_rag import LocalRepoContextExtractor

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set them using 'export VAR=value' or in a .env file")
    sys.exit(1)

async def main():
    # Initialize the repo context extractor
    repo_extractor = LocalRepoContextExtractor()
    
    # Set repository URL and branch
    repo_url = "https://github.com/huggingface/trl.git"
    branch = "main"
    
    try:
        # Load repository
        print(f"Loading repository from {repo_url}...")
        await repo_extractor.load_repository(repo_url, branch)
        
        # Example issue data
        issue_title = "PPOTrainer produces nan losses"
        issue_body = """
        When I try to use PPOTrainer, after a few steps I start getting NaN losses.
        I think there might be a problem with the KL penalty calculation or gradient updates.
        Has anyone else encountered this issue?
        """
        
        print(f"\nGetting context for issue: {issue_title}")
        
        # Get relevant context
        context = await repo_extractor.get_issue_context(issue_title, issue_body)
        
        print("\nRelevant context found:")
        print(f"Response summary: {context['response'][:200]}...")
        print(f"\nFound {len(context['sources'])} relevant sources")
        
        # Print first few sources
        for i, source in enumerate(context['sources'][:3], 1):
            print(f"\nSource {i}: {source['file']}")
            print(f"Content snippet: {source['content'][:100]}...")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 