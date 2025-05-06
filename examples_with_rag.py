import asyncio
import os
from dotenv import load_dotenv
from src.models import PromptRequest, IssueResponse
from src.github_client import GitHubIssueClient
from src.llm_client import LLMClient
from src.prompt_generator import PromptGenerator
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize clients
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()

async def run_example(issue_url: str, prompt_type: str, model: str = "gpt-4-turbo-preview"):
    """Run a single example with the given parameters."""
    print(f"\nRunning example for {prompt_type} prompt type...")
    print(f"Issue URL: {issue_url}")
    print(f"Model: {model}")
    
    # Get GitHub issue
    issue_response = await github_client.get_issue(issue_url)
    if not isinstance(issue_response, IssueResponse) or issue_response.status != "success":
        print(f"Error fetching issue: {issue_response.error if hasattr(issue_response, 'error') else 'Unknown error'}")
        return
    
    # Create prompt request
    request = PromptRequest(
        issue_url=issue_url,
        prompt_type=prompt_type,
        model=model,
        context={}  # No additional context needed as we're using RAG
    )
    
    # Generate and process prompt
    prompt_response = await prompt_generator.generate_prompt(request, issue_response.data)
    if prompt_response.status == "error":
        print(f"Error generating prompt: {prompt_response.error}")
        return
    
    print("\nGenerated Prompt:")
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
    # Example GitHub issue from llama-index repository
    issue = {
        "url": "https://github.com/run-llama/llama_index/issues/18632",
        "prompt_type": "explain",
        "model": "gpt-4o-mini"
    }
    
    await run_example(issue["url"], issue["prompt_type"], issue["model"])

if __name__ == "__main__":
    asyncio.run(main()) 