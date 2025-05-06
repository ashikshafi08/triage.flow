import asyncio
from src.github_client import GitHubIssueClient
from src.prompt_generator import PromptGenerator
from src.llm_client import LLMClient
from src.models import PromptRequest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_examples():
    # Initialize clients
    github_client = GitHubIssueClient()
    prompt_generator = PromptGenerator()
    llm_client = LLMClient()

    # Example 1: Explain an issue
    issue_url = "https://github.com/run-llama/llama_index/issues/18614"
    print("\nExample 1: Explaining an Issue")
    print("=" * 50)
    
    # First fetch the issue
    issue_response = await github_client.get_issue(issue_url)
    print(f"Issue Response Status: {issue_response.status}")
    if issue_response.status == "success":
        print(f"Issue Title: {issue_response.data.title}")
        # Generate explain prompt
        explain_request = PromptRequest(
            issue_url=issue_url,
            prompt_type="explain",
            model="gpt-4o-mini",
            context={}  # Context should be derived from the issue
        )
        explain_response = await prompt_generator.generate_prompt(
            explain_request,
            issue_response.data
        )
        print(f"Prompt Response Status: {explain_response.status}")
        
        # Process with LLM
        llm_response = await llm_client.process_prompt(
            prompt=explain_response.prompt,
            prompt_type="explain",
            context=explain_request.context,
            model=explain_request.model
        )
        print("LLM Response:")
        print(llm_response.prompt if llm_response.prompt else "No response")
    else:
        print(f"Error fetching issue: {issue_response.error}")

    # Example 2: Generate a fix
    print("\nExample 2: Generating a Fix")
    print("=" * 50)
    if issue_response.status == "success":
        fix_request = PromptRequest(
            issue_url=issue_url,
            prompt_type="fix",
            model="gpt-4",
            context={}  # Context should be derived from the issue
        )
        fix_response = await prompt_generator.generate_prompt(
            fix_request,
            issue_response.data
        )
        print(f"Prompt Response Status: {fix_response.status}")
        
        # Process with LLM
        llm_response = await llm_client.process_prompt(
            prompt=fix_response.prompt,
            prompt_type="fix",
            context=fix_request.context,
            model=fix_request.model
        )
        print("LLM Response:")
        print(llm_response.prompt if llm_response.prompt else "No response")
    else:
        print("Skipping fix example due to previous error")

    # Example 3: Create test cases
    print("\nExample 3: Creating Test Cases")
    print("=" * 50)
    if issue_response.status == "success":
        test_request = PromptRequest(
            issue_url=issue_url,
            prompt_type="test",
            model="gpt-4o-mini",
            context={}  # Context should be derived from the issue
        )
        test_response = await prompt_generator.generate_prompt(
            test_request,
            issue_response.data
        )
        print(f"Prompt Response Status: {test_response.status}")
        
        # Process with LLM
        llm_response = await llm_client.process_prompt(
            prompt=test_response.prompt,
            prompt_type="test",
            context=test_request.context,
            model=test_request.model
        )
        print("LLM Response:")
        print(llm_response.prompt if llm_response.prompt else "No response")
    else:
        print("Skipping test example due to previous error")

    # Example 4: Summarize the issue
    print("\nExample 4: Summarizing the Issue")
    print("=" * 50)
    if issue_response.status == "success":
        summarize_request = PromptRequest(
            issue_url=issue_url,
            prompt_type="summarize",
            model="gpt-4",
            context={}  # Context should be derived from the issue
        )
        summarize_response = await prompt_generator.generate_prompt(
            summarize_request,
            issue_response.data
        )
        print(f"Prompt Response Status: {summarize_response.status}")
        
        # Process with LLM
        llm_response = await llm_client.process_prompt(
            prompt=summarize_response.prompt,
            prompt_type="summarize",
            context=summarize_request.context,
            model=summarize_request.model
        )
        print("LLM Response:")
        print(llm_response.prompt if llm_response.prompt else "No response")
    else:
        print("Skipping summarize example due to previous error")

if __name__ == "__main__":
    asyncio.run(run_examples()) 