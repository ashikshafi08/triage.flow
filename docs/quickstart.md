# Quickstart Guide

Welcome to the GitHub Issue Prompt Generator with RAG! This guide will help you get started quickly.

## 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/gh-issue-prompt.git
cd gh-issue-prompt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Set Up Environment Variables

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token  # Optional, for private repos
```

## 3. Run an Example

Try the complete RAG example:

```bash
python examples/examples_complete_rag.py
```

This will fetch a GitHub issue, analyze the repository, and generate a detailed prompt for the LLM.

## 4. Minimal Python Example

You can use the tool in your own script:

```python
from src.github_client import GitHubIssueClient
from src.local_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
import asyncio

async def main():
    issue_url = "https://github.com/huggingface/smolagents/issues/1292"
    github_client = GitHubIssueClient()
    issue_response = await github_client.get_issue(issue_url)
    if issue_response.status != "success":
        print("Failed to fetch issue")
        return
    repo_extractor = LocalRepoContextExtractor()
    await repo_extractor.load_repository("https://github.com/huggingface/smolagents.git")
    context = await repo_extractor.get_issue_context(issue_response.data.title, issue_response.data.body)
    prompt_generator = PromptGenerator()
    prompt = await prompt_generator.generate_prompt(
        request=None,  # Fill in as needed
        issue=issue_response.data
    )
    print(prompt)

asyncio.run(main())
```

You're ready to go! 