# Advanced Usage Guide

This guide covers advanced features and integration options for the GitHub Issue Prompt Generator with RAG.

## 1. Using the Tool as a Python Library (API Integration)

You can import and use the core components in your own Python applications, scripts, or web services.

### Example: Integrate with FastAPI

```python
from fastapi import FastAPI
from src.github_client import GitHubIssueClient
from src.new_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
import asyncio

app = FastAPI()

@app.get("/generate-prompt/")
async def generate_prompt(issue_url: str, prompt_type: str = "explain"):
    github_client = GitHubIssueClient()
    issue_response = await github_client.get_issue(issue_url)
    if issue_response.status != "success":
        return {"error": "Failed to fetch issue"}
    repo_extractor = LocalRepoContextExtractor()
    await repo_extractor.load_repository(issue_url.rsplit("/issues/", 1)[0] + ".git")
    context = await repo_extractor.get_issue_context(issue_response.data.title, issue_response.data.body)
    prompt_generator = PromptGenerator()
    prompt = await prompt_generator.generate_prompt(
        request=None,  # Fill in as needed
        issue=issue_response.data
    )
    return {"prompt": prompt}
```

## 2. Customizing Prompt Templates

You can extend or modify the prompt templates in `src/prompt_generator.py` to fit your workflow. For example, add a new template:

```python
def _generate_custom_prompt(self, issue: Issue, context: Dict, repo_context: Dict) -> str:
    return f"""Custom prompt for {issue.title}..."""

# Register it:
self.templates["custom"] = self._generate_custom_prompt
```

Then use `prompt_type="custom"` in your requests.

## 3. Handling Multi-Language Repositories

The tool automatically detects and processes files in 20+ programming languages. You can:
- Add new languages or patterns in `src/language_config.py`
- Adjust the `LANGUAGE_CONFIG` dictionary to fine-tune file extension, doc, or import patterns
- The context extraction and prompt generation will automatically include language metadata and relevant code/doc for each language

## 4. Batch Processing and Automation

You can write scripts to process multiple issues or repositories in batch mode. For example:

```python
issue_urls = [
    "https://github.com/org/repo/issues/1",
    "https://github.com/org/repo/issues/2",
]

for url in issue_urls:
    # Use the same workflow as above to generate prompts for each issue
    ...
```

## 5. Extending the System

- Add new prompt types or context extractors
- Integrate with other LLMs or vector stores
- Build a web UI or Slack bot on top of the API

For more, see the codebase and README for entry points and extension hooks. 