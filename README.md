# GH Issue Prompt

<p align="center">
  <b>AI-powered GitHub Issue Context & Prompt Generator with RAG</b><br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
</p>

---

## Why Use This?

- **Instantly understand and triage GitHub issues** with deep, code-aware context
- **Automate prompt generation** for LLMs using real repo code, docs, and discussions
- **Works with 20+ programming languages** and any public or private repo
- **Save time for maintainers, contributors, and AI agents**

---

## Features at a Glance

| Feature                        | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| Multi-Language Support        | 20+ languages, auto-detected                                     |
| Local Repo Analysis           | Fast, privacy-friendly, no API rate limits                       |
| FAISS Vector Store            | Efficient, scalable code/document search                         |
| OpenAI Embeddings             | High-quality semantic understanding                              |
| Issue + Comments Extraction   | Full context from GitHub issues and discussions                  |
| Contextual Prompt Generation  | Explain, fix, test, summarize, or customize                     |
| CLI & Python API              | Use from terminal or integrate in your own apps                  |
| Extensible                    | Add new prompt types, languages, or LLMs easily                  |

---

## Who is this for?
- **Open source maintainers**: triage and understand issues faster
- **Contributors**: get up to speed on unfamiliar codebases
- **AI agents & bots**: generate actionable, context-rich prompts
- **Anyone** who wants to automate or enhance GitHub issue workflows

---

## How it Works

```
GitHub Issue URL
      |
      v
Extract Issue & Comments
      |
      v
Clone Repo Locally
      |
      v
Analyze Code & Docs (Multi-Language)
      |
      v
Build Vector Index (FAISS + OpenAI)
      |
      v
Retrieve Relevant Context
      |
      v
Generate LLM Prompt
      |
      v
LLM Response
```

---

## Quickstart

See [docs/quickstart.md](docs/quickstart.md) for full details.

```bash
git clone https://github.com/yourusername/gh-issue-prompt.git
cd gh-issue-prompt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add your .env file with OPENAI_API_KEY and GITHUB_TOKEN
python examples/examples_complete_rag.py
```

---

## Example Usage

```python
from src.github_client import GitHubIssueClient
from src.local_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
import asyncio

def get_repo_url_from_issue_url(issue_url: str) -> str:
    parts = issue_url.rstrip('/').split('/')
    if len(parts) >= 5:
        return f"https://github.com/{parts[3]}/{parts[4]}.git"
    raise ValueError("Invalid GitHub issue URL")

async def main():
    issue_url = "https://github.com/huggingface/smolagents/issues/1292"
    github_client = GitHubIssueClient()
    issue_response = await github_client.get_issue(issue_url)
    if issue_response.status != "success":
        print("Failed to fetch issue")
        return
    repo_extractor = LocalRepoContextExtractor()
    repo_url = get_repo_url_from_issue_url(issue_url)
    await repo_extractor.load_repository(repo_url)
    context = await repo_extractor.get_issue_context(issue_response.data.title, issue_response.data.body)
    prompt_generator = PromptGenerator()
    prompt = await prompt_generator.generate_prompt(
        request=None,  # Fill in as needed
        issue=issue_response.data
    )
    print(prompt)

asyncio.run(main())
```

---

## More Documentation
- [Quickstart](docs/quickstart.md)
- [CLI Usage](docs/usage_cli.md)
- [Advanced Usage](docs/advanced_usage.md)

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 