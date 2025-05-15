<p align="center">
  <img src="triage_flow_logo.png" alt="triage.flow logo" width="180" height="180"/>
</p>

# triage.flow

<p align="center">
  <b>AI-powered GitHub Issue Context & Prompt Generator with RAG and Multi-Model LLM Support</b><br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
</p>

---

## Why Use triage.flow?

- **Instantly understand and triage GitHub issues** with deep, code-aware context
- **Automate prompt generation** for LLMs using real repo code, docs, and discussions
- **Works with 20+ programming languages** and any public or private repo
- **Supports OpenAI, OpenRouter, Claude, Mistral, and more**
- **Save time for maintainers, contributors, and AI agents**

---

## Features at a Glance

| Feature                        | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| Multi-Language Support        | 20+ languages, auto-detected                                     |
| Local Repo Analysis           | Fast, privacy-friendly, no API rate limits                       |
| FAISS Vector Store            | Efficient, scalable code/document search                         |
| OpenAI & OpenRouter Embeddings| High-quality semantic understanding                              |
| Multi-Provider LLM Support    | OpenAI, OpenRouter, Claude, Mistral, and more                    |
| Issue + Comments Extraction   | Full context from GitHub issues and discussions                  |
| Contextual Prompt Generation  | Explain, fix, test, summarize, or customize                     |
| CLI & Python API              | Use from terminal or integrate in your own apps                  |
| Extensible                    | Add new prompt types, languages, or LLMs easily                  |

---

## Who is triage.flow for?
- **Open source maintainers**: triage and understand issues faster
- **Contributors**: get up to speed on unfamiliar codebases
- **AI agents & bots**: generate actionable, context-rich prompts
- **Anyone** who wants to automate or enhance GitHub issue workflows

---

## How triage.flow Works

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
Build Vector Index (FAISS + OpenAI/OpenRouter)
      |
      v
Retrieve Relevant Context
      |
      v
Generate LLM Prompt (Explain, Fix, Test, Summarize, ...)
      |
      v
LLM Response
```

---

## Quickstart

See [docs/quickstart.md](docs/quickstart.md) for full details.

```bash
git clone https://github.com/yourusername/triage.flow.git
cd triage.flow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add your .env file with OPENAI_API_KEY, OPENROUTER_API_KEY, and GITHUB_TOKEN
python examples/examples_openrouter.py
```

or Use uv to install dependencies (Recommended)

```bash
uv pip sync requirements.txt
python examples/examples_openrouter.py
```

---

## Environment Variables

| Variable              | Description                                 |
|----------------------|---------------------------------------------|
| `OPENAI_API_KEY`     | Your OpenAI API key (for OpenAI models)     |
| `OPENROUTER_API_KEY` | Your OpenRouter API key (for OpenRouter)    |
| `GITHUB_TOKEN`       | GitHub personal access token                |
| `LLM_PROVIDER`       | `openai`, `openrouter`, etc.                |
| `DEFAULT_MODEL`      | Default model name (e.g., `gpt-3.5-turbo`)  |

---

## Example Usage

```python
from src.models import PromptRequest, LLMConfig
from src.github_client import GitHubIssueClient
from src.local_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
import asyncio

async def main():
    issue_url = "https://github.com/vllm-project/vllm/issues/17747"
    github_client = GitHubIssueClient()
    issue_response = await github_client.get_issue(issue_url)
    if issue_response.status != "success":
        print("Failed to fetch issue")
        return
    repo_extractor = LocalRepoContextExtractor()
    repo_url = "https://github.com/vllm-project/vllm.git"
    await repo_extractor.load_repository(repo_url)
    context = await repo_extractor.get_issue_context(issue_response.data.title, issue_response.data.body)
    prompt_generator = PromptGenerator()
    llm_config = LLMConfig(
        provider="openrouter",
        name="openai/o4-mini-high",
        additional_params={"max_tokens": 8000}
    )
    request = PromptRequest(
        issue_url=issue_url,
        prompt_type="explain",
        llm_config=llm_config,
        context={"repo_context": context}
    )
    prompt_response = await prompt_generator.generate_prompt(request, issue_response.data)
    print(prompt_response.prompt)

asyncio.run(main())
```

---

## Available Prompt Types
- `explain`: Explain the issue in detail
- `fix`: Suggest a fix for the issue
- `test`: Generate test cases for the issue
- `summarize`: Summarize the issue
- *(Extensible: add your own!)*

---

## Roadmap / Coming Soon
- **MCP Server:** Expose repo context, vector index, and prompt/LLM APIs for agentic workflows and interactive codebase exploration
- **Custom Prompt Types:** Easily add new prompt types or LLM actions
- **Code Actions:** Suggest, review, or apply code changes via API
- **Session/Workspace Management:** Multi-user and agent support

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