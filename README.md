# GitHub Issue Prompt Generator with RAG

A powerful tool that uses Retrieval-Augmented Generation (RAG) to analyze GitHub issues and generate detailed prompts for understanding and solving them. The system combines local repository analysis with advanced language models to provide comprehensive context and insights.

## Features

- **Multi-Language Support**: Comprehensive support for 20+ programming languages including:
  - Python, JavaScript/TypeScript, Java, C/C++
  - Go, Rust, Ruby, PHP, Swift, Kotlin
  - Scala, Dart, Haskell, Elixir, Clojure
  - Erlang, Lua, Perl, and more
- **Language-Specific Processing**:
  - Automatic language detection
  - Language-specific documentation extraction
  - Import/require pattern recognition
  - Structured code analysis
- **Local Repository Analysis**: Clones and analyzes repositories locally for faster processing
- **FAISS Vector Store**: Uses FAISS for efficient similarity search and vector storage
- **OpenAI Integration**: Leverages OpenAI's embedding models for better text understanding
- **Contextual Prompt Generation**: Creates detailed prompts based on issue context and repository code
- **Multiple Example Scripts**: Various examples showing different use cases and configurations

## Prerequisites

- Python 3.8+
- Git
- OpenAI API key
- GitHub token (optional, for private repositories)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gh-issue-prompt.git
cd gh-issue-prompt
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file with:
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token  # Optional
```

## Usage

The project provides several example scripts in the `examples/` directory:

1. **Basic Example** (`examples.py`):
```bash
python examples/examples.py
```

2. **Complete RAG Example** (`examples_complete_rag.py`):
```bash
python examples/examples_complete_rag.py
```

3. **Local RAG Example** (`examples_local_rag.py`):
```bash
python examples/examples_local_rag.py
```

4. **Environment-based Example** (`example_with_env.py`):
```bash
python examples/example_with_env.py
```

## Project Structure

```
gh-issue-prompt/
├── examples/                 # Example scripts
├── src/                      # Source code
│   ├── config.py            # Configuration settings
│   ├── github_client.py     # GitHub API client
│   ├── language_config.py   # Language-specific configurations
│   ├── local_rag.py         # Local RAG implementation
│   ├── local_repo_loader.py # Repository cloning utilities
│   ├── llm_client.py        # LLM client interface
│   ├── main.py              # Main application logic
│   ├── models.py            # Data models
│   ├── prompt_generator.py  # Prompt generation logic
│   └── repo_context.py      # Repository context extraction
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── LICENSE                  # License file
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## How It Works

1. **Repository Analysis**:
   - Clones the target repository locally
   - Automatically detects programming languages
   - Processes code and documentation files with language-specific patterns
   - Extracts imports, dependencies, and documentation
   - Creates a FAISS vector index for efficient search

2. **Language-Specific Processing**:
   - Identifies file types and programming languages
   - Extracts language-specific documentation (e.g., JSDoc, Python docstrings)
   - Recognizes import/require patterns
   - Structures content with language context
   - Maintains language metadata throughout processing

3. **Issue Analysis**:
   - Extracts relevant context from the repository
   - Identifies related code and documentation
   - Considers language-specific patterns in search
   - Generates a comprehensive prompt with language context

4. **Prompt Generation**:
   - Combines issue details with repository context
   - Includes language-specific insights
   - Creates structured prompts for different use cases
   - Provides detailed analysis and potential solutions

---

## Example 1: Generate an Explanation for a GitHub Issue

This example demonstrates how to generate a detailed explanation for a GitHub issue using the complete RAG pipeline.

```bash
python examples/examples_complete_rag.py
```

- The script will prompt you for a GitHub issue URL, or you can edit the script to set your own.
- The tool will:
  1. Fetch the issue (title, body, comments, etc.)
  2. Clone the repository and analyze the codebase
  3. Retrieve relevant code and documentation
  4. Generate a comprehensive prompt for the LLM
  5. Output the explanation and LLM response

---

## Example 2: Use the Tool in Your Own Python Script

You can use the core components in your own Python code:

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

---

## Example 3: Customizing Prompt Types

You can generate different types of prompts (explain, fix, test, summarize) by changing the `prompt_type` in your script or API call. For example:

```python
request = PromptRequest(
    issue_url=issue_url,
    prompt_type="fix",  # Options: explain, fix, test, summarize
    model="gpt-4o-mini",
    context={"repo_context": context}
)
```

This will generate a prompt asking the LLM to propose a fix for the issue, using all the context from the repository and the issue discussion.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 