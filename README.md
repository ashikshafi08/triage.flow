# GH Issue Prompt

Transform GitHub issues into structured LLM prompts with context-aware intelligence.

## Features

- Extract GitHub issues with caching and retry logic
- Generate different types of prompts:
  - Explain issues
  - Generate fixes
  - Create test cases
  - Summarize discussions
- REST API interface
- Configurable caching and rate limiting
- Support for multiple LLM models (GPT-4, GPT-4o-mini, etc.)
- Automatic context extraction from GitHub issues
- Local RAG (Retrieval-Augmented Generation) for intelligent context retrieval
  - Automatic repository cloning and analysis
  - Semantic search across codebase
  - Relevant code and documentation extraction
  - FAISS-based vector storage for efficient retrieval

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

4. Create a `.env` file with your credentials:
```
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running Examples

The repository includes example usage in `examples.py`:

```bash
python examples.py
```

This will demonstrate:
- Explaining a GitHub issue
- Generating a fix
- Creating test cases
- Summarizing the issue

### API Server

1. Start the API server:
```bash
python -m src.main
```

2. The API will be available at `http://localhost:8000`

3. Generate a prompt:
```bash
curl -X POST "http://localhost:8000/generate-prompt" \
     -H "Content-Type: application/json" \
     -d '{
           "issue_url": "https://github.com/owner/repo/issues/123",
           "prompt_type": "explain",
           "model": "gpt-4o-mini"
         }'
```

### Available Prompt Types

- `explain`: Get a detailed explanation of the issue
- `fix`: Generate a solution for the issue
- `test`: Create test cases for the issue
- `summarize`: Get a concise summary of the issue

### Available Models

- `gpt-4`: OpenAI's GPT-4 model
- `gpt-4o-mini`: OpenAI's GPT-4o-mini model

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
gh-issue-prompt/
├── src/
│   ├── github_client.py    # GitHub API integration
│   ├── llm_client.py       # LLM integration
│   ├── prompt_generator.py # Prompt generation logic
│   ├── models.py          # Data models
│   ├── config.py          # Configuration management
│   ├── main.py            # FastAPI application
│   ├── local_rag.py       # Local RAG implementation
│   └── local_repo_loader.py # Repository cloning and loading
├── examples.py            # Example usage
├── examples_complete_rag.py # Complete RAG example
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## License

MIT License - see LICENSE file for details 