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

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` with your credentials:
```
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

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
           "model": "gpt-4",
           "context": {}
         }'
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

## License

MIT License - see LICENSE file for details 