# GitHub Issue Prompt Generator with RAG

A powerful tool that uses Retrieval-Augmented Generation (RAG) to analyze GitHub issues and generate detailed prompts for understanding and solving them. The system combines local repository analysis with advanced language models to provide comprehensive context and insights.

## Features

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
   - Processes code and documentation files
   - Creates a FAISS vector index for efficient search

2. **Issue Analysis**:
   - Extracts relevant context from the repository
   - Identifies related code and documentation
   - Generates a comprehensive prompt

3. **Prompt Generation**:
   - Combines issue details with repository context
   - Creates structured prompts for different use cases
   - Provides detailed analysis and potential solutions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 