# Using the CLI Examples

This guide explains how to use the command-line interface (CLI) examples provided with the GitHub Issue Prompt Generator with RAG.

## 1. Environment Setup

Before running any CLI examples, ensure you have:
- Installed all dependencies (see Quickstart)
- Set up your `.env` file with the required API keys

## 2. Running the Complete RAG Example

The main CLI entry point is:

```bash
python examples/examples_complete_rag.py
```

This script will:
- Prompt you for a GitHub issue URL (or you can edit the script to set your own)
- Fetch the issue (title, body, comments, etc.)
- Clone the repository and analyze the codebase
- Retrieve relevant code and documentation
- Generate a comprehensive prompt for the LLM
- Output the explanation and LLM response

## 3. Output Interpretation

After running the script, you will see:
- The issue title, description, and comments
- The number of relevant files found in the repository
- The generated prompt (with context from the repo and issue)
- The LLM's response (explanation, fix, test, or summary)

## 4. Customizing the Example

You can change the prompt type (explain, fix, test, summarize) by editing the `prompt_type` variable in the script:

```python
issue = {
    "url": "https://github.com/huggingface/trl/issues/3368",
    "prompt_type": "fix",  # Options: explain, fix, test, summarize
    "model": "gpt-4o-mini"
}
```

You can also change the model or the issue URL as needed.

## 5. Troubleshooting

- **ModuleNotFoundError:** Make sure you run the script from the project root and set `PYTHONPATH` if needed:
  ```bash
  PYTHONPATH=$PYTHONPATH:. python examples/examples_complete_rag.py
  ```
- **API Key Errors:** Ensure your `.env` file is present and contains valid keys.
- **Git Errors:** If the repo doesn't have a `main` branch, the tool will try `master` automatically.

## 6. Next Steps

Try the other example scripts in the `examples/` directory for different workflows and advanced usage. 