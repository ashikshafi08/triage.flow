# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- OpenRouter support with Claude 3 Sonnet model
- Example file for OpenRouter usage (`examples/examples_openrouter.py`)
- Full file path support in repository context
- Markdown cleaning and formatting improvements
- Test file reference highlighting
- Checkbox formatting for suggestions

### Changed
- Replaced llama_index GitHub client with direct GitHub API calls
- Updated prompt templates for better formatting
- Improved error handling in GitHub client
- Enhanced context formatting in prompts
- Simplified template system in PromptGenerator

### Removed
- Dependency on llama_index for GitHub operations
- Raw markdown tags from issue descriptions
- Redundant context formatting methods

### Fixed
- GitHub issue comment fetching
- Markdown formatting in prompts
- File path handling in repository context
- Suggestion formatting in prompts

## [0.1.0] - Initial Release

### Added
- Basic GitHub issue analysis functionality
- Support for multiple prompt types (explain, fix, test, summarize)
- Local repository context extraction
- OpenAI integration
- FAISS vector store for document indexing
- Multi-language support for code analysis 