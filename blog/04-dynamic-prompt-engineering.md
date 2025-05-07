# Dynamic Prompt Engineering: Adapting to Different Analysis Needs

## Introduction

When we first started working with Large Language Models (LLMs), we quickly realized that the quality of the prompts we send to these models is crucial for getting meaningful responses. After months of experimentation and iteration, we've developed a dynamic prompt engineering system that adapts to different analysis needs while maintaining consistency and quality.

## The Challenge

Creating effective prompts for LLMs is more art than science. We needed to address several key challenges:
- How to format prompts for different types of tasks
- How to integrate relevant context from the codebase
- How to provide clear and specific instructions
- How to maintain a consistent structure
- How to handle markdown and formatting properly

## Our Solution

### 1. Template-Based System
```python
self.prompt_templates = {
    "explain": """Please explain the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. A clear explanation of what the issue is about
2. The root cause of the problem
3. Any relevant technical details from the codebase
4. Potential impact if not addressed""",

    "fix": """Please provide a solution for the following GitHub issue:
    # ... other templates
}
```

We've built a comprehensive template system that:
- Defines task-specific templates for different types of analysis
- Maintains a consistent structure across all prompts
- Provides clear and specific instructions
- Integrates relevant context from the codebase

### 2. Markdown Processing
```python
def _clean_markdown(self, text: str) -> str:
    """Clean up markdown formatting in text."""
    # Remove <details> and <summary> tags and their content
    text = re.sub(r'<details>.*?</details>', '', text, flags=re.DOTALL)
    
    # Remove other HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
```

Our markdown processing system:
- Removes HTML tags that might confuse the model
- Cleans up formatting to maintain readability
- Preserves important structural elements
- Ensures consistent output format

### 3. Context Integration
```python
async def generate_prompt(self, request: PromptRequest, issue: Issue) -> PromptResponse:
    # Clean up the issue description
    clean_description = self._clean_markdown(issue.body)
    
    # Format the context
    context = request.context.get("repo_context", {})
    context_text = ""
    if context:
        context_text = "\nRepository Context:\n\n"
        if context.get("sources"):
            context_text += "Relevant Files:\n"
            for source in context["sources"]:
                context_text += f"- {source['file']}\n"
            context_text += "\n"
        if context.get("response"):
            context_text += f"Repository Context:\n{context['response']}\n"
```

Our context integration system:
- Integrates relevant repository context
- Formats file references clearly
- Maintains a clear structure
- Preserves important information

## Real-World Benefits

1. **Consistency**: Our template system ensures that all prompts follow a consistent structure, making it easier for the model to understand and respond.

2. **Clarity**: By providing clear instructions and formatting, we get more accurate and useful responses from the model.

3. **Context**: Our system integrates relevant context from the codebase, helping the model understand the full picture.

4. **Flexibility**: The template system makes it easy to modify and extend prompts for different types of analysis.

## Implementation Details

### 1. Template Management
- We maintain a library of task-specific templates
- We ensure consistent formatting across all prompts
- We provide clear instructions for each task
- We integrate context from the codebase

### 2. Markdown Handling
- We remove HTML tags that might confuse the model
- We clean up formatting to maintain readability
- We preserve important structural elements
- We ensure consistent output format

### 3. Context Integration
- We integrate relevant repository context
- We format file references clearly
- We maintain a clear structure
- We preserve important information

## Future Improvements

1. Add more prompt types for different analysis tasks
2. Improve context integration with better relevance scoring
3. Enhance markdown handling for complex formatting
4. Add prompt validation to ensure quality

## Conclusion

Building our dynamic prompt engineering system has been a fascinating journey. It's given us the ability to create effective prompts that lead to meaningful and useful responses from LLMs. By combining template-based generation, smart markdown processing, and context integration, we've created a system that adapts to different analysis needs while maintaining consistency and quality.

What's most exciting is that this is just the beginning. As we continue to improve the system, we're finding new ways to create more effective prompts, making our tool even more powerful and useful for developers. 