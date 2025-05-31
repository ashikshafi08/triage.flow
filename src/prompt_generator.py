from typing import Dict, Any, Optional
import re
from .models import Issue, PromptRequest, PromptResponse, IssueComment
from .config import settings
from .language_config import get_language_metadata

class PromptGenerator:
    def __init__(self):
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

Title: {title}
Description: {description}

{context}

Please provide:
1. A detailed explanation of the proposed fix
2. Code changes needed (with file paths and line numbers)
3. Test cases to verify the fix
4. Any potential side effects or considerations""",

            "test": """Please create test cases for the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. Test scenarios that cover the issue
2. Test code with file paths and line numbers
3. Expected results for each test
4. Any edge cases to consider""",

            "summarize": """Please summarize the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. A concise summary of the issue
2. Key technical details
3. Current status
4. Next steps or recommendations""",

            "document": """Please generate documentation for the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. A clear problem statement
2. Technical requirements and specifications
3. Usage examples
4. API reference (if applicable)
5. Troubleshooting tips""",

            "review": """Please review the code changes for the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. Code quality assessment
2. Potential bugs or vulnerabilities
3. Performance considerations
4. Style and consistency feedback
5. Suggestions for improvement""",

            "prioritize": """Please prioritize the following GitHub issue:

Title: {title}
Description: {description}

{context}

Please provide:
1. Severity assessment (critical, high, medium, low)
2. Impact analysis
3. Urgency level
4. Resource estimation
5. Recommended timeline"""
        }

    def _clean_markdown(self, text: str) -> str:
        """Clean up markdown formatting in text."""
        # Remove <details> and <summary> tags and their content
        text = re.sub(r'<details>.*?</details>', '', text, flags=re.DOTALL)
        
        # Remove other HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def _format_suggestions(self, text: str) -> str:
        """Format suggestions with proper markdown."""
        # Convert "Suggested next steps" to a proper section
        text = re.sub(
            r'Suggested next steps.*?$',
            '\nSuggested Next Steps:\n',
            text,
            flags=re.DOTALL | re.MULTILINE
        )
        
        # Convert suggestions to checkboxes
        text = re.sub(
            r'•\s*(.*?)(?=\n•|\n\n|$)',
            r'- [ ] \1',
            text,
            flags=re.MULTILINE
        )
        
        return text

    def _add_test_references(self, text: str, context: Dict[str, Any]) -> str:
        """Add references to relevant test files."""
        test_files = []
        for source in context.get("sources", []):
            if "test" in source.get("file", "").lower():
                test_files.append(f"- {source['file']}")
        
        if test_files:
            text += "\n\nRelated Test Files:\n" + "\n".join(test_files)
        
        return text

    async def generate_prompt(self, request: PromptRequest, issue: Issue) -> PromptResponse:
        """Generate a prompt based on the request type and issue, including comments, labels, and assignees."""
        try:
            # Get the appropriate template
            template = self.prompt_templates.get(request.prompt_type)
            if not template:
                return PromptResponse(
                    status="error",
                    error=f"Unsupported prompt type: {request.prompt_type}"
                )

            # Clean up the issue description
            clean_description = self._clean_markdown(issue.body)

            # Add labels and assignees metadata
            labels_text = f"Labels: {', '.join(issue.labels)}" if issue.labels else "Labels: None"
            assignees_text = f"Assignees: {', '.join(issue.assignees)}" if issue.assignees else "Assignees: None"

            # Add discussion/comments section (show up to 3 most recent comments)
            discussion_text = ""
            if issue.comments:
                discussion_text = "\nDiscussion (recent comments):\n"
                for comment in issue.comments[-3:]:
                    user = getattr(comment, 'user', '')
                    body = getattr(comment, 'body', '')
                    created = getattr(comment, 'created_at', '')
                    discussion_text += f"- {user} ({created}): {body.strip()}\n"

            # Format the context
            repo_context_data = request.context.get("repo_context", {})
            context_text = ""
            if repo_context_data:
                repo_info = repo_context_data.get("repo_info")
                if repo_info:
                    owner = repo_info.get('owner', 'N/A')
                    repo_name = repo_info.get('repo', 'N/A')
                    branch = repo_info.get('branch', 'N/A')
                    url = repo_info.get('url', 'N/A')
                    
                    context_text += f"You are analyzing an issue from the repository: {owner}/{repo_name} (branch: {branch}, URL: {url}).\n"
                    if repo_info.get("languages"):
                        lang_list = ", ".join(repo_info["languages"].values())
                        context_text += f"The primary languages in this repository appear to be: {lang_list}.\n"
                    context_text += "Consider this repository information when forming your response.\n\n"

                context_text += "Retrieved Context from Repository:\n"
                if repo_context_data.get("sources"):
                    context_text += "Relevant Files/Snippets:\n" # Changed heading
                    for source in repo_context_data["sources"][:3]: # Limit to 3 sources to keep prompt concise
                        context_text += f"  - File: {source.get('file', 'N/A')}"
                        if source.get('language') and source.get('language') != 'unknown':
                            context_text += f" (Language: {source.get('language')})"
                        context_text += "\n"
                        # Optionally include a snippet of content if desired, but can make prompts very long
                        # content_snippet = source.get('content', '').strip()[:200] + "..." if source.get('content') else "N/A"
                        # context_text += f"    Snippet: {content_snippet}\n"
                    context_text += "\n"
                if repo_context_data.get("response"): # This is the summary from RAG
                    context_text += f"Context Summary:\n{repo_context_data['response']}\n"

            # Compose the full prompt
            prompt = f"""
{labels_text}
{assignees_text}

{template.format(
    title=issue.title,
    description=clean_description,
    context=context_text
)}
{discussion_text}
"""

            # Format suggestions if present
            prompt = self._format_suggestions(prompt)
            # Add test references
            prompt = self._add_test_references(prompt, repo_context_data) # Changed context to repo_context_data

            # Debug print to verify inclusion of metadata
            print(f"[DEBUG] Labels: {issue.labels}, Assignees: {issue.assignees}, Comments: {len(issue.comments) if issue.comments else 0}")

            return PromptResponse(
                status="success",
                prompt=prompt
            )

        except Exception as e:
            return PromptResponse(
                status="error",
                error=f"Failed to generate prompt: {str(e)}"
            )
