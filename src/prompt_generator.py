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
4. Next steps or recommendations"""
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
            prompt = self._add_test_references(prompt, context)

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