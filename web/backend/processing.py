import asyncio
from src.github_client import GitHubIssueClient
from src.new_rag import LocalRepoContextExtractor
from src.prompt_generator import PromptGenerator
from src.llm_client import LLMClient
from src.models import IssueResponse, PromptRequest, LLMConfig
from store import jobs  # Import job store from store.py
import re
from datetime import datetime

def update_progress(job_id: str, message: str):
    """Helper function to update the progress log for a given job."""
    if job_id in jobs:
        jobs[job_id]["progress_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

def ensure_proper_code_formatting(text: str) -> str:
    """Ensure code blocks in markdown are properly formatted with language tags.
    
    This function looks for code blocks that might be missing language specifiers
    and attempts to add appropriate language tags based on content.
    """
    # Regular expression to find code blocks
    # This matches both ```language and ``` without language
    code_block_pattern = r'```(?:([a-zA-Z0-9_+-]+)\n)?(.*?)```'
    
    def process_match(match):
        language = match.group(1)
        code_content = match.group(2)
        
        # If language is already specified, keep it as is
        if language:
            return f'```{language}\n{code_content}```'
        
        # Try to detect language based on content
        detected_lang = detect_language(code_content)
        return f'```{detected_lang}\n{code_content}```'
    
    # Replace code blocks with properly formatted ones
    # Using re.DOTALL to make . match newlines
    processed_text = re.sub(code_block_pattern, process_match, text, flags=re.DOTALL)
    return processed_text

def detect_language(code_content: str) -> str:
    """Attempt to detect the programming language of a code snippet."""
    # Simple heuristics for common languages
    code_content = code_content.strip()
    
    # Check for Python
    if re.search(r'\bdef\b|\bclass\b|\bimport\b|\bfrom\s+\w+\s+import\b', code_content):
        return 'python'
    
    # Check for JavaScript/TypeScript
    if re.search(r'\bconst\b|\blet\b|\bvar\b|\bfunction\b|\b=>\b|\bexport\b|\bimport\s+.*\s+from\b', code_content):
        # Check for TypeScript-specific syntax
        if re.search(r'\b(interface|type)\s+\w+|:\s*[A-Z]\w+', code_content):
            return 'typescript'
        return 'javascript'
    
    # Check for HTML
    if re.search(r'<\s*(!DOCTYPE|html|head|body|div|span|a|p|h[1-6])', code_content):
        return 'html'
    
    # Check for CSS
    if re.search(r'\b(body|div|span|a|p)\s*{[^}]*}', code_content):
        return 'css'
    
    # Check for JSON
    if (code_content.startswith('{') and code_content.endswith('}')) or \
       (code_content.startswith('[') and code_content.endswith(']')):
        try:
            # Try to parse as JSON
            import json
            json.loads(code_content)
            return 'json'
        except:
            pass
    
    # Check for SQL
    if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*\b(FROM|INTO|TABLE|DATABASE)\b', 
                code_content, re.IGNORECASE):
        return 'sql'
    
    # Check for shell/bash
    if re.search(r'^\s*(#!.*sh|\$\s+|sudo\s+|apt\s+|yum\s+|brew\s+|cd\s+|ls\s+|mkdir\s+|rm\s+)', code_content, re.MULTILINE):
        return 'bash'
    
    # Default to text if we can't detect
    return 'text'

async def process_issue(job_id: str, issue_url: str, prompt_type: str):
    try:
        update_progress(job_id, "Starting analysis...")

        # Initialize clients
        github_client = GitHubIssueClient()
        repo_extractor = LocalRepoContextExtractor()
        prompt_generator = PromptGenerator()
        llm_client = LLMClient()
        
        update_progress(job_id, "Fetching GitHub issue details...")
        # Get GitHub issue
        issue_response = await github_client.get_issue(issue_url)
        if not isinstance(issue_response, IssueResponse) or issue_response.status != "success":
            jobs[job_id] = {
                "status": "error",
                "error": f"GitHub error: {issue_response.error}",
                "progress_log": jobs[job_id]["progress_log"]
            }
            return
        
        # Extract repo info from URL
        url_parts = issue_url.split('/')
        owner = url_parts[3]
        repo = url_parts[4]
        repo_url = f"https://github.com/{owner}/{repo}.git"
        
        update_progress(job_id, f"Cloning and processing repository: {owner}/{repo}...")
        # Load repository context
        await repo_extractor.load_repository(repo_url)
        
        update_progress(job_id, "Extracting relevant context from code and documentation...")
        context = await repo_extractor.get_issue_context(
            issue_response.data.title,
            issue_response.data.body
        )
        
        # Configure LLM (using OpenRouter by default)
        llm_config = LLMConfig(
            provider="openrouter",
            name="google/gemini-2.5-flash-preview-05-20", 

        )
        
        update_progress(job_id, "Generating prompt for LLM...")
        # Generate prompt
        request = PromptRequest(
            issue_url=issue_url,
            prompt_type=prompt_type,
            llm_config=llm_config,
            context={"repo_context": context}
        )
        prompt_response = await prompt_generator.generate_prompt(request, issue_response.data)
        
        if prompt_response.status == "error":
            jobs[job_id] = {
                "status": "error",
                "error": f"Prompt error: {prompt_response.error}",
                "progress_log": jobs[job_id]["progress_log"]
            }
            return
        
        update_progress(job_id, "Calling LLM for response...")
        # Get LLM response
        llm_response = await llm_client.process_prompt(
            prompt=prompt_response.prompt,
            prompt_type=prompt_type,
            model=llm_config.name
        )
        
        update_progress(job_id, "Finalizing results...")
        
        # Process the LLM response to ensure proper code block formatting
        processed_response = ensure_proper_code_formatting(llm_response.prompt)
        
        # Update job status with results
        jobs[job_id] = {
            "status": "completed",
            "prompt": prompt_response.prompt,
            "result": processed_response,
            "tokens_used": getattr(llm_response, "tokens_used", 0),
            "progress_log": jobs[job_id]["progress_log"]
        }
        
    except Exception as e:
        update_progress(job_id, f"Processing failed: {str(e)}")
        jobs[job_id] = {
            "status": "error",
            "error": f"Processing error: {str(e)}",
            "progress_log": jobs[job_id]["progress_log"]
        }
