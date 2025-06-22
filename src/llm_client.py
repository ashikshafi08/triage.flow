from typing import Optional, Dict, Any, AsyncGenerator
import httpx
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.prompts import PromptTemplate
from .config import settings
from .models import PromptResponse
import json
import aiohttp

# Try to import tiktoken for better token estimation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to word-based estimation if tiktoken fails
            pass
    
    # Fallback: rough estimation (words * 1.3)
    return int(len(text.split()) * 1.3)

def format_rag_context_for_llm(rag_data: Optional[Dict[str, Any]]) -> str:
    """Formats the RAG context dictionary into a string for the LLM prompt."""
    if not rag_data:
        return "No specific RAG context available for this query."

    context_parts = []
    repo_info = rag_data.get("repo_info")
    if repo_info:
        owner = repo_info.get('owner', 'N/A')
        repo_name = repo_info.get('repo', 'N/A')
        branch = repo_info.get('branch', 'N/A')
        url = repo_info.get('url', 'N/A')
        
        repo_info_str = f"Repository Information:\n"
        repo_info_str += f"- Name: {owner}/{repo_name}\n"
        repo_info_str += f"- Branch: {branch}\n"
        repo_info_str += f"- URL: {url}\n"
        if repo_info.get("languages"):
            lang_list = ", ".join(repo_info["languages"].values())
            repo_info_str += f"- Languages: {lang_list}\n"
        context_parts.append(repo_info_str)

    rag_summary = rag_data.get("response")
    if rag_summary:
        context_parts.append(f"Retrieved Context Summary:\n{rag_summary}")

    # Add related issues context if available
    related_issues = rag_data.get("related_issues")
    if related_issues and related_issues.get("issues"):
        issues_str = f"\nRELATED GITHUB ISSUES ({len(related_issues['issues'])} found):\n"
        issues_str += "=" * 60 + "\n"
        
        for i, issue in enumerate(related_issues["issues"], 1):
            issues_str += f"\n{i}. Issue #{issue['number']}: {issue['title']}\n"
            issues_str += f"   State: {issue['state']} | Similarity: {issue.get('similarity', 'N/A')}\n"
            issues_str += f"   URL: {issue['url']}\n"
            
            if issue.get('labels'):
                issues_str += f"   Labels: {', '.join(issue['labels'])}\n"
            
            if issue.get('body_preview'):
                issues_str += f"   Description: {issue['body_preview']}\n"
            
            if issue.get('patch_url'):
                issues_str += f"   Patch/Solution: {issue['patch_url']}\n"
            
            issues_str += "   " + "-" * 50 + "\n"
        
        issues_str += f"\nIMPORTANT: These are similar past issues from the same repository.\n"
        issues_str += "Use them as context for understanding common problems, solutions, and patterns.\n"
        issues_str += "When relevant, reference these issues and their solutions in your response.\n"
        
        context_parts.append(issues_str)

    # Add related patches context if available
    related_patches = rag_data.get("patches")
    if related_patches:
        patches_str = f"\nRELATED CODE PATCHES ({len(related_patches)} found):\n"
        patches_str += "=" * 60 + "\n"
        
        for i, patch_result in enumerate(related_patches, 1):
            patch = patch_result.patch
            patches_str += f"\n{i}. Patch for Issue #{patch.issue_id} (PR #{patch.pr_number})\n"
            patches_str += f"   Similarity: {patch_result.similarity:.3f}\n"
            patches_str += f"   Files Changed: {', '.join(patch.files_changed)}\n"
            
            # Truncate long patch summaries for context
            summary_preview = patch.diff_summary
            if len(summary_preview) > 1500:
                summary_preview = summary_preview[:1500] + "\n... [summary truncated] ..."
            patches_str += f"   Patch Summary:\n{summary_preview}\n"
            patches_str += "   " + "-" * 50 + "\n"
        
        patches_str += f"\nIMPORTANT: These are code patches that fixed similar past issues.\n"
        patches_str += "Use them as concrete examples of how to solve problems in this codebase.\n"
        
        context_parts.append(patches_str)

    sources = rag_data.get("sources")
    search_type = rag_data.get("search_type", "regular")
    
    if sources:
        # Set max sources using configuration
        max_sources = settings.MAX_RAG_SOURCES
        
        # If file-oriented search, prioritize showing more files
        if search_type == "file_oriented":
            max_sources = min(settings.MAX_RAG_SOURCES, 25)  # Keep reasonable limit for file-oriented
            content_preview_length = settings.MAX_CONTENT_PREVIEW_CHARS
        else:
            content_preview_length = settings.MAX_CONTENT_PREVIEW_CHARS
        
        if search_type == "file_oriented":
            sources_str = "FILE SEARCH RESULTS:\n"
            sources_str += "=" * 50 + "\n"
            sources_str += f"Found {len(sources)} relevant files"
            if len(sources) > max_sources:
                sources_str += f" (showing top {max_sources}):\n\n"
            else:
                sources_str += ":\n\n"
            
            for i, source in enumerate(sources[:max_sources], 1):
                file_path = source.get('file', 'UNKNOWN_FILE')
                language = source.get('language', 'unknown')
                match_reasons = source.get('match_reasons', [])
                
                sources_str += f"{i}. FILE: {file_path}\n"
                if language and language != 'unknown':
                    sources_str += f"   Language: {language}\n"
                if match_reasons:
                    sources_str += f"   Match reasons: {', '.join(match_reasons)}\n"
                
                # Add a preview of the content
                content = source.get('content', '')
                if content:
                    preview = content[:content_preview_length].strip()
                    if len(content) > content_preview_length:
                        preview += "..."
                    sources_str += f"   Content Preview:\n   {preview}\n"
                
                sources_str += "-" * 30 + "\n"
            
            sources_str += f"\nIMPORTANT: These files were found using file pattern matching.\n"
            sources_str += "The file paths listed above are the exact paths in the repository.\n"
        else:
            sources_str = "EXACT FILE PATHS AND CONTENT FROM REPOSITORY:\n"
            sources_str += "=" * 50 + "\n"
            
            for i, source in enumerate(sources[:max_sources], 1):
                file_path = source.get('file', 'UNKNOWN_FILE')
                language = source.get('language', 'unknown')
                
                sources_str += f"\n{i}. FILE: {file_path}\n"
                if language and language != 'unknown':
                    sources_str += f"   Language: {language}\n"
                sources_str += f"   Content Preview:\n"
                
                # Add a preview of the content
                content = source.get('content', '')
                if content:
                    preview = content[:content_preview_length].strip()
                    if len(content) > content_preview_length:
                        preview += "..."
                    sources_str += f"   {preview}\n"
                
                sources_str += "-" * 30 + "\n"
            
            sources_str += f"\nIMPORTANT: These are the ONLY file paths that exist in the retrieved context.\n"
            sources_str += "DO NOT reference any other file paths not listed above.\n"
        
        context_parts.append(sources_str)
    
    # Add user-selected files (from @) as a special section
    user_files = rag_data.get("user_selected_files")
    if user_files:
        user_files_str = "\nUSER-SELECTED FILES (via @):\n" + "="*50 + "\n"
        for file in user_files:
            user_files_str += f"\nFILE: {file['file']}\nCONTENT:\n{file['content'][:settings.MAX_CONTENT_PREVIEW_CHARS]}...\n" + "-"*30 + "\n"
        context_parts.append(user_files_str)
    
    if not context_parts:
        return "No specific RAG context was retrieved for this query."
        
    return "\n\n".join(context_parts)

class LLMClient:
    def __init__(self):
        self.default_model = settings.default_model
        
        # Define system prompts for different types
        self.system_prompts = {
            "explain": """You are triage.flow - an AI-powered repository analysis assistant. You help developers understand, explore, and triage codebases through intelligent conversation and tool usage.

When explaining code or issues, provide context-first analysis. Use retrieval tools to find relevant information, never hallucinate file paths or code content. Be concise but thorough, always cite source files when making claims about the codebase.""",
            
            "fix": """You are triage.flow - an AI-powered repository analysis assistant. You help developers understand, explore, and triage codebases through intelligent conversation and tool usage.

When suggesting fixes, always search for relevant context first. Provide practical solutions that integrate well with existing patterns. Never suggest untested code without warnings. Only reference files that exist in the retrieved context.""",
            
            "test": """You are triage.flow - an AI-powered repository analysis assistant. You help developers understand, explore, and triage codebases through intelligent conversation and tool usage.

When creating tests, examine existing test patterns first. Suggest comprehensive test cases that integrate with current frameworks. Only suggest safe test code that doesn't affect production systems.""",
            
            "summarize": """You are triage.flow - an AI-powered repository analysis assistant. You help developers understand, explore, and triage codebases through intelligent conversation and tool usage.

When summarizing, focus on actionable insights. Use search tools to gather comprehensive context before providing summaries. Never include sensitive information - focus on technical analysis only.""",
            
            "classification": """You are triage.flow - an AI-powered repository analysis assistant. Classify issues accurately based on their content. Return only valid JSON with 'label' and 'confidence' keys. Use provided examples as guidance."""
        }

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model configuration with fallback for unknown models"""
        # Try to get the specific model config
        if model in settings.model_configs:
            return settings.model_configs[model]
        
        # Try to get the default model config
        if self.default_model in settings.model_configs:
            return settings.model_configs[self.default_model]
        
        # Fallback: create a reasonable default config
        print(f"Warning: No configuration found for model '{model}', using default config")
        return {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 32000  # Reasonable default
        }

    def _get_openai_llm(self, model: Optional[str] = None):
        """Get OpenAI LLM instance."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAI(
            api_key=settings.openai_api_key,
            model=model or self.default_model
        )

    def _get_openrouter_llm(self, model: Optional[str] = None):
        """Get OpenRouter LLM instance."""
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        
        model_config = self._get_model_config(model or self.default_model)
        return OpenRouter(
            api_key=settings.openrouter_api_key,
            model=model or self.default_model,
            max_tokens=model_config["max_tokens"],
            temperature=model_config["temperature"]
        )

    def _format_message_with_cache_control(self, role: str, content: str, rag_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format message with cache_control for OpenRouter prompt caching."""
        if not settings.ENABLE_PROMPT_CACHING or not rag_context or not rag_context.get("sources"):
            # Caching disabled or no RAG context to cache, return simple message
            return {
                "role": role,
                "content": content
            }
        
        # Format RAG context for caching
        formatted_rag_context = format_rag_context_for_llm(rag_context)
        
        # Only cache if RAG context is substantial
        rag_tokens = estimate_tokens(formatted_rag_context)
        if rag_tokens < settings.PROMPT_CACHE_MIN_TOKENS:
            # Small context, don't use caching
            final_content = f"Relevant Context:\n{formatted_rag_context}\n\n{content}"
            return {
                "role": role,
                "content": final_content
            }
        
        print(f"Using prompt caching for {rag_tokens} tokens of RAG context")
        
        # Use multipart content with cache_control for large RAG context
        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant helping with code repository analysis. Below is the relevant repository context:"
                },
                {
                    "type": "text", 
                    "text": formatted_rag_context,
                    "cache_control": {
                        "type": "ephemeral"
                    }
                },
                {
                    "type": "text",
                    "text": f"Based on the repository context above, please respond to the following:\n\n{content}"
                }
            ]
        }

    async def process_prompt(self, prompt: str, prompt_type: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> PromptResponse:
        try:
            model = model or self.default_model
            system_prompt = self.system_prompts.get(prompt_type, "")
            
            if settings.llm_provider == "openai":
                # Format RAG context for OpenAI
                formatted_dynamic_context_str = ""
                if context:
                    formatted_dynamic_context_str = format_rag_context_for_llm(context)
                
                # Prepend RAG context to the main prompt (conversation history)
                final_prompt_for_template = f"Relevant Context:\n{formatted_dynamic_context_str}\n\nConversation History:\n{prompt}"

                template = PromptTemplate(
                    template=final_prompt_for_template,
                    system_prompt=system_prompt
                )
                
                llm = self._get_openai_llm(model)
                response = await llm.acomplete(template.format())
                
                return PromptResponse(
                    status="success",
                    prompt=response.text,
                    model_used=model
                )
                
            elif settings.llm_provider == "openrouter":
                # Use direct OpenRouter API with cache_control support
                return await self._process_openrouter_with_caching(prompt, system_prompt, context, model)
                
            else:
                return PromptResponse(
                    status="error",
                    error=f"Unsupported LLM provider: {settings.llm_provider}"
                )
            
        except Exception as e:
            print(f"Error in process_prompt: {str(e)}")
            return PromptResponse(
                status="error",
                error=f"Failed to process prompt: {str(e)}"
            )

    async def stream_openrouter_response(self, prompt: str, prompt_type: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream response from OpenRouter API with cache_control support."""
        try:
            # Get model configuration
            model_config = self._get_model_config(model)
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.APP_URL,
                "X-Title": "GH Issue Prompt"
            }
            
            # Prepare messages with system prompt
            messages = [
                {"role": "system", "content": self._get_system_prompt(prompt_type)}
            ]
            
            # Add context if provided
            if context:
                context_str = self._format_context(context)
                messages.append({"role": "system", "content": f"Context:\n{context_str}"})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request body
            request_body = {
                "model": model_config["name"],
                "messages": messages,
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000)
            }
            
            # Add cache control if enabled
            if settings.ENABLE_CACHE_CONTROL:
                request_body["cache_control"] = True
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=request_body
                ) as response:
                    if response.status_code != 200:
                        error_detail = f"OpenRouter API error: {response.status_code}"
                        yield f"data: {json.dumps({'error': error_detail})}\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                return
                            elif data:
                                try:
                                    # Parse and potentially log cache usage
                                    parsed_data = json.loads(data)
                                    if "usage" in parsed_data:
                                        usage = parsed_data["usage"]
                                        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                        if cached_tokens > 0:
                                            print(f"Cache hit: {cached_tokens} tokens cached")
                                    
                                    # Ensure proper UTF-8 encoding
                                    content = parsed_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                    if content:
                                        # Encode and decode to ensure proper UTF-8 handling
                                        content = content.encode('utf-8', errors='ignore').decode('utf-8')
                                        parsed_data['choices'][0]['delta']['content'] = content
                                    
                                    # Forward the chunk
                                    yield f"data: {json.dumps(parsed_data)}\n\n"
                                except json.JSONDecodeError:
                                    # Forward non-JSON data as-is
                                    yield f"data: {data}\n\n"
            
        except Exception as e:
            error_msg = f"Error in OpenRouter streaming with caching: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    async def _process_openrouter_with_caching(self, prompt: str, system_prompt: str, context: Optional[Dict[str, Any]], model: str) -> PromptResponse:
        """Process prompt using OpenRouter API with cache_control support."""
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        
        model_config = self._get_model_config(model)
        
        # Ensure model_config is not None
        if not model_config:
            print(f"Warning: Could not get model config for {model}, using defaults")
            model_config = {
                "max_tokens": 4096,
                "temperature": 0.7,
                "context_window": 32000
            }
        
        # Prepare messages array
        messages = []
        
        # Add system message with potential caching
        if system_prompt:
            if (settings.ENABLE_PROMPT_CACHING and context and context.get("sources") and 
                estimate_tokens(format_rag_context_for_llm(context)) >= settings.PROMPT_CACHE_MIN_TOKENS):
                # Large context - use cache_control in system message
                formatted_rag_context = format_rag_context_for_llm(context)
                rag_tokens = estimate_tokens(formatted_rag_context)
                print(f"Using prompt caching for {rag_tokens} tokens of RAG context in system message")
                
                messages.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{system_prompt}\n\nBelow is the relevant repository context for this conversation:"
                        },
                        {
                            "type": "text",
                            "text": formatted_rag_context,
                            "cache_control": {
                                "type": "ephemeral"
                            }
                        }
                    ]
                })
                # Add user message without RAG context (it's cached in system)
                messages.append({
                    "role": "user", 
                    "content": prompt
                })
            else:
                # Small or no context - traditional format
                if context:
                    formatted_rag_context = format_rag_context_for_llm(context)
                    final_content = f"Relevant Context:\n{formatted_rag_context}\n\nConversation History:\n{prompt}"
                else:
                    final_content = prompt
                    
                messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": final_content})
        else:
            # No system prompt - add context to user message if needed
            user_message = self._format_message_with_cache_control("user", prompt, context)
            messages.append(user_message)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": model_config.get("max_tokens", 4096),
            "temperature": model_config.get("temperature", 0.7),
            "stream": False
        }
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-repo",  # Replace with actual repo
                    "X-Title": "Triage Flow Repository Chat"
                },
                json=payload,
                timeout=120.0
            )
            
            if response.status_code != 200:
                error_detail = f"OpenRouter API error: {response.status_code} - {response.text}"
                print(error_detail)
                return PromptResponse(
                    status="error",
                    error=error_detail
                )
            
            result = response.json()
            
            # Extract response content
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Log cache usage if available
                if "usage" in result:
                    usage = result["usage"]
                    print(f"Token usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                          f"Completion: {usage.get('completion_tokens', 0)}, "
                          f"Cache: {usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)}")
                
                return PromptResponse(
                    status="success",
                    prompt=content,
                    model_used=model,
                    tokens_used=result.get("usage", {})
                )
            else:
                return PromptResponse(
                    status="error",
                    error="No response content received from OpenRouter"
                )
