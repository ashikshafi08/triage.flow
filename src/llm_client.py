from typing import Optional, Dict, Any, AsyncGenerator
import httpx
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.prompts import PromptTemplate
from .config import settings
from .models import PromptResponse
import json

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

    sources = rag_data.get("sources")
    search_type = rag_data.get("search_type", "regular")
    
    if sources:
        # Set max sources without token-based limiting
        max_sources = 15
        
        # If file-oriented search, prioritize showing more files
        if search_type == "file_oriented":
            max_sources = 25
            content_preview_length = 300
        else:
            content_preview_length = 500
        
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
            user_files_str += f"\nFILE: {file['file']}\nCONTENT:\n{file['content'][:1000]}...\n" + "-"*30 + "\n"
        context_parts.append(user_files_str)
    
    if not context_parts:
        return "No specific RAG context was retrieved for this query."
        
    return "\n\n".join(context_parts)

class LLMClient:
    def __init__(self):
        self.default_model = settings.default_model
        
        # Define system prompts for different types
        self.system_prompts = {
            "explain": """You are an expert software engineer. Your task is to explain GitHub issues clearly, concisely, and technically.
            Begin with a brief summary, then elaborate on the problem, its root cause, and potential impact.
            When relevant, refer to the specific repository details provided in the context.
            
            CRITICAL: When referencing files, use ONLY the exact file paths provided in the retrieved context. 
            Never invent, guess, or assume file paths. If a file path is not explicitly provided in the context, say so clearly.""",
            
            "fix": """You are an expert software engineer. Your task is to provide detailed yet concise solutions for GitHub issues.
            Include essential code changes, necessary tests, and consider edge cases.
            When relevant, refer to the specific repository details provided in the context.
            
            CRITICAL: When referencing files, use ONLY the exact file paths provided in the retrieved context. 
            Never invent, guess, or assume file paths. If a file path is not explicitly provided in the context, say so clearly.""",
            
            "test": """You are an expert software engineer. Your task is to create comprehensive and focused test cases for GitHub issues.
            Focus on verifying the issue and validating potential fixes efficiently.
            When relevant, refer to the specific repository details provided in the context.
            
            CRITICAL: When referencing files, use ONLY the exact file paths provided in the retrieved context. 
            Never invent, guess, or assume file paths. If a file path is not explicitly provided in the context, say so clearly.""",
            
            "summarize": """You are an expert software engineer. Your task is to summarize GitHub issues concisely and accurately.
            Focus on key points, current status, and actionable next steps.
            When relevant, refer to the specific repository details provided in the context.
            
            CRITICAL: When referencing files, use ONLY the exact file paths provided in the retrieved context. 
            Never invent, guess, or assume file paths. If a file path is not explicitly provided in the context, say so clearly."""
            # Add similar modifications for other prompt types if they exist (document, review, prioritize from prompt_generator.py)
        }

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return settings.model_configs.get(model, settings.model_configs[self.default_model])

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
        model = model or self.default_model
        system_prompt = self.system_prompts.get(prompt_type, "")
        
        if not settings.openrouter_api_key:
            yield f"data: {json.dumps({'error': 'OpenRouter API key not configured'})}\n\n"
            return
        
        model_config = self._get_model_config(model)
        
        # Prepare messages array with caching support
        messages = []
        
        # Add system message with potential caching
        if system_prompt:
            if (settings.ENABLE_PROMPT_CACHING and context and context.get("sources") and 
                estimate_tokens(format_rag_context_for_llm(context)) >= settings.PROMPT_CACHE_MIN_TOKENS):
                # Large context - use cache_control in system message
                formatted_rag_context = format_rag_context_for_llm(context)
                rag_tokens = estimate_tokens(formatted_rag_context)
                print(f"Using prompt caching for {rag_tokens} tokens of RAG context in streaming")
                
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
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/your-repo",  # Replace with actual repo
                        "X-Title": "Triage Flow Repository Chat"
                    },
                    json=payload,
                    timeout=120.0
                ) as response:
                    
                    if response.status_code != 200:
                        error_detail = f"OpenRouter API error: {response.status_code}"
                        yield f"data: {json.dumps({'error': error_detail})}\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                # Include final usage info if available
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
                                    
                                    # Forward the chunk
                                    yield f"data: {data}\n\n"
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
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
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
