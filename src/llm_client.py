from typing import Optional, Dict, Any, AsyncGenerator
import httpx
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from .config import settings
from .models import PromptResponse
import json

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
    if sources:
        sources_str = "EXACT FILE PATHS AND CONTENT FROM REPOSITORY:\n"
        sources_str += "=" * 50 + "\n"
        
        for i, source in enumerate(sources[:15], 1):  # Limit to top 5 for better focus
            file_path = source.get('file', 'UNKNOWN_FILE')
            language = source.get('language', 'unknown')
            
            sources_str += f"\n{i}. FILE: {file_path}\n"
            if language and language != 'unknown':
                sources_str += f"   Language: {language}\n"
            sources_str += f"   Content Preview:\n"
            
            # Add a preview of the content
            content = source.get('content', '')
            if content:
                # Show first 300 characters to give context without overwhelming
                preview = content[:300].strip()
                if len(content) > 300:
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

    async def _get_openrouter_response(self, prompt: str, model: str, system_prompt: str, dynamic_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get response from OpenRouter API, now with dynamic_context."""
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  # Consider making this dynamic if needed
            "X-Title": "GH Issue Prompt"
        }

        model_config = self._get_model_config(model)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if dynamic_context:
            formatted_dynamic_context_str = format_rag_context_for_llm(dynamic_context)
            # Add formatted RAG context as a system message before the user's prompt history
            messages.append({"role": "system", "name": "retrieved_context", "content": formatted_dynamic_context_str})
            
        messages.append({"role": "user", "content": prompt}) # prompt is the conversation history

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": model_config["max_tokens"],
                        "temperature": model_config["temperature"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract the response and token usage
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "tokens_used": data.get("usage", {}).get("total_tokens")
                }
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP Status Error: {e.response.status_code} - {e.response.text}"
            print(error_detail)
            raise ValueError(error_detail) from e
        except httpx.RequestError as e:
            error_detail = f"HTTP Request Error: An error occurred while requesting {e.request.url!r} - {str(e)}"
            print(error_detail)
            raise ValueError(error_detail) from e
        except Exception as e:
            error_detail = f"Unexpected error in OpenRouter response: {str(e)}"
            print(error_detail)
            raise ValueError(error_detail) from e

    async def process_prompt(self, prompt: str, prompt_type: str, context: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> PromptResponse:
        try:
            model = model or self.default_model
            system_prompt = self.system_prompts.get(prompt_type, "") # Base system prompt
            
            # The 'context' parameter is our fresh_rag_context from main.py
            
            if settings.llm_provider == "openai":
                # Format RAG context for OpenAI
                formatted_dynamic_context_str = ""
                if context:
                    formatted_dynamic_context_str = format_rag_context_for_llm(context)
                
                # Prepend RAG context to the main prompt (conversation history)
                # The system_prompt from self.system_prompts is handled by PromptTemplate's system_prompt arg
                final_prompt_for_template = f"Relevant Context:\n{formatted_dynamic_context_str}\n\nConversation History:\n{prompt}"

                template = PromptTemplate(
                    template=final_prompt_for_template, # Now includes RAG context + conversation
                    system_prompt=system_prompt 
                )
                
                llm = self._get_openai_llm(model)
                response = await llm.acomplete(template.format()) # .format() is for template variables if any
                
                return PromptResponse(
                    status="success",
                    prompt=response.text,
                    model_used=model
                )
                
            elif settings.llm_provider == "openrouter":
                # Pass the raw context dictionary to _get_openrouter_response, it will format it
                response_data = await self._get_openrouter_response(prompt, model, system_prompt, dynamic_context=context)
                return PromptResponse(
                    status="success",
                    prompt=response_data["text"],
                    model_used=model,
                    tokens_used=response_data.get("tokens_used")
                )
                
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
        """Stream response from OpenRouter API."""
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")

        model = model or self.default_model
        system_prompt = self.system_prompts.get(prompt_type, "")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "GH Issue Prompt",
            "Content-Type": "application/json"
        }

        model_config = self._get_model_config(model)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            formatted_dynamic_context_str = format_rag_context_for_llm(context)
            messages.append({"role": "system", "name": "retrieved_context", "content": formatted_dynamic_context_str})
            
        messages.append({"role": "user", "content": prompt})

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
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            if data == "[DONE]":
                                break
                                
                            try:
                                json_data = json.loads(data)
                                if "choices" in json_data and len(json_data["choices"]) > 0:
                                    delta = json_data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        # Format as SSE for frontend consumption
                                        yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                                
                    # Send final done message
                    yield "data: [DONE]\n\n"
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP Status Error: {e.response.status_code} - {e.response.text}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
        except httpx.RequestError as e:
            error_msg = f"HTTP Request Error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
