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
        # Token budget guard - estimate tokens and limit sources if too many
        estimated_tokens = sum(estimate_tokens(part) for part in context_parts)
        max_sources = 15
        
        # If file-oriented search, prioritize showing more files with less content
        if search_type == "file_oriented":
            max_sources = 25
            content_preview_length = 150
        else:
            content_preview_length = 300
        
        # Estimate tokens from sources and reduce if needed
        for source in sources[:max_sources]:
            estimated_tokens += estimate_tokens(source.get('content', ''))
            if estimated_tokens > 12000:  # Conservative limit for 16k context
                max_sources = max(5, max_sources - 5)
                content_preview_length = max(100, content_preview_length - 50)
                break
        
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
                
                # Final token budget check - ensure total doesn't exceed limit
                system_tokens = estimate_tokens(system_prompt)
                context_tokens = estimate_tokens(formatted_dynamic_context_str)
                prompt_tokens = estimate_tokens(prompt)
                total_tokens = system_tokens + context_tokens + prompt_tokens
                
                # If we're over the limit, trim the context
                if total_tokens > 15500:  # Conservative limit for 16k context
                    print(f"Token budget exceeded ({total_tokens}), trimming context...")
                    # Reduce context by trimming sources or shortening previews
                    if context and context.get("sources"):
                        sources = context["sources"]
                        # Try reducing to fewer sources first
                        while len(sources) > 3 and total_tokens > 15500:
                            sources = sources[:-1]  # Remove last source
                            temp_context = {**context, "sources": sources}
                            temp_formatted = format_rag_context_for_llm(temp_context)
                            total_tokens = system_tokens + estimate_tokens(temp_formatted) + prompt_tokens
                        
                        # If still too long, shorten content previews
                        if total_tokens > 15500:
                            for source in sources:
                                if "content" in source and len(source["content"]) > 100:
                                    source["content"] = source["content"][:100] + "..."
                            temp_context = {**context, "sources": sources}
                            formatted_dynamic_context_str = format_rag_context_for_llm(temp_context)
                        else:
                            formatted_dynamic_context_str = format_rag_context_for_llm({**context, "sources": sources})
                
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
                # Format RAG context for OpenRouter (same as OpenAI approach)
                formatted_dynamic_context_str = ""
                if context:
                    formatted_dynamic_context_str = format_rag_context_for_llm(context)
                
                # Apply same token budget logic as OpenAI
                system_tokens = estimate_tokens(system_prompt)
                context_tokens = estimate_tokens(formatted_dynamic_context_str)
                prompt_tokens = estimate_tokens(prompt)
                total_tokens = system_tokens + context_tokens + prompt_tokens
                
                # If we're over the limit, trim the context
                if total_tokens > 15500:  # Conservative limit for 16k context
                    print(f"Token budget exceeded ({total_tokens}), trimming context...")
                    # Reduce context by trimming sources or shortening previews
                    if context and context.get("sources"):
                        sources = context["sources"]
                        # Try reducing to fewer sources first
                        while len(sources) > 3 and total_tokens > 15500:
                            sources = sources[:-1]  # Remove last source
                            temp_context = {**context, "sources": sources}
                            temp_formatted = format_rag_context_for_llm(temp_context)
                            total_tokens = system_tokens + estimate_tokens(temp_formatted) + prompt_tokens
                        
                        # If still too long, shorten content previews
                        if total_tokens > 15500:
                            for source in sources:
                                if "content" in source and len(source["content"]) > 100:
                                    source["content"] = source["content"][:100] + "..."
                            temp_context = {**context, "sources": sources}
                            formatted_dynamic_context_str = format_rag_context_for_llm(temp_context)
                        else:
                            formatted_dynamic_context_str = format_rag_context_for_llm({**context, "sources": sources})
                
                # Prepend RAG context to the main prompt (conversation history)
                final_prompt_for_template = f"Relevant Context:\n{formatted_dynamic_context_str}\n\nConversation History:\n{prompt}"

                template = PromptTemplate(
                    template=final_prompt_for_template, # Now includes RAG context + conversation
                    system_prompt=system_prompt 
                )
                
                llm = self._get_openrouter_llm(model)
                response = await llm.acomplete(template.format())
                
                return PromptResponse(
                    status="success",
                    prompt=response.text,
                    model_used=model
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
        """Stream response from OpenRouter API using LlamaIndex."""
        model = model or self.default_model
        system_prompt = self.system_prompts.get(prompt_type, "")

        # Format RAG context (same logic as non-streaming)
        formatted_dynamic_context_str = ""
        if context:
            formatted_dynamic_context_str = format_rag_context_for_llm(context)
        
        # Apply token budget check for streaming too
        system_tokens = estimate_tokens(system_prompt)
        context_tokens = estimate_tokens(formatted_dynamic_context_str)
        prompt_tokens = estimate_tokens(prompt)
        total_tokens = system_tokens + context_tokens + prompt_tokens
        
        if total_tokens > 15500:  # Conservative limit for 16k context
            print(f"Token budget exceeded ({total_tokens}), trimming context...")
            if context and context.get("sources"):
                sources = context["sources"]
                # Try reducing to fewer sources first
                while len(sources) > 3 and total_tokens > 15500:
                    sources = sources[:-1]  # Remove last source
                    temp_context = {**context, "sources": sources}
                    temp_formatted = format_rag_context_for_llm(temp_context)
                    total_tokens = system_tokens + estimate_tokens(temp_formatted) + prompt_tokens
                
                # If still too long, shorten content previews
                if total_tokens > 15500:
                    for source in sources:
                        if "content" in source and len(source["content"]) > 100:
                            source["content"] = source["content"][:100] + "..."
                    temp_context = {**context, "sources": sources}
                    formatted_dynamic_context_str = format_rag_context_for_llm(temp_context)
                else:
                    formatted_dynamic_context_str = format_rag_context_for_llm({**context, "sources": sources})
        
        # Prepend RAG context to the main prompt (conversation history)
        final_prompt_for_template = f"Relevant Context:\n{formatted_dynamic_context_str}\n\nConversation History:\n{prompt}"

        template = PromptTemplate(
            template=final_prompt_for_template,
            system_prompt=system_prompt 
        )
        
        try:
            llm = self._get_openrouter_llm(model)
            response_stream = await llm.astream_complete(template.format())
            
            async for chunk in response_stream:
                # Format as SSE for frontend consumption
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.delta}}]})}\n\n"
                
            # Send final done message
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"Error in OpenRouter streaming: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
