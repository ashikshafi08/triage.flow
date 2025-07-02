# src/agent_tools/llm_config.py

import logging
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

# Assuming 'settings' will be available in the context where this is called,
# or passed in. For now, let's import it directly if it's a global config.
# If 'settings' is instance-specific, this approach will need adjustment.
try:
    from ..config import settings
except ImportError:
    # Fallback for cases where this might be run in a different context
    # or if settings are passed directly to the function.
    # This part might need to be adapted based on how settings are managed.
    class MockSettings:
        llm_provider = "openai" # Default or placeholder
        openrouter_api_key = None
        openai_api_key = None
        default_model = "gpt-3.5-turbo" # Default or placeholder
    settings = MockSettings()

logger = logging.getLogger(__name__)


def get_llm_instance(llm_provider: str = settings.llm_provider,
                     openrouter_api_key: str = settings.openrouter_api_key,
                     openai_api_key: str = settings.openai_api_key,
                     default_model: str = settings.default_model) -> LLM:
    """Get LLM instance based on settings."""
    
    # All models through OpenRouter support function calling capability
    # as per https://openrouter.ai/docs/features/tool-calling
    # No need to override models for function calling
    
    # Skip validation - let LlamaIndex handle model compatibility internally
    logger.debug(f"Creating LLM instance for model: {default_model}")
    logger.debug(f"Provider: {llm_provider}")
    
    if llm_provider == "openrouter":
        if not openrouter_api_key:
            logger.error("OpenRouter API key is required but not found.")
            raise ValueError("OpenRouter API key is required")
        
        # Use OpenAILike for OpenRouter which properly supports function calling
        # This bypasses LlamaIndex's model validation and uses OpenRouter's OpenAI-compatible API
        logger.info(f"Using OpenAILike with OpenRouter for {default_model} (function calling enabled)")
        
        return OpenAILike(
            api_base="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model=default_model,
            is_chat_model=True,
            is_function_calling_model=True,  # Critical flag for function calling
            context_window=32000,
            max_tokens=4096,
            temperature=0.7,
            timeout=120,
            # OpenRouter-specific headers
            default_headers={
                "HTTP-Referer": "https://github.com/triage-flow",
                "X-Title": "Triage Flow Repository Analysis"
            }
        )
        
    
    # Default to OpenAI if not openrouter or if provider is explicitly openai
    elif llm_provider == "openai":
        if not openai_api_key:
            logger.error("OpenAI API key is required but not found.")
            raise ValueError("OpenAI API key is required")
        
        # Clean the model name if it has openai/ prefix
        clean_model = default_model.replace("openai/", "")
        
        try:
            return OpenAI(
                api_key=openai_api_key,
                model=clean_model,
                max_tokens=4096,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM instance: {e}")
            raise ValueError(f"Failed to initialize OpenAI with model {clean_model}: {e}")
    else:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
