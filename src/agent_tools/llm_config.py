# src/agent_tools/llm_config.py

import logging
from llama_index.core.llms import LLM
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai import OpenAI

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
    if llm_provider == "openrouter":
        if not openrouter_api_key:
            logger.error("OpenRouter API key is required but not found.")
            raise ValueError("OpenRouter API key is required")
        return OpenRouter(
            api_key=openrouter_api_key,
            model=default_model,
            max_tokens=4096,
            temperature=0.7
        )
    # Default to OpenAI if not openrouter or if provider is explicitly openai
    elif llm_provider == "openai":
        if not openai_api_key:
            logger.error("OpenAI API key is required but not found.")
            raise ValueError("OpenAI API key is required")
        return OpenAI(
            api_key=openai_api_key,
            model=default_model,
            max_tokens=4096,
            temperature=0.7
        )
    else:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
