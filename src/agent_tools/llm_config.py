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
    
    # Skip validation - let LlamaIndex handle model compatibility internally
    logger.debug(f"Creating LLM instance for model: {default_model}")
    logger.debug(f"Provider: {llm_provider}")
    
    if llm_provider == "openrouter":
        if not openrouter_api_key:
            logger.error("OpenRouter API key is required but not found.")
            raise ValueError("OpenRouter API key is required")
        
        # OpenRouter supports tool calling for most models, especially OpenAI ones
        logger.info(f"Using OpenRouter with model {default_model} for function calling")
        
        try:
            return OpenRouter(
                api_key=openrouter_api_key,
                model=default_model,
                max_tokens=4096,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to create OpenRouter LLM instance: {e}")
            # If the model is an OpenAI model via OpenRouter, provide specific guidance
            if default_model.startswith("openai/"):
                logger.info(f"OpenRouter model {default_model} failed to initialize. This might be a temporary API issue.")
                # Don't raise here, let the calling code handle it
                raise ValueError(f"Failed to initialize OpenRouter with model {default_model}: {e}")
            else:
                raise
    
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
