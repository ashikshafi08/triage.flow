from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # GitHub Configuration
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # API Configuration
    cache_duration_minutes: int = int(os.getenv("CACHE_DURATION_MINUTES", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    backoff_factor: int = int(os.getenv("BACKOFF_FACTOR", "2"))
    
    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai, openrouter, etc.
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Model Configuration
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    
    # Model-specific configurations
    model_configs: Dict[str, Dict[str, Any]] = {
        "gpt-3.5-turbo": {
            "max_tokens": 2000,
            "temperature": 0.7,
        },
        "gpt-4": {
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        "claude-3-opus": {
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        "claude-3-sonnet": {
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        "mistral-large": {
            "max_tokens": 4000,
            "temperature": 0.7,
        }
    }
    
    class Config:
        env_file = ".env"

settings = Settings() 