from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN")
    
    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")  # "openai" or "openrouter"
    default_model: str = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash-preview-05-20")
    
    # Model configurations
    model_configs: Dict[str, Any] = {
        "gpt-4": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 8192
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 4096
        },
        "anthropic/claude-3.5-sonnet": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 200000
        },
        "anthropic/claude-3-opus": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 200000
        },
        "google/gemini-2.5-flash-preview-05-20": {
            "max_tokens": 8192,
            "temperature": 0.7,
            "context_window": 1000000  # Gemini 2.5 Flash has 1M context window
        },
        "meta-llama/llama-3.1-70b-instruct": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 131072
        }
    }
    
    # Cache Configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL_RAG: int = int(os.getenv("CACHE_TTL_RAG", "300"))  # 5 minutes
    CACHE_TTL_FOLDER: int = int(os.getenv("CACHE_TTL_FOLDER", "1800"))  # 30 minutes
    CACHE_TTL_RESPONSE: int = int(os.getenv("CACHE_TTL_RESPONSE", "600"))  # 10 minutes
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    MAX_CACHE_MEMORY_MB: int = int(os.getenv("MAX_CACHE_MEMORY_MB", "500"))
    
    # Feature Flags
    ENABLE_RAG_CACHING: bool = os.getenv("ENABLE_RAG_CACHING", "true").lower() == "true"
    ENABLE_RESPONSE_CACHING: bool = os.getenv("ENABLE_RESPONSE_CACHING", "true").lower() == "true"
    ENABLE_SMART_SIZING: bool = os.getenv("ENABLE_SMART_SIZING", "true").lower() == "true"
    ENABLE_REPO_SUMMARIES: bool = os.getenv("ENABLE_REPO_SUMMARIES", "true").lower() == "true"
    ENABLE_ASYNC_RAG: bool = os.getenv("ENABLE_ASYNC_RAG", "true").lower() == "true"
    ENABLE_PROMPT_CACHING: bool = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
    
    # Smart Sizing Configuration
    MIN_RAG_SOURCES: int = int(os.getenv("MIN_RAG_SOURCES", "10"))
    MAX_RAG_SOURCES: int = int(os.getenv("MAX_RAG_SOURCES", "25"))
    DEFAULT_RAG_SOURCES: int = int(os.getenv("DEFAULT_RAG_SOURCES", "15"))
    
    # Query Complexity Thresholds
    SIMPLE_QUERY_WORD_LIMIT: int = int(os.getenv("SIMPLE_QUERY_WORD_LIMIT", "10"))
    COMPLEX_QUERY_WORD_THRESHOLD: int = int(os.getenv("COMPLEX_QUERY_WORD_THRESHOLD", "30"))
    FILE_MENTION_WEIGHT: int = int(os.getenv("FILE_MENTION_WEIGHT", "5"))  # Each @file adds this to complexity
    
    # Performance Settings
    ASYNC_BATCH_SIZE: int = int(os.getenv("ASYNC_BATCH_SIZE", "10"))
    PARALLEL_FILE_PROCESSING: bool = os.getenv("PARALLEL_FILE_PROCESSING", "true").lower() == "true"
    MAX_CONCURRENT_OPERATIONS: int = int(os.getenv("MAX_CONCURRENT_OPERATIONS", "20"))
    
    # Prompt Caching Configuration
    PROMPT_CACHE_MIN_TOKENS: int = int(os.getenv("PROMPT_CACHE_MIN_TOKENS", "1000"))  # Minimum tokens to enable caching
    
    # Agentic System Configuration
    AGENTIC_MAX_ITERATIONS: int = int(os.getenv("AGENTIC_MAX_ITERATIONS", "10"))  # Maximum steps for agentic exploration
    
    class Config:
        env_file = ".env"

settings = Settings() 