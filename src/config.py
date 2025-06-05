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
    MAX_RAG_SOURCES: int = int(os.getenv("MAX_RAG_SOURCES", "50"))
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
    AGENTIC_MAX_ITERATIONS: int = int(os.getenv("AGENTIC_MAX_ITERATIONS", "50"))  # Increased from 30 to 50
    AGENTIC_DEBUG_MODE: bool = os.getenv("AGENTIC_DEBUG_MODE", "false").lower() == "true"  # Enable detailed agentic logging
    FORCE_AGENTIC_APPROACH: bool = os.getenv("FORCE_AGENTIC_APPROACH", "false").lower() == "true"  # Force all queries to use agentic
    
    # Content and Response Limits Configuration
    MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", "10485760"))  # 10MB default (up from 5MB)
    MAX_USER_FILE_CONTENT_CHARS: int = int(os.getenv("MAX_USER_FILE_CONTENT_CHARS", "50000"))  # 50KB default (up from 20KB)
    MAX_CONTENT_PREVIEW_CHARS: int = int(os.getenv("MAX_CONTENT_PREVIEW_CHARS", "5000"))  # 5KB default (up from 2KB)
    MAX_AGENTIC_FILE_SIZE_BYTES: int = int(os.getenv("MAX_AGENTIC_FILE_SIZE_BYTES", "5242880"))  # 5MB for agentic tools (up from 1MB)

    # Token-based limits (more intelligent)
    ENABLE_SMART_TRUNCATION: bool = os.getenv("ENABLE_SMART_TRUNCATION", "true").lower() == "true"
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "200000"))  # Increased context window
    TOKEN_BUFFER_RATIO: float = float(os.getenv("TOKEN_BUFFER_RATIO", "0.8"))  # Use 80% of context window for safety
    
    # Dynamic Content Handling
    ENABLE_DYNAMIC_CONTENT: bool = os.getenv("ENABLE_DYNAMIC_CONTENT", "true").lower() == "true"
    CONTENT_CHUNK_SIZE: int = int(os.getenv("CONTENT_CHUNK_SIZE", "10000"))  # Size of content chunks for processing
    MAX_CHUNKS_PER_REQUEST: int = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "10"))  # Maximum chunks to process at once
    ENABLE_CONTENT_STREAMING: bool = os.getenv("ENABLE_CONTENT_STREAMING", "true").lower() == "true"
    STREAM_CHUNK_SIZE: int = int(os.getenv("STREAM_CHUNK_SIZE", "1000"))  # Size of streaming chunks
    
    class Config:
        env_file = ".env"

settings = Settings() 