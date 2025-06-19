from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional, ClassVar
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
    # Primary, high-quality model used for final synthesis
    default_model: str = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash-preview-05-20")
    # Cost-efficient model used for iterative reasoning (ReAct steps)
    cheap_model: str = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash-preview-05-20")
    
    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    
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
        },
        "deepseek/deepseek-r1-0528-qwen3-8b": {
            "max_tokens": 4096,
            "temperature": 0.5,
            "context_window": 32768  # DeepSeek R1 context window
        }
    }
    
    # Cache Configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL_RAG: int = int(os.getenv("CACHE_TTL_RAG", "300"))  # 5 minutes
    CACHE_TTL_FOLDER: int = int(os.getenv("CACHE_TTL_FOLDER", "1800"))  # 30 minutes
    CACHE_TTL_RESPONSE: int = int(os.getenv("CACHE_TTL_RESPONSE", "600"))  # 10 minutes
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    MAX_CACHE_MEMORY_MB: int = int(os.getenv("MAX_CACHE_MEMORY_MB", "500"))
    
    # Index Storage Configuration
    INDEX_STORAGE_DIR: str = os.getenv("INDEX_STORAGE_DIR", os.path.expanduser("~/.triage_flow/indices"))
    DISABLE_AUTO_INDEXING: bool = os.getenv("DISABLE_AUTO_INDEXING", "false").lower() == "true"

    # Bot Configuration
    BOT_NAME: str = os.getenv("BOT_NAME", "Triage.Flow Bot")
    BOT_REPO_URL: str = os.getenv("BOT_REPO_URL", "https://github.com/ashikshafi08/triage.flow")
    
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
    # Increased limit for complex analysis like PR reviews
    AGENTIC_MAX_ITERATIONS: int = int(os.getenv("AGENTIC_MAX_ITERATIONS", "12"))
    AGENTIC_DEBUG_MODE: bool = os.getenv("AGENTIC_DEBUG_MODE", "true").lower() == "true"  # Enable detailed agentic logging by default
    FORCE_AGENTIC_APPROACH: bool = os.getenv("FORCE_AGENTIC_APPROACH", "false").lower() == "true"  # Force all queries to use agentic
    
    # Context-Aware Tools Configuration
    ENABLE_CONTEXT_AWARE_TOOLS: bool = os.getenv("ENABLE_CONTEXT_AWARE_TOOLS", "true").lower() == "true"  # Enable enhanced context sharing
    ENABLE_CONTEXT_ENHANCEMENT: bool = os.getenv("ENABLE_CONTEXT_ENHANCEMENT", "true").lower() == "true"  # Enable query context enhancement
    CONTEXT_CACHE_TTL: int = int(os.getenv("CONTEXT_CACHE_TTL", "300"))  # 5 minutes for context cache
    MAX_CONTEXT_EXECUTIONS: int = int(os.getenv("MAX_CONTEXT_EXECUTIONS", "50"))  # Maximum executions to track in context
    
    # Content and Response Limits Configuration
    MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", "10485760"))  # 10MB default (up from 5MB)
    MAX_USER_FILE_CONTENT_CHARS: int = int(os.getenv("MAX_USER_FILE_CONTENT_CHARS", "50000"))  # 50KB default (up from 20KB)
    MAX_CONTENT_PREVIEW_CHARS: int = int(os.getenv("MAX_CONTENT_PREVIEW_CHARS", "5000"))  # 5KB default (up from 2KB)
    MAX_AGENTIC_FILE_SIZE_BYTES: int = int(os.getenv("MAX_AGENTIC_FILE_SIZE_BYTES", "5242880"))  # 5MB for agentic tools (up from 1MB)

    # Issue Processing Configuration
    MAX_ISSUES_TO_PROCESS: Optional[int] = int(os.getenv("MAX_ISSUES_TO_PROCESS", "300"))  
    MAX_PR_TO_PROCESS: Optional[int] = int(os.getenv("MAX_PR_TO_PROCESS", "300"))  
    MAX_PATCH_LINKAGE_ISSUES: int = int(os.getenv("MAX_PATCH_LINKAGE_ISSUES", "100"))  # Conservative limit for diff downloads

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
    
    # Redis Configuration (computed property)
    @property
    def REDIS_CONFIG(self) -> Dict[str, Any]:
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "password": self.redis_password,
            "ssl": self.redis_ssl,
            "decode_responses": True,
        }

    # Chunk Store Configuration
    CHUNK_STORE_CONFIG: ClassVar[Dict[str, Any]] = {
        "store_type": "redis",  # or "memory"
        "chunk_ttl": 3600,  # 1 hour expiry for chunks
        "max_chunk_size": 8192,  # 8KB max chunk size
    }
    
    # Enhanced Persistence Settings
    ENABLE_ENHANCED_PERSISTENCE: bool = True
    PERSISTENCE_BASE_DIR: str = ".index_cache"
    INDEX_REBUILD_THRESHOLD: float = 0.1  # Rebuild if >10% of files changed
    INDEX_REBUILD_MIN_FILES: int = 50  # Or if >50 files changed
    
    # NEW: Composite Agentic Retrieval Settings
    ENABLE_COMPOSITE_RETRIEVAL: bool = os.getenv("ENABLE_COMPOSITE_RETRIEVAL", "true").lower() == "true"
    COMPOSITE_MAX_CONCURRENT_QUERIES: int = int(os.getenv("COMPOSITE_MAX_CONCURRENT_QUERIES", "5"))
    COMPOSITE_MAX_RESULTS_PER_INDEX: int = int(os.getenv("COMPOSITE_MAX_RESULTS_PER_INDEX", "10"))
    COMPOSITE_ENABLE_RERANKING: bool = os.getenv("COMPOSITE_ENABLE_RERANKING", "true").lower() == "true"
    COMPOSITE_CACHE_ROUTING_DECISIONS: bool = os.getenv("COMPOSITE_CACHE_ROUTING_DECISIONS", "true").lower() == "true"
    COMPOSITE_CONFIDENCE_MIN_THRESHOLD: float = float(os.getenv("COMPOSITE_CONFIDENCE_MIN_THRESHOLD", "0.4"))
    COMPOSITE_AGENTIC_THRESHOLD: float = float(os.getenv("COMPOSITE_AGENTIC_THRESHOLD", "0.5"))
    
    class Config:
        env_file = ".env"

settings = Settings() 