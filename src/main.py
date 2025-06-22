from fastapi import FastAPI, HTTPException, Query
import nest_asyncio
import asyncio
import logging
from typing import Optional
from .api.middleware import setup_cors
from .api.dependencies import session_manager
from .cache import cleanup_caches_periodically, initialize_redis_cache
from .chunk_store import ChunkStoreFactory

# Import routers
from .api.routers import chat, sessions, repository, issues, timeline, agentic

# Enable nested event loops for Jupyter notebooks
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.2.0"
)

# Setup middleware
setup_cors(app)

# Include routers
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(repository.router)
app.include_router(issues.router)
app.include_router(timeline.router)
app.include_router(agentic.router)

# Background task to clean up old sessions
async def cleanup_sessions_periodically():
    while True:
        await session_manager.cleanup_sessions()
        await asyncio.sleep(600)  # Clean up every 10 minutes

@app.on_event("startup")
async def startup_event():
    # Initialize Redis cache system
    await initialize_redis_cache()
    
    # Start background tasks
    asyncio.create_task(cleanup_sessions_periodically())
    asyncio.create_task(cleanup_caches_periodically())
    
    # Initialize chunk store
    ChunkStoreFactory.get_instance()

@app.get("/")
async def root():
    return {"message": "GH Issue Prompt API"}

@app.get("/cache-stats")
async def get_cache_statistics():
    """Get enhanced cache statistics for monitoring performance"""
    from .cache import rag_cache, response_cache, folder_cache, issue_cache
    from .config import settings
    
    return {
        "rag_cache": rag_cache.get_stats(),
        "response_cache": response_cache.get_stats(),
        "folder_cache": folder_cache.get_stats(),
        "issue_cache": issue_cache.get_stats(),
        "cache_enabled": settings.CACHE_ENABLED,
        "redis_url": settings.REDIS_URL,
        "feature_flags": {
            "rag_caching": settings.ENABLE_RAG_CACHING,
            "response_caching": settings.ENABLE_RESPONSE_CACHING,
            "smart_sizing": settings.ENABLE_SMART_SIZING,
            "repo_summaries": settings.ENABLE_REPO_SUMMARIES,
            "prompt_caching": settings.ENABLE_PROMPT_CACHING
        },
        "prompt_caching": {
            "enabled": settings.ENABLE_PROMPT_CACHING,
            "min_tokens": settings.PROMPT_CACHE_MIN_TOKENS,
            "provider": settings.llm_provider
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup Redis connections on shutdown"""
    from .cache import redis_manager
    await redis_manager.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
