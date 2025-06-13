"""
Migration utility to transition from old cache_manager.py to new Redis cache system
"""
import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class CacheMigration:
    """Utility to migrate cache data and update imports"""
    
    def __init__(self):
        self.migration_log: List[str] = []
    
    async def migrate_existing_caches(self):
        """Migrate data from old cache system if it exists"""
        try:
            # Try to import old cache system
            from ..cache_manager import rag_cache as old_rag_cache
            from ..cache_manager import response_cache as old_response_cache
            from ..cache_manager import folder_cache as old_folder_cache
            
            # Import new cache system
            from . import rag_cache, response_cache, folder_cache
            
            migration_stats = {
                "rag_migrated": 0,
                "response_migrated": 0, 
                "folder_migrated": 0,
                "errors": []
            }
            
            # Migrate RAG cache
            if hasattr(old_rag_cache, 'cache') and old_rag_cache.cache:
                for key, entry in old_rag_cache.cache.items():
                    try:
                        if not entry.is_expired():
                            remaining_ttl = max(60, int(entry.ttl - (time.time() - entry.timestamp)))
                            await rag_cache.set(key, entry.value, remaining_ttl)
                            migration_stats["rag_migrated"] += 1
                    except Exception as e:
                        migration_stats["errors"].append(f"RAG key {key}: {e}")
            
            # Migrate Response cache
            if hasattr(old_response_cache, 'cache') and old_response_cache.cache:
                for key, entry in old_response_cache.cache.items():
                    try:
                        if not entry.is_expired():
                            remaining_ttl = max(60, int(entry.ttl - (time.time() - entry.timestamp)))
                            await response_cache.set(key, entry.value, remaining_ttl)
                            migration_stats["response_migrated"] += 1
                    except Exception as e:
                        migration_stats["errors"].append(f"Response key {key}: {e}")
            
            # Migrate Folder cache
            if hasattr(old_folder_cache, 'cache') and old_folder_cache.cache:
                for key, entry in old_folder_cache.cache.items():
                    try:
                        if not entry.is_expired():
                            remaining_ttl = max(60, int(entry.ttl - (time.time() - entry.timestamp)))
                            await folder_cache.set(key, entry.value, remaining_ttl)
                            migration_stats["folder_migrated"] += 1
                    except Exception as e:
                        migration_stats["errors"].append(f"Folder key {key}: {e}")
            
            logger.info(f"Cache migration completed: {migration_stats}")
            return migration_stats
            
        except ImportError:
            logger.info("No old cache system found, starting fresh")
            return {"status": "fresh_start"}
        except Exception as e:
            logger.error(f"Cache migration failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_import_replacement_guide(self) -> Dict[str, str]:
        """Get guide for updating imports in your codebase"""
        return {
            "old_imports": [
                "from .cache_manager import rag_cache, response_cache, folder_cache",
                "from .cache_manager import cache_rag_result, cache_response", 
                "from .cache_manager import cleanup_caches_periodically"
            ],
            "new_imports": [
                "from .cache import rag_cache, response_cache, folder_cache, issue_cache",
                "from .cache import cache_rag_result, cache_response",
                "from .cache import cleanup_caches_periodically, initialize_redis_cache"
            ],
            "additional_features": [
                "# New Redis-specific features:",
                "from .cache import CacheLayer, redis_manager",
                "# Use cache layers: CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS",
                "# Manual cache control: await rag_cache.invalidate_pattern('query_*')"
            ]
        }

# Global migration instance
cache_migration = CacheMigration() 