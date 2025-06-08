import redis
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import json
import hashlib
import logging
from src.config import settings
import uuid

logger = logging.getLogger(__name__)

class ChunkStore(ABC):
    """Abstract base class for chunk storage"""
    @abstractmethod
    def store(self, content: str) -> str:
        """Store content and return chunk ID"""
        pass
    
    @abstractmethod
    def retrieve(self, chunk_id: str) -> Optional[str]:
        """Retrieve content by chunk ID"""
        pass
    
    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by ID"""
        pass

class InMemoryChunkStore(ChunkStore):
    """In-memory implementation of chunk storage"""
    def __init__(self):
        self._store: Dict[str, str] = {}
        self.max_size = settings.CHUNK_STORE_CONFIG["max_chunk_size"]
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate a unique chunk ID based on content hash"""
        return f"chunk_{hashlib.md5(content.encode()).hexdigest()}"
    
    def store(self, content: str) -> str:
        """Store content in memory and return chunk ID"""
        if len(content) > self.max_size:
            # If content is too large, store it in chunks
            chunks = [content[i:i + self.max_size] for i in range(0, len(content), self.max_size)]
            chunk_ids = []
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(chunk)
                self._store[chunk_id] = chunk
                chunk_ids.append(chunk_id)
            # Store chunk IDs as a list
            meta_id = self._generate_chunk_id(content)
            self._store[meta_id] = json.dumps({"type": "chunked", "chunks": chunk_ids})
            return meta_id
        else:
            # Store single chunk
            chunk_id = self._generate_chunk_id(content)
            self._store[chunk_id] = content
            return chunk_id
    
    def retrieve(self, chunk_id: str) -> Optional[str]:
        """Retrieve content from memory by chunk ID"""
        content = self._store.get(chunk_id)
        if not content:
            return None
        
        try:
            # Check if this is a chunked content
            meta = json.loads(content)
            if meta.get("type") == "chunked":
                # Retrieve and combine chunks
                chunks = []
                for sub_chunk_id in meta["chunks"]:
                    sub_content = self._store.get(sub_chunk_id)
                    if sub_content:
                        chunks.append(sub_content)
                return "".join(chunks)
        except json.JSONDecodeError:
            # Not a chunked content, return as is
            pass
        
        return content
    
    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk and its sub-chunks if any"""
        content = self._store.get(chunk_id)
        if not content:
            return False
        
        try:
            # Check if this is a chunked content
            meta = json.loads(content)
            if meta.get("type") == "chunked":
                # Delete all sub-chunks
                for sub_chunk_id in meta["chunks"]:
                    self._store.pop(sub_chunk_id, None)
        except json.JSONDecodeError:
            pass
        
        # Delete the main chunk
        return bool(self._store.pop(chunk_id, None))

class RedisChunkStore(ChunkStore):
    """Redis-based implementation of chunk storage"""
    def __init__(self, config: dict):
        self.redis = None
        self.config = config
        try:
            self.redis = redis.StrictRedis(**config)
            self.redis.ping()
        except Exception as e:
            logging.warning(f"Redis unavailable: {e}. Falling back to in-memory chunk store (non-persistent).")
            self.redis = None
        self.ttl = settings.CHUNK_STORE_CONFIG["chunk_ttl"]
        self.max_size = settings.CHUNK_STORE_CONFIG["max_chunk_size"]
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate a unique chunk ID based on content hash"""
        return f"chunk_{hashlib.md5(content.encode()).hexdigest()}"
    
    def store(self, content: str) -> str:
        """Store content in Redis and return chunk ID"""
        try:
            if len(content) > self.max_size:
                # If content is too large, store it in chunks
                chunks = [content[i:i + self.max_size] for i in range(0, len(content), self.max_size)]
                chunk_ids = []
                for chunk in chunks:
                    chunk_id = self._generate_chunk_id(chunk)
                    self.redis.setex(chunk_id, self.ttl, chunk)
                    chunk_ids.append(chunk_id)
                # Store chunk IDs as a list
                meta_id = self._generate_chunk_id(content)
                self.redis.setex(meta_id, self.ttl, json.dumps({"type": "chunked", "chunks": chunk_ids}))
                return meta_id
            else:
                # Store single chunk
                chunk_id = self._generate_chunk_id(content)
                self.redis.setex(chunk_id, self.ttl, content)
                return chunk_id
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis error during store: {e}")
            raise
    
    def retrieve(self, chunk_id: str) -> Optional[str]:
        """Retrieve content from Redis by chunk ID"""
        try:
            content = self.redis.get(chunk_id)
            if not content:
                return None
            
            try:
                # Check if this is a chunked content
                meta = json.loads(content)
                if meta.get("type") == "chunked":
                    # Retrieve and combine chunks
                    chunks = []
                    for sub_chunk_id in meta["chunks"]:
                        sub_content = self.redis.get(sub_chunk_id)
                        if sub_content:
                            chunks.append(sub_content)
                    return "".join(chunks)
            except json.JSONDecodeError:
                # Not a chunked content, return as is
                pass
            
            return content
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis error during retrieve: {e}")
            raise
    
    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk and its sub-chunks if any"""
        try:
            content = self.redis.get(chunk_id)
            if not content:
                return False
            
            try:
                # Check if this is a chunked content
                meta = json.loads(content)
                if meta.get("type") == "chunked":
                    # Delete all sub-chunks
                    for sub_chunk_id in meta["chunks"]:
                        self.redis.delete(sub_chunk_id)
            except json.JSONDecodeError:
                pass
            
            # Delete the main chunk
            return bool(self.redis.delete(chunk_id))
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis error during delete: {e}")
            raise

class ChunkStoreFactory:
    """Factory for creating chunk stores"""
    _instance = None
    
    @classmethod
    def get_instance(cls) -> ChunkStore:
        if cls._instance is None:
            store_type = settings.CHUNK_STORE_CONFIG["store_type"]
            if store_type == "redis":
                try:
                    cls._instance = RedisChunkStore(settings.REDIS_CONFIG)
                    logger.info("Using Redis chunk store")
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    logger.warning(f"Failed to initialize Redis chunk store: {e}. Falling back to in-memory store.")
                    cls._instance = InMemoryChunkStore()
            else:
                cls._instance = InMemoryChunkStore()
                logger.info("Using in-memory chunk store")
        return cls._instance
