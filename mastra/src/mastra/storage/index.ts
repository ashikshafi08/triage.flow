/**
 * Storage Module
 *
 * LibSQL storage utilities for sessions, embeddings, and caching.
 * Replaces Python enhanced_persistence.py and redis_cache_manager.py.
 *
 * @module storage
 */

// Configuration
export {
  loadStorageConfig,
  getStorageConfig,
  VECTOR_INDEXES,
  type StorageConfig,
  type VectorIndexName,
} from "./config";

// Types
export {
  type Session,
  type SearchResult,
  type CreateSessionParams,
  type UpdateSessionParams,
  type FileChecksum,
  type FileInfo,
  type ChangeSet,
  type CacheEntry,
  CACHE_NAMESPACES,
  type CacheNamespace,
  type EmbeddingMetadata,
  type StoreEmbeddingParams,
  type StoreEmbeddingsParams,
  type VectorSearchParams,
  type VectorSearchResult,
  type VectorIndexStats,
} from "./types";

// Session Storage (sessions, checksums, cache)
export { SessionStorage, getSessionStorage } from "./sessionStorage";

// Vector Store (embeddings, similarity search)
export { VectorStore, getVectorStore } from "./vectorStore";
