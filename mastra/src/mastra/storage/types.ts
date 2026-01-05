/**
 * Storage Types
 *
 * TypeScript interfaces for storage operations.
 * Extends the shared types with storage-specific interfaces.
 *
 * @module storage/types
 */

import type { Session, SearchResult } from "../types";

// Re-export shared types for convenience
export type { Session, SearchResult };

/**
 * Parameters for creating a new session.
 */
export interface CreateSessionParams {
  repoUrl: string;
  repoOwner: string;
  repoName: string;
  branch?: string;
  embeddingModel?: string;
}

/**
 * Session update parameters (all fields optional).
 */
export type UpdateSessionParams = Partial<
  Pick<Session, "status" | "totalFiles" | "totalChunks">
> & {
  metadata?: Record<string, unknown>;
};

/**
 * File checksum for incremental indexing.
 * Tracks file state to detect changes.
 */
export interface FileChecksum {
  sessionId: string;
  filePath: string;
  fileSize: number;
  fileMtime: number;
  checksum: string;
  createdAt: Date;
}

/**
 * File info from filesystem scan.
 */
export interface FileInfo {
  path: string;
  size: number;
  mtime: number;
}

/**
 * Change detection result.
 */
export interface ChangeSet {
  added: string[];
  modified: string[];
  deleted: string[];
  unchanged: string[];
}

/**
 * Cache entry with TTL.
 */
export interface CacheEntry<T = unknown> {
  key: string;
  namespace: string;
  value: T;
  ttlSeconds: number;
  createdAt: Date;
  expiresAt: Date;
}

/**
 * Cache namespaces for different data types.
 */
export const CACHE_NAMESPACES = {
  RAG: "rag",
  FOLDER: "folder",
  RESPONSE: "response",
  EMBEDDING: "embedding",
} as const;

export type CacheNamespace =
  (typeof CACHE_NAMESPACES)[keyof typeof CACHE_NAMESPACES];

/**
 * Vector embedding metadata stored alongside vectors.
 */
export interface EmbeddingMetadata {
  sessionId: string;
  filePath: string;
  language: string;
  chunkIndex: number;
  startLine?: number;
  endLine?: number;
  content: string;
  [key: string]: unknown;
}

/**
 * Parameters for storing embeddings.
 */
export interface StoreEmbeddingParams {
  indexName: string;
  id?: string;
  vector: number[];
  metadata: EmbeddingMetadata;
}

/**
 * Parameters for batch embedding storage.
 */
export interface StoreEmbeddingsParams {
  indexName: string;
  vectors: number[][];
  metadata: EmbeddingMetadata[];
  ids?: string[];
}

/**
 * Parameters for vector similarity search.
 */
export interface VectorSearchParams {
  indexName: string;
  queryVector: number[];
  topK?: number;
  minScore?: number;
  filter?: Record<string, unknown>;
  includeVector?: boolean;
}

/**
 * Vector search result from LibSQLVector.
 */
export interface VectorSearchResult {
  id: string;
  score: number;
  metadata: EmbeddingMetadata;
  vector?: number[];
}

/**
 * Vector index statistics.
 */
export interface VectorIndexStats {
  dimension: number;
  count: number;
  metric: "cosine" | "euclidean" | "dotproduct";
}

/**
 * Database row types for session storage.
 */
export interface SessionRow {
  id: string;
  repo_url: string;
  repo_owner: string;
  repo_name: string;
  branch: string;
  status: string;
  total_files: number;
  total_chunks: number;
  embedding_model: string;
  created_at: string;
  updated_at: string;
  metadata: string | null;
}

/**
 * Database row types for file checksums.
 */
export interface FileChecksumRow {
  id: number;
  session_id: string;
  file_path: string;
  file_size: number;
  file_mtime: number;
  checksum: string;
  created_at: string;
}

/**
 * Database row types for cache entries.
 */
export interface CacheRow {
  cache_key: string;
  namespace: string;
  value: string;
  ttl_seconds: number;
  created_at: string;
  expires_at: string;
}
