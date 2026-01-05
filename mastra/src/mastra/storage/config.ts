// Environment-based configuration for LibSQL storage and vector operations.

import { createSingleton } from "../utils/singleton";

export interface StorageConfig {
  // Database URLs
  databaseUrl: string;
  vectorDatabaseUrl: string;
  authToken?: string;

  // Embedding settings
  embeddingModel: string;
  embeddingDimension: number;

  // Cache TTLs (seconds)
  cacheTtlRag: number;
  cacheTtlFolder: number;
  cacheTtlResponse: number;
  cacheCleanupInterval: number;

  // Vector search defaults
  vectorTopKDefault: number;
  vectorMinScoreDefault: number;
}

/**
 * Loads storage configuration from environment variables.
 * Uses sensible defaults for local development.
 */
export function loadStorageConfig(): StorageConfig {
  const databaseUrl = process.env.DATABASE_URL || "file:./mastra.db";

  return {
    // Database
    databaseUrl,
    vectorDatabaseUrl: process.env.VECTOR_DATABASE_URL || databaseUrl,
    authToken: process.env.DATABASE_AUTH_TOKEN,

    // Embedding
    embeddingModel: process.env.EMBEDDING_MODEL || "text-embedding-3-small",
    embeddingDimension: parseInt(process.env.EMBEDDING_DIMENSION || "1536"),

    // Cache TTLs (in seconds)
    cacheTtlRag: parseInt(process.env.CACHE_TTL_RAG || "300"), // 5 min
    cacheTtlFolder: parseInt(process.env.CACHE_TTL_FOLDER || "1800"), // 30 min
    cacheTtlResponse: parseInt(process.env.CACHE_TTL_RESPONSE || "600"), // 10 min
    cacheCleanupInterval: parseInt(process.env.CACHE_CLEANUP_INTERVAL || "60"),

    // Vector search
    vectorTopKDefault: parseInt(process.env.VECTOR_TOP_K_DEFAULT || "10"),
    vectorMinScoreDefault: parseFloat(process.env.VECTOR_MIN_SCORE_DEFAULT || "0.3"),
  };
}

export const getStorageConfig = createSingleton(loadStorageConfig);

// Vector index names used across the application
export const VECTOR_INDEXES = {
  CODE_EMBEDDINGS: "code_embeddings",
  ISSUE_EMBEDDINGS: "issue_embeddings",
  DOC_EMBEDDINGS: "doc_embeddings",
} as const;

export type VectorIndexName = (typeof VECTOR_INDEXES)[keyof typeof VECTOR_INDEXES];
