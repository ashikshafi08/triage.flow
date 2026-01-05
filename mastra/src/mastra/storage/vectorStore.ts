/**
 * Vector Store
 *
 * LibSQLVector wrapper for embedding storage and similarity search.
 * Replaces Python FAISS-based vector storage.
 *
 * @module storage/vectorStore
 */

import { LibSQLVector } from "@mastra/libsql";
import {
  getStorageConfig,
  VECTOR_INDEXES,
  type VectorIndexName,
} from "./config";
import type {
  EmbeddingMetadata,
  VectorSearchParams,
  VectorSearchResult,
  VectorIndexStats,
} from "./types";

/**
 * Vector store manager for embedding operations.
 *
 * Wraps LibSQLVector with application-specific helpers:
 * - Automatic index creation with correct dimensions
 * - Session-scoped operations (filter by sessionId)
 * - Batch operations for efficient indexing
 */
export class VectorStore {
  private store: LibSQLVector;
  private initializedIndexes = new Set<string>();

  constructor(connectionUrl?: string, authToken?: string) {
    const config = getStorageConfig();
    this.store = new LibSQLVector({
      connectionUrl: connectionUrl || config.vectorDatabaseUrl,
      authToken: authToken || config.authToken,
    });
  }

  /**
   * Ensure an index exists with the correct configuration.
   */
  async ensureIndex(
    indexName: VectorIndexName,
    dimension?: number
  ): Promise<void> {
    if (this.initializedIndexes.has(indexName)) return;

    const config = getStorageConfig();
    const dim = dimension || config.embeddingDimension;

    try {
      await this.store.createIndex({
        indexName,
        dimension: dim,
        metric: "cosine",
      });
    } catch (error) {
      // Index might already exist - check if it's the right dimension
      const stats = await this.describeIndex(indexName);
      if (stats && stats.dimension !== dim) {
        throw new Error(
          `Index ${indexName} exists with dimension ${stats.dimension}, expected ${dim}`
        );
      }
    }

    this.initializedIndexes.add(indexName);
  }

  /**
   * Initialize all standard indexes.
   */
  async initializeAllIndexes(): Promise<void> {
    for (const indexName of Object.values(VECTOR_INDEXES)) {
      await this.ensureIndex(indexName);
    }
  }

  /**
   * Store a single embedding with metadata.
   */
  async storeEmbedding(
    indexName: VectorIndexName,
    vector: number[],
    metadata: EmbeddingMetadata,
    id?: string
  ): Promise<string> {
    await this.ensureIndex(indexName);

    const generatedId = id || crypto.randomUUID();

    await this.store.upsert({
      indexName,
      vectors: [vector],
      metadata: [metadata],
      ids: [generatedId],
    });

    return generatedId;
  }

  /**
   * Store multiple embeddings in a batch.
   * More efficient than individual inserts.
   */
  async storeEmbeddings(
    indexName: VectorIndexName,
    vectors: number[][],
    metadata: EmbeddingMetadata[],
    ids?: string[]
  ): Promise<string[]> {
    await this.ensureIndex(indexName);

    if (vectors.length === 0) return [];

    if (vectors.length !== metadata.length) {
      throw new Error("Vectors and metadata arrays must have equal length");
    }

    const generatedIds =
      ids || vectors.map(() => crypto.randomUUID());

    // Batch in chunks of 100 for memory efficiency
    const BATCH_SIZE = 100;
    for (let i = 0; i < vectors.length; i += BATCH_SIZE) {
      const end = Math.min(i + BATCH_SIZE, vectors.length);
      await this.store.upsert({
        indexName,
        vectors: vectors.slice(i, end),
        metadata: metadata.slice(i, end),
        ids: generatedIds.slice(i, end),
      });
    }

    return generatedIds;
  }

  /**
   * Search for similar vectors.
   */
  async search(params: VectorSearchParams): Promise<VectorSearchResult[]> {
    const config = getStorageConfig();

    const queryOptions: {
      indexName: string;
      queryVector: number[];
      topK: number;
      includeVector: boolean;
      minScore: number;
      filter?: Record<string, unknown>;
    } = {
      indexName: params.indexName,
      queryVector: params.queryVector,
      topK: params.topK || config.vectorTopKDefault,
      includeVector: params.includeVector || false,
      minScore: params.minScore || config.vectorMinScoreDefault,
    };

    // Only add filter if provided
    if (params.filter) {
      queryOptions.filter = params.filter;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const results = await this.store.query(queryOptions as any);

    return results.map((r) => ({
      id: r.id,
      score: r.score,
      metadata: r.metadata as EmbeddingMetadata,
      vector: r.vector,
    }));
  }

  /**
   * Search within a specific session.
   */
  async searchBySession(
    indexName: VectorIndexName,
    sessionId: string,
    queryVector: number[],
    options?: Omit<VectorSearchParams, "indexName" | "queryVector" | "filter">
  ): Promise<VectorSearchResult[]> {
    return this.search({
      indexName,
      queryVector,
      filter: { sessionId },
      ...options,
    });
  }

  /**
   * Delete all embeddings for a session.
   */
  async deleteBySession(
    indexName: VectorIndexName,
    sessionId: string
  ): Promise<void> {
    // LibSQLVector doesn't have direct bulk delete by filter
    // We need to query first, then delete individually
    const results = await this.store.query({
      indexName,
      queryVector: new Array(getStorageConfig().embeddingDimension).fill(0),
      topK: 10000, // Get all matching
      filter: { sessionId },
    });

    for (const result of results) {
      await this.store.deleteVector({
        indexName,
        id: result.id,
      });
    }
  }

  /**
   * Delete embeddings for specific files in a session.
   */
  async deleteByFiles(
    indexName: VectorIndexName,
    sessionId: string,
    filePaths: string[]
  ): Promise<void> {
    if (filePaths.length === 0) return;

    // Query for each file path and delete
    for (const filePath of filePaths) {
      const results = await this.store.query({
        indexName,
        queryVector: new Array(getStorageConfig().embeddingDimension).fill(0),
        topK: 1000,
        filter: { sessionId, filePath },
      });

      for (const result of results) {
        await this.store.deleteVector({
          indexName,
          id: result.id,
        });
      }
    }
  }

  /**
   * Get index statistics.
   */
  async describeIndex(indexName: string): Promise<VectorIndexStats | null> {
    try {
      const stats = await this.store.describeIndex({ indexName });
      return stats as VectorIndexStats;
    } catch {
      return null;
    }
  }

  /**
   * List all indexes.
   */
  async listIndexes(): Promise<string[]> {
    return this.store.listIndexes();
  }

  /**
   * Delete an entire index.
   */
  async deleteIndex(indexName: string): Promise<void> {
    await this.store.deleteIndex({ indexName });
    this.initializedIndexes.delete(indexName);
  }

  /**
   * Clear all vectors from an index but keep structure.
   */
  async truncateIndex(indexName: string): Promise<void> {
    await this.store.truncateIndex({ indexName });
  }

  /**
   * Get the underlying LibSQLVector instance for advanced operations.
   */
  getRawStore(): LibSQLVector {
    return this.store;
  }
}

// Singleton instance
let _vectorStore: VectorStore | null = null;

export function getVectorStore(): VectorStore {
  if (!_vectorStore) {
    _vectorStore = new VectorStore();
  }
  return _vectorStore;
}
