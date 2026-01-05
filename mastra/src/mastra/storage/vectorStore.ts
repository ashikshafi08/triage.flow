// LibSQLVector wrapper for embedding storage and similarity search.

import { LibSQLVector } from "@mastra/libsql";
import { createSingleton } from "../utils/singleton";
import { getStorageConfig, VECTOR_INDEXES, type VectorIndexName } from "./config";
import type {
  EmbeddingMetadata,
  VectorSearchParams,
  VectorSearchResult,
  VectorIndexStats,
} from "./types";

// Extract LibSQLVector's filter type from its query method
type LibSQLVectorFilter = Parameters<LibSQLVector["query"]>[0]["filter"];

/** Result of a batch delete operation */
export interface BatchDeleteResult {
  deletedCount: number;
  failedCount: number;
  failedIds: string[];
}

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
  async ensureIndex(indexName: VectorIndexName, dimension?: number): Promise<void> {
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
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      // Only treat "already exists" as non-fatal
      const isAlreadyExistsError =
        errorMessage.includes("already exists") ||
        errorMessage.includes("duplicate") ||
        errorMessage.includes("SQLITE_CONSTRAINT");

      if (!isAlreadyExistsError) {
        throw new Error(`Failed to create index "${indexName}": ${errorMessage}`);
      }

      // Verify existing index has correct dimension
      const stats = await this.describeIndex(indexName);
      if (stats && stats.dimension !== dim) {
        throw new Error(
          `Index ${indexName} exists with dimension ${stats.dimension}, expected ${dim}`
        );
      }
      if (!stats) {
        throw new Error(
          `Index "${indexName}" creation reported conflict but index doesn't exist. Check database connectivity.`
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

    const generatedIds = ids || vectors.map(() => crypto.randomUUID());

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

    const results = await this.store.query({
      indexName: params.indexName,
      queryVector: params.queryVector,
      topK: params.topK ?? config.vectorTopKDefault,
      includeVector: params.includeVector ?? false,
      minScore: params.minScore ?? config.vectorMinScoreDefault,
      filter: params.filter as LibSQLVectorFilter,
    });

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
   * Returns details about what was deleted for debugging.
   */
  async deleteBySession(
    indexName: VectorIndexName,
    sessionId: string
  ): Promise<BatchDeleteResult> {
    // LibSQLVector doesn't have direct bulk delete by filter
    // Query first, then delete in parallel batches with error tracking
    const results = await this.store.query({
      indexName,
      queryVector: new Array(getStorageConfig().embeddingDimension).fill(0),
      topK: 10000,
      filter: { sessionId } as LibSQLVectorFilter,
    });

    return this.batchDelete(indexName, results.map((r) => r.id));
  }

  /**
   * Delete embeddings for specific files in a session.
   */
  async deleteByFiles(
    indexName: VectorIndexName,
    sessionId: string,
    filePaths: string[]
  ): Promise<BatchDeleteResult> {
    if (filePaths.length === 0) {
      return { deletedCount: 0, failedCount: 0, failedIds: [] };
    }

    // Collect all IDs to delete
    const idsToDelete: string[] = [];

    for (const filePath of filePaths) {
      const results = await this.store.query({
        indexName,
        queryVector: new Array(getStorageConfig().embeddingDimension).fill(0),
        topK: 1000,
        filter: { sessionId, filePath } as LibSQLVectorFilter,
      });
      idsToDelete.push(...results.map((r) => r.id));
    }

    return this.batchDelete(indexName, idsToDelete);
  }

  /**
   * Batch delete with error tracking using Promise.allSettled.
   * Continues on individual failures and reports results.
   */
  private async batchDelete(
    indexName: VectorIndexName,
    ids: string[]
  ): Promise<BatchDeleteResult> {
    let deletedCount = 0;
    const failedIds: string[] = [];

    const BATCH_SIZE = 50;
    for (let i = 0; i < ids.length; i += BATCH_SIZE) {
      const batch = ids.slice(i, i + BATCH_SIZE);
      const settlements = await Promise.allSettled(
        batch.map((id) => this.store.deleteVector({ indexName, id }))
      );

      for (let j = 0; j < settlements.length; j++) {
        if (settlements[j].status === "fulfilled") {
          deletedCount++;
        } else {
          failedIds.push(batch[j]);
        }
      }
    }

    if (failedIds.length > 0) {
      console.error(
        `[VectorStore] Partial deletion failure: ${deletedCount} succeeded, ${failedIds.length} failed`,
        { failedIds: failedIds.slice(0, 10) } // Log first 10 for debugging
      );
    }

    return { deletedCount, failedCount: failedIds.length, failedIds };
  }

  /**
   * Get index statistics.
   * Returns null only if the index doesn't exist; throws on other errors.
   */
  async describeIndex(indexName: string): Promise<VectorIndexStats | null> {
    try {
      const stats = await this.store.describeIndex({ indexName });
      return stats as VectorIndexStats;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      // Only return null for "index not found" errors
      if (
        errorMessage.includes("not found") ||
        errorMessage.includes("does not exist") ||
        errorMessage.includes("no such")
      ) {
        return null;
      }
      
      // All other errors should propagate
      throw new Error(`Failed to describe index "${indexName}": ${errorMessage}`);
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

export const getVectorStore = createSingleton(() => new VectorStore());
