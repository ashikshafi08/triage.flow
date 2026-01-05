// Main RAG class for codebase indexing and semantic search.

import { openai } from "@ai-sdk/openai";
import { embedMany, embed } from "ai";
import {
  getVectorStore,
  getSessionStorage,
  VECTOR_INDEXES,
  type VectorIndexName,
  type EmbeddingMetadata,
} from "../storage";
import { cloneRepository, loadFiles, searchFiles } from "./repositoryLoader";
import { chunkFiles, getChunkStats } from "./codeChunker";
import { getLanguageMetadata } from "./languageConfig";
import type {
  RepoInfo,
  ContextResult,
  SourceDocument,
  CodebaseRagConfig,
  SearchOptions,
  QueryClassification,
} from "./types";

// File-oriented query patterns
const FILE_QUERY_PATTERNS = [
  /which files?/i,
  /what files?/i,
  /find files?/i,
  /list files?/i,
  /show files?/i,
  /\*\.\w+/, // Glob patterns like *.ts
  /\.ts\b|\.py\b|\.js\b|\.go\b/, // File extensions
];

// Default configuration
const DEFAULT_CONFIG: Required<CodebaseRagConfig> = {
  indexName: VECTOR_INDEXES.CODE_EMBEDDINGS,
  chunkSize: 1500,
  chunkOverlap: 200,
  topK: 10,
  minScore: 0.3,
};

/**
 * CodebaseRag - Main class for codebase indexing and retrieval.
 *
 * Provides:
 * - Repository cloning and file loading
 * - Code-aware chunking and embedding
 * - Vector similarity search
 * - Hybrid search (semantic + keyword)
 */
export class CodebaseRag {
  private config: Required<CodebaseRagConfig>;
  private repoInfo: RepoInfo | null = null;
  private sessionId: string | null = null;
  private indexed = false;

  constructor(config: CodebaseRagConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Load and index a repository.
   */
  async loadRepository(
    repoUrl: string,
    options: { branch?: string; sessionId?: string } = {}
  ): Promise<RepoInfo> {
    const vectorStore = getVectorStore();
    const sessionStorage = getSessionStorage();

    // Clone or update repository
    const { repoPath, repoInfo } = await cloneRepository(repoUrl, {
      branch: options.branch,
    });

    this.repoInfo = repoInfo;

    // Create or get session
    if (options.sessionId) {
      this.sessionId = options.sessionId;
    } else {
      const existingSession = await sessionStorage.findSessionByRepo(
        repoInfo.owner,
        repoInfo.repo,
        repoInfo.branch
      );

      if (existingSession) {
        this.sessionId = existingSession.id;
        // Check if we need to re-index
        if (existingSession.status === "ready") {
          this.indexed = true;
          return repoInfo;
        }
      } else {
        const session = await sessionStorage.createSession({
          repoUrl,
          repoOwner: repoInfo.owner,
          repoName: repoInfo.repo,
          branch: repoInfo.branch,
        });
        this.sessionId = session.id;
      }
    }

    // Update session status
    await sessionStorage.updateSession(this.sessionId, { status: "indexing" });

    try {
      // Load files
      const files = await loadFiles(repoPath);

      // Chunk files
      const chunks = await chunkFiles(files, {
        maxSize: this.config.chunkSize,
        overlap: this.config.chunkOverlap,
      });

      // Generate embeddings in batches
      const BATCH_SIZE = 100;
      const allVectors: number[][] = [];
      const allMetadata: EmbeddingMetadata[] = [];

      const embeddingModel = openai.embedding("text-embedding-3-small");

      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);
        const texts = batch.map((c) => c.content);

        const { embeddings } = await embedMany({
          model: embeddingModel,
          values: texts,
        });

        for (let j = 0; j < batch.length; j++) {
          allVectors.push(embeddings[j]);
          allMetadata.push({
            sessionId: this.sessionId!,
            filePath: batch[j].metadata.filePath,
            language: batch[j].metadata.language,
            chunkIndex: batch[j].metadata.chunkIndex,
            startLine: batch[j].metadata.startLine,
            endLine: batch[j].metadata.endLine,
            content: batch[j].content,
          });
        }
      }

      // Store in vector database
      await vectorStore.storeEmbeddings(this.config.indexName, allVectors, allMetadata);

      // Update session with stats
      const stats = getChunkStats(chunks);
      await sessionStorage.updateSession(this.sessionId, {
        status: "ready",
        totalFiles: files.length,
        totalChunks: stats.totalChunks,
      });

      this.indexed = true;
    } catch (error) {
      await sessionStorage.updateSession(this.sessionId, { status: "error" });
      throw error;
    }

    return repoInfo;
  }

  /**
   * Get relevant context for a query.
   */
  async getRelevantContext(query: string, options: SearchOptions = {}): Promise<ContextResult> {
    if (!this.repoInfo || !this.sessionId) {
      throw new Error("Repository not loaded. Call loadRepository first.");
    }

    const vectorStore = getVectorStore();

    // Classify the query
    const classification = this.classifyQuery(query);

    let sources: SourceDocument[] = [];
    let searchType: ContextResult["searchType"] = "semantic";

    if (classification.isFileQuery) {
      // File-oriented search
      searchType = "file_oriented";
      const filePatterns =
        classification.filePatterns.length > 0 ? classification.filePatterns : ["**/*"];

      for (const pattern of filePatterns) {
        const matchedFiles = await searchFiles(this.repoInfo.repoPath, pattern);

        for (const filePath of matchedFiles.slice(0, 10)) {
          const metadata = getLanguageMetadata(filePath);
          sources.push({
            file: filePath,
            language: metadata.language,
            description: `File matching pattern: ${pattern}`,
            content: `File: ${filePath}`,
            score: 1.0,
          });
        }
      }
    } else {
      // Semantic vector search
      const { embedding } = await embed({
        model: openai.embedding("text-embedding-3-small"),
        value: query,
      });

      const results = await vectorStore.searchBySession(
        this.config.indexName,
        this.sessionId,
        embedding,
        {
          topK: options.topK || this.config.topK,
          minScore: options.minScore || this.config.minScore,
        }
      );

      // Filter by restrictFiles if specified
      const filteredResults = options.restrictFiles
        ? results.filter((r) =>
            options.restrictFiles!.some((pattern) => r.metadata.filePath.includes(pattern))
          )
        : results;

      sources = filteredResults.map((r) => ({
        file: r.metadata.filePath,
        language: r.metadata.language,
        description: `Lines ${r.metadata.startLine}-${r.metadata.endLine}`,
        content: options.includeContent !== false ? r.metadata.content : "",
        score: r.score,
        startLine: r.metadata.startLine,
        endLine: r.metadata.endLine,
      }));
    }

    return {
      sources,
      repoInfo: this.repoInfo,
      searchType,
      complexity: classification.complexity,
      query,
    };
  }

  /**
   * Get context for an issue (convenience method).
   */
  async getIssueContext(
    title: string,
    body: string,
    options: SearchOptions = {}
  ): Promise<ContextResult> {
    const query = `${title}\n\n${body}`;
    return this.getRelevantContext(query, options);
  }

  /**
   * Classify a query to determine search strategy.
   */
  private classifyQuery(query: string): QueryClassification {
    const isFileQuery = FILE_QUERY_PATTERNS.some((pattern) => pattern.test(query));

    // Extract file patterns (glob-like)
    const filePatterns: string[] = [];
    const globMatch = query.match(/\*+\.\w+/g);
    if (globMatch) {
      filePatterns.push(...globMatch.map((p) => `**/${p}`));
    }

    // Extract extension patterns (e.g., ".ts" in "find .ts files")
    const extMatch = query.match(/\.(\w+)\b/g);
    if (extMatch) {
      // Add extension patterns as glob patterns
      for (const ext of extMatch) {
        if (!filePatterns.some((p) => p.includes(ext))) {
          filePatterns.push(`**/*${ext}`);
        }
      }
    }

    // Extract keywords
    const keywords = query
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3 && !["what", "where", "which", "find", "show"].includes(w));

    // Calculate complexity (simple heuristic)
    const complexity = Math.min(1.0, keywords.length * 0.1 + (isFileQuery ? 0.2 : 0.5));

    return {
      isFileQuery,
      filePatterns,
      keywords,
      complexity,
    };
  }

  /**
   * Check if repository is indexed.
   */
  isIndexed(): boolean {
    return this.indexed;
  }

  /**
   * Get current repository info.
   */
  getRepoInfo(): RepoInfo | null {
    return this.repoInfo;
  }

  /**
   * Get current session ID.
   */
  getSessionId(): string | null {
    return this.sessionId;
  }

  /**
   * Delete index for current session.
   */
  async deleteIndex(): Promise<void> {
    if (!this.sessionId) return;

    const vectorStore = getVectorStore();
    const sessionStorage = getSessionStorage();

    await vectorStore.deleteBySession(this.config.indexName, this.sessionId);
    await sessionStorage.deleteSession(this.sessionId);

    this.indexed = false;
    this.sessionId = null;
    this.repoInfo = null;
  }
}

// Factory function for creating CodebaseRag instances
export function createCodebaseRag(config?: CodebaseRagConfig): CodebaseRag {
  return new CodebaseRag(config);
}
