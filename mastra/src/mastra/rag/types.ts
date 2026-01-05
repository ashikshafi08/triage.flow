// TypeScript interfaces for the RAG module.

import type { VectorIndexName } from "../storage";

/**
 * Repository information extracted from URL.
 */
export interface RepoInfo {
  owner: string;
  repo: string;
  branch: string;
  url: string;
  repoPath: string;
}

/**
 * Language metadata for a file.
 */
export interface LanguageMetadata {
  language: string;
  displayName: string;
  description: string;
  docPattern: string | null;
  importPattern: string | null;
}

/**
 * Language configuration entry.
 */
export interface LanguageConfigEntry {
  extensions: string[];
  docPattern: string | null;
  importPattern: string | null;
  displayName: string;
  description: string;
}

/**
 * Loaded file from repository.
 */
export interface LoadedFile {
  path: string;
  relativePath: string;
  content: string;
  language: string;
  metadata: LanguageMetadata;
}

/**
 * Document chunk after processing.
 */
export interface DocumentChunk {
  id: string;
  content: string;
  metadata: ChunkMetadata;
}

/**
 * Metadata for a document chunk.
 */
export interface ChunkMetadata {
  filePath: string;
  language: string;
  chunkIndex: number;
  startLine?: number;
  endLine?: number;
  totalChunks?: number;
}

/**
 * Source document in search results.
 */
export interface SourceDocument {
  file: string;
  language: string;
  description: string;
  content: string;
  score?: number;
  startLine?: number;
  endLine?: number;
}

/**
 * Result from getRelevantContext.
 */
export interface ContextResult {
  sources: SourceDocument[];
  repoInfo: RepoInfo;
  searchType: "file_oriented" | "semantic" | "hybrid";
  complexity: number;
  query: string;
}

/**
 * Configuration for CodebaseRag.
 */
export interface CodebaseRagConfig {
  indexName?: VectorIndexName;
  chunkSize?: number;
  chunkOverlap?: number;
  topK?: number;
  minScore?: number;
}

/**
 * Query classification result.
 */
export interface QueryClassification {
  isFileQuery: boolean;
  filePatterns: string[];
  keywords: string[];
  complexity: number;
}

/**
 * Options for repository loading.
 */
export interface LoadOptions {
  branch?: string;
  excludePatterns?: string[];
  maxFileSize?: number;
}

/**
 * Options for chunking.
 */
export interface ChunkOptions {
  maxSize?: number;
  overlap?: number;
  strategy?: "recursive" | "markdown" | "html" | "semantic";
}

/**
 * Search options for RAG queries.
 */
export interface SearchOptions {
  topK?: number;
  minScore?: number;
  restrictFiles?: string[];
  includeContent?: boolean;
}
