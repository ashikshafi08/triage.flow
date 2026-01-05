// RAG: Codebase indexing and semantic search capabilities.

// Types
export type {
  RepoInfo,
  LanguageMetadata,
  LanguageConfigEntry,
  LoadedFile,
  DocumentChunk,
  ChunkMetadata,
  SourceDocument,
  ContextResult,
  CodebaseRagConfig,
  QueryClassification,
  LoadOptions,
  ChunkOptions,
  SearchOptions,
} from "./types";

// Language configuration
export {
  LANGUAGE_CONFIG,
  getAllExtensions,
  getLanguageMetadata,
  isSupportedExtension,
  getLanguageByExtension,
  getExtensionsForLanguage,
} from "./languageConfig";

// Repository loading
export {
  cloneRepository,
  loadFiles,
  parseRepoUrl,
  searchFiles,
  getFileStats,
  getRepoStorageDir,
  isValidRepository,
  getCurrentBranch,
} from "./repositoryLoader";

// Code chunking
export { chunkFile, chunkFiles, estimateChunkCount, getChunkStats } from "./codeChunker";

// Main RAG class
export { CodebaseRag, createCodebaseRag } from "./codebaseRag";
