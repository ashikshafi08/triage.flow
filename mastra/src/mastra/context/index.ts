/**
 * Context Management Module
 *
 * Handles tool execution tracking, context sharing between agents,
 * and cache invalidation. Replaces Python context_manager.py.
 *
 * @module context
 */

// Types
export type {
  ToolExecution,
  FileMetadata,
  ComponentMetadata,
  Conflict,
  ConflictResolution,
  ExecutionContext,
  CacheEntry,
  CacheStats,
  ContextForTool,
  ExecutionSummary,
  ContextManagerOptions,
  RecordExecutionOptions,
} from "./types";

// Tool relationships
export {
  TOOL_RELATIONSHIPS,
  PARAMETER_ALIASES,
  getRelatedTools,
  normalizeParameters,
  areToolsRelated,
} from "./toolRelationships";

// Cache
export { ContextCache } from "./contextCache";

// Execution tracker
export { ExecutionTracker, type TrackerSummary } from "./executionTracker";

// Main context manager
export { ContextManager, createContextManager } from "./contextManager";
