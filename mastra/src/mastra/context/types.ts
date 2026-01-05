// TypeScript interfaces for context management.

/** Record of a single tool execution.
 */
export interface ToolExecution {
  id: string;
  toolName: string;
  parameters: Record<string, unknown>;
  result: unknown;
  executionTime: number;
  timestamp: Date;
  decisionsMade: Record<string, unknown>;
  contextUsed: Record<string, unknown>;
  filesAccessed: string[];
  relatedExecutions: string[];
}

/**
 * Metadata about a discovered file.
 */
export interface FileMetadata {
  path: string;
  language: string;
  size?: number;
  discoveredBy: string;
  discoveredAt: Date;
  summary?: string;
}

/**
 * Metadata about an analyzed component.
 */
export interface ComponentMetadata {
  name: string;
  type: "function" | "class" | "module" | "interface" | "type";
  filePath: string;
  analyzedBy: string;
  analyzedAt: Date;
  dependencies?: string[];
  summary?: string;
}

/**
 * Conflict between tool executions.
 */
export interface Conflict {
  type: "decision" | "file" | "parameter";
  key: string;
  previousValue: unknown;
  newValue: unknown;
  previousExecution: string;
  newExecution: string;
}

/**
 * Resolution of a conflict.
 */
export interface ConflictResolution {
  timestamp: Date;
  executionId: string;
  conflicts: Conflict[];
  resolutionStrategy: "latest_wins" | "merge" | "skip";
  resolvedDecisions: Record<string, unknown>;
}

/**
 * Shared execution context across tool executions.
 */
export interface ExecutionContext {
  sessionId: string;
  query: string;
  startedAt: Date;
  discoveredFiles: Map<string, FileMetadata>;
  analyzedComponents: Map<string, ComponentMetadata>;
  decisionsMade: Map<string, unknown>;
  executionTrace: ToolExecution[];
  conflictResolutions: ConflictResolution[];
}

/**
 * Cache entry with TTL.
 */
export interface CacheEntry {
  value: unknown;
  timestamp: Date;
  expiresAt: Date;
  toolName: string;
  hitCount: number;
}

/**
 * Cache statistics.
 */
export interface CacheStats {
  size: number;
  maxSize: number;
  hitCount: number;
  missCount: number;
  hitRate: number;
  evictionCount: number;
}

/**
 * Context provided to a tool before execution.
 */
export interface ContextForTool {
  sessionId: string;
  query: string;
  relatedExecutions: ToolExecution[];
  relevantFiles: Map<string, FileMetadata>;
  relevantDecisions: Record<string, unknown>;
  cachedResult?: unknown;
  hasConflicts: boolean;
  suggestedResolution?: string;
}

/**
 * Summary of current execution context.
 */
export interface ExecutionSummary {
  sessionId: string;
  query: string;
  totalExecutions: number;
  totalDuration: number;
  filesDiscovered: number;
  componentsAnalyzed: number;
  decisionsMade: number;
  conflictsResolved: number;
  toolBreakdown: Record<string, number>;
}

/**
 * Options for creating a ContextManager.
 */
export interface ContextManagerOptions {
  sessionId: string;
  repoPath?: string;
  cacheMaxSize?: number;
  cacheTtlMs?: number;
}

/**
 * Options for recording a tool execution.
 */
export interface RecordExecutionOptions {
  toolName: string;
  parameters: Record<string, unknown>;
  result: unknown;
  executionTime: number;
  contextUsed?: Record<string, unknown>;
}
