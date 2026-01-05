/**
 * Context Manager
 *
 * Main class for managing execution context across tool calls.
 * Replaces Python context_manager.py.
 *
 * @module context/contextManager
 */

import { ExecutionTracker, type TrackerSummary } from "./executionTracker";
import { ContextCache } from "./contextCache";
import type {
  ExecutionContext,
  ToolExecution,
  ContextForTool,
  ExecutionSummary,
  ContextManagerOptions,
  RecordExecutionOptions,
  FileMetadata,
  ConflictResolution,
  CacheStats,
} from "./types";

// Default configuration
const DEFAULT_CACHE_MAX_SIZE = 1000;
const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * ContextManager coordinates tool executions and shares context.
 *
 * Features:
 * - Start/end execution contexts for queries
 * - Track tool executions with timing and metadata
 * - Share discovered files and decisions between tools
 * - Cache tool results with TTL
 * - Detect and resolve conflicts between executions
 */
export class ContextManager {
  private currentContext: ExecutionContext | null = null;
  private readonly executionTracker: ExecutionTracker;
  private readonly cache: ContextCache;
  private readonly sessionId: string;
  private readonly repoPath: string;

  constructor(options: ContextManagerOptions) {
    this.sessionId = options.sessionId;
    this.repoPath = options.repoPath || "";
    this.executionTracker = new ExecutionTracker();
    this.cache = new ContextCache({
      maxSize: options.cacheMaxSize ?? DEFAULT_CACHE_MAX_SIZE,
      defaultTtlMs: options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS,
    });
  }

  // ========================================
  // Lifecycle Methods
  // ========================================

  /**
   * Start a new execution context for a query.
   */
  startExecution(query: string): ExecutionContext {
    // Clean up previous context if exists
    if (this.currentContext) {
      this.cleanup();
    }

    this.currentContext = {
      sessionId: this.sessionId,
      query,
      startedAt: new Date(),
      discoveredFiles: new Map(),
      analyzedComponents: new Map(),
      decisionsMade: new Map(),
      executionTrace: [],
      conflictResolutions: [],
    };

    return this.currentContext;
  }

  /**
   * End the current execution and return summary.
   */
  endExecution(): ExecutionSummary | null {
    if (!this.currentContext) {
      return null;
    }

    const summary = this.getExecutionSummary();

    // Transfer execution trace to context
    this.currentContext.executionTrace = this.executionTracker.getAllExecutions();

    return summary;
  }

  /**
   * Clean up the context manager.
   */
  cleanup(): void {
    this.executionTracker.clear();
    this.currentContext = null;
    // Note: Cache is NOT cleared on cleanup (persists across executions)
  }

  /**
   * Clear all state including cache.
   */
  clearAll(): void {
    this.cleanup();
    this.cache.clear();
  }

  // ========================================
  // Context Retrieval
  // ========================================

  /**
   * Get context relevant to a tool before execution.
   */
  getContextForTool(
    toolName: string,
    parameters: Record<string, unknown>
  ): ContextForTool {
    // Check cache first
    const cachedResult = this.cache.getForTool(toolName, parameters);

    // Get related executions
    const relatedExecutions = this.executionTracker.findRelatedExecutions(
      toolName,
      parameters
    );

    // Get relevant files
    const relevantFiles = this.executionTracker.getRelevantFiles(
      toolName,
      parameters
    );

    // Merge with discovered files from current context
    if (this.currentContext) {
      for (const [path, metadata] of this.currentContext.discoveredFiles) {
        if (!relevantFiles.has(path)) {
          relevantFiles.set(path, metadata);
        }
      }
    }

    // Get relevant decisions
    const relevantDecisions = this.executionTracker.getRelevantDecisions(
      toolName
    );

    // Check for potential conflicts
    const hasConflicts = this.checkForPotentialConflicts(
      toolName,
      parameters,
      relatedExecutions
    );

    return {
      sessionId: this.sessionId,
      query: this.currentContext?.query || "",
      relatedExecutions,
      relevantFiles,
      relevantDecisions,
      cachedResult: cachedResult ?? undefined,
      hasConflicts,
      suggestedResolution: hasConflicts
        ? "Review related executions for conflicting decisions"
        : undefined,
    };
  }

  // ========================================
  // Execution Recording
  // ========================================

  /**
   * Record a tool execution.
   */
  recordExecution(options: RecordExecutionOptions): ToolExecution {
    const execution = this.executionTracker.record(
      options.toolName,
      options.parameters,
      options.result,
      options.executionTime,
      options.contextUsed
    );

    // Update cache
    this.cache.setForTool(
      options.toolName,
      options.parameters,
      options.result
    );

    // Update current context
    this.updateContextFromExecution(execution);

    return execution;
  }

  /**
   * Add a discovered file to the context.
   */
  addDiscoveredFile(path: string, metadata: Partial<FileMetadata>): void {
    if (!this.currentContext) return;

    this.currentContext.discoveredFiles.set(path, {
      path,
      language: metadata.language || "unknown",
      discoveredBy: metadata.discoveredBy || "unknown",
      discoveredAt: metadata.discoveredAt || new Date(),
      size: metadata.size,
      summary: metadata.summary,
    });
  }

  /**
   * Add a decision to the context.
   */
  addDecision(key: string, value: unknown): void {
    if (!this.currentContext) return;
    this.currentContext.decisionsMade.set(key, value);
  }

  // ========================================
  // Summary and State
  // ========================================

  /**
   * Get summary of current execution.
   */
  getExecutionSummary(): ExecutionSummary {
    const trackerSummary = this.executionTracker.getSummary();

    return {
      sessionId: this.sessionId,
      query: this.currentContext?.query || "",
      totalExecutions: trackerSummary.totalExecutions,
      totalDuration: trackerSummary.totalDuration,
      filesDiscovered:
        this.currentContext?.discoveredFiles.size ||
        trackerSummary.uniqueFiles,
      componentsAnalyzed: this.currentContext?.analyzedComponents.size || 0,
      decisionsMade: this.currentContext?.decisionsMade.size || 0,
      conflictsResolved: this.currentContext?.conflictResolutions.length || 0,
      toolBreakdown: trackerSummary.toolCounts,
    };
  }

  /**
   * Get current execution context.
   */
  getCurrentContext(): ExecutionContext | null {
    return this.currentContext;
  }

  /**
   * Get cache statistics.
   */
  getCacheStats(): CacheStats {
    return this.cache.getStats();
  }

  /**
   * Get all tracked executions.
   */
  getAllExecutions(): ToolExecution[] {
    return this.executionTracker.getAllExecutions();
  }

  /**
   * Get all discovered files.
   */
  getAllDiscoveredFiles(): Map<string, FileMetadata> {
    const files = this.executionTracker.getAllFiles();

    // Merge with current context files
    if (this.currentContext) {
      for (const [path, metadata] of this.currentContext.discoveredFiles) {
        if (!files.has(path)) {
          files.set(path, metadata);
        }
      }
    }

    return files;
  }

  // ========================================
  // Conflict Detection
  // ========================================

  /**
   * Detect conflicts for a potential execution.
   */
  detectConflicts(
    toolName: string,
    parameters: Record<string, unknown>,
    result: unknown
  ) {
    return this.executionTracker.detectConflicts(toolName, parameters, result);
  }

  /**
   * Resolve conflicts for an execution.
   */
  resolveConflicts(
    executionId: string,
    strategy: "latest_wins" | "merge" | "skip" = "latest_wins"
  ): ConflictResolution | null {
    const execution = this.executionTracker.getExecution(executionId);
    if (!execution) return null;

    const conflicts = this.executionTracker.detectConflicts(
      execution.toolName,
      execution.parameters,
      execution.result
    );

    if (conflicts.length === 0) return null;

    const resolution = this.executionTracker.resolveConflicts(
      executionId,
      conflicts,
      strategy
    );

    // Add to context
    if (this.currentContext) {
      this.currentContext.conflictResolutions.push(resolution);
    }

    return resolution;
  }

  // ========================================
  // Private Helpers
  // ========================================

  /**
   * Check for potential conflicts before execution.
   */
  private checkForPotentialConflicts(
    toolName: string,
    parameters: Record<string, unknown>,
    relatedExecutions: ToolExecution[]
  ): boolean {
    // Check if any related execution made decisions that might conflict
    for (const execution of relatedExecutions) {
      if (
        execution.toolName === toolName &&
        Object.keys(execution.decisionsMade).length > 0
      ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Update context from a completed execution.
   */
  private updateContextFromExecution(execution: ToolExecution): void {
    if (!this.currentContext) return;

    // Add execution to trace
    this.currentContext.executionTrace.push(execution);

    // Update discovered files
    for (const filePath of execution.filesAccessed) {
      if (!this.currentContext.discoveredFiles.has(filePath)) {
        this.currentContext.discoveredFiles.set(filePath, {
          path: filePath,
          language: this.inferLanguage(filePath),
          discoveredBy: execution.toolName,
          discoveredAt: execution.timestamp,
        });
      }
    }

    // Update decisions
    for (const [key, value] of Object.entries(execution.decisionsMade)) {
      this.currentContext.decisionsMade.set(key, value);
    }
  }

  /**
   * Infer language from file path.
   */
  private inferLanguage(filePath: string): string {
    const ext = filePath.split(".").pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      ts: "typescript",
      tsx: "typescript",
      js: "javascript",
      jsx: "javascript",
      py: "python",
      go: "go",
      rs: "rust",
      java: "java",
      rb: "ruby",
      md: "markdown",
    };
    return langMap[ext || ""] || "unknown";
  }
}

// Factory function
export function createContextManager(
  options: ContextManagerOptions
): ContextManager {
  return new ContextManager(options);
}
