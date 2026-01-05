// Tracks tool executions and their relationships.

import type { ToolExecution, FileMetadata, Conflict, ConflictResolution } from "./types";
import { TOOL_RELATIONSHIPS, normalizeParameters, areToolsRelated } from "./toolRelationships";
import { inferLanguage } from "../utils/language";

/**
 * Summary of tracked executions.
 */
export interface TrackerSummary {
  totalExecutions: number;
  totalDuration: number;
  uniqueTools: number;
  uniqueFiles: number;
  toolCounts: Record<string, number>;
}

/**
 * Tracks tool executions within an execution context.
 *
 * Responsibilities:
 * - Record tool executions with timing and context
 * - Find related executions based on tool relationships
 * - Extract files accessed from parameters and results
 * - Detect and resolve conflicts between executions
 */
export class ExecutionTracker {
  private executions: ToolExecution[] = [];
  private filesAccessed: Map<string, FileMetadata> = new Map();
  private decisions: Map<string, unknown> = new Map();
  private executionCounter = 0;

  /**
   * Record a tool execution.
   */
  record(
    toolName: string,
    parameters: Record<string, unknown>,
    result: unknown,
    executionTime: number,
    contextUsed?: Record<string, unknown>
  ): ToolExecution {
    const id = `exec-${++this.executionCounter}-${Date.now()}`;
    const normalizedParams = normalizeParameters(toolName, parameters);

    // Extract files from parameters and result
    const filesFromParams = this.extractFilesFromParams(normalizedParams);
    const filesFromResult = this.extractFilesFromResult(result);
    const allFiles = [...filesFromParams, ...filesFromResult];

    // Find related executions
    const relatedExecutions = this.findRelatedExecutions(toolName, normalizedParams);

    // Extract decisions from result
    const decisionsMade = this.extractDecisions(toolName, result);

    const execution: ToolExecution = {
      id,
      toolName,
      parameters: normalizedParams,
      result,
      executionTime,
      timestamp: new Date(),
      decisionsMade,
      contextUsed: contextUsed || {},
      filesAccessed: allFiles,
      relatedExecutions: relatedExecutions.map((e) => e.id),
    };

    this.executions.push(execution);

    // Update tracked files
    for (const file of allFiles) {
      if (!this.filesAccessed.has(file)) {
        this.filesAccessed.set(file, {
          path: file,
          language: inferLanguage(file),
          discoveredBy: toolName,
          discoveredAt: new Date(),
        });
      }
    }

    // Update decisions
    for (const [key, value] of Object.entries(decisionsMade)) {
      this.decisions.set(key, value);
    }

    return execution;
  }

  /**
   * Find related executions for a tool call.
   */
  findRelatedExecutions(toolName: string, parameters: Record<string, unknown>): ToolExecution[] {
    const related: ToolExecution[] = [];

    for (const execution of this.executions) {
      // Check tool relationship
      if (areToolsRelated(toolName, execution.toolName)) {
        related.push(execution);
        continue;
      }

      // Check shared files
      const sharedFiles = this.hasSharedFiles(parameters, execution.parameters);
      if (sharedFiles) {
        related.push(execution);
      }
    }

    return related;
  }

  /**
   * Get files relevant to a tool.
   */
  getRelevantFiles(
    toolName: string,
    parameters: Record<string, unknown>
  ): Map<string, FileMetadata> {
    const relevant = new Map<string, FileMetadata>();

    // Get files from related executions
    const relatedExecutions = this.findRelatedExecutions(toolName, parameters);
    for (const execution of relatedExecutions) {
      for (const filePath of execution.filesAccessed) {
        const metadata = this.filesAccessed.get(filePath);
        if (metadata) {
          relevant.set(filePath, metadata);
        }
      }
    }

    // Get files from parameters
    const filesFromParams = this.extractFilesFromParams(parameters);
    for (const filePath of filesFromParams) {
      const metadata = this.filesAccessed.get(filePath);
      if (metadata) {
        relevant.set(filePath, metadata);
      }
    }

    return relevant;
  }

  /**
   * Get decisions relevant to a tool.
   */
  getRelevantDecisions(toolName: string): Record<string, unknown> {
    const relevant: Record<string, unknown> = {};
    const relatedTools = TOOL_RELATIONSHIPS[toolName] || [];

    for (const execution of this.executions) {
      if (execution.toolName === toolName || relatedTools.includes(execution.toolName)) {
        for (const [key, value] of Object.entries(execution.decisionsMade)) {
          relevant[key] = value;
        }
      }
    }

    return relevant;
  }

  /**
   * Detect conflicts between a new execution and previous ones.
   */
  detectConflicts(
    toolName: string,
    parameters: Record<string, unknown>,
    result: unknown
  ): Conflict[] {
    const conflicts: Conflict[] = [];
    const newDecisions = this.extractDecisions(toolName, result);

    for (const execution of this.executions) {
      if (execution.toolName === toolName) {
        // Check for conflicting decisions
        for (const [key, newValue] of Object.entries(newDecisions)) {
          const prevValue = execution.decisionsMade[key];
          if (prevValue !== undefined && JSON.stringify(prevValue) !== JSON.stringify(newValue)) {
            conflicts.push({
              type: "decision",
              key,
              previousValue: prevValue,
              newValue,
              previousExecution: execution.id,
              newExecution: "",
            });
          }
        }
      }
    }

    return conflicts;
  }

  /**
   * Resolve conflicts using the specified strategy.
   */
  resolveConflicts(
    executionId: string,
    conflicts: Conflict[],
    strategy: "latest_wins" | "merge" | "skip" = "latest_wins"
  ): ConflictResolution {
    const resolved: Record<string, unknown> = {};

    for (const conflict of conflicts) {
      switch (strategy) {
        case "latest_wins":
          resolved[conflict.key] = conflict.newValue;
          break;
        case "merge":
          // For merge, try to combine values if they're arrays or objects
          if (Array.isArray(conflict.previousValue) && Array.isArray(conflict.newValue)) {
            resolved[conflict.key] = [
              ...new Set([...conflict.previousValue, ...conflict.newValue]),
            ];
          } else if (
            typeof conflict.previousValue === "object" &&
            typeof conflict.newValue === "object"
          ) {
            resolved[conflict.key] = {
              ...conflict.previousValue,
              ...conflict.newValue,
            };
          } else {
            resolved[conflict.key] = conflict.newValue;
          }
          break;
        case "skip":
          resolved[conflict.key] = conflict.previousValue;
          break;
      }
    }

    // Update decisions
    for (const [key, value] of Object.entries(resolved)) {
      this.decisions.set(key, value);
    }

    return {
      timestamp: new Date(),
      executionId,
      conflicts,
      resolutionStrategy: strategy,
      resolvedDecisions: resolved,
    };
  }

  /**
   * Get execution by ID.
   */
  getExecution(id: string): ToolExecution | undefined {
    return this.executions.find((e) => e.id === id);
  }

  /**
   * Get all executions.
   */
  getAllExecutions(): ToolExecution[] {
    return [...this.executions];
  }

  /**
   * Get executions for a specific tool.
   */
  getExecutionsForTool(toolName: string): ToolExecution[] {
    return this.executions.filter((e) => e.toolName === toolName);
  }

  /**
   * Get all tracked files.
   */
  getAllFiles(): Map<string, FileMetadata> {
    return new Map(this.filesAccessed);
  }

  /**
   * Get all decisions.
   */
  getAllDecisions(): Map<string, unknown> {
    return new Map(this.decisions);
  }

  /**
   * Get tracker summary.
   */
  getSummary(): TrackerSummary {
    const toolCounts: Record<string, number> = {};
    let totalDuration = 0;
    const uniqueFiles = new Set<string>();

    for (const execution of this.executions) {
      toolCounts[execution.toolName] = (toolCounts[execution.toolName] || 0) + 1;
      totalDuration += execution.executionTime;
      for (const file of execution.filesAccessed) {
        uniqueFiles.add(file);
      }
    }

    return {
      totalExecutions: this.executions.length,
      totalDuration,
      uniqueTools: Object.keys(toolCounts).length,
      uniqueFiles: uniqueFiles.size,
      toolCounts,
    };
  }

  /**
   * Clear all tracked data.
   */
  clear(): void {
    this.executions = [];
    this.filesAccessed.clear();
    this.decisions.clear();
    this.executionCounter = 0;
  }

  /**
   * Extract file paths from parameters.
   */
  private extractFilesFromParams(parameters: Record<string, unknown>): string[] {
    const files: string[] = [];
    const fileKeys = ["filePath", "path", "file", "directoryPath", "directory"];

    for (const key of fileKeys) {
      const value = parameters[key];
      if (typeof value === "string" && value.length > 0) {
        files.push(value);
      }
    }

    // Check for file arrays
    if (Array.isArray(parameters.files)) {
      for (const file of parameters.files) {
        if (typeof file === "string") {
          files.push(file);
        }
      }
    }

    return files;
  }

  /**
   * Extract file paths from result.
   */
  private extractFilesFromResult(result: unknown): string[] {
    const files: string[] = [];

    if (typeof result !== "object" || result === null) {
      return files;
    }

    const resultObj = result as Record<string, unknown>;

    // Look for common file-related fields
    const fileFields = ["files", "filePath", "paths", "matchedFiles"];
    for (const field of fileFields) {
      const value = resultObj[field];
      if (typeof value === "string") {
        files.push(value);
      } else if (Array.isArray(value)) {
        for (const item of value) {
          if (typeof item === "string") {
            files.push(item);
          } else if (typeof item === "object" && item !== null) {
            const itemObj = item as Record<string, unknown>;
            if (typeof itemObj.path === "string") {
              files.push(itemObj.path);
            } else if (typeof itemObj.filePath === "string") {
              files.push(itemObj.filePath);
            }
          }
        }
      }
    }

    return files;
  }

  /**
   * Extract decisions from result.
   */
  private extractDecisions(toolName: string, result: unknown): Record<string, unknown> {
    const decisions: Record<string, unknown> = {};

    if (typeof result !== "object" || result === null) {
      return decisions;
    }

    const resultObj = result as Record<string, unknown>;

    // Look for decision-related fields
    const decisionFields = ["decision", "decisions", "recommendation", "analysis", "conclusion"];

    for (const field of decisionFields) {
      if (resultObj[field] !== undefined) {
        decisions[`${toolName}:${field}`] = resultObj[field];
      }
    }

    return decisions;
  }

  /**
   * Check if two parameter sets share file references.
   */
  private hasSharedFiles(
    params1: Record<string, unknown>,
    params2: Record<string, unknown>
  ): boolean {
    const files1 = new Set(this.extractFilesFromParams(params1));
    const files2 = new Set(this.extractFilesFromParams(params2));

    for (const file of files1) {
      if (files2.has(file)) {
        return true;
      }
    }

    return false;
  }
}
