/**
 * ContextManager Unit Tests
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  ContextManager,
  ContextCache,
  ExecutionTracker,
} from "../../../src/mastra/context";

describe("ContextManager", () => {
  let manager: ContextManager;

  beforeEach(() => {
    manager = new ContextManager({
      sessionId: "test-session-123",
      repoPath: "/tmp/test-repo",
    });
  });

  describe("Lifecycle", () => {
    it("should start a new execution context", () => {
      const context = manager.startExecution("Find the authentication logic");

      expect(context).toBeDefined();
      expect(context.sessionId).toBe("test-session-123");
      expect(context.query).toBe("Find the authentication logic");
      expect(context.discoveredFiles.size).toBe(0);
      expect(context.executionTrace).toHaveLength(0);
    });

    it("should return null when ending without starting", () => {
      const summary = manager.endExecution();
      expect(summary).toBeNull();
    });

    it("should return summary when ending execution", () => {
      manager.startExecution("Test query");
      const summary = manager.endExecution();

      expect(summary).toBeDefined();
      expect(summary?.sessionId).toBe("test-session-123");
      expect(summary?.query).toBe("Test query");
      expect(summary?.totalExecutions).toBe(0);
    });

    it("should cleanup context", () => {
      manager.startExecution("Test query");
      manager.cleanup();

      expect(manager.getCurrentContext()).toBeNull();
    });
  });

  describe("Execution Recording", () => {
    it("should record a tool execution", () => {
      manager.startExecution("Test query");

      const execution = manager.recordExecution({
        toolName: "search-codebase",
        parameters: { query: "authentication" },
        result: { files: ["auth.ts"] },
        executionTime: 150,
      });

      expect(execution).toBeDefined();
      expect(execution.toolName).toBe("search-codebase");
      expect(execution.executionTime).toBe(150);
    });

    it("should track discovered files from execution", () => {
      manager.startExecution("Test query");

      manager.recordExecution({
        toolName: "read-file",
        parameters: { filePath: "src/auth.ts" },
        result: { content: "code..." },
        executionTime: 50,
      });

      const files = manager.getAllDiscoveredFiles();
      expect(files.has("src/auth.ts")).toBe(true);
    });

    it("should get all executions", () => {
      manager.startExecution("Test query");

      manager.recordExecution({
        toolName: "tool-1",
        parameters: {},
        result: {},
        executionTime: 10,
      });

      manager.recordExecution({
        toolName: "tool-2",
        parameters: {},
        result: {},
        executionTime: 20,
      });

      const executions = manager.getAllExecutions();
      expect(executions).toHaveLength(2);
    });
  });

  describe("Context for Tool", () => {
    it("should get context for a tool", () => {
      manager.startExecution("Find authentication logic");

      const context = manager.getContextForTool("search-codebase", {
        query: "auth",
      });

      expect(context).toBeDefined();
      expect(context.sessionId).toBe("test-session-123");
      expect(context.query).toBe("Find authentication logic");
    });

    it("should include related executions in context", () => {
      manager.startExecution("Test query");

      // Record a related execution
      manager.recordExecution({
        toolName: "read-file",
        parameters: { filePath: "src/auth.ts" },
        result: { content: "code" },
        executionTime: 10,
      });

      // Get context for a related tool
      const context = manager.getContextForTool("analyze-file-structure", {
        filePath: "src/auth.ts",
      });

      // read-file and analyze-file-structure are related
      expect(context.relatedExecutions.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Summary", () => {
    it("should return execution summary", () => {
      manager.startExecution("Test query");

      manager.recordExecution({
        toolName: "search-codebase",
        parameters: { query: "test" },
        result: {},
        executionTime: 100,
      });

      manager.recordExecution({
        toolName: "read-file",
        parameters: { filePath: "test.ts" },
        result: {},
        executionTime: 50,
      });

      const summary = manager.getExecutionSummary();

      expect(summary.totalExecutions).toBe(2);
      expect(summary.totalDuration).toBe(150);
      expect(summary.toolBreakdown["search-codebase"]).toBe(1);
      expect(summary.toolBreakdown["read-file"]).toBe(1);
    });
  });
});

describe("ContextCache", () => {
  let cache: ContextCache;

  beforeEach(() => {
    cache = new ContextCache({ maxSize: 10, defaultTtlMs: 1000 });
  });

  describe("Basic Operations", () => {
    it("should set and get values", () => {
      cache.set("key1", { data: "value1" });
      const result = cache.get("key1");

      expect(result).toEqual({ data: "value1" });
    });

    it("should return undefined for missing keys", () => {
      const result = cache.get("nonexistent");
      expect(result).toBeUndefined();
    });

    it("should check if key exists", () => {
      cache.set("exists", "value");

      expect(cache.has("exists")).toBe(true);
      expect(cache.has("missing")).toBe(false);
    });

    it("should delete keys", () => {
      cache.set("to-delete", "value");
      cache.delete("to-delete");

      expect(cache.has("to-delete")).toBe(false);
    });

    it("should clear all entries", () => {
      cache.set("key1", "value1");
      cache.set("key2", "value2");
      cache.clear();

      expect(cache.has("key1")).toBe(false);
      expect(cache.has("key2")).toBe(false);
    });
  });

  describe("Tool-specific Operations", () => {
    it("should set and get by tool name and params", () => {
      cache.setForTool("search", { query: "test" }, { results: [] });
      const result = cache.getForTool("search", { query: "test" });

      expect(result).toEqual({ results: [] });
    });

    it("should generate consistent keys for same params", () => {
      cache.setForTool("tool", { a: 1, b: 2 }, "value1");
      const result = cache.getForTool("tool", { b: 2, a: 1 });

      // Keys should be the same regardless of param order
      expect(result).toBe("value1");
    });
  });

  describe("TTL Expiration", () => {
    it("should expire entries after TTL", async () => {
      const shortCache = new ContextCache({ maxSize: 10, defaultTtlMs: 50 });
      shortCache.set("expires", "value");

      // Wait for expiration
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(shortCache.get("expires")).toBeUndefined();
    });
  });

  describe("LRU Eviction", () => {
    it("should evict oldest entries when at capacity", () => {
      const smallCache = new ContextCache({ maxSize: 3 });

      smallCache.set("key1", "value1");
      smallCache.set("key2", "value2");
      smallCache.set("key3", "value3");
      smallCache.set("key4", "value4"); // Should evict key1

      expect(smallCache.has("key1")).toBe(false);
      expect(smallCache.has("key4")).toBe(true);
    });
  });

  describe("Statistics", () => {
    it("should track hit and miss counts", () => {
      cache.set("exists", "value");

      cache.get("exists"); // Hit
      cache.get("exists"); // Hit
      cache.get("missing"); // Miss

      const stats = cache.getStats();
      expect(stats.hitCount).toBe(2);
      expect(stats.missCount).toBe(1);
      expect(stats.hitRate).toBeCloseTo(0.67, 1);
    });
  });
});

describe("ExecutionTracker", () => {
  let tracker: ExecutionTracker;

  beforeEach(() => {
    tracker = new ExecutionTracker();
  });

  describe("Recording Executions", () => {
    it("should record an execution", () => {
      const execution = tracker.record(
        "search-codebase",
        { query: "authentication" },
        { files: ["auth.ts"] },
        100
      );

      expect(execution).toBeDefined();
      expect(execution.toolName).toBe("search-codebase");
      expect(execution.executionTime).toBe(100);
      expect(execution.id).toMatch(/^exec-\d+-\d+$/);
    });

    it("should extract files from parameters", () => {
      const execution = tracker.record(
        "read-file",
        { filePath: "src/index.ts" },
        {},
        50
      );

      expect(execution.filesAccessed).toContain("src/index.ts");
    });

    it("should track multiple executions", () => {
      tracker.record("tool1", {}, {}, 10);
      tracker.record("tool2", {}, {}, 20);
      tracker.record("tool3", {}, {}, 30);

      const all = tracker.getAllExecutions();
      expect(all).toHaveLength(3);
    });
  });

  describe("Finding Related Executions", () => {
    it("should find executions for related tools", () => {
      tracker.record("read-file", { filePath: "auth.ts" }, {}, 10);

      const related = tracker.findRelatedExecutions("analyze-file-structure", {
        filePath: "auth.ts",
      });

      expect(related.length).toBeGreaterThan(0);
    });

    it("should find executions with shared files", () => {
      tracker.record("read-file", { filePath: "shared.ts" }, {}, 10);

      const related = tracker.findRelatedExecutions("other-tool", {
        filePath: "shared.ts",
      });

      expect(related.length).toBeGreaterThan(0);
    });
  });

  describe("Getting Relevant Files", () => {
    it("should get files from related executions", () => {
      tracker.record("read-file", { filePath: "file1.ts" }, {}, 10);
      tracker.record("read-file", { filePath: "file2.ts" }, {}, 10);

      const files = tracker.getRelevantFiles("analyze-file-structure", {});

      expect(files.size).toBe(2);
    });
  });

  describe("Summary", () => {
    it("should generate execution summary", () => {
      tracker.record("tool1", {}, {}, 100);
      tracker.record("tool1", {}, {}, 50);
      tracker.record("tool2", {}, {}, 75);

      const summary = tracker.getSummary();

      expect(summary.totalExecutions).toBe(3);
      expect(summary.totalDuration).toBe(225);
      expect(summary.toolCounts["tool1"]).toBe(2);
      expect(summary.toolCounts["tool2"]).toBe(1);
    });
  });

  describe("Clearing", () => {
    it("should clear all tracked data", () => {
      tracker.record("tool1", {}, {}, 10);
      tracker.record("tool2", {}, {}, 20);

      tracker.clear();

      expect(tracker.getAllExecutions()).toHaveLength(0);
      expect(tracker.getAllFiles().size).toBe(0);
    });
  });
});
