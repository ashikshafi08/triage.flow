/**
 * Context Cache
 *
 * TTL-based caching with LRU eviction for tool results.
 *
 * @module context/contextCache
 */

import type { CacheEntry, CacheStats } from "./types";

// Default configuration
const DEFAULT_MAX_SIZE = 1000;
const DEFAULT_TTL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * TTL-based cache with LRU eviction.
 *
 * Features:
 * - Automatic expiration based on TTL
 * - LRU eviction when size limit reached
 * - Hit/miss statistics
 * - Cache key generation from tool name + parameters
 */
export class ContextCache {
  private cache: Map<string, CacheEntry>;
  private readonly maxSize: number;
  private readonly defaultTtlMs: number;
  private hitCount = 0;
  private missCount = 0;
  private evictionCount = 0;

  constructor(options?: { maxSize?: number; defaultTtlMs?: number }) {
    this.cache = new Map();
    this.maxSize = options?.maxSize ?? DEFAULT_MAX_SIZE;
    this.defaultTtlMs = options?.defaultTtlMs ?? DEFAULT_TTL_MS;
  }

  /**
   * Get a cached value by key.
   * Returns undefined if not found or expired.
   */
  get(key: string): unknown | undefined {
    const entry = this.cache.get(key);

    if (!entry) {
      this.missCount++;
      return undefined;
    }

    if (this.isExpired(entry)) {
      this.cache.delete(key);
      this.missCount++;
      return undefined;
    }

    // Update hit count and move to end (most recently used)
    entry.hitCount++;
    this.hitCount++;
    this.cache.delete(key);
    this.cache.set(key, entry);

    return entry.value;
  }

  /**
   * Get a cached value for a tool call.
   */
  getForTool(
    toolName: string,
    parameters: Record<string, unknown>
  ): unknown | undefined {
    const key = this.generateKey(toolName, parameters);
    return this.get(key);
  }

  /**
   * Set a cached value with optional TTL.
   */
  set(key: string, value: unknown, ttlMs?: number): void {
    // Evict if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictOldest(1);
    }

    const now = new Date();
    const ttl = ttlMs ?? this.defaultTtlMs;

    const entry: CacheEntry = {
      value,
      timestamp: now,
      expiresAt: new Date(now.getTime() + ttl),
      toolName: "",
      hitCount: 0,
    };

    this.cache.set(key, entry);
  }

  /**
   * Set a cached value for a tool call.
   */
  setForTool(
    toolName: string,
    parameters: Record<string, unknown>,
    value: unknown,
    ttlMs?: number
  ): void {
    const key = this.generateKey(toolName, parameters);

    // Evict if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictOldest(1);
    }

    const now = new Date();
    const ttl = ttlMs ?? this.defaultTtlMs;

    const entry: CacheEntry = {
      value,
      timestamp: now,
      expiresAt: new Date(now.getTime() + ttl),
      toolName,
      hitCount: 0,
    };

    this.cache.set(key, entry);
  }

  /**
   * Check if a key exists and is not expired.
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;
    if (this.isExpired(entry)) {
      this.cache.delete(key);
      return false;
    }
    return true;
  }

  /**
   * Check if a tool call is cached.
   */
  hasForTool(
    toolName: string,
    parameters: Record<string, unknown>
  ): boolean {
    const key = this.generateKey(toolName, parameters);
    return this.has(key);
  }

  /**
   * Delete a cached value.
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Delete a cached tool call.
   */
  deleteForTool(
    toolName: string,
    parameters: Record<string, unknown>
  ): boolean {
    const key = this.generateKey(toolName, parameters);
    return this.cache.delete(key);
  }

  /**
   * Clear all cache entries.
   */
  clear(): void {
    this.cache.clear();
    this.hitCount = 0;
    this.missCount = 0;
    this.evictionCount = 0;
  }

  /**
   * Invalidate all entries for a specific tool.
   */
  invalidateTool(toolName: string): number {
    let count = 0;
    for (const [key, entry] of this.cache.entries()) {
      if (entry.toolName === toolName) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  /**
   * Clean up expired entries.
   */
  cleanup(): number {
    let count = 0;
    for (const [key, entry] of this.cache.entries()) {
      if (this.isExpired(entry)) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  /**
   * Get cache statistics.
   */
  getStats(): CacheStats {
    const totalRequests = this.hitCount + this.missCount;
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate: totalRequests > 0 ? this.hitCount / totalRequests : 0,
      evictionCount: this.evictionCount,
    };
  }

  /**
   * Check if an entry is expired.
   */
  private isExpired(entry: CacheEntry): boolean {
    return new Date() > entry.expiresAt;
  }

  /**
   * Evict the oldest entries.
   */
  private evictOldest(count: number): void {
    const entries = Array.from(this.cache.entries());
    const toEvict = entries.slice(0, count);

    for (const [key] of toEvict) {
      this.cache.delete(key);
      this.evictionCount++;
    }
  }

  /**
   * Generate a cache key from tool name and parameters.
   */
  private generateKey(
    toolName: string,
    parameters: Record<string, unknown>
  ): string {
    // Sort keys for consistent hashing
    const sortedParams = Object.keys(parameters)
      .sort()
      .reduce(
        (acc, key) => {
          acc[key] = parameters[key];
          return acc;
        },
        {} as Record<string, unknown>
      );

    return `${toolName}:${JSON.stringify(sortedParams)}`;
  }
}
