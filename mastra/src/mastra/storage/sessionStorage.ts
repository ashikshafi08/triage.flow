// LibSQL-based session management, file checksums, and caching.

import { createClient, type Client } from "@libsql/client";
import { createSingleton } from "../utils/singleton";
import { getStorageConfig } from "./config";
import type {
  Session,
  CreateSessionParams,
  UpdateSessionParams,
  FileChecksum,
  FileInfo,
  ChangeSet,
  CacheNamespace,
  SessionRow,
  FileChecksumRow,
  CacheRow,
} from "./types";

/**
 * Session storage manager with cache operations.
 *
 * This class handles:
 * - Session CRUD operations
 * - File checksum tracking for incremental indexing
 * - TTL-based caching (replaces Redis)
 */
export class SessionStorage {
  private client: Client;
  private initialized = false;

  constructor(connectionUrl?: string, authToken?: string) {
    const config = getStorageConfig();
    this.client = createClient({
      url: connectionUrl || config.databaseUrl,
      authToken: authToken || config.authToken,
    });
  }

  /**
   * Initialize storage tables.
   * Creates custom tables for sessions, checksums, and cache.
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    // Sessions table
    await this.client.execute(`
      CREATE TABLE IF NOT EXISTS triage_sessions (
        id TEXT PRIMARY KEY,
        repo_url TEXT NOT NULL,
        repo_owner TEXT NOT NULL,
        repo_name TEXT NOT NULL,
        branch TEXT DEFAULT 'main',
        status TEXT DEFAULT 'initializing',
        total_files INTEGER DEFAULT 0,
        total_chunks INTEGER DEFAULT 0,
        embedding_model TEXT DEFAULT 'text-embedding-3-small',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT
      )
    `);

    // File checksums for incremental indexing
    await this.client.execute(`
      CREATE TABLE IF NOT EXISTS triage_file_checksums (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        file_mtime REAL NOT NULL,
        checksum TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(session_id, file_path)
      )
    `);

    // Cache table (replaces Redis)
    await this.client.execute(`
      CREATE TABLE IF NOT EXISTS triage_cache (
        cache_key TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        value TEXT NOT NULL,
        ttl_seconds INTEGER DEFAULT 3600,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
      )
    `);

    // Indexes for efficient queries
    await this.client.execute(`
      CREATE INDEX IF NOT EXISTS idx_checksums_session
      ON triage_file_checksums(session_id)
    `);
    await this.client.execute(`
      CREATE INDEX IF NOT EXISTS idx_cache_namespace
      ON triage_cache(namespace)
    `);
    await this.client.execute(`
      CREATE INDEX IF NOT EXISTS idx_cache_expires
      ON triage_cache(expires_at)
    `);

    this.initialized = true;
  }

  // ========================================
  // Session Operations
  // ========================================

  /**
   * Create a new session.
   */
  async createSession(params: CreateSessionParams): Promise<Session> {
    await this.init();

    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    const config = getStorageConfig();

    await this.client.execute({
      sql: `
        INSERT INTO triage_sessions
        (id, repo_url, repo_owner, repo_name, branch, embedding_model, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `,
      args: [
        id,
        params.repoUrl,
        params.repoOwner,
        params.repoName,
        params.branch || "main",
        params.embeddingModel || config.embeddingModel,
        now,
        now,
      ],
    });

    return {
      id,
      repoUrl: params.repoUrl,
      repoOwner: params.repoOwner,
      repoName: params.repoName,
      branch: params.branch || "main",
      status: "initializing",
      totalFiles: 0,
      totalChunks: 0,
      createdAt: new Date(now),
      updatedAt: new Date(now),
    };
  }

  /**
   * Get session by ID.
   */
  async getSession(sessionId: string): Promise<Session | null> {
    await this.init();

    const result = await this.client.execute({
      sql: "SELECT * FROM triage_sessions WHERE id = ?",
      args: [sessionId],
    });

    if (result.rows.length === 0) return null;

    return this.rowToSession(result.rows[0] as unknown as SessionRow);
  }

  /**
   * Find session by repository.
   */
  async findSessionByRepo(
    repoOwner: string,
    repoName: string,
    branch = "main"
  ): Promise<Session | null> {
    await this.init();

    const result = await this.client.execute({
      sql: `
        SELECT * FROM triage_sessions
        WHERE repo_owner = ? AND repo_name = ? AND branch = ?
        ORDER BY created_at DESC LIMIT 1
      `,
      args: [repoOwner, repoName, branch],
    });

    if (result.rows.length === 0) return null;

    return this.rowToSession(result.rows[0] as unknown as SessionRow);
  }

  /**
   * Update session fields.
   */
  async updateSession(sessionId: string, updates: UpdateSessionParams): Promise<Session | null> {
    await this.init();

    const now = new Date().toISOString();
    const setClauses: string[] = ["updated_at = ?"];
    const args: (string | number)[] = [now];

    if (updates.status !== undefined) {
      setClauses.push("status = ?");
      args.push(updates.status);
    }
    if (updates.totalFiles !== undefined) {
      setClauses.push("total_files = ?");
      args.push(updates.totalFiles);
    }
    if (updates.totalChunks !== undefined) {
      setClauses.push("total_chunks = ?");
      args.push(updates.totalChunks);
    }
    if (updates.metadata !== undefined) {
      setClauses.push("metadata = ?");
      args.push(JSON.stringify(updates.metadata));
    }

    args.push(sessionId);

    await this.client.execute({
      sql: `UPDATE triage_sessions SET ${setClauses.join(", ")} WHERE id = ?`,
      args,
    });

    return this.getSession(sessionId);
  }

  /**
   * Delete session and associated data.
   */
  async deleteSession(sessionId: string): Promise<void> {
    await this.init();

    await this.client.batch([
      {
        sql: "DELETE FROM triage_file_checksums WHERE session_id = ?",
        args: [sessionId],
      },
      {
        sql: "DELETE FROM triage_sessions WHERE id = ?",
        args: [sessionId],
      },
    ]);
  }

  /**
   * List all sessions.
   */
  async listSessions(): Promise<Session[]> {
    await this.init();

    const result = await this.client.execute(
      "SELECT * FROM triage_sessions ORDER BY created_at DESC"
    );

    return result.rows.map((row: Record<string, unknown>) =>
      this.rowToSession(row as unknown as SessionRow)
    );
  }

  // ========================================
  // File Checksum Operations
  // ========================================

  /**
   * Save file checksums for a session.
   */
  async saveFileChecksums(
    sessionId: string,
    checksums: Omit<FileChecksum, "sessionId" | "createdAt">[]
  ): Promise<void> {
    await this.init();

    if (checksums.length === 0) return;

    const now = new Date().toISOString();
    const statements = checksums.map((cs) => ({
      sql: `
        INSERT OR REPLACE INTO triage_file_checksums
        (session_id, file_path, file_size, file_mtime, checksum, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `,
      args: [sessionId, cs.filePath, cs.fileSize, cs.fileMtime, cs.checksum, now],
    }));

    await this.client.batch(statements);
  }

  /**
   * Get all file checksums for a session.
   */
  async getFileChecksums(sessionId: string): Promise<FileChecksum[]> {
    await this.init();

    const result = await this.client.execute({
      sql: "SELECT * FROM triage_file_checksums WHERE session_id = ?",
      args: [sessionId],
    });

    return result.rows.map((row: Record<string, unknown>) => {
      const r = row as unknown as FileChecksumRow;
      return {
        sessionId: r.session_id,
        filePath: r.file_path,
        fileSize: r.file_size,
        fileMtime: r.file_mtime,
        checksum: r.checksum,
        createdAt: new Date(r.created_at),
      };
    });
  }

  /**
   * Detect file changes by comparing current files to stored checksums.
   */
  async detectChanges(sessionId: string, currentFiles: FileInfo[]): Promise<ChangeSet> {
    const storedChecksums = await this.getFileChecksums(sessionId);
    const storedMap = new Map(storedChecksums.map((c) => [c.filePath, c]));
    const currentPaths = new Set(currentFiles.map((f) => f.path));

    const added: string[] = [];
    const modified: string[] = [];
    const unchanged: string[] = [];
    const deleted: string[] = [];

    for (const file of currentFiles) {
      const stored = storedMap.get(file.path);
      if (!stored) {
        added.push(file.path);
      } else if (stored.fileSize !== file.size || stored.fileMtime !== file.mtime) {
        modified.push(file.path);
      } else {
        unchanged.push(file.path);
      }
    }

    for (const stored of storedChecksums) {
      if (!currentPaths.has(stored.filePath)) {
        deleted.push(stored.filePath);
      }
    }

    return { added, modified, deleted, unchanged };
  }

  /**
   * Delete checksums for specific files.
   */
  async deleteFileChecksums(sessionId: string, filePaths: string[]): Promise<void> {
    await this.init();

    if (filePaths.length === 0) return;

    const placeholders = filePaths.map(() => "?").join(", ");
    await this.client.execute({
      sql: `
        DELETE FROM triage_file_checksums
        WHERE session_id = ? AND file_path IN (${placeholders})
      `,
      args: [sessionId, ...filePaths],
    });
  }

  // ========================================
  // Cache Operations (replaces Redis)
  // ========================================

  /**
   * Get cached value by namespace and key.
   */
  async cacheGet<T>(namespace: CacheNamespace, key: string): Promise<T | null> {
    await this.init();

    const cacheKey = `${namespace}:${key}`;
    const now = new Date().toISOString();

    const result = await this.client.execute({
      sql: `
        SELECT value FROM triage_cache
        WHERE cache_key = ? AND expires_at > ?
      `,
      args: [cacheKey, now],
    });

    if (result.rows.length === 0) return null;

    try {
      return JSON.parse(result.rows[0].value as string) as T;
    } catch {
      return null;
    }
  }

  /**
   * Set cached value with TTL.
   */
  async cacheSet<T>(
    namespace: CacheNamespace,
    key: string,
    value: T,
    ttlSeconds?: number
  ): Promise<void> {
    await this.init();

    const config = getStorageConfig();
    const ttl =
      ttlSeconds ??
      (namespace === "rag"
        ? config.cacheTtlRag
        : namespace === "folder"
          ? config.cacheTtlFolder
          : config.cacheTtlResponse);

    const cacheKey = `${namespace}:${key}`;
    const now = new Date();
    const expiresAt = new Date(now.getTime() + ttl * 1000);

    await this.client.execute({
      sql: `
        INSERT OR REPLACE INTO triage_cache
        (cache_key, namespace, value, ttl_seconds, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `,
      args: [
        cacheKey,
        namespace,
        JSON.stringify(value),
        ttl,
        now.toISOString(),
        expiresAt.toISOString(),
      ],
    });
  }

  /**
   * Delete cached value.
   */
  async cacheDelete(namespace: CacheNamespace, key: string): Promise<void> {
    await this.init();

    const cacheKey = `${namespace}:${key}`;
    await this.client.execute({
      sql: "DELETE FROM triage_cache WHERE cache_key = ?",
      args: [cacheKey],
    });
  }

  /**
   * Invalidate all cache entries matching a pattern.
   * Pattern uses SQL LIKE syntax (% for wildcard).
   */
  async cacheInvalidatePattern(pattern: string): Promise<number> {
    await this.init();

    const result = await this.client.execute({
      sql: "DELETE FROM triage_cache WHERE cache_key LIKE ?",
      args: [pattern],
    });

    return result.rowsAffected;
  }

  /**
   * Clean expired cache entries.
   * Returns number of entries removed.
   */
  async cleanExpiredCache(): Promise<number> {
    await this.init();

    const now = new Date().toISOString();
    const result = await this.client.execute({
      sql: "DELETE FROM triage_cache WHERE expires_at <= ?",
      args: [now],
    });

    return result.rowsAffected;
  }

  // ========================================
  // Utility Methods
  // ========================================

  /**
   * Convert database row to Session object.
   */
  private rowToSession(row: SessionRow): Session {
    return {
      id: row.id,
      repoUrl: row.repo_url,
      repoOwner: row.repo_owner,
      repoName: row.repo_name,
      branch: row.branch,
      status: row.status as Session["status"],
      totalFiles: row.total_files,
      totalChunks: row.total_chunks,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
    };
  }

  /**
   * Close database connection.
   */
  async close(): Promise<void> {
    this.client.close();
    this.initialized = false;
  }
}

export const getSessionStorage = createSingleton(() => new SessionStorage());
