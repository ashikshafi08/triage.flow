// Common interfaces and type definitions used across modules.

// Session types
export interface Session {
  id: string;
  repoUrl: string;
  repoOwner: string;
  repoName: string;
  branch: string;
  status: "initializing" | "indexing" | "ready" | "error";
  totalFiles: number;
  totalChunks: number;
  createdAt: Date;
  updatedAt: Date;
}

// Repository types
export interface RepoInfo {
  owner: string;
  repo: string;
  branch: string;
  url: string;
  repoPath: string;
}

// Search result types
export interface SearchResult {
  id: string;
  content: string;
  filePath: string;
  language: string;
  score: number;
  metadata: Record<string, unknown>;
}

// Tool execution types
export interface ToolExecution {
  toolName: string;
  parameters: Record<string, unknown>;
  result: unknown;
  executionTime: number;
  timestamp: Date;
}
