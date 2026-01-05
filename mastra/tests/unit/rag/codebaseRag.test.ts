/**
 * CodebaseRag Unit Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { RepoInfo, ContextResult, SourceDocument } from "../../../src/mastra/rag/types";

// Mock VectorStore
const mockVectorStore = {
  ensureIndex: vi.fn().mockResolvedValue(undefined),
  storeEmbeddings: vi.fn().mockResolvedValue(["id-1", "id-2"]),
  search: vi.fn().mockResolvedValue([
    {
      id: "id-1",
      score: 0.95,
      metadata: {
        sessionId: "session-123",
        filePath: "src/index.ts",
        content: "export const main = () => {};",
        language: "typescript",
        chunkIndex: 0,
        startLine: 1,
        endLine: 10,
      },
    },
    {
      id: "id-2",
      score: 0.85,
      metadata: {
        sessionId: "session-123",
        filePath: "src/utils.ts",
        content: "export const utils = {};",
        language: "typescript",
        chunkIndex: 0,
        startLine: 1,
        endLine: 5,
      },
    },
  ]),
  searchBySession: vi.fn().mockResolvedValue([
    {
      id: "id-1",
      score: 0.95,
      metadata: {
        sessionId: "session-123",
        filePath: "src/index.ts",
        content: "export const main = () => {};",
        language: "typescript",
        chunkIndex: 0,
        startLine: 1,
        endLine: 10,
      },
    },
  ]),
  deleteBySession: vi.fn().mockResolvedValue(undefined),
  close: vi.fn().mockResolvedValue(undefined),
};

// Mock SessionStorage
const mockSessionStorage = {
  createSession: vi.fn().mockResolvedValue({
    id: "session-123",
    repoUrl: "https://github.com/test/repo",
    repoOwner: "test",
    repoName: "repo",
    branch: "main",
    status: "initializing",
  }),
  findSessionByRepo: vi.fn().mockResolvedValue(null),
  updateSession: vi.fn().mockResolvedValue(undefined),
  deleteSession: vi.fn().mockResolvedValue(undefined),
  close: vi.fn().mockResolvedValue(undefined),
};

// Mock storage module
vi.mock("../../../src/mastra/storage", () => ({
  getVectorStore: vi.fn(() => mockVectorStore),
  getSessionStorage: vi.fn(() => mockSessionStorage),
  VECTOR_INDEXES: {
    CODE_EMBEDDINGS: "code_embeddings",
    ISSUE_EMBEDDINGS: "issue_embeddings",
    DOC_EMBEDDINGS: "doc_embeddings",
  },
}));

// Mock repositoryLoader
vi.mock("../../../src/mastra/rag/repositoryLoader", () => ({
  cloneRepository: vi.fn().mockResolvedValue({
    repoPath: "/tmp/test-repo",
    repoInfo: {
      owner: "test",
      repo: "repo",
      branch: "main",
      url: "https://github.com/test/repo",
      repoPath: "/tmp/test-repo",
    },
  }),
  loadFiles: vi.fn().mockResolvedValue([
    {
      path: "/tmp/test-repo/src/index.ts",
      relativePath: "src/index.ts",
      content: "export const main = () => console.log('main');",
      language: "typescript",
      metadata: { language: "typescript", displayName: "TypeScript", description: "", docPattern: null, importPattern: null },
    },
    {
      path: "/tmp/test-repo/src/utils.ts",
      relativePath: "src/utils.ts",
      content: "export const utils = {};",
      language: "typescript",
      metadata: { language: "typescript", displayName: "TypeScript", description: "", docPattern: null, importPattern: null },
    },
  ]),
  searchFiles: vi.fn().mockResolvedValue([
    "src/index.ts",
    "src/utils.ts",
  ]),
  parseRepoUrl: vi.fn().mockImplementation((url: string) => {
    const match = url.match(/github\.com[\/:]([^\/]+)\/([^\/\.]+)/);
    if (match) {
      return { owner: match[1], repo: match[2].replace(/\.git$/, "") };
    }
    return { owner: "unknown", repo: "unknown" };
  }),
}));

// Mock codeChunker
vi.mock("../../../src/mastra/rag/codeChunker", () => ({
  chunkFiles: vi.fn().mockResolvedValue([
    {
      id: "src/index.ts:0",
      content: "export const main = () => console.log('main');",
      metadata: { filePath: "src/index.ts", language: "typescript", chunkIndex: 0, startLine: 1, endLine: 1 },
    },
    {
      id: "src/utils.ts:0",
      content: "export const utils = {};",
      metadata: { filePath: "src/utils.ts", language: "typescript", chunkIndex: 0, startLine: 1, endLine: 1 },
    },
  ]),
  getChunkStats: vi.fn().mockReturnValue({
    totalChunks: 2,
    byLanguage: { typescript: 2 },
    avgChunkSize: 40,
  }),
}));

// Mock AI SDK
vi.mock("ai", () => ({
  embed: vi.fn().mockResolvedValue({
    embedding: new Array(1536).fill(0).map(() => Math.random()),
  }),
  embedMany: vi.fn().mockResolvedValue({
    embeddings: [
      new Array(1536).fill(0).map(() => Math.random()),
      new Array(1536).fill(0).map(() => Math.random()),
    ],
  }),
}));

vi.mock("@ai-sdk/openai", () => ({
  openai: {
    embedding: vi.fn().mockReturnValue({ modelId: "text-embedding-3-small" }),
  },
}));

// Import after mocking
import { CodebaseRag, createCodebaseRag } from "../../../src/mastra/rag/codebaseRag";
import { parseRepoUrl } from "../../../src/mastra/rag/repositoryLoader";

describe("CodebaseRag", () => {
  let rag: CodebaseRag;

  beforeEach(() => {
    vi.clearAllMocks();
    rag = new CodebaseRag();
  });

  describe("Constructor", () => {
    it("should create a CodebaseRag instance", () => {
      expect(rag).toBeDefined();
      expect(rag).toBeInstanceOf(CodebaseRag);
    });

    it("should accept optional configuration", () => {
      const customRag = new CodebaseRag({
        chunkSize: 500,
        chunkOverlap: 100,
        topK: 5,
        minScore: 0.5,
      });

      expect(customRag).toBeDefined();
    });
  });

  describe("createCodebaseRag factory", () => {
    it("should create a CodebaseRag instance", () => {
      const instance = createCodebaseRag();
      expect(instance).toBeInstanceOf(CodebaseRag);
    });

    it("should pass config to instance", () => {
      const instance = createCodebaseRag({ topK: 20 });
      expect(instance).toBeInstanceOf(CodebaseRag);
    });
  });

  describe("loadRepository", () => {
    it("should load a repository from URL", async () => {
      const repoUrl = "https://github.com/test/repo";

      const repoInfo = await rag.loadRepository(repoUrl);

      expect(repoInfo).toBeDefined();
      expect(repoInfo.owner).toBe("test");
      expect(repoInfo.repo).toBe("repo");
    });

    it("should load with specific branch", async () => {
      const repoUrl = "https://github.com/test/repo";

      const repoInfo = await rag.loadRepository(repoUrl, { branch: "develop" });

      expect(repoInfo).toBeDefined();
    });

    it("should associate with session ID", async () => {
      const repoUrl = "https://github.com/test/repo";

      const repoInfo = await rag.loadRepository(repoUrl, {
        sessionId: "custom-session",
      });

      expect(repoInfo).toBeDefined();
      expect(rag.getSessionId()).toBe("custom-session");
    });

    it("should set indexed to true after loading", async () => {
      const repoUrl = "https://github.com/test/repo";

      await rag.loadRepository(repoUrl);

      expect(rag.isIndexed()).toBe(true);
    });
  });

  describe("getRelevantContext", () => {
    beforeEach(async () => {
      await rag.loadRepository("https://github.com/test/repo");
    });

    it("should return relevant context for a query", async () => {
      const result = await rag.getRelevantContext("authentication logic");

      expect(result).toBeDefined();
      expect(result.sources).toBeInstanceOf(Array);
      expect(result.query).toBe("authentication logic");
    });

    it("should include file paths in sources", async () => {
      const result = await rag.getRelevantContext("main function");

      for (const source of result.sources) {
        expect(source.file).toBeDefined();
        expect(typeof source.file).toBe("string");
      }
    });

    it("should include relevance scores", async () => {
      const result = await rag.getRelevantContext("utilities");

      for (const source of result.sources) {
        expect(source.score).toBeDefined();
        expect(typeof source.score).toBe("number");
      }
    });

    it("should include repoInfo in result", async () => {
      const result = await rag.getRelevantContext("test");

      expect(result.repoInfo).toBeDefined();
      expect(result.repoInfo.owner).toBe("test");
      expect(result.repoInfo.repo).toBe("repo");
    });

    it("should include searchType in result", async () => {
      const result = await rag.getRelevantContext("test query");

      expect(result.searchType).toBeDefined();
      expect(["semantic", "file_oriented", "hybrid"]).toContain(result.searchType);
    });
  });

  describe("getIssueContext", () => {
    beforeEach(async () => {
      await rag.loadRepository("https://github.com/test/repo");
    });

    it("should get context for an issue", async () => {
      const result = await rag.getIssueContext(
        "Bug in authentication",
        "The login function fails when..."
      );

      expect(result).toBeDefined();
      expect(result.sources).toBeInstanceOf(Array);
    });

    it("should combine title and body for search", async () => {
      const result = await rag.getIssueContext(
        "Add dark mode",
        "We need to implement a dark mode theme"
      );

      expect(result).toBeDefined();
      expect(result.query).toContain("Add dark mode");
      expect(result.query).toContain("dark mode theme");
    });

    it("should handle empty body", async () => {
      const result = await rag.getIssueContext("Simple bug fix", "");

      expect(result).toBeDefined();
    });
  });

  describe("State methods", () => {
    it("should return false for isIndexed before loading", () => {
      expect(rag.isIndexed()).toBe(false);
    });

    it("should return null for getRepoInfo before loading", () => {
      expect(rag.getRepoInfo()).toBeNull();
    });

    it("should return null for getSessionId before loading", () => {
      expect(rag.getSessionId()).toBeNull();
    });

    it("should return repoInfo after loading", async () => {
      await rag.loadRepository("https://github.com/test/repo");

      const repoInfo = rag.getRepoInfo();
      expect(repoInfo).not.toBeNull();
      expect(repoInfo?.owner).toBe("test");
    });
  });

  describe("deleteIndex", () => {
    it("should reset state after deleting", async () => {
      await rag.loadRepository("https://github.com/test/repo");
      expect(rag.isIndexed()).toBe(true);

      await rag.deleteIndex();

      expect(rag.isIndexed()).toBe(false);
      expect(rag.getRepoInfo()).toBeNull();
      expect(rag.getSessionId()).toBeNull();
    });

    it("should call vectorStore.deleteBySession", async () => {
      await rag.loadRepository("https://github.com/test/repo");
      await rag.deleteIndex();

      expect(mockVectorStore.deleteBySession).toHaveBeenCalled();
    });
  });
});

describe("parseRepoUrl", () => {
  it("should parse HTTPS URLs", () => {
    const result = parseRepoUrl("https://github.com/owner/repo");

    expect(result.owner).toBe("owner");
    expect(result.repo).toBe("repo");
  });

  it("should parse SSH URLs", () => {
    const result = parseRepoUrl("git@github.com:owner/repo.git");

    expect(result.owner).toBe("owner");
    expect(result.repo).toBe("repo");
  });

  it("should handle .git suffix", () => {
    const result = parseRepoUrl("https://github.com/owner/repo.git");

    expect(result.owner).toBe("owner");
    expect(result.repo).toBe("repo");
  });
});

describe("CodebaseRag Error Handling", () => {
  let rag: CodebaseRag;

  beforeEach(() => {
    vi.clearAllMocks();
    rag = new CodebaseRag();
  });

  it("should throw when getting context without loading", async () => {
    await expect(rag.getRelevantContext("test")).rejects.toThrow(
      "Repository not loaded"
    );
  });

  it("should throw when getting issue context without loading", async () => {
    await expect(
      rag.getIssueContext("title", "body")
    ).rejects.toThrow("Repository not loaded");
  });
});
