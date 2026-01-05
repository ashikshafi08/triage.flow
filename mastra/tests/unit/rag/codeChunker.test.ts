/**
 * Code Chunker Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { LoadedFile, DocumentChunk, LanguageMetadata } from "../../../src/mastra/rag/types";

// Mock @mastra/rag
vi.mock("@mastra/rag", () => ({
  MDocument: {
    fromText: vi.fn().mockImplementation((text: string) => ({
      chunk: vi.fn().mockResolvedValue([
        { text: text.slice(0, Math.min(500, text.length)), metadata: { loc: { startIndex: 0 } } },
        ...(text.length > 500
          ? [{ text: text.slice(500), metadata: { loc: { startIndex: 500 } } }]
          : []),
      ]),
    })),
    fromMarkdown: vi.fn().mockImplementation((text: string) => ({
      chunk: vi.fn().mockResolvedValue([
        { text, metadata: { loc: { startIndex: 0 } } },
      ]),
    })),
    fromHTML: vi.fn().mockImplementation((text: string) => ({
      chunk: vi.fn().mockResolvedValue([
        { text, metadata: { loc: { startIndex: 0 } } },
      ]),
    })),
    fromJSON: vi.fn().mockImplementation((text: string) => ({
      chunk: vi.fn().mockResolvedValue([
        { text, metadata: { loc: { startIndex: 0 } } },
      ]),
    })),
  },
}));

// Import after mocking
import { chunkFile, chunkFiles, estimateChunkCount, getChunkStats } from "../../../src/mastra/rag/codeChunker";

// Helper to create metadata
function createMetadata(language: string): LanguageMetadata {
  return {
    language,
    displayName: language.charAt(0).toUpperCase() + language.slice(1),
    description: `${language} file`,
    docPattern: null,
    importPattern: null,
  };
}

describe("chunkFile", () => {
  const sampleFile: LoadedFile = {
    path: "/tmp/repo/src/index.ts",
    relativePath: "src/index.ts",
    content: `
import { Mastra } from "@mastra/core";

export class MyApp {
  private mastra: Mastra;

  constructor() {
    this.mastra = new Mastra({
      agents: [],
      tools: [],
    });
  }

  async run() {
    console.log("Running app");
  }
}
`.trim(),
    language: "typescript",
    metadata: createMetadata("typescript"),
  };

  it("should chunk a file and return DocumentChunks", async () => {
    const chunks = await chunkFile(sampleFile);

    expect(Array.isArray(chunks)).toBe(true);
    expect(chunks.length).toBeGreaterThan(0);
  });

  it("should preserve file metadata in chunks", async () => {
    const chunks = await chunkFile(sampleFile);

    for (const chunk of chunks) {
      expect(chunk.metadata.filePath).toBe(sampleFile.relativePath);
      expect(chunk.metadata.language).toBe(sampleFile.language);
    }
  });

  it("should assign chunk indices", async () => {
    const chunks = await chunkFile(sampleFile);

    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].metadata.chunkIndex).toBe(i);
    }
  });

  it("should have unique chunk IDs", async () => {
    const chunks = await chunkFile(sampleFile);
    const ids = new Set(chunks.map((c) => c.id));

    expect(ids.size).toBe(chunks.length);
  });

  it("should use provided chunk options", async () => {
    const chunks = await chunkFile(sampleFile, {
      maxSize: 100,
      overlap: 20,
    });

    expect(Array.isArray(chunks)).toBe(true);
  });
});

describe("chunkFiles", () => {
  const sampleFiles: LoadedFile[] = [
    {
      path: "/tmp/repo/src/index.ts",
      relativePath: "src/index.ts",
      content: "export const main = () => console.log('main');",
      language: "typescript",
      metadata: createMetadata("typescript"),
    },
    {
      path: "/tmp/repo/src/utils.ts",
      relativePath: "src/utils.ts",
      content: "export const utils = { format: (s: string) => s.trim() };",
      language: "typescript",
      metadata: createMetadata("typescript"),
    },
    {
      path: "/tmp/repo/src/config.ts",
      relativePath: "src/config.ts",
      content: "export const config = { debug: true, version: '1.0.0' };",
      language: "typescript",
      metadata: createMetadata("typescript"),
    },
  ];

  it("should chunk multiple files", async () => {
    const chunks = await chunkFiles(sampleFiles);

    expect(Array.isArray(chunks)).toBe(true);
    expect(chunks.length).toBeGreaterThanOrEqual(sampleFiles.length);
  });

  it("should preserve file paths in chunks", async () => {
    const chunks = await chunkFiles(sampleFiles);

    const filePaths = new Set(chunks.map((c) => c.metadata.filePath));

    for (const file of sampleFiles) {
      expect(filePaths.has(file.relativePath)).toBe(true);
    }
  });

  it("should handle empty file list", async () => {
    const chunks = await chunkFiles([]);

    expect(chunks).toEqual([]);
  });

  it("should apply options to all files", async () => {
    const chunks = await chunkFiles(sampleFiles, {
      maxSize: 200,
      overlap: 50,
    });

    expect(Array.isArray(chunks)).toBe(true);
  });
});

describe("estimateChunkCount", () => {
  it("should estimate chunk count for content", () => {
    const content = "x".repeat(3000);
    const estimate = estimateChunkCount(content, 1000);

    expect(estimate).toBe(3);
  });

  it("should return at least 1 for small content", () => {
    const content = "small";
    const estimate = estimateChunkCount(content);

    expect(estimate).toBeGreaterThanOrEqual(1);
  });

  it("should use default max size", () => {
    const content = "x".repeat(4500);
    const estimate = estimateChunkCount(content); // Default 1500

    expect(estimate).toBe(3);
  });
});

describe("getChunkStats", () => {
  it("should calculate stats for chunks", () => {
    const chunks: DocumentChunk[] = [
      {
        id: "file1:0",
        content: "content1",
        metadata: { filePath: "file1.ts", language: "typescript", chunkIndex: 0 },
      },
      {
        id: "file1:1",
        content: "content2",
        metadata: { filePath: "file1.ts", language: "typescript", chunkIndex: 1 },
      },
      {
        id: "file2:0",
        content: "python content",
        metadata: { filePath: "file2.py", language: "python", chunkIndex: 0 },
      },
    ];

    const stats = getChunkStats(chunks);

    expect(stats.totalChunks).toBe(3);
    expect(stats.byLanguage.typescript).toBe(2);
    expect(stats.byLanguage.python).toBe(1);
    expect(stats.avgChunkSize).toBeGreaterThan(0);
  });

  it("should handle empty chunks array", () => {
    const stats = getChunkStats([]);

    expect(stats.totalChunks).toBe(0);
    expect(stats.avgChunkSize).toBe(0);
    expect(Object.keys(stats.byLanguage)).toHaveLength(0);
  });
});

describe("Chunking Edge Cases", () => {
  it("should handle empty file content", async () => {
    const emptyFile: LoadedFile = {
      path: "/tmp/empty.ts",
      relativePath: "empty.ts",
      content: "",
      language: "typescript",
      metadata: createMetadata("typescript"),
    };

    const chunks = await chunkFile(emptyFile);

    expect(Array.isArray(chunks)).toBe(true);
  });

  it("should handle very long files", async () => {
    const longContent = "x".repeat(5000);
    const longFile: LoadedFile = {
      path: "/tmp/long.ts",
      relativePath: "long.ts",
      content: longContent,
      language: "typescript",
      metadata: createMetadata("typescript"),
    };

    const chunks = await chunkFile(longFile);

    expect(chunks.length).toBeGreaterThan(1);
  });

  it("should handle files with special characters", async () => {
    const specialFile: LoadedFile = {
      path: "/tmp/special.ts",
      relativePath: "special.ts",
      content: "const emoji = '🎉'; const unicode = '你好';",
      language: "typescript",
      metadata: createMetadata("typescript"),
    };

    const chunks = await chunkFile(specialFile);

    expect(Array.isArray(chunks)).toBe(true);
    expect(chunks.length).toBeGreaterThan(0);
  });

  it("should handle markdown files", async () => {
    const mdFile: LoadedFile = {
      path: "/tmp/README.md",
      relativePath: "README.md",
      content: `# Title

## Section 1

Some content here.

## Section 2

More content.

\`\`\`typescript
const code = 'example';
\`\`\`
`,
      language: "markdown",
      metadata: createMetadata("markdown"),
    };

    const chunks = await chunkFile(mdFile);

    expect(Array.isArray(chunks)).toBe(true);
  });

  it("should handle JSON files", async () => {
    const jsonFile: LoadedFile = {
      path: "/tmp/package.json",
      relativePath: "package.json",
      content: JSON.stringify(
        {
          name: "test-package",
          version: "1.0.0",
          dependencies: {
            typescript: "^5.0.0",
          },
        },
        null,
        2
      ),
      language: "json",
      metadata: createMetadata("json"),
    };

    const chunks = await chunkFile(jsonFile);

    expect(Array.isArray(chunks)).toBe(true);
  });
});
