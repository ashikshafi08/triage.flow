// Language-aware document chunking using Mastra RAG.

import { MDocument } from "@mastra/rag";
import type { ChunkParams, RecursiveChunkOptions } from "@mastra/rag";
import type { LoadedFile, DocumentChunk, ChunkOptions } from "./types";

// Default chunk settings
const DEFAULT_MAX_SIZE = 1500;
const DEFAULT_OVERLAP = 200;

// Chunking strategies supported by Mastra RAG
type ChunkStrategy = "recursive" | "markdown" | "html" | "json" | "character";

/**
 * Get optimal chunking strategy based on language.
 */
function getChunkingStrategy(language: string): {
  strategy: ChunkStrategy;
  separators?: string[];
} {
  switch (language) {
    case "markdown":
      return { strategy: "markdown" };
    case "html":
      return { strategy: "html" };
    case "json":
      return { strategy: "json" };
    case "python":
      return {
        strategy: "recursive",
        separators: ["\nclass ", "\ndef ", "\n\n", "\n", " "],
      };
    case "javascript":
    case "typescript":
      return {
        strategy: "recursive",
        separators: ["\nfunction ", "\nconst ", "\nexport ", "\nclass ", "\n\n", "\n", " "],
      };
    case "java":
    case "kotlin":
    case "scala":
      return {
        strategy: "recursive",
        separators: ["\npublic ", "\nprivate ", "\nprotected ", "\nclass ", "\n\n", "\n", " "],
      };
    case "go":
      return {
        strategy: "recursive",
        separators: ["\nfunc ", "\ntype ", "\n\n", "\n", " "],
      };
    case "rust":
      return {
        strategy: "recursive",
        separators: ["\nfn ", "\nimpl ", "\nstruct ", "\nenum ", "\n\n", "\n", " "],
      };
    case "ruby":
      return {
        strategy: "recursive",
        separators: ["\nclass ", "\ndef ", "\nmodule ", "\n\n", "\n", " "],
      };
    default:
      return {
        strategy: "recursive",
        separators: ["\n\n", "\n", " "],
      };
  }
}

/**
 * Chunk a single file into document chunks.
 */
export async function chunkFile(
  file: LoadedFile,
  options: ChunkOptions = {}
): Promise<DocumentChunk[]> {
  const maxSize = options.maxSize || DEFAULT_MAX_SIZE;
  const overlap = options.overlap || DEFAULT_OVERLAP;

  // Create MDocument from file content
  const doc =
    file.language === "markdown"
      ? MDocument.fromMarkdown(file.content)
      : file.language === "html"
        ? MDocument.fromHTML(file.content)
        : file.language === "json"
          ? MDocument.fromJSON(file.content)
          : MDocument.fromText(file.content);

  // Get language-specific chunking strategy
  const { strategy, separators } = getChunkingStrategy(file.language);

  // Use the strategy from options or fallback
  const selectedStrategy = options.strategy || strategy;

  // Build typed chunk params based on strategy
  const chunkParams: ChunkParams =
    selectedStrategy === "recursive"
      ? ({
          strategy: "recursive",
          maxSize,
          overlap,
          separators,
          addStartIndex: true,
        } satisfies { strategy: "recursive" } & RecursiveChunkOptions)
      : selectedStrategy === "markdown"
        ? { strategy: "markdown", maxSize, overlap, addStartIndex: true }
        : selectedStrategy === "html"
          ? { strategy: "html", maxSize, overlap, addStartIndex: true, headers: [] }
          : selectedStrategy === "json"
            ? { strategy: "json", maxSize, overlap, addStartIndex: true }
            : { strategy: "character", maxSize, overlap, addStartIndex: true };

  const chunkedDocs = await doc.chunk(chunkParams);

  // Convert to DocumentChunk format
  const chunks: DocumentChunk[] = chunkedDocs.map(
    (node: { text: string; metadata?: Record<string, unknown> }, index: number) => {
      // Calculate approximate line numbers from start index
      const startIndex = (node.metadata?.loc as { startIndex?: number })?.startIndex || 0;
      const contentBeforeChunk = file.content.substring(0, startIndex);
      const startLine = contentBeforeChunk.split("\n").length;
      const endLine = startLine + node.text.split("\n").length - 1;

      return {
        id: `${file.relativePath}:${index}`,
        content: node.text,
        metadata: {
          filePath: file.relativePath,
          language: file.language,
          chunkIndex: index,
          startLine,
          endLine,
          totalChunks: chunkedDocs.length,
        },
      };
    }
  );

  return chunks;
}

/**
 * Chunk multiple files in batch.
 */
export async function chunkFiles(
  files: LoadedFile[],
  options: ChunkOptions = {}
): Promise<DocumentChunk[]> {
  const allChunks: DocumentChunk[] = [];

  for (const file of files) {
    try {
      const chunks = await chunkFile(file, options);
      allChunks.push(...chunks);
    } catch (error) {
      // Log error but continue with other files
      console.warn(`Failed to chunk file ${file.relativePath}:`, error);
    }
  }

  return allChunks;
}

/** Estimate the number of chunks a file will produce. */
export const estimateChunkCount = (content: string, maxSize = DEFAULT_MAX_SIZE) =>
  Math.max(1, Math.ceil(content.length / maxSize));

/**
 * Get chunk statistics for a set of files.
 */
export function getChunkStats(chunks: DocumentChunk[]): {
  totalChunks: number;
  byLanguage: Record<string, number>;
  avgChunkSize: number;
} {
  const byLanguage: Record<string, number> = {};
  let totalSize = 0;

  for (const chunk of chunks) {
    const lang = chunk.metadata.language;
    byLanguage[lang] = (byLanguage[lang] || 0) + 1;
    totalSize += chunk.content.length;
  }

  return {
    totalChunks: chunks.length,
    byLanguage,
    avgChunkSize: chunks.length > 0 ? Math.round(totalSize / chunks.length) : 0,
  };
}
