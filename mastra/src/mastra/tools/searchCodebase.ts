import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API, assertResponseOk } from "./httpClient";

export const searchCodebase = createTool({
  id: "search-codebase",
  description:
    "Search the indexed codebase for code patterns, functions, classes, or concepts using semantic search",
  inputSchema: z.object({
    sessionId: z.string().describe("The session ID for the repository"),
    query: z.string().describe("Search query - can be natural language or code pattern"),
    topK: z.number().default(10).describe("Number of results to return"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        file: z.string(),
        content: z.string(),
        score: z.number(),
      })
    ),
  }),
  execute: async ({ context }) => {
    const { sessionId, query, topK } = context;
    const response = await fetchWithTimeout(
      `${PYTHON_API}/assistant/sessions/${sessionId}/agentic-query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
      },
      60000 // 60s timeout for search (can be slow)
    );

    await assertResponseOk(response, "Search");
    return response.json();
  },
});
