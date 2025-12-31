import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API } from "./httpClient";

export const getFileTree = createTool({
  id: "get-file-tree",
  description: "Get the directory structure of the repository",
  inputSchema: z.object({
    sessionId: z.string().describe("The session ID for the repository"),
  }),
  outputSchema: z.object({
    tree: z.array(z.any()),
  }),
  execute: async ({ context }) => {
    const { sessionId } = context;
    const params = new URLSearchParams({ session_id: sessionId });
    const response = await fetchWithTimeout(
      `${PYTHON_API}/api/tree?${params}`
    );

    if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      throw new Error(`Tree fetch failed: ${errorText}`);
    }

    const tree = await response.json();
    return { tree };
  },
});
