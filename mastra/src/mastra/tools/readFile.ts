import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API, assertResponseOk } from "./httpClient";

export const readFile = createTool({
  id: "read-file",
  description: "Read the contents of a specific file from the repository",
  inputSchema: z.object({
    sessionId: z.string().describe("The session ID for the repository"),
    filePath: z.string().describe("Path to the file relative to repository root"),
  }),
  outputSchema: z.object({
    content: z.string(),
    size: z.number(),
    type: z.string().optional(),
    encoding: z.string().optional(),
  }),
  execute: async ({ context }) => {
    const { sessionId, filePath } = context;
    const params = new URLSearchParams({
      session_id: sessionId,
      file_path: filePath,
    });
    const response = await fetchWithTimeout(`${PYTHON_API}/api/file-content?${params}`);

    await assertResponseOk(response, "Read");
    return response.json();
  },
});
