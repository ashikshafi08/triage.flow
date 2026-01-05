import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API, assertResponseOk } from "./httpClient";

export const getSessionStatus = createTool({
  id: "get-session-status",
  description: "Check if a session is ready (repository cloned and indexed)",
  inputSchema: z.object({
    sessionId: z.string().describe("The session ID to check"),
  }),
  outputSchema: z.object({
    sessionId: z.string(),
    status: z.string(),
    indexingProgress: z.number().optional(),
    error: z.string().optional(),
  }),
  execute: async ({ context }) => {
    const { sessionId } = context;
    const response = await fetchWithTimeout(`${PYTHON_API}/assistant/sessions/${sessionId}/status`);

    await assertResponseOk(response, "Status check");
    const data = await response.json();
    return {
      sessionId: data.session_id,
      status: data.status,
      indexingProgress: data.metadata?.progress,
      error: data.error,
    };
  },
});
