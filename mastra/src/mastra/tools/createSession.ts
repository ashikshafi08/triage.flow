import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API, assertResponseOk } from "./httpClient";

export const createSession = createTool({
  id: "create-session",
  description:
    "Create a new session for a GitHub repository. This must be called first before using other tools.",
  inputSchema: z.object({
    repoUrl: z
      .string()
      .describe("Full GitHub repository URL (e.g., https://github.com/owner/repo)"),
  }),
  outputSchema: z.object({
    sessionId: z.string(),
    status: z.string(),
    repoMetadata: z.record(z.any()).optional(),
  }),
  execute: async ({ context }) => {
    const { repoUrl } = context;
    const response = await fetchWithTimeout(`${PYTHON_API}/assistant/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: repoUrl }),
    });

    await assertResponseOk(response, "Session creation");
    const data = await response.json();
    return {
      sessionId: data.session_id,
      status: data.status,
      repoMetadata: data.repo_metadata,
    };
  },
});
