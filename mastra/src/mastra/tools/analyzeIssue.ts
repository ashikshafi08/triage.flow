import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { fetchWithTimeout, PYTHON_API, assertResponseOk } from "./httpClient";

export const analyzeIssue = createTool({
  id: "analyze-issue",
  description:
    "Analyze a GitHub issue to understand the problem and find related code. The repo URL is extracted from the issue URL automatically.",
  inputSchema: z.object({
    issueUrl: z
      .string()
      .describe("Full GitHub issue URL (e.g., https://github.com/owner/repo/issues/123)"),
  }),
  outputSchema: z.object({
    sessionId: z.string().optional(),
    status: z.string(),
    steps: z.array(z.any()).optional(),
    finalResult: z
      .object({
        classification: z.any().optional(),
        relatedFiles: z.array(z.string()).optional(),
        remediationPlan: z.string().optional(),
        agenticInsights: z.any().optional(),
      })
      .optional(),
    issueTitle: z.string().optional(),
    issueNumber: z.number().optional(),
    error: z.string().optional(),
  }),
  execute: async ({ context }) => {
    const { issueUrl } = context;
    const response = await fetchWithTimeout(
      `${PYTHON_API}/api/analyze-issue`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ issue_url: issueUrl }),
      },
      60000 // 60s timeout for analysis
    );

    await assertResponseOk(response, "Analysis");
    return response.json();
  },
});
