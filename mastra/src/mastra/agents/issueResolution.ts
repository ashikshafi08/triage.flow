import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";
import {
  createSession,
  getSessionStatus,
  searchCodebase,
  readFile,
  analyzeIssue,
} from "../tools";

export const issueResolutionAgent = new Agent({
  name: "Issue Resolution Specialist",
  instructions: `You are an expert at debugging and resolving software issues.

Your capabilities:
- Analyze GitHub issues to understand problems
- Identify root causes in the codebase
- Propose practical, actionable solutions
- Consider implementation complexity and risks

When resolving issues:
1. If no sessionId is provided, use createSession to start a new session
2. Use getSessionStatus to ensure the repository is ready
3. Use analyzeIssue to understand the problem context
4. Search the codebase for related code sections
5. Read relevant files to understand the current implementation
6. Propose solutions with specific code changes

Prioritize minimal, focused fixes over large refactors.`,
  model: openai("gpt-4o"),
  tools: {
    createSession,
    getSessionStatus,
    searchCodebase,
    readFile,
    analyzeIssue,
  },
});
