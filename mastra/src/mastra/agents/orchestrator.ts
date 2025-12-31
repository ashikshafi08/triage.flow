import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";
import { createSession, getSessionStatus } from "../tools";

export const orchestratorAgent = new Agent({
  name: "Analysis Orchestrator",
  instructions: `You coordinate analysis across specialized agents.

Available specialists:
- Code Analysis Specialist: For understanding code structure and architecture
- Issue Resolution Specialist: For debugging and fixing issues

Your role:
1. If a repository URL is provided, use createSession to bootstrap a session
2. Use getSessionStatus to verify the repository is indexed before delegating
3. Analyze incoming requests to determine required expertise
4. Delegate to appropriate specialists
5. Synthesize results from multiple agents
6. Provide comprehensive final recommendations

When delegating, be specific about what each specialist should focus on.`,
  model: openai("gpt-4o"),
  tools: {
    createSession,
    getSessionStatus,
  },
});
