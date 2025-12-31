import "dotenv/config"; // Explicit dotenv load (per audit recommendation)

import { Mastra } from "@mastra/core/mastra";
import { PinoLogger } from "@mastra/loggers";
import { LibSQLStore } from "@mastra/libsql";

import {
  codeAnalysisAgent,
  issueResolutionAgent,
  orchestratorAgent,
} from "./agents";

export const mastra = new Mastra({
  agents: {
    codeAnalysisAgent,
    issueResolutionAgent,
    orchestratorAgent,
  },
  storage: new LibSQLStore({
    url: process.env.DATABASE_URL || "file:./mastra.db",
  }),
  logger: new PinoLogger({
    name: "TriageFlow",
    level: "info",
  }),
});
