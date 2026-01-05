// Defines relationships between tools for context sharing.

/** Maps tool names to related tools for context sharing.
 * When a tool executes, it can access context from related tools.
 */
export const TOOL_RELATIONSHIPS: Record<string, string[]> = {
  // File operations
  "read-file": ["analyze-file-structure", "find-related-files", "git-blame", "explore-directory"],
  "explore-directory": [
    "read-file",
    "find-related-files",
    "analyze-file-structure",
    "get-file-tree",
  ],

  // Search operations
  "search-codebase": ["find-related-files", "semantic-content-search", "read-file"],
  "semantic-content-search": ["search-codebase", "find-related-files", "read-file"],
  "find-related-files": ["search-codebase", "read-file", "explore-directory"],

  // Analysis operations
  "analyze-file-structure": ["read-file", "find-related-files", "explore-directory"],
  "get-file-tree": ["explore-directory", "find-related-files"],

  // Git operations
  "git-blame": ["read-file", "git-log"],
  "git-log": ["git-blame", "read-file"],

  // Session operations
  "create-session": ["get-session-status"],
  "get-session-status": ["create-session", "search-codebase"],
};

/**
 * Parameter aliases for normalization.
 * Maps non-standard parameter names to their canonical form.
 */
export const PARAMETER_ALIASES: Record<string, Record<string, string>> = {
  "explore-directory": {
    dir_path: "directoryPath",
    path: "directoryPath",
    directory: "directoryPath",
    folder: "directoryPath",
  },
  "read-file": {
    file_path: "filePath",
    path: "filePath",
    file: "filePath",
    filename: "filePath",
  },
  "search-codebase": {
    search_query: "query",
    q: "query",
    search: "query",
    term: "query",
  },
  "find-related-files": {
    file_path: "filePath",
    path: "filePath",
    source: "filePath",
  },
};

/** Get related tools for a given tool. */
export const getRelatedTools = (toolName: string) => TOOL_RELATIONSHIPS[toolName] ?? [];

/**
 * Normalize parameters using aliases.
 */
export function normalizeParameters(
  toolName: string,
  parameters: Record<string, unknown>
): Record<string, unknown> {
  const aliases = PARAMETER_ALIASES[toolName];
  if (!aliases) {
    return parameters;
  }

  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(parameters)) {
    const canonicalKey = aliases[key] || key;
    normalized[canonicalKey] = value;
  }

  return normalized;
}

/** Check if two tools are related. */
export const areToolsRelated = (tool1: string, tool2: string) => {
  const related1 = TOOL_RELATIONSHIPS[tool1] ?? [];
  const related2 = TOOL_RELATIONSHIPS[tool2] ?? [];
  return related1.includes(tool2) || related2.includes(tool1);
};
