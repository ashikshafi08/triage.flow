/**
 * Test Setup
 *
 * Global test configuration and utilities.
 */

import { beforeAll, afterAll, vi } from "vitest";

// Mock environment variables for tests
beforeAll(() => {
  process.env.DATABASE_URL = ":memory:";
  process.env.OPENAI_API_KEY = "test-key";
  process.env.GITHUB_TOKEN = "test-token";
});

// Cleanup after all tests
afterAll(() => {
  vi.clearAllMocks();
});

// Export test utilities
export const TEST_REPO_URL = "https://github.com/test-owner/test-repo";
export const TEST_SESSION_ID = "test-session-123";
