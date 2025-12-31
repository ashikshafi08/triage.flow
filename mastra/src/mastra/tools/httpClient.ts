// Shared HTTP client with timeout handling and HTTP 202 support
const PYTHON_API = process.env.PYTHON_API_URL || "http://localhost:8000";
const DEFAULT_TIMEOUT = 30000; // 30 seconds

export async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout = DEFAULT_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    // Handle HTTP 202 (repository still initializing)
    if (response.status === 202) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(
        `Repository still initializing: ${detail.detail || detail.message || "Please wait and retry"}`
      );
    }

    return response;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeout}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export { PYTHON_API };
