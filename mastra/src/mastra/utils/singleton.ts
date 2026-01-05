// Generic singleton factory with lazy initialization and error tracking.

/**
 * Creates a singleton factory that lazily initializes on first call.
 * If initialization fails, the error is cached and rethrown on subsequent calls.
 */
export function createSingleton<T>(factory: () => T): () => T {
  let instance: T | undefined;
  let initialized = false;
  let initError: Error | undefined;

  return () => {
    // If already initialized (success or failure), return cached result
    if (initialized) {
      if (initError) {
        throw new Error(`Singleton initialization failed: ${initError.message}`);
      }
      return instance as T;
    }

    // First call - attempt initialization
    try {
      instance = factory();
      initialized = true;
      return instance;
    } catch (error) {
      initError = error instanceof Error ? error : new Error(String(error));
      initialized = true; // Mark as attempted to prevent retry loops
      throw new Error(`Singleton initialization failed: ${initError.message}`);
    }
  };
}
