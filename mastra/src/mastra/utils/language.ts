// File extension to language inference - delegates to comprehensive languageConfig.

import { getLanguageMetadata } from "../rag/languageConfig";

/**
 * Infer programming language from file path.
 * Uses the comprehensive language config from RAG module.
 */
export const inferLanguage = (filePath: string): string =>
  getLanguageMetadata(filePath).language;
