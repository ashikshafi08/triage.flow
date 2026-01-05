/**
 * Language Configuration
 *
 * Language-specific configurations for the RAG system.
 * Port of Python language_config.py.
 *
 * @module rag/languageConfig
 */

import * as path from "path";
import type { LanguageConfigEntry, LanguageMetadata } from "./types";

/**
 * Language-specific file extensions and metadata.
 * Maps language identifiers to their configuration.
 */
export const LANGUAGE_CONFIG: Record<string, LanguageConfigEntry> = {
  python: {
    extensions: [".py", ".pyw", ".pyi"],
    docPattern: '""".*?"""|\'\'\'.*?\'\'\'|#.*?$',
    importPattern: "^(?:from|import)\\s+([\\w\\.]+)",
    displayName: "Python",
    description:
      "A high-level, interpreted programming language known for its readability and versatility.",
  },
  javascript: {
    extensions: [".js", ".jsx", ".mjs", ".cjs"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^(?:import|require)\\s*\\(?['\"]([^'\"]+)['\"]\\)?",
    displayName: "JavaScript",
    description: "A scripting language primarily used for web development.",
  },
  typescript: {
    extensions: [".ts", ".tsx", ".mts", ".cts"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^(?:import|require)\\s*\\(?['\"]([^'\"]+)['\"]\\)?",
    displayName: "TypeScript",
    description:
      "A typed superset of JavaScript that compiles to plain JavaScript.",
  },
  java: {
    extensions: [".java"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Java",
    description: "A class-based, object-oriented programming language.",
  },
  c: {
    extensions: [".c", ".h"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: '^#include\\s*[<"]([^>"]+)[>"]',
    displayName: "C",
    description: "A general-purpose, procedural programming language.",
  },
  cpp: {
    extensions: [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: '^#include\\s*[<"]([^>"]+)[>"]',
    displayName: "C++",
    description: "An extension of C with object-oriented features.",
  },
  go: {
    extensions: [".go"],
    docPattern: "//.*?$",
    importPattern: "^import\\s*\\(?['\"]([^'\"]+)['\"]\\)?",
    displayName: "Go",
    description:
      "A statically typed, compiled programming language designed for simplicity and efficiency.",
  },
  rust: {
    extensions: [".rs"],
    docPattern: "//!.*?$|//.*?$",
    importPattern: "^use\\s+([\\w:]+)",
    displayName: "Rust",
    description:
      "A systems programming language focused on safety and performance.",
  },
  ruby: {
    extensions: [".rb", ".rake"],
    docPattern: "=begin.*?=end|#.*?$",
    importPattern: "^(?:require|require_relative)\\s*['\"]([^'\"]+)['\"]",
    displayName: "Ruby",
    description:
      "A dynamic, object-oriented programming language with a focus on simplicity and productivity.",
  },
  php: {
    extensions: [".php"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^(?:require|include)\\s*['\"]([^'\"]+)['\"]",
    displayName: "PHP",
    description:
      "A server-side scripting language designed for web development.",
  },
  swift: {
    extensions: [".swift"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Swift",
    description:
      "A powerful and intuitive programming language for iOS, macOS, and other Apple platforms.",
  },
  kotlin: {
    extensions: [".kt", ".kts"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Kotlin",
    description:
      "A modern programming language that makes developers happier.",
  },
  scala: {
    extensions: [".scala", ".sc"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Scala",
    description:
      "A general-purpose programming language providing support for functional programming.",
  },
  dart: {
    extensions: [".dart"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+['\"]([^'\"]+)['\"]",
    displayName: "Dart",
    description:
      "A client-optimized language for fast apps on any platform.",
  },
  haskell: {
    extensions: [".hs", ".lhs"],
    docPattern: "--.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Haskell",
    description:
      "A purely functional programming language with strong static typing.",
  },
  elixir: {
    extensions: [".ex", ".exs"],
    docPattern: "#.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Elixir",
    description:
      "A dynamic, functional language designed for building scalable applications.",
  },
  clojure: {
    extensions: [".clj", ".cljs", ".cljc", ".edn"],
    docPattern: ";.*?$",
    importPattern: "^\\(require\\s+'\\[([\\w\\.]+)\\]\\)",
    displayName: "Clojure",
    description:
      "A dynamic, general-purpose programming language combining the approachability of a scripting language with robust infrastructure.",
  },
  erlang: {
    extensions: [".erl", ".hrl"],
    docPattern: "%.*?$",
    importPattern: "^-import\\(([\\w\\.]+)\\)",
    displayName: "Erlang",
    description:
      "A general-purpose, concurrent, functional programming language.",
  },
  lua: {
    extensions: [".lua"],
    docPattern: "--.*?$",
    importPattern: "^require\\s*['\"]([^'\"]+)['\"]",
    displayName: "Lua",
    description:
      "A lightweight, high-level, multi-paradigm programming language designed primarily for embedded use in applications.",
  },
  perl: {
    extensions: [".pl", ".pm"],
    docPattern: "#.*?$",
    importPattern: "^use\\s+([\\w:]+)",
    displayName: "Perl",
    description:
      "A general-purpose programming language originally developed for text manipulation.",
  },
  markdown: {
    extensions: [".md", ".markdown", ".mdx"],
    docPattern: null,
    importPattern: null,
    displayName: "Markdown",
    description:
      "A lightweight markup language for creating formatted text using a plain-text editor.",
  },
  html: {
    extensions: [".html", ".htm"],
    docPattern: "<!--.*?-->",
    importPattern:
      '<script\\s+src=["\']([^"\']+)["\']|<link\\s+href=["\']([^"\']+)["\']',
    displayName: "HTML",
    description:
      "The standard markup language for documents designed to be displayed in a web browser.",
  },
  css: {
    extensions: [".css", ".scss", ".sass", ".less"],
    docPattern: "/\\*.*?\\*/",
    importPattern: "@import\\s+url\\(['\"]?([^'\"\\)]+)['\"]?\\)",
    displayName: "CSS",
    description:
      "A style sheet language used for describing the presentation of a document written in HTML.",
  },
  shell: {
    extensions: [".sh", ".bash", ".zsh", ".fish"],
    docPattern: "#.*?$",
    importPattern: "^(?:source|\\.)\\s+([^\\s]+)",
    displayName: "Shell",
    description: "Shell scripting languages for Unix-like operating systems.",
  },
  dockerfile: {
    extensions: [".dockerfile"],
    docPattern: "#.*?$",
    importPattern: "^FROM\\s+([^\\s]+)",
    displayName: "Dockerfile",
    description: "Instructions for building Docker container images.",
  },
  yaml: {
    extensions: [".yaml", ".yml"],
    docPattern: "#.*?$",
    importPattern: null,
    displayName: "YAML",
    description: "A human-readable data serialization standard.",
  },
  json: {
    extensions: [".json", ".jsonc"],
    docPattern: null,
    importPattern: null,
    displayName: "JSON",
    description:
      "JavaScript Object Notation, a lightweight data interchange format.",
  },
  xml: {
    extensions: [".xml", ".xsl", ".xslt"],
    docPattern: "<!--.*?-->",
    importPattern: null,
    displayName: "XML",
    description: "Extensible Markup Language for structured data.",
  },
  toml: {
    extensions: [".toml"],
    docPattern: "#.*?$",
    importPattern: null,
    displayName: "TOML",
    description:
      "Tom's Obvious, Minimal Language for configuration files.",
  },
  sql: {
    extensions: [".sql"],
    docPattern: "--.*?$|/\\*.*?\\*/",
    importPattern: null,
    displayName: "SQL",
    description:
      "Structured Query Language for managing and querying relational databases.",
  },
  graphql: {
    extensions: [".graphql", ".gql"],
    docPattern: "#.*?$",
    importPattern: null,
    displayName: "GraphQL",
    description: "A query language for APIs and a runtime for executing those queries.",
  },
  vue: {
    extensions: [".vue"],
    docPattern: "<!--.*?-->|/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+.*?from\\s+['\"]([^'\"]+)['\"]",
    displayName: "Vue",
    description: "A progressive JavaScript framework for building user interfaces.",
  },
  svelte: {
    extensions: [".svelte"],
    docPattern: "<!--.*?-->|/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+.*?from\\s+['\"]([^'\"]+)['\"]",
    displayName: "Svelte",
    description: "A radical new approach to building user interfaces.",
  },
  groovy: {
    extensions: [".groovy", ".gradle"],
    docPattern: "/\\*\\*.*?\\*/|//.*?$",
    importPattern: "^import\\s+([\\w\\.]+)",
    displayName: "Groovy",
    description: "A powerful, optionally typed dynamic language for the JVM.",
  },
  csharp: {
    extensions: [".cs"],
    docPattern: "///.*?$|/\\*\\*.*?\\*/",
    importPattern: "^using\\s+([\\w\\.]+)",
    displayName: "C#",
    description: "A modern, object-oriented programming language developed by Microsoft.",
  },
};

// Extension to language lookup map (built once)
const extensionToLanguage = new Map<string, string>();
for (const [lang, config] of Object.entries(LANGUAGE_CONFIG)) {
  for (const ext of config.extensions) {
    extensionToLanguage.set(ext.toLowerCase(), lang);
  }
}

// Special filename handlers
const specialFilenames = new Map<string, string>([
  ["dockerfile", "dockerfile"],
  ["dockerfile.dev", "dockerfile"],
  ["dockerfile.prod", "dockerfile"],
  ["dockerfile.local", "dockerfile"],
  ["makefile", "shell"],
  ["gemfile", "ruby"],
  ["rakefile", "ruby"],
  ["vagrantfile", "ruby"],
  ["jenkinsfile", "groovy"],
]);

/**
 * Get all supported file extensions.
 */
export function getAllExtensions(): string[] {
  const extensions: string[] = [];
  for (const config of Object.values(LANGUAGE_CONFIG)) {
    extensions.push(...config.extensions);
  }
  return extensions;
}

/**
 * Get language metadata for a given file path.
 */
export function getLanguageMetadata(filePath: string): LanguageMetadata {
  const filename = path.basename(filePath).toLowerCase();
  const ext = path.extname(filePath).toLowerCase();

  // Check special filenames first
  const specialLang = specialFilenames.get(filename);
  if (specialLang) {
    const config = LANGUAGE_CONFIG[specialLang];
    if (config) {
      return {
        language: specialLang,
        displayName: config.displayName,
        description: config.description,
        docPattern: config.docPattern,
        importPattern: config.importPattern,
      };
    }
  }

  // Check by extension
  const lang = extensionToLanguage.get(ext);
  if (lang) {
    const config = LANGUAGE_CONFIG[lang];
    return {
      language: lang,
      displayName: config.displayName,
      description: config.description,
      docPattern: config.docPattern,
      importPattern: config.importPattern,
    };
  }

  // Unknown language
  return {
    language: "unknown",
    displayName: "Unknown",
    description: "Unknown file type",
    docPattern: null,
    importPattern: null,
  };
}

/**
 * Check if a file extension is supported.
 */
export function isSupportedExtension(ext: string): boolean {
  return extensionToLanguage.has(ext.toLowerCase());
}

/**
 * Get language by extension.
 */
export function getLanguageByExtension(ext: string): string | null {
  return extensionToLanguage.get(ext.toLowerCase()) ?? null;
}

/**
 * Get extensions for a specific language.
 */
export function getExtensionsForLanguage(language: string): string[] {
  return LANGUAGE_CONFIG[language]?.extensions ?? [];
}
