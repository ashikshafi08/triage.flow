/**
 * Language Configuration Unit Tests
 */

import { describe, it, expect } from "vitest";
import {
  LANGUAGE_CONFIG,
  getLanguageMetadata,
  getAllExtensions,
  isSupportedExtension,
  getLanguageByExtension,
  getExtensionsForLanguage,
} from "../../../src/mastra/rag/languageConfig";

describe("LANGUAGE_CONFIG", () => {
  it("should contain common programming languages", () => {
    const expectedLanguages = [
      "typescript",
      "javascript",
      "python",
      "go",
      "rust",
      "java",
      "cpp",
      "ruby",
    ];

    for (const lang of expectedLanguages) {
      expect(LANGUAGE_CONFIG[lang]).toBeDefined();
      expect(LANGUAGE_CONFIG[lang].extensions).toBeInstanceOf(Array);
      expect(LANGUAGE_CONFIG[lang].extensions.length).toBeGreaterThan(0);
    }
  });

  it("should have extensions starting with dot", () => {
    for (const [lang, config] of Object.entries(LANGUAGE_CONFIG)) {
      for (const ext of config.extensions) {
        expect(ext.startsWith(".")).toBe(true);
      }
    }
  });

  it("should have displayName for each language", () => {
    for (const [lang, config] of Object.entries(LANGUAGE_CONFIG)) {
      expect(config.displayName).toBeDefined();
      expect(typeof config.displayName).toBe("string");
    }
  });
});

describe("getLanguageMetadata", () => {
  describe("TypeScript files", () => {
    it("should detect .ts files as typescript", () => {
      const metadata = getLanguageMetadata("src/index.ts");

      expect(metadata.language).toBe("typescript");
      expect(metadata.displayName).toBe("TypeScript");
    });

    it("should detect .tsx files as typescript", () => {
      const metadata = getLanguageMetadata("components/Button.tsx");

      expect(metadata.language).toBe("typescript");
    });

    it("should detect .mts files as typescript", () => {
      const metadata = getLanguageMetadata("module.mts");

      expect(metadata.language).toBe("typescript");
    });
  });

  describe("JavaScript files", () => {
    it("should detect .js files as javascript", () => {
      const metadata = getLanguageMetadata("app.js");

      expect(metadata.language).toBe("javascript");
      expect(metadata.displayName).toBe("JavaScript");
    });

    it("should detect .jsx files as javascript", () => {
      const metadata = getLanguageMetadata("Component.jsx");

      expect(metadata.language).toBe("javascript");
    });

    it("should detect .mjs files as javascript", () => {
      const metadata = getLanguageMetadata("esm-module.mjs");

      expect(metadata.language).toBe("javascript");
    });
  });

  describe("Python files", () => {
    it("should detect .py files as python", () => {
      const metadata = getLanguageMetadata("main.py");

      expect(metadata.language).toBe("python");
      expect(metadata.displayName).toBe("Python");
    });

    it("should detect .pyi stub files as python", () => {
      const metadata = getLanguageMetadata("types.pyi");

      expect(metadata.language).toBe("python");
    });
  });

  describe("Go files", () => {
    it("should detect .go files as go", () => {
      const metadata = getLanguageMetadata("main.go");

      expect(metadata.language).toBe("go");
      expect(metadata.displayName).toBe("Go");
    });
  });

  describe("Rust files", () => {
    it("should detect .rs files as rust", () => {
      const metadata = getLanguageMetadata("lib.rs");

      expect(metadata.language).toBe("rust");
      expect(metadata.displayName).toBe("Rust");
    });
  });

  describe("Markdown files", () => {
    it("should detect .md files as markdown", () => {
      const metadata = getLanguageMetadata("README.md");

      expect(metadata.language).toBe("markdown");
      expect(metadata.displayName).toBe("Markdown");
    });

    it("should detect .mdx files as markdown", () => {
      const metadata = getLanguageMetadata("docs/guide.mdx");

      expect(metadata.language).toBe("markdown");
    });
  });

  describe("Configuration files", () => {
    it("should detect .json files as json", () => {
      const metadata = getLanguageMetadata("package.json");

      expect(metadata.language).toBe("json");
      expect(metadata.displayName).toBe("JSON");
    });

    it("should detect .yaml files as yaml", () => {
      const metadata = getLanguageMetadata("config.yaml");

      expect(metadata.language).toBe("yaml");
    });

    it("should detect .yml files as yaml", () => {
      const metadata = getLanguageMetadata("ci.yml");

      expect(metadata.language).toBe("yaml");
    });

    it("should detect .toml files as toml", () => {
      const metadata = getLanguageMetadata("Cargo.toml");

      expect(metadata.language).toBe("toml");
      expect(metadata.displayName).toBe("TOML");
    });
  });

  describe("Unknown files", () => {
    it("should return unknown for unrecognized extensions", () => {
      const metadata = getLanguageMetadata("data.xyz");

      expect(metadata.language).toBe("unknown");
      expect(metadata.displayName).toBe("Unknown");
    });

    it("should return unknown for files without extension", () => {
      const metadata = getLanguageMetadata("LICENSE");

      expect(metadata.language).toBe("unknown");
    });
  });

  describe("Path handling", () => {
    it("should handle deeply nested paths", () => {
      const metadata = getLanguageMetadata("src/lib/utils/helpers/format.ts");

      expect(metadata.language).toBe("typescript");
    });

    it("should handle paths with dots in directory names", () => {
      const metadata = getLanguageMetadata("node_modules/@types/node/index.d.ts");

      expect(metadata.language).toBe("typescript");
    });
  });
});

describe("getAllExtensions", () => {
  it("should return an array of extensions", () => {
    const extensions = getAllExtensions();

    expect(Array.isArray(extensions)).toBe(true);
    expect(extensions.length).toBeGreaterThan(0);
  });

  it("should include common extensions", () => {
    const extensions = getAllExtensions();

    expect(extensions).toContain(".ts");
    expect(extensions).toContain(".js");
    expect(extensions).toContain(".py");
    expect(extensions).toContain(".go");
    expect(extensions).toContain(".rs");
    expect(extensions).toContain(".md");
  });

  it("should all start with a dot", () => {
    const extensions = getAllExtensions();

    for (const ext of extensions) {
      expect(ext.startsWith(".")).toBe(true);
    }
  });
});

describe("isSupportedExtension", () => {
  it("should return true for supported extensions", () => {
    expect(isSupportedExtension(".ts")).toBe(true);
    expect(isSupportedExtension(".js")).toBe(true);
    expect(isSupportedExtension(".py")).toBe(true);
    expect(isSupportedExtension(".go")).toBe(true);
  });

  it("should return false for unsupported extensions", () => {
    expect(isSupportedExtension(".xyz")).toBe(false);
    expect(isSupportedExtension(".abc")).toBe(false);
  });

  it("should be case insensitive", () => {
    expect(isSupportedExtension(".TS")).toBe(true);
    expect(isSupportedExtension(".Py")).toBe(true);
  });
});

describe("getLanguageByExtension", () => {
  it("should return language for known extensions", () => {
    expect(getLanguageByExtension(".ts")).toBe("typescript");
    expect(getLanguageByExtension(".py")).toBe("python");
    expect(getLanguageByExtension(".go")).toBe("go");
  });

  it("should return null for unknown extensions", () => {
    expect(getLanguageByExtension(".xyz")).toBeNull();
  });

  it("should be case insensitive", () => {
    expect(getLanguageByExtension(".TS")).toBe("typescript");
  });
});

describe("getExtensionsForLanguage", () => {
  it("should return extensions for known languages", () => {
    const tsExtensions = getExtensionsForLanguage("typescript");
    expect(tsExtensions).toContain(".ts");
    expect(tsExtensions).toContain(".tsx");

    const pyExtensions = getExtensionsForLanguage("python");
    expect(pyExtensions).toContain(".py");
  });

  it("should return empty array for unknown languages", () => {
    const extensions = getExtensionsForLanguage("unknown-lang");
    expect(extensions).toEqual([]);
  });
});
