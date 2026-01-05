/**
 * Repository Loader
 *
 * Git operations and file loading for codebase indexing.
 *
 * @module rag/repositoryLoader
 */

import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import { simpleGit, SimpleGit } from "simple-git";
import { glob } from "glob";
import { getAllExtensions, getLanguageMetadata } from "./languageConfig";
import type { RepoInfo, LoadedFile, LoadOptions } from "./types";

// Default patterns to exclude
const DEFAULT_EXCLUDE_PATTERNS = [
  "**/node_modules/**",
  "**/.git/**",
  "**/dist/**",
  "**/build/**",
  "**/.next/**",
  "**/__pycache__/**",
  "**/.pytest_cache/**",
  "**/venv/**",
  "**/.venv/**",
  "**/vendor/**",
  "**/.idea/**",
  "**/.vscode/**",
  "**/coverage/**",
  "**/*.min.js",
  "**/*.min.css",
  "**/package-lock.json",
  "**/yarn.lock",
  "**/pnpm-lock.yaml",
];

// Maximum file size to load (1MB default)
const DEFAULT_MAX_FILE_SIZE = 1024 * 1024;

/**
 * Parse a GitHub repository URL to extract owner and repo.
 */
export function parseRepoUrl(repoUrl: string): { owner: string; repo: string } {
  // Handle various GitHub URL formats
  const patterns = [
    /github\.com\/([^\/]+)\/([^\/\.]+)(?:\.git)?/,
    /github\.com:([^\/]+)\/([^\/\.]+)(?:\.git)?/,
  ];

  for (const pattern of patterns) {
    const match = repoUrl.match(pattern);
    if (match) {
      return { owner: match[1], repo: match[2] };
    }
  }

  throw new Error(`Invalid GitHub URL: ${repoUrl}`);
}

/**
 * Get or create a persistent directory for cloned repositories.
 */
export function getRepoStorageDir(): string {
  const storageDir = path.join(os.tmpdir(), "triage-flow-repos");
  return storageDir;
}

/**
 * Clone a repository to a persistent temp directory.
 * Returns the path to the cloned repository.
 */
export async function cloneRepository(
  repoUrl: string,
  options: LoadOptions = {}
): Promise<{ repoPath: string; repoInfo: RepoInfo }> {
  const branch = options.branch || "main";
  const { owner, repo } = parseRepoUrl(repoUrl);

  const storageDir = getRepoStorageDir();
  await fs.mkdir(storageDir, { recursive: true });

  const repoPath = path.join(storageDir, `${owner}_${repo}`);

  const git: SimpleGit = simpleGit();

  // Check if repo already exists
  const exists = await fs
    .access(path.join(repoPath, ".git"))
    .then(() => true)
    .catch(() => false);

  if (exists) {
    // Pull latest changes
    const repoGit = simpleGit(repoPath);
    try {
      await repoGit.fetch();
      await repoGit.checkout(branch);
      await repoGit.pull("origin", branch);
    } catch (error) {
      // If pull fails, try a fresh clone
      await fs.rm(repoPath, { recursive: true, force: true });
      await git.clone(repoUrl, repoPath, ["--branch", branch, "--depth", "1"]);
    }
  } else {
    // Fresh clone
    await git.clone(repoUrl, repoPath, ["--branch", branch, "--depth", "1"]);
  }

  const repoInfo: RepoInfo = {
    owner,
    repo,
    branch,
    url: repoUrl,
    repoPath,
  };

  return { repoPath, repoInfo };
}

/**
 * Load all supported files from a directory.
 */
export async function loadFiles(
  inputDir: string,
  options: LoadOptions = {}
): Promise<LoadedFile[]> {
  const excludePatterns = options.excludePatterns || DEFAULT_EXCLUDE_PATTERNS;
  const maxFileSize = options.maxFileSize || DEFAULT_MAX_FILE_SIZE;

  // Get all supported extensions
  const extensions = getAllExtensions();
  const extensionPatterns = extensions.map((ext) => `**/*${ext}`);

  // Find all matching files
  const files = await glob(extensionPatterns, {
    cwd: inputDir,
    ignore: excludePatterns,
    nodir: true,
    absolute: false,
  });

  const loadedFiles: LoadedFile[] = [];

  for (const relativePath of files) {
    const absolutePath = path.join(inputDir, relativePath);

    try {
      // Check file size
      const stats = await fs.stat(absolutePath);
      if (stats.size > maxFileSize) {
        continue;
      }

      // Read file content
      const content = await fs.readFile(absolutePath, "utf-8");

      // Skip binary files (simple heuristic)
      if (content.includes("\0")) {
        continue;
      }

      // Get language metadata
      const metadata = getLanguageMetadata(relativePath);

      loadedFiles.push({
        path: absolutePath,
        relativePath,
        content,
        language: metadata.language,
        metadata,
      });
    } catch (error) {
      // Skip files that can't be read
      continue;
    }
  }

  return loadedFiles;
}

/**
 * Get file statistics for a directory.
 */
export async function getFileStats(inputDir: string): Promise<{
  totalFiles: number;
  byLanguage: Record<string, number>;
  totalSize: number;
}> {
  const files = await loadFiles(inputDir);

  const byLanguage: Record<string, number> = {};
  let totalSize = 0;

  for (const file of files) {
    byLanguage[file.language] = (byLanguage[file.language] || 0) + 1;
    totalSize += file.content.length;
  }

  return {
    totalFiles: files.length,
    byLanguage,
    totalSize,
  };
}

/**
 * Search files by pattern (for file-oriented queries).
 */
export async function searchFiles(
  inputDir: string,
  pattern: string,
  options: LoadOptions = {}
): Promise<string[]> {
  const excludePatterns = options.excludePatterns || DEFAULT_EXCLUDE_PATTERNS;

  const files = await glob(pattern, {
    cwd: inputDir,
    ignore: excludePatterns,
    nodir: true,
    absolute: false,
  });

  return files;
}

/**
 * Check if a local path is a valid repository.
 */
export async function isValidRepository(repoPath: string): Promise<boolean> {
  try {
    const git = simpleGit(repoPath);
    await git.status();
    return true;
  } catch {
    return false;
  }
}

/**
 * Get the current branch of a repository.
 */
export async function getCurrentBranch(repoPath: string): Promise<string> {
  const git = simpleGit(repoPath);
  const status = await git.status();
  return status.current || "main";
}
