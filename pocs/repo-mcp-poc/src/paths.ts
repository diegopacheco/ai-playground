import { homedir } from 'node:os';
import { join } from 'node:path';

export const HOME = homedir();
export const ROOT = join(HOME, '.mcp', 'repo-tool');
export const BIN_DIR = join(ROOT, 'bin');
export const REPOS_DIR = join(ROOT, 'repos');
export const REGISTRY_PATH = join(ROOT, 'registry.json');

export function repoNameFromUrl(url: string): string {
  const cleaned = url.replace(/\.git$/, '').replace(/\/+$/, '');
  const parts = cleaned.split('/').filter(Boolean);
  const repo = parts.pop();
  const owner = parts.pop();
  if (!repo || !owner) throw new Error(`Cannot derive repo name from URL: ${url}`);
  return `${owner}__${repo}`;
}

export function repoPath(name: string): string {
  return join(REPOS_DIR, name);
}

export function repoMetaPath(name: string): string {
  return join(repoPath(name), '.repo-tool.meta.json');
}

export function repoLockPath(name: string): string {
  return join(repoPath(name), '.repo-tool.lock');
}
