import { existsSync, mkdirSync } from 'node:fs';
import { loadRegistry, saveRegistry } from '../registry.js';
import { repoNameFromUrl, repoPath, REPOS_DIR } from '../paths.js';
import { gitClone } from '../git.js';
import { initMeta } from '../freshness.js';

export async function addRepo(input: { url: string; branch?: string }) {
  const name = repoNameFromUrl(input.url);
  const reg = loadRegistry();
  if (reg.repos.some(r => r.name === name)) {
    throw new Error(`Repo already registered: ${name}. Remove it first.`);
  }
  const dest = repoPath(name);
  if (existsSync(dest)) {
    throw new Error(`Path exists but not in registry: ${dest}`);
  }
  mkdirSync(REPOS_DIR, { recursive: true });
  await gitClone(input.url, dest, input.branch);
  reg.repos.push({
    name,
    url: input.url,
    path: dest,
    branch: input.branch,
    added_at: new Date().toISOString(),
  });
  saveRegistry(reg);
  await initMeta(name);
  return { name, path: dest };
}
