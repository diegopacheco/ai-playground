import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { REGISTRY_PATH } from './paths.js';

export type RepoEntry = {
  name: string;
  url: string;
  path: string;
  branch?: string;
  added_at: string;
};

export type Registry = {
  version: number;
  repos: RepoEntry[];
};

export function loadRegistry(): Registry {
  if (!existsSync(REGISTRY_PATH)) {
    return { version: 1, repos: [] };
  }
  return JSON.parse(readFileSync(REGISTRY_PATH, 'utf8')) as Registry;
}

export function saveRegistry(reg: Registry): void {
  mkdirSync(dirname(REGISTRY_PATH), { recursive: true });
  writeFileSync(REGISTRY_PATH, JSON.stringify(reg, null, 2));
}

export function findRepo(name: string): RepoEntry | undefined {
  return loadRegistry().repos.find(r => r.name === name);
}

export function requireRepo(name: string): RepoEntry {
  const r = findRepo(name);
  if (!r) {
    const known = loadRegistry().repos.map(x => x.name).join(', ') || '(none)';
    throw new Error(`Unknown repo: ${name}. Registered: ${known}`);
  }
  return r;
}
