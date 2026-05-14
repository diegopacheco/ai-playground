import { rmSync } from 'node:fs';
import { loadRegistry, saveRegistry } from '../registry.js';
import { repoPath } from '../paths.js';

export async function removeRepo(input: { name: string }) {
  const reg = loadRegistry();
  const idx = reg.repos.findIndex(r => r.name === input.name);
  if (idx < 0) throw new Error(`Unknown repo: ${input.name}`);
  reg.repos.splice(idx, 1);
  saveRegistry(reg);
  rmSync(repoPath(input.name), { recursive: true, force: true });
  return { removed: true, name: input.name };
}
