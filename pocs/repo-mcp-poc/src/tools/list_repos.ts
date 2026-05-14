import { readFileSync, existsSync } from 'node:fs';
import { loadRegistry } from '../registry.js';
import { repoMetaPath } from '../paths.js';

export async function listRepos() {
  const reg = loadRegistry();
  return reg.repos.map(r => {
    let last_pull_at: string | null = null;
    const mp = repoMetaPath(r.name);
    if (existsSync(mp)) {
      try {
        last_pull_at = JSON.parse(readFileSync(mp, 'utf8')).last_pull_at;
      } catch {}
    }
    return { name: r.name, url: r.url, branch: r.branch ?? null, last_pull_at };
  });
}
