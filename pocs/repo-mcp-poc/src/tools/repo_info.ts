import { readFileSync, existsSync } from 'node:fs';
import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { git } from '../git.js';
import { repoMetaPath } from '../paths.js';

export async function repoInfo(input: { repo: string }) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  const head = (await git(r.path, ['rev-parse', 'HEAD'])).trim();
  let branch = '';
  try {
    branch = (await git(r.path, ['symbolic-ref', '--short', 'HEAD'])).trim();
  } catch {
    branch = '';
  }
  let last_pull_at: string | null = null;
  let last_pull_status: string | null = null;
  const mp = repoMetaPath(input.repo);
  if (existsSync(mp)) {
    try {
      const meta = JSON.parse(readFileSync(mp, 'utf8'));
      last_pull_at = meta.last_pull_at ?? null;
      last_pull_status = meta.last_pull_status ?? null;
    } catch {}
  }
  return {
    stale: fresh.stale,
    name: r.name,
    url: r.url,
    default_branch: branch,
    head,
    last_pull_at,
    last_pull_status,
  };
}
