import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { git } from '../git.js';
import { isBlocked } from '../filter.js';

export async function listFiles(input: { repo: string; glob?: string }) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  const args = ['ls-files'];
  if (input.glob) args.push('--', input.glob);
  const out = await git(r.path, args);
  const files = out.split('\n').filter(Boolean).filter(p => !isBlocked(p));
  return { stale: fresh.stale, count: files.length, files };
}
