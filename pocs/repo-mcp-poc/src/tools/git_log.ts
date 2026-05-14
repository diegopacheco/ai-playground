import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { git } from '../git.js';

export async function gitLog(input: { repo: string; path?: string; limit?: number }) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  const limit = input.limit ?? 20;
  const args = ['log', `-${limit}`, '--pretty=format:%H%x09%an%x09%aI%x09%s'];
  if (input.path) args.push('--', input.path);
  const out = await git(r.path, args);
  const commits = out.split('\n').filter(Boolean).map(line => {
    const [sha, author, date, subject] = line.split('\t');
    return { sha, author, date, subject };
  });
  return { stale: fresh.stale, commits };
}
