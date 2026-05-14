import { execFile } from 'node:child_process';
import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { isBlocked } from '../filter.js';

export async function grep(input: {
  repo: string;
  pattern: string;
  glob?: string;
  context?: number;
  max_results?: number;
}) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  const ctx = input.context ?? 2;
  const args = ['-n', '--no-heading', '--color', 'never', '-C', String(ctx)];
  if (input.glob) args.push('-g', input.glob);
  args.push('--', input.pattern);

  const stdout = await new Promise<string>((resolve, reject) => {
    execFile(
      'rg',
      args,
      { cwd: r.path, maxBuffer: 1024 * 1024 * 256 },
      (err, out) => {
        const code = (err as NodeJS.ErrnoException | null)?.code;
        if (err && code !== undefined && code !== '1' && Number(code) !== 1) {
          if (!out) return reject(err);
        }
        resolve(out);
      },
    );
  });

  const blocks = stdout.split(/\n--\n/);
  const kept: string[] = [];
  let matchCount = 0;
  const max = input.max_results ?? Infinity;

  for (const block of blocks) {
    if (!block.trim()) continue;
    const lines = block.split('\n').filter(Boolean);
    if (lines.length === 0) continue;
    const firstSep = lines[0].search(/[:\-]/);
    if (firstSep < 0) continue;
    const path = lines[0].slice(0, firstSep);
    if (isBlocked(path)) continue;
    const blockMatches = lines.filter(l => {
      const sep = l.indexOf(':', path.length);
      const dash = l.indexOf('-', path.length);
      return sep >= 0 && (dash < 0 || sep < dash);
    }).length;
    if (matchCount + blockMatches > max) break;
    matchCount += blockMatches;
    kept.push(block);
  }

  return {
    stale: fresh.stale,
    total_matches: matchCount,
    output: kept.join('\n--\n'),
  };
}
