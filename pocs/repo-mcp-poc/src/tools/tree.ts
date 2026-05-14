import { readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { isBlocked } from '../filter.js';

export async function tree(input: { repo: string; path?: string; depth?: number }) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  const root = input.path ? join(r.path, input.path) : r.path;
  const maxDepth = input.depth ?? 3;
  const lines: string[] = [];
  walk(root, r.path, '', maxDepth, lines);
  return { stale: fresh.stale, tree: lines.join('\n') };
}

function walk(dir: string, repoRoot: string, prefix: string, depth: number, out: string[]): void {
  if (depth < 0) return;
  let entries: string[];
  try {
    entries = readdirSync(dir).sort();
  } catch {
    return;
  }
  const visible = entries.filter(e => {
    if (e === '.git') return false;
    const rel = relative(repoRoot, join(dir, e));
    return !isBlocked(rel);
  });
  visible.forEach((e, i) => {
    const full = join(dir, e);
    const isLast = i === visible.length - 1;
    const connector = isLast ? '└── ' : '├── ';
    let isDir = false;
    try { isDir = statSync(full).isDirectory(); } catch {}
    out.push(prefix + connector + e + (isDir ? '/' : ''));
    if (isDir) {
      walk(full, repoRoot, prefix + (isLast ? '    ' : '│   '), depth - 1, out);
    }
  });
}
