import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { requireRepo } from '../registry.js';
import { ensureFresh } from '../freshness.js';
import { isBlocked, looksBinaryByExt, isBinaryBuffer } from '../filter.js';

export async function readFile(input: {
  repo: string;
  path: string;
  start_line?: number;
  end_line?: number;
}) {
  const r = requireRepo(input.repo);
  const fresh = await ensureFresh(input.repo);
  if (isBlocked(input.path)) {
    throw new Error(`Path blocked by filter: ${input.path}`);
  }
  const full = join(r.path, input.path);
  let st;
  try { st = statSync(full); } catch { throw new Error(`Not found: ${input.path}`); }
  if (!st.isFile()) throw new Error(`Not a file: ${input.path}`);
  if (looksBinaryByExt(input.path)) {
    return { stale: fresh.stale, error: 'binary file', path: input.path };
  }
  const buf = readFileSync(full);
  if (isBinaryBuffer(buf)) {
    return { stale: fresh.stale, error: 'binary file', path: input.path };
  }
  const text = buf.toString('utf8');
  const lines = text.split('\n');
  const start = Math.max(1, input.start_line ?? 1);
  const end = Math.min(lines.length, input.end_line ?? lines.length);
  const slice = lines.slice(start - 1, end);
  const width = String(end).length;
  const numbered = slice
    .map((l, i) => `${String(start + i).padStart(width, ' ')}\t${l}`)
    .join('\n');
  return {
    stale: fresh.stale,
    path: input.path,
    start_line: start,
    end_line: end,
    total_lines: lines.length,
    content: numbered,
  };
}
