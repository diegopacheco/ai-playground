import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { repoMetaPath, repoPath, repoLockPath } from './paths.js';
import { git, gitDefaultBranch } from './git.js';

const DAY_MS = 24 * 60 * 60 * 1000;

type Meta = {
  last_pull_at: string;
  last_pull_status: string;
  default_branch: string;
};

function loadMeta(name: string): Meta | null {
  const p = repoMetaPath(name);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, 'utf8')) as Meta;
}

function saveMeta(name: string, meta: Meta): void {
  writeFileSync(repoMetaPath(name), JSON.stringify(meta, null, 2));
}

export async function initMeta(name: string): Promise<void> {
  const branch = await gitDefaultBranch(repoPath(name));
  saveMeta(name, {
    last_pull_at: new Date().toISOString(),
    last_pull_status: 'ok',
    default_branch: branch,
  });
}

function tryLock(name: string): boolean {
  const lock = repoLockPath(name);
  if (existsSync(lock)) {
    try {
      const t = Number(readFileSync(lock, 'utf8'));
      if (Date.now() - t > 60_000) {
        unlinkSync(lock);
      } else {
        return false;
      }
    } catch {
      return false;
    }
  }
  try {
    writeFileSync(lock, String(Date.now()), { flag: 'wx' });
    return true;
  } catch {
    return false;
  }
}

function releaseLock(name: string): void {
  try { unlinkSync(repoLockPath(name)); } catch {}
}

export async function ensureFresh(name: string): Promise<{ stale: boolean }> {
  const meta = loadMeta(name);
  if (!meta) {
    await initMeta(name);
    return { stale: false };
  }
  const age = Date.now() - new Date(meta.last_pull_at).getTime();
  if (age < DAY_MS) return { stale: false };

  if (!tryLock(name)) return { stale: true };
  try {
    await git(repoPath(name), ['pull', '--ff-only']);
    saveMeta(name, { ...meta, last_pull_at: new Date().toISOString(), last_pull_status: 'ok' });
    return { stale: false };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    saveMeta(name, { ...meta, last_pull_status: msg.slice(0, 200) });
    return { stale: true };
  } finally {
    releaseLock(name);
  }
}
