import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const exec = promisify(execFile);

export async function git(cwd: string, args: string[], timeoutMs = 30000): Promise<string> {
  const { stdout } = await exec('git', args, {
    cwd,
    timeout: timeoutMs,
    maxBuffer: 1024 * 1024 * 64,
  });
  return stdout;
}

export async function gitClone(url: string, dest: string, branch?: string): Promise<void> {
  const args = ['clone'];
  if (branch) args.push('--branch', branch);
  args.push(url, dest);
  await exec('git', args, { timeout: 600000, maxBuffer: 1024 * 1024 * 64 });
}

export async function gitDefaultBranch(cwd: string): Promise<string> {
  try {
    const out = await git(cwd, ['symbolic-ref', '--short', 'HEAD']);
    return out.trim();
  } catch {
    return 'main';
  }
}
