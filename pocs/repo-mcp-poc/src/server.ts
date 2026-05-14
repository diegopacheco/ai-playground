import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { listRepos } from './tools/list_repos.js';
import { addRepo } from './tools/add_repo.js';
import { removeRepo } from './tools/remove_repo.js';
import { tree } from './tools/tree.js';
import { listFiles } from './tools/list_files.js';
import { readFile } from './tools/read_file.js';
import { grep } from './tools/grep.js';
import { gitLog } from './tools/git_log.js';
import { repoInfo } from './tools/repo_info.js';

function ok(v: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(v, null, 2) }] };
}

export function buildServer(): McpServer {
  const server = new McpServer({ name: 'repo-mcp', version: '1.0.0' });

  server.tool('list_repos', {}, async () => ok(await listRepos()));

  server.tool(
    'add_repo',
    {
      url: z.string().describe('GitHub URL of the repo to register and clone'),
      branch: z.string().optional().describe('Optional branch; defaults to repo default'),
    },
    async (args) => ok(await addRepo(args)),
  );

  server.tool(
    'remove_repo',
    { name: z.string() },
    async (args) => ok(await removeRepo(args)),
  );

  server.tool(
    'tree',
    {
      repo: z.string(),
      path: z.string().optional(),
      depth: z.number().int().positive().optional(),
    },
    async (args) => ok(await tree(args)),
  );

  server.tool(
    'list_files',
    {
      repo: z.string(),
      glob: z.string().optional(),
    },
    async (args) => ok(await listFiles(args)),
  );

  server.tool(
    'read_file',
    {
      repo: z.string(),
      path: z.string(),
      start_line: z.number().int().positive().optional(),
      end_line: z.number().int().positive().optional(),
    },
    async (args) => ok(await readFile(args)),
  );

  server.tool(
    'grep',
    {
      repo: z.string(),
      pattern: z.string(),
      glob: z.string().optional(),
      context: z.number().int().nonnegative().optional(),
      max_results: z.number().int().positive().optional(),
    },
    async (args) => ok(await grep(args)),
  );

  server.tool(
    'git_log',
    {
      repo: z.string(),
      path: z.string().optional(),
      limit: z.number().int().positive().optional(),
    },
    async (args) => ok(await gitLog(args)),
  );

  server.tool(
    'repo_info',
    { repo: z.string() },
    async (args) => ok(await repoInfo(args)),
  );

  return server;
}
