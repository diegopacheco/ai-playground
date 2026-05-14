import { addRepo } from './tools/add_repo.js';

async function main(): Promise<void> {
  const url = process.argv[2];
  const branch = process.argv[3];
  if (!url) {
    console.error('Usage: cli-add <url> [branch]');
    process.exit(1);
  }
  const result = await addRepo({ url, branch: branch || undefined });
  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
