export type CliResult = {
  code: number;
  stdout: string;
  stderr: string;
};

export type CliRunner = (args: string[]) => Promise<CliResult>;

export class CliError extends Error {
  constructor(readonly args: string[], readonly result: CliResult) {
    super(result.stderr.trim() || result.stdout.trim() || `openshell ${args[0]} failed with code ${result.code}`);
  }
}

export const runOpenshell: CliRunner = async (args) => {
  const proc = Bun.spawn(["openshell", ...args], {
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, NO_COLOR: "1" },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { code, stdout, stderr };
};
