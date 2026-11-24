export type Agent = { readonly run: (prompt: string, signal: AbortSignal) => Promise<string> };

export function claudeCliAgent(model: string): Agent {
  return {
    run: async (prompt, signal) => {
      const proc = Bun.spawn(["claude", "-p", "--model", model, prompt], { stdout: "pipe", stderr: "pipe" });
      const kill = () => proc.kill();
      signal.addEventListener("abort", kill, { once: true });
      try {
        const [out, err, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
        if (signal.aborted) throw new Error("aborted");
        if (code !== 0) throw new Error(`claude exited with ${code}: ${(err || out).trim()}`);
        return out.trim();
      } finally {
        signal.removeEventListener("abort", kill);
      }
    },
  };
}
