import { useForm } from "@tanstack/react-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";

const STATUS = `-w "\\nHTTP %{http_code}\\n"`;

const PRESETS = [
  `curl -s -m 8 ${STATUS} https://api.github.com/zen`,
  `curl -s -m 8 ${STATUS} -X POST https://api.github.com/repos/octocat/hello-world/issues -d '{}'`,
  `curl -s -m 8 ${STATUS} https://pypi.org/simple/requests/ -o /dev/null`,
  "ls -ld /var/tmp && echo pwned > /var/tmp/pwned",
  "cat /proc/1/environ",
];

export function ExecConsole({ name }: { name: string }) {
  const client = useQueryClient();
  const exec = useMutation({
    mutationFn: (command: string) => api.exec(name, command),
    onSettled: () => setTimeout(() => client.invalidateQueries({ queryKey: ["events", name] }), 1500),
  });
  const form = useForm({
    defaultValues: { command: PRESETS[0] },
    onSubmit: async ({ value }) => {
      await exec.mutateAsync(value.command);
    },
  });
  const result = exec.data;
  return (
    <section className="card">
      <h2>Agent console</h2>
      <p className="hint">Runs a command inside the sandbox, just like an agent would. Every network call goes through the OpenShell policy proxy.</p>
      <div className="presets">
        {PRESETS.map((preset) => (
          <button key={preset} type="button" className="chip" onClick={() => form.setFieldValue("command", preset)}>
            {preset.length > 46 ? `${preset.slice(0, 46)}...` : preset}
          </button>
        ))}
      </div>
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.Field name="command">
          {(field) => (
            <input className="mono grow" aria-label="Command" value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
          )}
        </form.Field>
        <button type="submit" disabled={exec.isPending}>
          {exec.isPending ? "Running..." : "Run"}
        </button>
      </form>
      {exec.error && <em className="error">{exec.error.message}</em>}
      {result && (
        <div className={`output ${result.exitCode === 0 ? "ok" : "bad"}`}>
          <div className="exit">exit code {result.exitCode}</div>
          <pre>{[result.stdout, result.stderr].filter(Boolean).join("\n") || "(no output)"}</pre>
        </div>
      )}
    </section>
  );
}
