import { useForm } from "@tanstack/react-form";
import { useMutation } from "@tanstack/react-query";
import { mcp } from "../mcp/client";
import type { DeployResult, ToolResult } from "../mcp/types";

type DeployInput = { service: string; env: string; simulate_failure: boolean };

const deploy = async (input: DeployInput) =>
  (await mcp.request<ToolResult<DeployResult>>("tools/call", { name: "deploy_service", arguments: input })).structuredContent;

export function DeployForm() {
  const mutation = useMutation({ mutationFn: deploy });
  const form = useForm({
    defaultValues: { service: "checkout", env: "staging", simulate_failure: false } as DeployInput,
    onSubmit: ({ value }) => mutation.mutate(value),
  });
  return (
    <section className="card deploy">
      <header className="card-head">
        <div>
          <h3>Trigger a deploy</h3>
          <p className="dim">
            The UI calls <span className="mono">tools/call deploy_service</span>, the same tool Claude Code and Codex call.
          </p>
        </div>
      </header>
      <form
        className="deploy-form"
        onSubmit={(e) => {
          e.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.Field
          name="service"
          validators={{ onChange: ({ value }) => (value.trim() ? undefined : "service is required") }}
          children={(field) => (
            <label className="field">
              <span>service</span>
              <input value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
              {field.state.meta.errors.length > 0 && <em className="error-text">{field.state.meta.errors.join(", ")}</em>}
            </label>
          )}
        />
        <form.Field
          name="env"
          children={(field) => (
            <label className="field">
              <span>env</span>
              <select value={field.state.value} onChange={(e) => field.handleChange(e.target.value)}>
                <option value="dev">dev</option>
                <option value="staging">staging</option>
                <option value="prod">prod</option>
              </select>
            </label>
          )}
        />
        <form.Field
          name="simulate_failure"
          children={(field) => (
            <label className="check">
              <input type="checkbox" checked={field.state.value} onChange={(e) => field.handleChange(e.target.checked)} />
              simulate failure
            </label>
          )}
        />
        <button type="submit" disabled={mutation.isPending}>
          Deploy
        </button>
      </form>
      {mutation.data && (
        <p className="mono result">
          {mutation.data.deployId} started by {mutation.data.triggeredBy}, cursor before first event {mutation.data.cursor}
        </p>
      )}
      {mutation.error && <p className="error-text">{mutation.error.message}</p>}
    </section>
  );
}
