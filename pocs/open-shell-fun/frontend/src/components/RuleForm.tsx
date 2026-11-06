import { useForm } from "@tanstack/react-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import type { Access, Enforcement, NetworkRule } from "../types";

const ACCESS: Access[] = ["read-only", "read-write", "full"];
const ENFORCEMENT: Enforcement[] = ["enforce", "audit"];

const initial: NetworkRule = {
  ruleName: "github_api",
  binary: "/usr/bin/curl",
  host: "api.github.com",
  port: 443,
  access: "read-only",
  enforcement: "enforce",
};

export function RuleForm({ name }: { name: string }) {
  const client = useQueryClient();
  const add = useMutation({
    mutationFn: (rule: NetworkRule) => api.addRule(name, rule),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["policy", name] });
      void client.invalidateQueries({ queryKey: ["sandboxes"] });
    },
  });
  const form = useForm({
    defaultValues: initial,
    onSubmit: async ({ value }) => {
      await add.mutateAsync(value);
    },
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    void form.handleSubmit();
  };
  return (
    <form className="rule-form" onSubmit={submit}>
      <form.Field name="ruleName">
        {(f) => (
          <label className="field"><span>Rule name</span><input value={f.state.value} onChange={(e) => f.handleChange(e.target.value)} /></label>
        )}
      </form.Field>
      <form.Field name="binary">
        {(f) => (
          <label className="field"><span>Binary</span><input className="mono" value={f.state.value} onChange={(e) => f.handleChange(e.target.value)} /></label>
        )}
      </form.Field>
      <form.Field name="host">
        {(f) => (
          <label className="field"><span>Host</span><input className="mono" value={f.state.value} onChange={(e) => f.handleChange(e.target.value)} /></label>
        )}
      </form.Field>
      <form.Field name="port">
        {(f) => (
          <label className="field small"><span>Port</span><input type="number" value={f.state.value} onChange={(e) => f.handleChange(Number(e.target.value))} /></label>
        )}
      </form.Field>
      <form.Field name="access">
        {(f) => (
          <label className="field small">
            <span>Access</span>
            <select value={f.state.value} onChange={(e) => f.handleChange(e.target.value as Access)}>
              {ACCESS.map((a) => <option key={a}>{a}</option>)}
            </select>
          </label>
        )}
      </form.Field>
      <form.Field name="enforcement">
        {(f) => (
          <label className="field small">
            <span>Mode</span>
            <select value={f.state.value} onChange={(e) => f.handleChange(e.target.value as Enforcement)}>
              {ENFORCEMENT.map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
        )}
      </form.Field>
      <button type="submit" disabled={add.isPending}>{add.isPending ? "Applying..." : "Apply rule"}</button>
      {add.error && <em className="error">{add.error.message}</em>}
      {add.data?.output.trim() && <pre className="cli-out">{add.data.output.trim()}</pre>}
    </form>
  );
}
