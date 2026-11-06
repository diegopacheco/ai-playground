import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { ruleNames } from "../policyRules";
import { RuleForm } from "./RuleForm";

export function PolicyPanel({ name }: { name: string }) {
  const client = useQueryClient();
  const policy = useQuery({ queryKey: ["policy", name], queryFn: () => api.policy(name) });
  const remove = useMutation({
    mutationFn: (rule: string) => api.removeRule(name, rule),
    onSuccess: () => client.invalidateQueries({ queryKey: ["policy", name] }),
  });
  const rules = ruleNames(policy.data?.yaml ?? "");
  return (
    <section className="card">
      <h2>Network policy</h2>
      <p className="hint">Default deny. Add an L7 rule and OpenShell hot reloads it, no restart. The rule lets one binary reach one host with an HTTP method preset.</p>
      <RuleForm name={name} />
      <div className="rules">
        <span className="label">Active rules:</span>
        {rules.length === 0 && <span className="muted">none, all egress denied</span>}
        {rules.map((rule) => (
          <span key={rule} className="tag">
            {rule}
            <button type="button" aria-label={`remove ${rule}`} onClick={() => remove.mutate(rule)} disabled={remove.isPending}>x</button>
          </span>
        ))}
      </div>
      {remove.error && <em className="error">{remove.error.message}</em>}
      <details open>
        <summary>Effective policy YAML</summary>
        <pre className="yaml">{policy.isLoading ? "loading..." : policy.error ? policy.error.message : policy.data?.yaml}</pre>
      </details>
    </section>
  );
}
