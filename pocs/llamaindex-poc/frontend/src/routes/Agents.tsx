import { useForm } from "@tanstack/react-form";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import type { AgentAnswer, AgentConfigResponse, AgentInfo, AgentPreferences } from "../types";

interface FormValues {
  active: string;
  models: Record<string, string>;
  timeout: number;
}

function ConfigForm({
  agents,
  preferences,
}: {
  agents: AgentInfo[];
  preferences: AgentPreferences;
}) {
  const queryClient = useQueryClient();
  const save = useMutation<AgentConfigResponse, Error, FormValues>({
    mutationFn: (values) =>
      api.saveAgents({ active: values.active, models: values.models, timeout: values.timeout }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agents"] }),
  });

  const form = useForm({
    defaultValues: {
      active: preferences.active,
      models: { ...preferences.models },
      timeout: preferences.timeout,
    } as FormValues,
    onSubmit: async ({ value }) => {
      await save.mutateAsync(value);
    },
  });

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void form.handleSubmit();
      }}
    >
      <form.Field name="active">
        {(field) => (
          <div style={{ marginBottom: 14 }}>
            <label className="muted">default agent</label>
            <select
              value={field.state.value}
              onChange={(event) => field.handleChange(event.target.value)}
            >
              {agents.map((agent) => (
                <option key={agent.key} value={agent.key} disabled={!agent.installed}>
                  {agent.label} {agent.installed ? "" : "(not installed)"}
                </option>
              ))}
            </select>
          </div>
        )}
      </form.Field>

      {agents.map((agent) => (
        <form.Field key={agent.key} name={`models.${agent.key}`}>
          {(field) => (
            <div style={{ marginBottom: 12 }}>
              <label className="muted">
                {agent.label} · <code>{agent.binary}</code>{" "}
                <span className={`badge ${agent.installed ? "on" : "off"}`}>
                  {agent.installed ? "installed" : "missing"}
                </span>
              </label>
              <input
                type="text"
                value={String(field.state.value ?? "")}
                placeholder={agent.default_model}
                onChange={(event) => field.handleChange(event.target.value)}
              />
            </div>
          )}
        </form.Field>
      ))}

      <form.Field name="timeout">
        {(field) => (
          <div style={{ marginBottom: 14 }}>
            <label className="muted">timeout (seconds)</label>
            <input
              type="number"
              min={10}
              max={900}
              style={{ width: 120 }}
              value={field.state.value}
              onChange={(event) => field.handleChange(Number(event.target.value))}
            />
          </div>
        )}
      </form.Field>

      <div className="row">
        <button className="primary" type="submit" disabled={save.isPending}>
          {save.isPending ? "saving…" : "Save choices"}
        </button>
        {save.isSuccess && <span className="muted">saved to data/agents.json</span>}
      </div>
      {save.isError && (
        <div className="error" style={{ marginTop: 10 }}>
          {save.error.message}
        </div>
      )}
    </form>
  );
}

export function Agents() {
  const [prompt, setPrompt] = useState("");
  const [agent, setAgent] = useState("");
  const config = useQuery({ queryKey: ["agents"], queryFn: api.agents });

  const ask = useMutation<AgentAnswer, Error, void>({
    mutationFn: () => api.askAgent({ prompt, agent: agent || undefined }),
  });

  return (
    <>
      <div className="card">
        <h2>Agent CLIs</h2>
        <p className="hint">
          The backend runs these as subprocesses: <code>claude -p --model …</code>,{" "}
          <code>codex exec -m …</code> and <code>agy -p … --model …</code>. Your choice is written to
          disk and reloaded on every start.
        </p>
        {config.data ? (
          <ConfigForm agents={config.data.agents} preferences={config.data.preferences} />
        ) : (
          <p className="muted">loading…</p>
        )}
      </div>

      <div className="card">
        <h2>
          <span className="spark-icon">✨</span>Run a prompt
        </h2>
        <p className="hint">
          The same spark button sits on every tab, so you can ask an agent from anywhere in the app.
        </p>
        <textarea
          value={prompt}
          placeholder="Explain how LlamaIndex SentenceSplitter picks chunk boundaries."
          onChange={(event) => setPrompt(event.target.value)}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <select
            value={agent || config.data?.preferences.active || ""}
            style={{ width: 220 }}
            onChange={(event) => setAgent(event.target.value)}
          >
            {(config.data?.agents ?? [])
              .filter((entry) => entry.installed)
              .map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.label}
                </option>
              ))}
          </select>
          <button className="primary" disabled={!prompt.trim() || ask.isPending} onClick={() => ask.mutate()}>
            {ask.isPending ? "running…" : "Run"}
          </button>
        </div>
        {ask.isError && (
          <div className="error" style={{ marginTop: 10 }}>
            {ask.error.message}
          </div>
        )}
        {ask.data && (
          <>
            <p className="hint" style={{ marginTop: 12 }}>
              {ask.data.label} · {ask.data.model} · {ask.data.elapsed_seconds}s · exit{" "}
              {ask.data.exit_code}
            </p>
            <div className="bubble">{ask.data.answer || "(empty response)"}</div>
          </>
        )}
      </div>
    </>
  );
}
