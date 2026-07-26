import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import type { AgentAnswer } from "../types";

interface Props {
  context: string;
}

export function SparkPanel({ context }: Props) {
  const [open, setOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [agent, setAgent] = useState("");

  const config = useQuery({ queryKey: ["agents"], queryFn: api.agents });
  const ask = useMutation<AgentAnswer, Error, void>({
    mutationFn: () => api.askAgent({ prompt, agent: agent || undefined, context }),
  });

  const active = agent || config.data?.preferences.active || "";
  const available = config.data?.agents.filter((entry) => entry.installed) ?? [];

  if (!open) {
    return (
      <button className="spark-fab" onClick={() => setOpen(true)}>
        <span className="spark-icon">✨</span>Ask an agent
      </button>
    );
  }

  return (
    <div className="spark-panel">
      <div className="spark-head">
        <span>
          <span className="spark-icon">✨</span>Agent assistant
        </span>
        <button onClick={() => setOpen(false)}>close</button>
      </div>

      <div className="spark-body">
        <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
          You are talking directly to a CLI coding agent, not to the RAG index. Context sent:{" "}
          <code>{context}</code>
        </p>
        {ask.isError && <div className="error">{ask.error.message}</div>}
        {ask.data && (
          <>
            <p className="muted" style={{ fontSize: 11.5, margin: "0 0 6px" }}>
              {ask.data.label} · {ask.data.model} · {ask.data.elapsed_seconds}s
            </p>
            <div className="spark-answer">{ask.data.answer || "(empty response)"}</div>
          </>
        )}
      </div>

      <div className="spark-foot">
        <textarea
          value={prompt}
          placeholder="Ask the agent anything about this screen…"
          onChange={(event) => setPrompt(event.target.value)}
        />
        <div className="row">
          <select
            className="grow"
            value={active}
            onChange={(event) => setAgent(event.target.value)}
          >
            {available.length === 0 && <option value="">no agent CLI found</option>}
            {available.map((entry) => (
              <option key={entry.key} value={entry.key}>
                {entry.label} ({entry.model})
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={!prompt.trim() || ask.isPending || available.length === 0}
            onClick={() => ask.mutate()}
          >
            {ask.isPending ? "running…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
