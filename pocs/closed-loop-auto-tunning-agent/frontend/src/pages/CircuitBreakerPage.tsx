import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getCbConfig, getCbMetrics, getTuneStatus, setScenario } from "../api";
import { useTrafficRunner } from "../useTrafficRunner";
import { ScenarioControls, type ScenarioForm } from "../components/ScenarioControls";
import { LiveMetrics } from "../components/LiveMetrics";
import { ActiveConfig } from "../components/ActiveConfig";
import { TuningPanel } from "../components/TuningPanel";
import { CompareView } from "../components/CompareView";
import type { RunResult, RunSummary } from "../types";

const DEFAULT_FORM: ScenarioForm = {
  failRate: 0.6,
  latencyMs: 300,
  jitterMs: 120,
  total: 80,
  rps: 25,
};

export function CircuitBreakerPage() {
  const qc = useQueryClient();
  const [form, setForm] = useState<ScenarioForm>(DEFAULT_FORM);
  const [runs, setRuns] = useState<RunResult[]>([]);
  const [justApplied, setJustApplied] = useState(false);
  const runner = useTrafficRunner();

  const config = useQuery({ queryKey: ["cbConfig"], queryFn: getCbConfig, refetchInterval: 5000 });
  const status = useQuery({ queryKey: ["tuneStatus"], queryFn: getTuneStatus });
  const metrics = useQuery({
    queryKey: ["cbMetrics"],
    queryFn: getCbMetrics,
    refetchInterval: runner.running ? 500 : 2500,
  });

  async function onRun() {
    await setScenario({ failRate: form.failRate, latencyMs: form.latencyMs, jitterMs: form.jitterMs });
    const cfg = await getCbConfig();
    const label = `fail ${cfg.failureRateThreshold}% · win ${cfg.slidingWindowSize} · wait ${cfg.waitDurationInOpenStateSeconds}s`;
    const result = await runner.run({
      endpoint: "/cb/call",
      total: form.total,
      rps: form.rps,
      label,
      resetBreaker: true,
      pollMetrics: true,
    });
    setRuns((prev) => [...prev, result]);
    metrics.refetch();
  }

  async function onApplied() {
    setJustApplied(true);
    await qc.invalidateQueries({ queryKey: ["cbConfig"] });
    await config.refetch();
    await onRun();
  }

  const lastRun: RunSummary | null = runs.length ? runs[runs.length - 1] : null;

  return (
    <div className="page">
      <p className="lead">
        Drive real traffic through the Circuit Breaker, then ask the model for a better config. The proposal is
        clamped to safe bounds and applied only when you click Apply. Apply rebuilds the breaker and automatically
        re-runs the same scenario, so the before / after comparison fills in on its own.
      </p>
      <div className="grid-2">
        <ScenarioControls
          form={form}
          onChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
          onRun={() => {
            setJustApplied(false);
            onRun();
          }}
          running={runner.running}
          runLabel="Run traffic"
        />
        <LiveMetrics metrics={metrics.data} tally={runner.tally} points={runner.points} />
      </div>
      <ActiveConfig config={config.data} highlight={justApplied} />
      <div className="grid-2">
        <TuningPanel status={status.data} lastRun={lastRun} onApplied={onApplied} />
        <CompareView runs={runs} />
      </div>
    </div>
  );
}
