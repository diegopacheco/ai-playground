import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPatternMetrics, setScenario } from "../api";
import { useTrafficRunner } from "../useTrafficRunner";
import { ScenarioControls, type ScenarioForm } from "../components/ScenarioControls";
import { OutcomeTally } from "../components/OutcomeTally";

type Props = {
  title: string;
  endpoint: string;
  metricsKey: string;
  description: string;
  defaults: ScenarioForm;
};

export function SimplePatternPage({ title, endpoint, metricsKey, description, defaults }: Props) {
  const [form, setForm] = useState<ScenarioForm>(defaults);
  const runner = useTrafficRunner();

  const metrics = useQuery({
    queryKey: ["patternMetrics", metricsKey],
    queryFn: () => getPatternMetrics(metricsKey),
    refetchInterval: runner.running ? 500 : 3000,
  });

  async function onRun() {
    await setScenario({ failRate: form.failRate, latencyMs: form.latencyMs, jitterMs: form.jitterMs });
    await runner.run({
      endpoint,
      total: form.total,
      rps: form.rps,
      label: title,
      resetBreaker: false,
      pollMetrics: false,
    });
    metrics.refetch();
  }

  return (
    <div className="page">
      <p className="lead">{description}</p>
      <div className="grid-2">
        <ScenarioControls
          form={form}
          onChange={(patch) => setForm((f) => ({ ...f, ...patch }))}
          onRun={onRun}
          running={runner.running}
          runLabel="Run traffic"
        />
        <div className="card">
          <h3>{title} outcomes</h3>
          <OutcomeTally tally={runner.tally} />
          <h4 className="sub">Resilience4j metrics</h4>
          <div className="metric-grid">
            {Object.entries(metrics.data ?? {})
              .filter(([k]) => k !== "ts")
              .map(([k, v]) => (
                <div className="metric" key={k}>
                  <span className="metric-value">{v}</span>
                  <span className="metric-label">{k}</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}
