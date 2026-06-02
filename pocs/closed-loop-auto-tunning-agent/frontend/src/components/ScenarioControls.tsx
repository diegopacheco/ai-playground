export type ScenarioForm = {
  failRate: number;
  latencyMs: number;
  jitterMs: number;
  total: number;
  rps: number;
};

type Props = {
  form: ScenarioForm;
  onChange: (patch: Partial<ScenarioForm>) => void;
  onRun: () => void;
  running: boolean;
  runLabel: string;
};

export function ScenarioControls({ form, onChange, onRun, running, runLabel }: Props) {
  return (
    <div className="card">
      <h3>Scenario & traffic</h3>
      <div className="controls">
        <label>
          <span>Downstream failure rate</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={form.failRate}
            onChange={(e) => onChange({ failRate: Number(e.target.value) })}
          />
          <em>{Math.round(form.failRate * 100)}%</em>
        </label>
        <label>
          <span>Latency (ms)</span>
          <input
            type="number"
            min={0}
            value={form.latencyMs}
            onChange={(e) => onChange({ latencyMs: Number(e.target.value) })}
          />
        </label>
        <label>
          <span>Jitter (ms)</span>
          <input
            type="number"
            min={0}
            value={form.jitterMs}
            onChange={(e) => onChange({ jitterMs: Number(e.target.value) })}
          />
        </label>
        <label>
          <span>Requests</span>
          <input
            type="number"
            min={1}
            value={form.total}
            onChange={(e) => onChange({ total: Number(e.target.value) })}
          />
        </label>
        <label>
          <span>Requests / sec</span>
          <input
            type="number"
            min={1}
            value={form.rps}
            onChange={(e) => onChange({ rps: Number(e.target.value) })}
          />
        </label>
      </div>
      <button className="primary" onClick={onRun} disabled={running}>
        {running ? "Running traffic…" : runLabel}
      </button>
    </div>
  );
}
