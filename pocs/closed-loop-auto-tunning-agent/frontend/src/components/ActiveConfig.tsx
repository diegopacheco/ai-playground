import type { CircuitBreakerSettings } from "../types";
import { SETTING_KEYS } from "../types";

type Props = { config?: CircuitBreakerSettings; highlight?: boolean };

export function ActiveConfig({ config, highlight }: Props) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>Active circuit breaker config</h3>
        {highlight && <span className="badge">just applied</span>}
      </div>
      {config ? (
        <div className="config-grid">
          {SETTING_KEYS.map((k) => (
            <div className="config-item" key={k}>
              <span className="config-value">{String(config[k])}</span>
              <span className="config-key">{k}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">loading…</p>
      )}
    </div>
  );
}
