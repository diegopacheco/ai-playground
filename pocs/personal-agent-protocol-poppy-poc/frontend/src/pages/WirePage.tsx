import { useSyncExternalStore } from "react";
import { Card, JsonBlock } from "../components";
import { decodeJwt } from "../poppy/jwt";
import { wire } from "../poppy/wire";

export function WirePage() {
  const entries = useSyncExternalStore(wire.subscribe, wire.list);
  return (
    <Card title={`Wire log (${entries.length} requests, newest first)`}>
      {entries.length === 0 ? <p className="dim">Use the other tabs; every request appears here with its decoded Session JWT.</p> : null}
      {entries.map((entry) => (
        <details key={entry.id} className="wire">
          <summary>
            <span className={`pill ${entry.status < 400 ? "pass" : "fail"}`}>{entry.status}</span> <strong>{entry.method}</strong> {entry.url}
            {entry.session ? <span className="dim"> signed</span> : null}
          </summary>
          {entry.session ? (
            <>
              <h3>Poppy-Session JWT</h3>
              <JsonBlock value={decodeJwt(entry.session)} />
            </>
          ) : null}
          {entry.request ? (
            <>
              <h3>Request body</h3>
              <pre className="json">{entry.request}</pre>
            </>
          ) : null}
          <h3>Response</h3>
          <pre className="json">{entry.response}</pre>
        </details>
      ))}
    </Card>
  );
}
