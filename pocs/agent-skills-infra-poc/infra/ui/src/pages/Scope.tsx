import { useQuery } from "@tanstack/react-query";
import { fetchClients } from "../api";
import { isAdmin, useAccess, viewOf, type Capability } from "../access";

export function AppFilter() {
  const { session, app, setApp } = useAccess();
  const admin = isAdmin(session.scopes);
  const clients = useQuery({ queryKey: ["clients"], queryFn: fetchClients, enabled: admin });
  if (!admin) {
    return (
      <span className="own-only" title="This client can only read its own tenant">
        own data only
      </span>
    );
  }
  return (
    <label className="app-filter">
      App
      <select value={app} onChange={(e) => setApp(e.target.value)} aria-label="Filter by app">
        <option value="">All apps</option>
        {(clients.data ?? [])
          .filter((c) => c.kind !== "admin")
          .map((c) => (
            <option key={c.client_id} value={c.client_id}>
              {c.client_id} ({c.kind})
            </option>
          ))}
      </select>
    </label>
  );
}

export function ScopeBanner({ capability }: { capability: Capability }) {
  const { session, app } = useAccess();
  const view = viewOf(session.scopes, capability);
  if (view === "none") return null;
  return (
    <div className={`scope-banner ${view}`}>
      {view === "all" ? (
        <>
          <b>{capability}:read:all</b> · showing {app ? <span className="tenant">{app}</span> : "every app"}
        </>
      ) : (
        <>
          <b>{capability}:read</b> · showing only <span className="tenant">{session.clientId}</span>, enforced by the server
        </>
      )}
    </div>
  );
}

export function NoScope({ capability }: { capability: Capability }) {
  const { session } = useAccess();
  return (
    <div className="panel">
      <h2>No access</h2>
      <p className="muted">
        <span className="tenant">{session.clientId}</span> has no <code>{capability}:read</code> or{" "}
        <code>{capability}:read:all</code> scope, so the {capability} API answers 403. Sign in with an app client or with{" "}
        <b>infra-admin</b>.
      </p>
    </div>
  );
}
