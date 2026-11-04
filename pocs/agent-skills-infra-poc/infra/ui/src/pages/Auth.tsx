import { useQuery } from "@tanstack/react-query";
import { fetchClients, fetchMe } from "../api";
import { isAdmin, useAccess } from "../access";
import { decodeClaims } from "../format";

export function Auth() {
  const { session, app } = useAccess();
  const admin = isAdmin(session.scopes);
  const me = useQuery({ queryKey: ["me", session.accessToken], queryFn: fetchMe });
  const clients = useQuery({ queryKey: ["clients"], queryFn: fetchClients, enabled: admin });
  const claims = decodeClaims(session.accessToken);
  const left = Math.max(0, Math.round((session.expiresAt - Date.now()) / 1000));

  return (
    <>
      <section className="panel">
        <h2>Your token</h2>
        <p className="muted">
          Issued by the auth service with the OAuth 2 client credentials grant, signed HS256, verified locally by every API.
          The <code>sub</code> claim is the tenant.
        </p>
        {me.isError && <div className="alert">{me.error.message}</div>}
        <div className="token-grid">
          <div>
            <span className="label">Client</span>
            <b className="mono">{me.data?.client_id ?? session.clientId}</b> <span className={`kind kind-${me.data?.kind ?? "unknown"}`}>{me.data?.kind}</span>
            <p className="muted">{me.data?.description}</p>
            <span className="label">Expires in</span>
            <b>{Math.floor(left / 60)}m {left % 60}s</b>
            <span className="label">Scopes</span>
            <div className="chips">
              {session.scopes.map((s) => (
                <span key={s} className="chip">
                  {s}
                </span>
              ))}
            </div>
          </div>
          <div>
            <span className="label">Decoded claims</span>
            <pre>{JSON.stringify(claims, null, 2)}</pre>
          </div>
        </div>
      </section>
      {admin && (
        <section className="panel">
          <h2>OAuth clients</h2>
          <p className="muted">Every tenant registered in the auth service. Secrets are stored hashed and never returned by the API.</p>
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>Kind</th>
                <th>Description</th>
                <th>Scopes</th>
              </tr>
            </thead>
            <tbody>
              {(clients.data ?? [])
                .filter((c) => !app || c.client_id === app)
                .map((c) => (
                  <tr key={c.client_id}>
                    <td className="mono">{c.client_id}</td>
                    <td>
                      <span className={`kind kind-${c.kind}`}>{c.kind}</span>
                    </td>
                    <td>{c.description}</td>
                    <td>
                      <div className="chips">
                        {c.scopes.map((s) => (
                          <span key={s} className="chip">
                            {s}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}
