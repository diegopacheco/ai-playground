import { useConnectedClients } from "../mcp/hooks";

const ago = (iso: string) => `${Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000))}s ago`;

export function AgentsPage() {
  const clients = useConnectedClients();
  return (
    <div className="page">
      <p className="lead">
        Claude Code and Codex connect over Streamable HTTP. They do not speak <span className="mono">events/*</span> yet, so they use the
        bridge tools <span className="mono">deploy_service</span>, <span className="mono">list_event_types</span> and{" "}
        <span className="mono">poll_events</span>. Every event they cause shows up live on the Live tab with their client name.
      </p>
      <div className="agents-grid">
        <section className="card">
          <h3>Install</h3>
          <pre className="json">{`./scripts/install.sh            # asks y/N for Claude Code and Codex
./scripts/install.sh claude codex
./scripts/uninstall.sh claude codex`}</pre>
          <h3>What it runs</h3>
          <pre className="json">{`claude mcp add --transport http --scope user mcp-events ${__MCP_URL__}
codex mcp add mcp-events --url ${__MCP_URL__}`}</pre>
          <h3>Try it</h3>
          <pre className="json">{`claude -p "deploy checkout to prod with mcp-events, then poll its events until it finishes"
codex exec "deploy search to staging with simulate_failure using mcp-events and report the alert"`}</pre>
        </section>
        <section className="card">
          <h3>Connected MCP clients</h3>
          {clients.data?.length ? (
            <table>
              <thead>
                <tr>
                  <th>client</th>
                  <th>version</th>
                  <th>protocol</th>
                  <th>session</th>
                  <th>requests</th>
                  <th>last seen</th>
                  <th>state</th>
                </tr>
              </thead>
              <tbody>
                {clients.data.map((c) => (
                  <tr key={c.sessionId}>
                    <td>
                      <span className="client-badge">{c.clientName}</span>
                    </td>
                    <td className="mono">{c.clientVersion}</td>
                    <td className="mono">{c.protocolVersion}</td>
                    <td className="mono">{c.sessionId}</td>
                    <td className="mono">{c.requests}</td>
                    <td className="mono">{ago(c.lastSeen)}</td>
                    <td className="mono">{c.closed ? "closed" : "open"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty">No MCP sessions yet.</div>
          )}
        </section>
      </div>
    </div>
  );
}
