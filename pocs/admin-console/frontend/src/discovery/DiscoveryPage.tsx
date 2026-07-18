import { useCallback, useEffect, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { EngineLogo } from "@design/EngineLogo/EngineLogo";
import { api } from "@lib/api";
import type { ApiError, ConnectionKind, DiscoveredContainer } from "@lib/types";
import "./DiscoveryPage.css";

export default function DiscoveryPage() {
  const [containers, setContainers] = useState<DiscoveredContainer[]>([]);
  const [runtime, setRuntime] = useState<string | null>(null);
  const [available, setAvailable] = useState(true);
  const [reason, setReason] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [projectName, setProjectName] = useState("discovered");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ projectName: string; imported: string[] } | null>(null);

  const scan = useCallback(() => {
    setBusy(true);
    setError(null);
    setResult(null);
    api
      .discover()
      .then((response) => {
        setRuntime(response.runtime);
        setAvailable(response.available);
        setReason(response.reason ?? null);
        setContainers(response.containers);
        setSelected(new Set(response.containers.filter((c) => c.importable).map((c) => c.id)));
      })
      .catch((caught: ApiError) => setError(caught.message))
      .finally(() => setBusy(false));
  }, []);

  useEffect(scan, [scan]);

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const importSelected = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await api.importDiscovered(projectName, [...selected]);
      setResult({ projectName: response.projectName, imported: response.imported });
    } catch (caught) {
      setError((caught as ApiError).message);
    } finally {
      setBusy(false);
    }
  };

  const importable = containers.filter((container) => container.importable);

  return (
    <div className="page">
      <h1 className="discovery-title">Discovery</h1>
      <p className="discovery-lead">
        Running containers that look like an engine this console supports. Check the ones you want and import
        them as a new project — the console then works exactly the same for them as for anything you configured
        by hand.
      </p>

      <div className="discovery-bar">
        <Button onClick={scan} disabled={busy}>
          {busy ? "scanning…" : "rescan"}
        </Button>
        {runtime ? <Badge tone="accent">{runtime}</Badge> : null}
        <span className="discovery-count">
          {importable.length} importable · {containers.length} found
        </span>
      </div>

      {!available ? (
        <p className="discovery-error" role="alert">
          {reason ?? "no container runtime is available to the console"}
        </p>
      ) : null}
      {error ? <p className="discovery-error" role="alert">{error}</p> : null}

      {result ? (
        <p className="discovery-success">
          imported {result.imported.length} connection{result.imported.length === 1 ? "" : "s"} into{" "}
          <strong>{result.projectName}</strong> — <a href="/">open the console</a> or{" "}
          <a href="/projects">review them in projects</a>.
        </p>
      ) : null}

      {containers.length > 0 ? (
        <>
          <table className="discovery-table">
            <thead>
              <tr>
                <th />
                <th>container</th>
                <th>engine</th>
                <th>image</th>
                <th>port</th>
                <th>credentials</th>
              </tr>
            </thead>
            <tbody>
              {containers.map((container) => (
                <tr key={container.id} className={container.importable ? "" : "discovery-row-blocked"}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.has(container.id)}
                      disabled={!container.importable}
                      onChange={() => toggle(container.id)}
                      aria-label={`import ${container.name}`}
                    />
                  </td>
                  <td className="discovery-name">
                    <EngineLogo kind={container.kind as ConnectionKind} size={20} />
                    {container.name}
                  </td>
                  <td><Badge tone="accent">{container.kind}</Badge></td>
                  <td className="discovery-image">{container.image}</td>
                  <td className="discovery-port">
                    {container.importable ? `localhost:${container.hostPort}` : "—"}
                  </td>
                  <td>
                    {container.username ? (
                      <span className="discovery-credentials">
                        {container.username}
                        {container.hasPassword ? <Badge>password detected</Badge> : null}
                      </span>
                    ) : (
                      <span className="discovery-muted">no auth</span>
                    )}
                    {container.reason ? <div className="discovery-reason">{container.reason}</div> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="discovery-import">
            <label>
              new project name
              <input
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                aria-label="new project name"
              />
            </label>
            <Button
              variant="primary"
              onClick={importSelected}
              disabled={busy || selected.size === 0 || !projectName.trim()}
            >
              import {selected.size} connection{selected.size === 1 ? "" : "s"}
            </Button>
          </div>

          <p className="discovery-warning">
            Detected credentials often belong to the container's <strong>superuser</strong>. The console still
            refuses every write, but for anything beyond local work you should replace them with a
            <code> SELECT</code>-only account in projects.
          </p>
        </>
      ) : available && !busy ? (
        <p className="discovery-empty">
          no running container matches a supported engine. Start one — or run <code>./demo/demo-start.sh</code>.
        </p>
      ) : null}
    </div>
  );
}
