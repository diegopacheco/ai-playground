import { useEffect, useMemo, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { api } from "@lib/api";
import type { Connection, Project, Session } from "@lib/types";
import { ConnectionPicker } from "./ConnectionPicker";
import { ConsolePane } from "./ConsolePane";
import { LoginForm } from "./LoginForm";
import "./ConsoleWorkspace.css";

export default function ConsoleWorkspace() {
  const [session, setSession] = useState<Session | null>(null);
  const [checking, setChecking] = useState(true);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [connectionId, setConnectionId] = useState<number | null>(null);
  const [pickOnLoad, setPickOnLoad] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    if (params.get("pick") === "1") {
      setPickOnLoad(true);
      params.delete("pick");
      const query = params.toString();
      window.history.replaceState({}, "", window.location.pathname + (query ? `?${query}` : ""));
    }
  }, []);

  useEffect(() => {
    api
      .session()
      .then(setSession)
      .catch(() => setSession(null))
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    if (!session) {
      return;
    }
    api.projects().then((loaded) => {
      setProjects(loaded);
      const first = loaded[0];
      setProjectId((current) => current ?? first?.id ?? null);
      setConnectionId((current) => current ?? first?.connections[0]?.id ?? null);
    });
  }, [session]);

  const project = useMemo(
    () => projects.find((candidate) => candidate.id === projectId) ?? null,
    [projects, projectId]
  );

  const connection: Connection | null = useMemo(
    () => project?.connections.find((candidate) => candidate.id === connectionId) ?? null,
    [project, connectionId]
  );

  if (checking) {
    return <p className="workspace-status">loading…</p>;
  }

  if (!session) {
    return <LoginForm onAuthenticated={setSession} />;
  }

  return (
    <div className="workspace">
      {session.usingBootstrapPassword ? (
        <div className="banner-warning">
          This install still uses the default <strong>admin/admin</strong> password. Change it in users.
        </div>
      ) : null}
      <div className="workspace-bar">
        <label>
          project
          <select
            value={projectId ?? ""}
            onChange={(event) => {
              const id = Number(event.target.value);
              setProjectId(id);
              const target = projects.find((candidate) => candidate.id === id);
              setConnectionId(target?.connections[0]?.id ?? null);
            }}
          >
            {projects.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.name}
              </option>
            ))}
          </select>
        </label>

        <ConnectionPicker
          connections={project?.connections ?? []}
          selected={connection}
          onSelect={(chosen) => setConnectionId(chosen.id)}
          autoOpen={pickOnLoad}
        />

        <span className="workspace-user">
          {session.username}
          <Badge tone={session.role === "admin" ? "accent" : "neutral"}>{session.role}</Badge>
        </span>
      </div>

      {connection ? (
        <ConsolePane key={connection.id} connection={connection} />
      ) : (
        <p className="workspace-status">
          no connections yet — add one in <a href="/projects">projects</a>
        </p>
      )}
    </div>
  );
}
