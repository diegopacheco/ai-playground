import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { ApiError, Connection, Project } from "@lib/types";
import "./ProjectsPage.css";

const KINDS = ["postgres", "mysql", "cassandra", "redis", "etcd", "kafka", "elasticsearch"] as const;

const DEFAULT_PORTS: Record<string, number> = {
  postgres: 5432,
  mysql: 3306,
  cassandra: 9042,
  redis: 6379,
  etcd: 2379,
  kafka: 9092,
  elasticsearch: 9200
};

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [health, setHealth] = useState<Record<number, string>>({});
  const [form, setForm] = useState({
    projectId: 0,
    name: "",
    kind: "postgres",
    host: "localhost",
    port: 5432,
    database: "",
    keyspace: "",
    datacenter: "",
    username: "",
    password: ""
  });

  const load = useCallback(() => {
    api
      .projects()
      .then((loaded) => {
        setProjects(loaded);
        setForm((current) => ({ ...current, projectId: current.projectId || loaded[0]?.id || 0 }));
      })
      .catch((caught: ApiError) => setError(caught.message));
  }, []);

  useEffect(load, [load]);

  const createProject = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await api.createProject(projectName);
      setProjectName("");
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const addConnection = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        name: form.name,
        kind: form.kind,
        host: form.host,
        port: Number(form.port)
      };
      for (const field of ["database", "keyspace", "datacenter", "username", "password"] as const) {
        if (form[field]) {
          payload[field] = form[field];
        }
      }
      await api.addConnection(form.projectId, payload);
      setForm({ ...form, name: "", username: "", password: "" });
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const check = async (connection: Connection) => {
    setHealth((current) => ({ ...current, [connection.id]: "checking…" }));
    try {
      const response = await api.ping(connection.id);
      setHealth((current) => ({
        ...current,
        [connection.id]: response.healthy ? "reachable" : response.error ?? "unreachable"
      }));
    } catch (caught) {
      setHealth((current) => ({ ...current, [connection.id]: (caught as ApiError).message }));
    }
  };

  const remove = async (projectId: number, connectionId: number) => {
    await api.deleteConnection(projectId, connectionId);
    load();
  };

  return (
    <div className="page">
      {error ? <p className="projects-error" role="alert">{error}</p> : null}

      <section className="projects-section">
        <h2>projects</h2>
        {projects.map((project) => (
          <div className="project-card" key={project.id}>
            <header>
              <strong>{project.name}</strong>
              <span className="project-meta">created by {project.createdBy}</span>
            </header>
            <table className="project-connections">
              <thead>
                <tr>
                  <th>name</th>
                  <th>kind</th>
                  <th>target</th>
                  <th>user</th>
                  <th>secret</th>
                  <th>health</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {project.connections.map((connection) => (
                  <tr key={connection.id}>
                    <td>{connection.name}</td>
                    <td><Badge tone="accent">{connection.kind}</Badge></td>
                    <td className="project-target">
                      {connection.host}:{connection.port}
                      {connection.database ? ` / ${connection.database}` : ""}
                      {connection.keyspace ? ` / ${connection.keyspace}` : ""}
                    </td>
                    <td>{connection.username ?? "—"}</td>
                    <td>{connection.hasPassword ? <Badge>encrypted</Badge> : "—"}</td>
                    <td className="project-health">{health[connection.id] ?? ""}</td>
                    <td className="project-actions">
                      <Button variant="ghost" onClick={() => check(connection)}>test</Button>
                      <Button variant="ghost" onClick={() => remove(project.id, connection.id)}>remove</Button>
                    </td>
                  </tr>
                ))}
                {project.connections.length === 0 ? (
                  <tr><td colSpan={7} className="project-empty">no connections yet</td></tr>
                ) : null}
              </tbody>
            </table>
          </div>
        ))}
      </section>

      <section className="projects-section">
        <h2>new project</h2>
        <form className="projects-form" onSubmit={createProject}>
          <input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="project name" />
          <Button variant="primary" type="submit">create</Button>
        </form>
      </section>

      <section className="projects-section">
        <h2>new connection</h2>
        <form className="projects-form projects-form-grid" onSubmit={addConnection}>
          <label>
            project
            <select
              value={form.projectId}
              onChange={(event) => setForm({ ...form, projectId: Number(event.target.value) })}
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>{project.name}</option>
              ))}
            </select>
          </label>
          <label>
            kind
            <select
              value={form.kind}
              onChange={(event) =>
                setForm({ ...form, kind: event.target.value, port: DEFAULT_PORTS[event.target.value] })
              }
            >
              {KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
            </select>
          </label>
          <label>name<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
          <label>host<input value={form.host} onChange={(event) => setForm({ ...form, host: event.target.value })} /></label>
          <label>port<input type="number" value={form.port} onChange={(event) => setForm({ ...form, port: Number(event.target.value) })} /></label>
          <label>database<input value={form.database} onChange={(event) => setForm({ ...form, database: event.target.value })} /></label>
          <label>keyspace / schema<input value={form.keyspace} onChange={(event) => setForm({ ...form, keyspace: event.target.value })} /></label>
          <label>datacenter<input value={form.datacenter} onChange={(event) => setForm({ ...form, datacenter: event.target.value })} /></label>
          <label>username<input value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} /></label>
          <label>password<input type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} /></label>
          <Button variant="primary" type="submit">add connection</Button>
        </form>
        <p className="projects-hint">
          Use a database account granted only <code>SELECT</code>. The console refuses writes, but a read-only
          account is the layer that still holds if that check is ever bypassed.
        </p>
      </section>
    </div>
  );
}
