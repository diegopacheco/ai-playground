import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { ApiError, Session } from "@lib/types";
import "./UsersPage.css";

interface UserRow {
  id: number;
  username: string;
  role: string;
  enabled: boolean;
  lastLoginAt: string | null;
}

export default function UsersPage() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [form, setForm] = useState({ username: "", password: "", role: "user" });
  const [ownPassword, setOwnPassword] = useState("");

  const load = useCallback(() => {
    api.session().then(setSession).catch(() => setSession(null));
    api
      .users()
      .then((rows) => setUsers(rows as unknown as UserRow[]))
      .catch((caught: ApiError) => setError(caught.message));
  }, []);

  useEffect(load, [load]);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await api.createUser(form.username, form.password, form.role);
      setForm({ username: "", password: "", role: "user" });
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const remove = async (id: number) => {
    setError(null);
    try {
      await api.deleteUser(id);
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  const changeOwn = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setNotice(null);
    try {
      await api.changeOwnPassword(ownPassword);
      setOwnPassword("");
      setNotice("password changed");
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  return (
    <div className="page">
      {error ? <p className="users-error" role="alert">{error}</p> : null}
      {notice ? <p className="users-notice">{notice}</p> : null}
      {session?.usingBootstrapPassword ? (
        <div className="banner-warning">
          The <strong>admin</strong> account still uses the default password. Change it below.
        </div>
      ) : null}

      <section className="users-section">
        <h2>users</h2>
        <table className="users-table">
          <thead>
            <tr>
              <th>username</th>
              <th>role</th>
              <th>enabled</th>
              <th>last login</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.id}>
                <td>{user.username}</td>
                <td><Badge tone={user.role === "admin" ? "accent" : "neutral"}>{user.role}</Badge></td>
                <td>{user.enabled ? "yes" : "no"}</td>
                <td className="users-when">
                  {user.lastLoginAt ? new Date(user.lastLoginAt).toLocaleString() : "never"}
                </td>
                <td className="users-actions">
                  <Button variant="ghost" onClick={() => remove(user.id)}>delete</Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="users-section">
        <h2>new user</h2>
        <form className="users-form" onSubmit={create}>
          <input
            value={form.username}
            onChange={(event) => setForm({ ...form, username: event.target.value })}
            placeholder="username"
          />
          <input
            type="password"
            value={form.password}
            onChange={(event) => setForm({ ...form, password: event.target.value })}
            placeholder="password"
          />
          <select value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value })}>
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>
          <Button variant="primary" type="submit">create</Button>
        </form>
      </section>

      <section className="users-section">
        <h2>change my password</h2>
        <form className="users-form" onSubmit={changeOwn}>
          <input
            type="password"
            value={ownPassword}
            onChange={(event) => setOwnPassword(event.target.value)}
            placeholder="new password"
          />
          <Button variant="primary" type="submit">change</Button>
        </form>
      </section>
    </div>
  );
}
