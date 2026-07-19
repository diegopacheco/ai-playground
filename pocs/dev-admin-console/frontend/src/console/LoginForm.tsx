import { useState, type FormEvent } from "react";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { ApiError, Session } from "@lib/types";
import "./LoginForm.css";

export interface LoginFormProps {
  onAuthenticated: (session: Session) => void;
}

export function LoginForm({ onAuthenticated }: LoginFormProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onAuthenticated(await api.login(username, password));
    } catch (caught) {
      setError((caught as ApiError).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="login" onSubmit={submit}>
      <h1>Dev Admin Console</h1>
      <label>
        username
        <input value={username} onChange={(event) => setUsername(event.target.value)} autoFocus />
      </label>
      <label>
        password
        <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
      </label>
      {error ? <p className="login-error" role="alert">{error}</p> : null}
      <Button variant="primary" type="submit" disabled={busy}>
        {busy ? "signing in…" : "Sign in"}
      </Button>
    </form>
  );
}
