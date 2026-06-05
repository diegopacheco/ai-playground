import { useState } from "react";
import { Card } from "../components/Card";
import { hasToken, setToken } from "../lib/token";

export function SettingsTab() {
  const [value, setValue] = useState("");
  const [active, setActive] = useState(hasToken());

  const save = () => {
    setToken(value);
    setActive(hasToken());
    setValue("");
  };

  const clear = () => {
    setToken(null);
    setActive(false);
    setValue("");
  };

  return (
    <div className="stack narrow">
      <Card title="GitHub token">
        <p className="hint">
          Optional. Without a token, syncing uses GitHub's unauthenticated limit (60 requests/hour), which is fine for a
          few repos. Paste a Personal Access Token to raise the limit to 5,000/hour.
        </p>
        <div className="token-state">
          Status:{" "}
          <strong className={active ? "ok" : "muted"}>
            {active ? "a token is active for this session" : "no token — syncing unauthenticated"}
          </strong>
        </div>
        <input
          className="token-input"
          type="password"
          placeholder="ghp_…"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          autoComplete="off"
        />
        <div className="row">
          <button className="primary" onClick={save} disabled={!value.trim()}>
            Use token
          </button>
          <button className="ghost" onClick={clear} disabled={!active}>
            Clear
          </button>
        </div>
      </Card>

      <Card title="How the token is handled">
        <ul className="bullet">
          <li>Kept only in memory in this browser tab.</li>
          <li>Never written to disk, never written to a database, never logged.</li>
          <li>Sent to the backend only on a Sync request, then discarded.</li>
          <li>Reloading this page clears it — you re-enter it when you want it.</li>
        </ul>
      </Card>
    </div>
  );
}
