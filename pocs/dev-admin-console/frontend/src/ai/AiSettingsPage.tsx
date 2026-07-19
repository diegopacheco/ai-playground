import { useCallback, useEffect, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { AiCli, ApiError } from "@lib/types";
import "./AiSettingsPage.css";

export default function AiSettingsPage() {
  const [clis, setClis] = useState<AiCli[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.aiClis().then(setClis).catch((caught: ApiError) => setError(caught.message));
    api
      .aiSettings()
      .then((settings) => {
        setChosen(settings.cli);
        setModel(settings.model ?? "");
      })
      .catch(() => setChosen(null));
  }, []);

  useEffect(load, [load]);

  const save = async (cli: string, nextModel: string) => {
    setError(null);
    setNotice(null);
    try {
      await api.saveAiSettings(cli, nextModel);
      setChosen(cli);
      setModel(nextModel);
      setNotice(`queries will be written by ${cli}${nextModel ? ` using ${nextModel}` : ""}`);
      load();
    } catch (caught) {
      setError((caught as ApiError).message);
    }
  };

  return (
    <div className="page">
      <h1 className="ai-settings-title">AI query authoring</h1>
      <p className="ai-settings-lead">
        Pick which agent CLI writes queries for you. The choice is remembered for your account and follows you
        across machines. Suggestions are always loaded into the editor for review — nothing runs on its own, and
        every suggestion passes the same read-only guard as a statement you type.
      </p>

      {error ? <p className="ai-settings-error" role="alert">{error}</p> : null}
      {notice ? <p className="ai-settings-notice">{notice}</p> : null}

      <div className="ai-cli-grid">
        {clis.map((cli) => {
          const active = cli.cli === chosen;
          return (
            <div
              key={cli.cli}
              className={active ? "ai-cli-card ai-cli-card-active" : "ai-cli-card"}
              aria-current={active ? "true" : undefined}
            >
              <header>
                <code>{cli.label}</code>
                {active ? <Badge tone="accent">in use</Badge> : null}
                {cli.installed ? (
                  <Badge tone="ok">installed</Badge>
                ) : (
                  <Badge tone="error">not found</Badge>
                )}
              </header>

              {cli.reason ? <p className="ai-cli-reason">{cli.reason}</p> : null}

              <label>
                model
                <input
                  value={active ? model : cli.model ?? ""}
                  onChange={(event) => (active ? setModel(event.target.value) : undefined)}
                  onFocus={() => {
                    if (!active) {
                      setChosen(cli.cli);
                      setModel(cli.model ?? "");
                    }
                  }}
                  placeholder="cli default"
                  disabled={!cli.installed}
                />
              </label>

              <Button
                variant={active ? "primary" : "secondary"}
                disabled={!cli.installed || !cli.enabled}
                onClick={() => save(cli.cli, active ? model : cli.model ?? "")}
              >
                {active ? "save" : "use this cli"}
              </Button>
            </div>
          );
        })}
      </div>

      <p className="ai-settings-warning">
        Asking AI sends your <strong>schema names</strong> — tables, columns, keys, topics and fields — to the model
        provider behind the CLI you choose. It never sends credentials, hostnames, or any row of data. If your table
        names are themselves sensitive, leave this feature unused.
      </p>
    </div>
  );
}
