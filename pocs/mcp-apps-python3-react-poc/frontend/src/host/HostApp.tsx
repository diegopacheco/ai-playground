import { useCallback, useEffect, useState } from "react";
import { connectServer, loadView, type ServerView } from "./mcpServer";
import { AppFrame, type HostEvent } from "./components/AppFrame";
import { EventLog } from "./components/EventLog";
import { Prompt } from "./components/Prompt";

type Turn = {
  id: number;
  expression: string;
};

export function HostApp() {
  const [view, setView] = useState<ServerView | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [events, setEvents] = useState<HostEvent[]>([]);

  useEffect(() => {
    connectServer(new URL("/mcp", window.location.origin))
      .then(loadView)
      .then(setView)
      .catch((error: unknown) => setFailure(error instanceof Error ? error.message : String(error)));
  }, []);

  const onEvent = useCallback((event: HostEvent) => setEvents((previous) => [...previous, event]), []);

  const ask = (expression: string) => setTurns((previous) => [...previous, { id: previous.length + 1, expression }]);

  return (
    <div className="host">
      <header className="topbar">
        <h1>Calculator MCP App</h1>
        <span className={view ? "pill ok" : "pill"}>{view ? `tool: ${view.toolName}` : failure ? "offline" : "connecting"}</span>
      </header>
      <div className="layout">
        <main className="thread">
          {failure ? <p className="failure">{failure}</p> : null}
          {turns.length === 0 && view ? (
            <p className="muted intro">
              This page plays the chat host. Pick an expression and the host calls the <code>calculate</code> tool,
              then renders the server's <code>ui://</code> view inline, exactly like Claude or ChatGPT would.
            </p>
          ) : null}
          {view
            ? turns.map((turn) => (
                <article key={turn.id} className="turn">
                  <div className="bubble">calculate {turn.expression}</div>
                  <AppFrame view={view} expression={turn.expression} onEvent={onEvent} />
                </article>
              ))
            : null}
          <Prompt disabled={!view} onSubmit={ask} />
        </main>
        <EventLog events={events} />
      </div>
    </div>
  );
}
