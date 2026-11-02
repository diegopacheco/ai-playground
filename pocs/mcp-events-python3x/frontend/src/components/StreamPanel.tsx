import { useState } from "react";
import { useEventStream } from "../mcp/hooks";
import type { EventDescriptor } from "../mcp/types";
import { FeedList } from "./FeedList";
import { FilterFields } from "./FilterFields";

export function StreamPanel({ descriptor }: { descriptor: EventDescriptor }) {
  const { feed, status, connect, disconnect, clear } = useEventStream(descriptor.name);
  const [args, setArgs] = useState<Record<string, string>>({});
  const live = status === "streaming" || status === "connecting";
  return (
    <section className="card stream">
      <header className="card-head">
        <div>
          <h3 className="mono">{descriptor.name}</h3>
          <p className="dim">{descriptor.description}</p>
        </div>
        <span className={`pill pill-${status}`}>{status}</span>
      </header>
      <FilterFields schema={descriptor.inputSchema} values={args} onChange={setArgs} disabled={live} />
      <div className="actions">
        {live ? (
          <button className="secondary" onClick={disconnect}>
            Disconnect
          </button>
        ) : (
          <>
            <button onClick={() => connect(args, null)}>Subscribe from now</button>
            <button className="secondary" disabled={!feed.cursor} onClick={() => connect(args, feed.cursor)}>
              Resume from cursor
            </button>
          </>
        )}
        <button className="ghost" onClick={clear}>
          Clear
        </button>
        <span className="mono dim cursor-label">
          client cursor <b>{feed.cursor ?? "none"}</b>
          {feed.duplicates > 0 && ` · ${feed.duplicates} duplicates dropped`}
        </span>
      </div>
      <FeedList entries={feed.entries} empty="Not subscribed yet. Subscribe, then trigger a deploy here or from Claude Code / Codex." />
    </section>
  );
}
