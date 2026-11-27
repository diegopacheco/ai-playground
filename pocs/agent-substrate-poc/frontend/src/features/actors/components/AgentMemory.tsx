import { memoryVerdict, type MemoryVerdict } from "../rules";
import type { AgentState, Note } from "../types";

const VERDICTS: Readonly<Record<MemoryVerdict, string>> = {
  empty: "No notes yet. Send one to wake the agent.",
  restored: "RAM matches the journal: process memory survived every suspend and resume.",
  "cold-boot": "RAM is missing notes the journal has: the process cold booted and only the durable volume survived.",
};

function NoteList({ title, hint, notes }: { readonly title: string; readonly hint: string; readonly notes: readonly Note[] }) {
  return (
    <div className="card">
      <h3>
        {title} <span className="count">{notes.length}</span>
      </h3>
      <p className="muted small">{hint}</p>
      <ol className="notes">
        {notes.map((n) => (
          <li key={n.id}>
            <span className="note-id">#{n.id}</span> {n.text}
          </li>
        ))}
      </ol>
    </div>
  );
}

type AgentMemoryProps = { readonly state: AgentState };

export function AgentMemory({ state }: AgentMemoryProps) {
  const verdict = memoryVerdict(state);
  return (
    <section className="stack">
      <div className="facts">
        <div className="fact">
          <span className="fact-label">Boot id</span>
          <span className="fact-value mono">{state.bootId}</span>
        </div>
        <div className="fact">
          <span className="fact-label">Process started</span>
          <span className="fact-value">{new Date(state.startedAt).toLocaleTimeString()}</span>
        </div>
        <div className="fact">
          <span className="fact-label">Requests served</span>
          <span className="fact-value">{state.requests}</span>
        </div>
        <div className="fact">
          <span className="fact-label">Sandbox hostname</span>
          <span className="fact-value mono">{state.hostname}</span>
        </div>
      </div>
      <p className={`verdict verdict-${verdict}`}>{VERDICTS[verdict]}</p>
      <div className="two">
        <NoteList title="In memory (RAM)" hint="A Go slice inside the process. Survives only a memory snapshot." notes={state.memoryNotes} />
        <NoteList title="Journal (durable volume)" hint="/data/journal.log on the durableDir volume." notes={state.journalNotes} />
      </div>
    </section>
  );
}
