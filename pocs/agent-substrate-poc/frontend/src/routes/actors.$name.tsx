import { createFileRoute } from "@tanstack/react-router";
import { ActionButtons } from "../features/actors/components/ActionButtons";
import { AgentMemory } from "../features/actors/components/AgentMemory";
import { NoteForm } from "../features/actors/components/NoteForm";
import { StateBadge } from "../features/actors/components/StateBadge";
import { useActor, useAgentState } from "../features/actors/hooks";
import { isTransient } from "../features/actors/rules";

export const Route = createFileRoute("/actors/$name")({
  component: ActorPage,
});

function ActorPage() {
  const { name } = Route.useParams();
  const actor = useActor(name);
  const running = actor.data?.state === "RUNNING";
  const agent = useAgentState(name, running);
  if (actor.isPending) return <p className="muted">Loading...</p>;
  if (actor.error) return <p className="error">{actor.error.message}</p>;
  const a = actor.data;
  return (
    <section className="stack">
      <div className="card head">
        <div>
          <h2>{a.name}</h2>
          <p className="muted">
            template {a.template} - version {a.version} - {a.snapshots} snapshots - worker {a.workerPod || "none"}
          </p>
        </div>
        <div className="head-right">
          <StateBadge state={a.state} />
          <ActionButtons name={a.name} state={a.state} />
        </div>
      </div>
      <div className="card">
        <NoteForm name={a.name} onSaved={() => void agent.refetch()} />
      </div>
      {!running && (
        <p className="muted card">
          {isTransient(a.state)
            ? "The actor is moving between workers..."
            : "Hibernated: no worker pod is assigned. Its RAM and disk live in a snapshot. Send a note or press resume to wake it."}
        </p>
      )}
      {running && agent.error && <p className="error">{agent.error.message}</p>}
      {running && agent.data && <AgentMemory state={agent.data} />}
    </section>
  );
}
