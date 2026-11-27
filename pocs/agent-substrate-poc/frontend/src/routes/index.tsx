import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { ActorsTable } from "../features/actors/components/ActorsTable";
import { CreateActorForm } from "../features/actors/components/CreateActorForm";
import { useActors } from "../features/actors/hooks";

export const Route = createFileRoute("/")({
  component: ActorsPage,
});

function ActorsPage() {
  const { data, isPending, error } = useActors();
  const rows = useMemo(() => [...(data ?? [])], [data]);
  const running = rows.filter((a) => a.state === "RUNNING").length;
  return (
    <section className="stack">
      <div className="card">
        <h2>Memory agents</h2>
        <p className="muted">
          Each actor is a stateful Go agent in a gVisor sandbox. Suspend it to snapshot RAM and disk and free the worker.
          Any request resumes it on whichever worker is free.
        </p>
        <CreateActorForm />
      </div>
      <div className="card">
        <h3>
          {rows.length} actors, {running} running, {rows.length - running} not holding a worker
        </h3>
        {isPending && <p className="muted">Loading...</p>}
        {error && <p className="error">{error.message}</p>}
        {rows.length > 0 && <ActorsTable rows={rows} />}
      </div>
    </section>
  );
}
