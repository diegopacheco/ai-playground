import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { WorkersTable } from "../features/actors/components/WorkersTable";
import { useWorkers } from "../features/actors/hooks";

export const Route = createFileRoute("/workers")({
  component: WorkersPage,
});

function WorkersPage() {
  const { data, isPending, error } = useWorkers();
  const rows = useMemo(() => [...(data ?? [])], [data]);
  const busy = rows.filter((w) => w.actors > 0).length;
  return (
    <section className="card">
      <h2>Worker pool</h2>
      <p className="muted">
        Warm gVisor worker pods. {busy} of {rows.length} hold an actor right now. The rest wait for the next resume.
      </p>
      {isPending && <p className="muted">Loading...</p>}
      {error && <p className="error">{error.message}</p>}
      {rows.length > 0 && <WorkersTable rows={rows} />}
    </section>
  );
}
