import { Link } from "@tanstack/react-router";

export function JournalButton({ runId, label = "See it running in DBOS" }: { runId: string; label?: string }) {
  return (
    <Link className="engine" to="/runs/$runId/journal" params={{ runId }}>
      {label}
    </Link>
  );
}
