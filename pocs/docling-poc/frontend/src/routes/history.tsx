import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { HistoryTable } from "../features/conversions/components/HistoryTable";
import { useHistory } from "../features/conversions/hooks";

export const Route = createFileRoute("/history")({
  component: HistoryPage,
});

function HistoryPage() {
  const { data, isPending, error } = useHistory();
  const rows = useMemo(() => [...(data ?? [])], [data]);
  if (isPending) return <p className="muted">Loading...</p>;
  if (error) return <p className="error">{error.message}</p>;
  return (
    <section className="card">
      <h2>Conversion history</h2>
      {rows.length === 0 ? <p className="muted">No documents converted yet.</p> : <HistoryTable rows={rows} />}
    </section>
  );
}
