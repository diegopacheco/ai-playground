export const DASHBOARD_URL = import.meta.env.VITE_DASHBOARD_URL;

export function DashboardButton({ runId, label = "See it running in DBOS" }: { runId: string; label?: string }) {
  return (
    <a className="engine" href={`${DASHBOARD_URL}/workflows/${encodeURIComponent(runId)}`} target="_blank" rel="noreferrer">
      {label}
    </a>
  );
}
