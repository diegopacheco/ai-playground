import { restateUiUrl } from "./restate.ts";

export function RestateButton({ invocationId, label = "See it running in Restate" }: { invocationId?: string; label?: string }) {
  if (!invocationId) return <button className="restate" disabled>Waiting for Restate</button>;
  return (
    <a className="restate" href={restateUiUrl(import.meta.env.VITE_RESTATE_ADMIN, invocationId)} target="_blank" rel="noreferrer">
      {label}
    </a>
  );
}
