export function TriggerButton({ url, label = "See it running in Trigger.dev" }: { url?: string; label?: string }) {
  if (!url) return <button className="engine" disabled>Waiting for Trigger.dev</button>;
  return (
    <a className="engine" href={url} target="_blank" rel="noreferrer">
      {label}
    </a>
  );
}
