export function TemporalButton({ url, label = "See it running in Temporal" }: { url?: string; label?: string }) {
  if (!url) return <button className="engine" disabled>Waiting for Temporal</button>;
  return (
    <a className="engine" href={url} target="_blank" rel="noreferrer">
      {label}
    </a>
  );
}
