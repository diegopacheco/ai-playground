export function Loading({ what }: { what?: string }) {
  return <div className="status loading">Loading {what ?? "data"}…</div>;
}

export function ErrorView({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "unexpected error";
  return <div className="status error">Could not load: {message}</div>;
}

export function Empty({ message }: { message: string }) {
  return <div className="status empty">{message}</div>;
}
