type ErrorNoteProps = { readonly error: Error | null };

export function ErrorNote({ error }: ErrorNoteProps) {
  if (!error) return null;
  return <p className="error" role="alert">{error.message}</p>;
}
