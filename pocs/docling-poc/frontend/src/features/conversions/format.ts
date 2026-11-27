export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function acceptsFile(name: string, extensions: readonly string[]): boolean {
  const dot = name.lastIndexOf(".");
  if (dot < 0) return false;
  return extensions.includes(name.slice(dot).toLowerCase());
}
