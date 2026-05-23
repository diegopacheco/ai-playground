export interface AsyncDisposable {
  close(): Promise<void>;
}

export async function withDisposable<R extends AsyncDisposable, T>(
  acquire: () => Promise<R>,
  use: (resource: R) => Promise<T>,
): Promise<T> {
  const resource = await acquire();
  try {
    return await use(resource);
  } finally {
    try {
      await resource.close();
    } catch (closeError) {
      const message =
        closeError instanceof Error ? closeError.message : String(closeError);
      console.warn(`[lifecycle] close failed: ${message}`);
    }
  }
}
