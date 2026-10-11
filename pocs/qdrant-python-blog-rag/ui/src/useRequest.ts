import { useState } from "react";

export type RequestState<T> = {
  data: T | null;
  error: string;
  loading: boolean;
  run: (task: () => Promise<T>) => Promise<void>;
};

export function useRequest<T>(): RequestState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function run(task: () => Promise<T>): Promise<void> {
    setLoading(true);
    setError("");
    try {
      setData(await task());
    } catch (failure) {
      setData(null);
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setLoading(false);
    }
  }

  return { data, error, loading, run };
}
