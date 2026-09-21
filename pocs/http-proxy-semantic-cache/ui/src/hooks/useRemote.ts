import { useEffect, useState } from "react";

export interface Remote<T> {
  data: T | null;
  error: string | null;
}

export function useRemote<T>(load: () => Promise<T>, version: number): Remote<T> {
  const [remote, setRemote] = useState<Remote<T>>({ data: null, error: null });

  useEffect(() => {
    let active = true;
    load()
      .then((data) => active && setRemote({ data, error: null }))
      .catch((error: Error) => active && setRemote((previous) => ({ data: previous.data, error: error.message })));
    return () => {
      active = false;
    };
  }, [load, version]);

  return remote;
}
