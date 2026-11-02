import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { mcp } from "./client";
import { applyNotification, emptyFeed, recordError, type FeedState } from "./feed";
import type { ConnectedClient, EventDescriptor } from "./types";

export type StreamStatus = "idle" | "connecting" | "streaming" | "closed" | "error";

export function useEventTypes() {
  return useQuery({
    queryKey: ["events/list"],
    queryFn: async () => (await mcp.request<{ events: EventDescriptor[] }>("events/list")).events,
  });
}

export function useConnectedClients() {
  return useQuery({
    queryKey: ["clients"],
    queryFn: async () => ((await (await fetch("/api/clients")).json()) as { clients: ConnectedClient[] }).clients,
    refetchInterval: 2000,
  });
}

export function useEventStream(name: string) {
  const [feed, setFeed] = useState<FeedState>(emptyFeed);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const controller = useRef<AbortController | null>(null);

  const disconnect = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setStatus("closed");
  }, []);

  const connect = useCallback(
    (args: Record<string, string>, cursor: string | null) => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setStatus("connecting");
      mcp
        .stream(
          { name, arguments: args, cursor },
          (message) => {
            setStatus("streaming");
            setFeed((state) => applyNotification(state, message, Date.now()));
          },
          abort.signal,
        )
        .then(() => setStatus("closed"))
        .catch((error: unknown) => {
          if (abort.signal.aborted) return;
          setStatus("error");
          setFeed((state) => recordError(state, error instanceof Error ? error.message : String(error), Date.now()));
        });
    },
    [name],
  );

  useEffect(() => () => controller.current?.abort(), []);

  const clear = useCallback(() => setFeed((state) => ({ ...emptyFeed, cursor: state.cursor, seen: state.seen })), []);

  return { feed, status, connect, disconnect, clear };
}
