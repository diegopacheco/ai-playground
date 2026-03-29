import { useEffect, useRef } from "react";
import type {
  AgentAction,
  AgentLog,
  Counters,
  TestClassification,
} from "../types";

interface SSECallbacks {
  onAction?: (action: AgentAction) => void;
  onCounterUpdate?: (counters: Counters) => void;
  onTestClassification?: (tc: TestClassification) => void;
  onNewLog?: (log: AgentLog) => void;
  onCycleStart?: (data: { cycle: number }) => void;
  onCycleEnd?: (data: { cycle: number; status: string }) => void;
}

export function useSSE(callbacks: SSECallbacks) {
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useEffect(() => {
    let es: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      es = new EventSource("/api/events");

      es.addEventListener("action", (e) => {
        cbRef.current.onAction?.(JSON.parse(e.data));
      });

      es.addEventListener("counter_update", (e) => {
        cbRef.current.onCounterUpdate?.(JSON.parse(e.data));
      });

      es.addEventListener("test_classification", (e) => {
        cbRef.current.onTestClassification?.(JSON.parse(e.data));
      });

      es.addEventListener("new_log", (e) => {
        cbRef.current.onNewLog?.(JSON.parse(e.data));
      });

      es.addEventListener("cycle_start", (e) => {
        cbRef.current.onCycleStart?.(JSON.parse(e.data));
      });

      es.addEventListener("cycle_end", (e) => {
        cbRef.current.onCycleEnd?.(JSON.parse(e.data));
      });

      es.onerror = () => {
        es?.close();
        reconnectTimer = setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      es?.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, []);
}
