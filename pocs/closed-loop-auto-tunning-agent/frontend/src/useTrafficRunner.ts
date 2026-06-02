import { useRef, useState } from "react";
import { callEndpoint, getCbMetrics, resetCb } from "./api";
import type { RunResult } from "./types";

type RunParams = {
  endpoint: string;
  total: number;
  rps: number;
  label: string;
  resetBreaker: boolean;
  pollMetrics: boolean;
};

const zeroTally = (): Record<string, number> => ({
  SUCCESS: 0,
  FAILURE: 0,
  SHORT_CIRCUITED: 0,
  RATE_LIMITED: 0,
  REJECTED: 0,
});

function buildResult(
  id: number,
  params: RunParams,
  tally: Record<string, number>,
  latencies: number[],
  points: number[],
): RunResult {
  const sorted = [...latencies].sort((a, b) => a - b);
  const mean = sorted.length ? sorted.reduce((a, b) => a + b, 0) / sorted.length : 0;
  const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : 0;
  return {
    id,
    label: params.label,
    total: params.total,
    success: tally.SUCCESS || 0,
    failure: tally.FAILURE || 0,
    shortCircuited: tally.SHORT_CIRCUITED || 0,
    rateLimited: tally.RATE_LIMITED || 0,
    rejected: tally.REJECTED || 0,
    meanLatencyMs: Math.round(mean),
    p95LatencyMs: Math.round(p95),
    failureRatePoints: points,
  };
}

export function useTrafficRunner() {
  const [running, setRunning] = useState(false);
  const [tally, setTally] = useState<Record<string, number>>(zeroTally());
  const [points, setPoints] = useState<number[]>([]);
  const idRef = useRef(0);

  async function run(params: RunParams): Promise<RunResult> {
    setRunning(true);
    const tallyLocal = zeroTally();
    const latencies: number[] = [];
    const pts: number[] = [];
    setTally({ ...tallyLocal });
    setPoints([]);

    if (params.resetBreaker) {
      await resetCb();
    }

    return new Promise<RunResult>((resolve) => {
      let sent = 0;
      let received = 0;
      const interval = Math.max(5, Math.round(1000 / params.rps));

      const metricsTimer = params.pollMetrics
        ? window.setInterval(() => {
            getCbMetrics()
              .then((m) => {
                pts.push(Math.max(0, m.failureRate));
                setPoints([...pts]);
              })
              .catch(() => {});
          }, 400)
        : null;

      const fireTimer = window.setInterval(() => {
        if (sent >= params.total) {
          window.clearInterval(fireTimer);
          return;
        }
        sent++;
        callEndpoint(params.endpoint)
          .then((r) => {
            tallyLocal[r.outcome] = (tallyLocal[r.outcome] || 0) + 1;
            latencies.push(r.latencyMs);
          })
          .catch(() => {
            tallyLocal.FAILURE = (tallyLocal.FAILURE || 0) + 1;
          })
          .finally(() => {
            received++;
            setTally({ ...tallyLocal });
            if (received >= params.total) {
              if (metricsTimer) window.clearInterval(metricsTimer);
              setRunning(false);
              resolve(buildResult(idRef.current++, params, tallyLocal, latencies, pts));
            }
          });
      }, interval);
    });
  }

  return { running, tally, points, run };
}
