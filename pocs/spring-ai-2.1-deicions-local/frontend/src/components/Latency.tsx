type LatencyProps = { readonly ms: number; readonly model?: string };

export function Latency({ ms, model }: LatencyProps) {
  return (
    <p className="latency">
      {model ? <span className="mono">{model}</span> : null}
      <span className="mono">{ms} ms</span>
      <span>one decision call</span>
    </p>
  );
}
