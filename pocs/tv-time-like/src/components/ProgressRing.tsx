export function ProgressRing({ value, label }: { value: number; label: string }) {
  return <div className="progress-ring" style={{ "--progress": `${value * 3.6}deg` } as React.CSSProperties}>
    <div><strong>{value}%</strong><small>{label}</small></div>
  </div>
}
