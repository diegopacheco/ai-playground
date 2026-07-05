import { Icon } from "./Icon"

type MetricIcon = "film" | "tv" | "play" | "clock"

export function MetricCard({ label, value, detail, icon }: { label: string; value: string; detail: string; icon: MetricIcon }) {
  return <article className="metric-card"><span className="metric-icon"><Icon name={icon}/></span><div><small>{label}</small><strong>{value}</strong><span>{detail}</span></div></article>
}
