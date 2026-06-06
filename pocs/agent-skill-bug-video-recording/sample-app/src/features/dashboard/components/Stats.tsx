import type { Stat } from '../api/dashboard'

export function Stats({ metrics }: { metrics: Stat[] }) {
  return (
    <ul className="stats">
      {metrics.map(m => (
        <li key={m.id} className="stat">
          <span className="stat-label">{m.label}</span>
          <span className="stat-value">{m.value}</span>
        </li>
      ))}
    </ul>
  )
}
