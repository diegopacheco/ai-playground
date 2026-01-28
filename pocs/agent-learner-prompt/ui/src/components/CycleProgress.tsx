import type { TaskStatus, ProgressEvent } from '../types'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '20px', marginBottom: '20px' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' },
  title: { fontSize: '16px', fontWeight: 600, color: '#f0f6fc' },
  status: { padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 600 },
  running: { background: '#1f6feb', color: '#fff' },
  completed: { background: '#238636', color: '#fff' },
  failed: { background: '#da3633', color: '#fff' },
  progressBar: { height: '8px', background: '#21262d', borderRadius: '4px', overflow: 'hidden', marginBottom: '16px' },
  progressFill: { height: '100%', background: '#238636', transition: 'width 0.3s' },
  info: { display: 'flex', gap: '24px', color: '#8b949e', fontSize: '14px' },
  events: { marginTop: '16px', maxHeight: '200px', overflowY: 'auto' as const },
  event: { padding: '8px 0', borderBottom: '1px solid #21262d', fontSize: '13px', color: '#8b949e' },
}

interface Props {
  status: TaskStatus | null
  events: ProgressEvent[]
}

export function CycleProgress({ status, events }: Props) {
  if (!status) return null
  const progress = status.total_cycles > 0 ? (status.current_cycle / status.total_cycles) * 100 : 0
  const statusStyle = status.completed ? (status.success ? styles.completed : styles.failed) : styles.running
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Task: {status.task_id.slice(0, 8)}...</span>
        <span style={{ ...styles.status, ...statusStyle }}>{status.status}</span>
      </div>
      <div style={styles.progressBar}>
        <div style={{ ...styles.progressFill, width: `${progress}%` }} />
      </div>
      <div style={styles.info}>
        <span>Cycle: {status.current_cycle}/{status.total_cycles}</span>
        <span>Phase: {status.phase}</span>
      </div>
      {events.length > 0 && (
        <div style={styles.events}>
          {events.map((e, i) => (
            <div key={i} style={styles.event}>
              [{e.event_type}] Cycle {e.cycle} - {e.phase}: {e.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
