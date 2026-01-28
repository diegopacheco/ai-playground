import type { TaskStatus } from '../types'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '20px' },
  title: { fontSize: '16px', fontWeight: 600, color: '#f0f6fc', marginBottom: '16px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' },
  stat: { background: '#0d1117', padding: '16px', borderRadius: '6px', textAlign: 'center' as const },
  value: { fontSize: '28px', fontWeight: 600, color: '#f0f6fc' },
  label: { fontSize: '12px', color: '#8b949e', marginTop: '4px' },
  success: { color: '#3fb950' },
  failed: { color: '#f85149' },
  running: { color: '#1f6feb' },
}

interface Props {
  status: TaskStatus | null
}

export function SessionSummary({ status }: Props) {
  if (!status) return null
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Session Summary</h3>
      <div style={styles.grid}>
        <div style={styles.stat}>
          <div style={styles.value}>{status.total_cycles}</div>
          <div style={styles.label}>Total Cycles</div>
        </div>
        <div style={styles.stat}>
          <div style={{ ...styles.value, ...(status.completed ? (status.success ? styles.success : styles.failed) : styles.running) }}>
            {status.completed ? (status.success ? 'OK' : 'FAIL') : 'RUN'}
          </div>
          <div style={styles.label}>Status</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.value}>{status.current_cycle}</div>
          <div style={styles.label}>Current Cycle</div>
        </div>
      </div>
    </div>
  )
}
