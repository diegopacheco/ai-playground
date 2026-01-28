import type { CycleInfo } from '../types'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '16px', marginBottom: '12px' },
  header: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' },
  badge: { background: '#238636', color: '#fff', padding: '4px 10px', borderRadius: '4px', fontSize: '12px', fontWeight: 600 },
  files: { display: 'flex', gap: '8px', flexWrap: 'wrap' as const },
  file: { padding: '4px 8px', background: '#21262d', borderRadius: '4px', fontSize: '12px', color: '#8b949e' },
  present: { color: '#3fb950' },
  absent: { color: '#6e7681' },
}

interface Props {
  cycle: CycleInfo
}

export function CycleReport({ cycle }: Props) {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.badge}>Cycle {cycle.cycle_number}</span>
      </div>
      <div style={styles.files}>
        <span style={{ ...styles.file, ...(cycle.has_prompt ? styles.present : styles.absent) }}>
          prompt.txt {cycle.has_prompt ? '(ok)' : '(-)'}
        </span>
        <span style={{ ...styles.file, ...(cycle.has_output ? styles.present : styles.absent) }}>
          output.txt {cycle.has_output ? '(ok)' : '(-)'}
        </span>
        <span style={{ ...styles.file, ...(cycle.has_review ? styles.present : styles.absent) }}>
          review.txt {cycle.has_review ? '(ok)' : '(-)'}
        </span>
      </div>
    </div>
  )
}
