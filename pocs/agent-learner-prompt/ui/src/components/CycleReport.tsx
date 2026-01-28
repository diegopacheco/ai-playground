import { useState } from 'react'
import type { CycleInfo } from '../types'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '16px', marginBottom: '12px' },
  header: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', cursor: 'pointer' },
  badge: { background: '#238636', color: '#fff', padding: '4px 10px', borderRadius: '4px', fontSize: '12px', fontWeight: 600 },
  expandIcon: { marginLeft: 'auto', color: '#8b949e', fontSize: '14px' },
  files: { display: 'flex', gap: '8px', flexWrap: 'wrap' as const, marginBottom: '12px' },
  file: { padding: '4px 8px', background: '#21262d', borderRadius: '4px', fontSize: '12px', color: '#8b949e' },
  present: { color: '#3fb950' },
  absent: { color: '#6e7681' },
  content: { marginTop: '12px', borderTop: '1px solid #30363d', paddingTop: '12px' },
  section: { marginBottom: '12px' },
  sectionTitle: { fontSize: '13px', fontWeight: 600, color: '#c9d1d9', marginBottom: '6px' },
  pre: { background: '#0d1117', padding: '10px', borderRadius: '4px', fontSize: '11px', color: '#8b949e', whiteSpace: 'pre-wrap' as const, maxHeight: '150px', overflowY: 'auto' as const, margin: 0 },
  empty: { color: '#6e7681', fontStyle: 'italic' as const },
}

interface Props {
  cycle: CycleInfo
}

export function CycleReport({ cycle }: Props) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div style={styles.container}>
      <div style={styles.header} onClick={() => setExpanded(!expanded)}>
        <span style={styles.badge}>Cycle {cycle.cycle_number}</span>
        <span style={styles.expandIcon}>{expanded ? '[-]' : '[+]'}</span>
      </div>
      <div style={styles.files}>
        <span style={{ ...styles.file, ...(cycle.has_learnings ? styles.present : styles.absent) }}>
          learnings.txt {cycle.has_learnings ? '(ok)' : '(-)'}
        </span>
        <span style={{ ...styles.file, ...(cycle.has_mistakes ? styles.present : styles.absent) }}>
          mistakes.txt {cycle.has_mistakes ? '(ok)' : '(-)'}
        </span>
        <span style={{ ...styles.file, ...(cycle.has_improved_prompt ? styles.present : styles.absent) }}>
          improved_prompt.txt {cycle.has_improved_prompt ? '(ok)' : '(-)'}
        </span>
      </div>
      {expanded && (
        <div style={styles.content}>
          <div style={styles.section}>
            <div style={styles.sectionTitle}>learnings.txt</div>
            <pre style={styles.pre}>{cycle.learnings_content || <span style={styles.empty}>No learnings</span>}</pre>
          </div>
          <div style={styles.section}>
            <div style={styles.sectionTitle}>mistakes.txt</div>
            <pre style={styles.pre}>{cycle.mistakes_content || <span style={styles.empty}>No mistakes</span>}</pre>
          </div>
          <div style={styles.section}>
            <div style={styles.sectionTitle}>improved_prompt.txt</div>
            <pre style={styles.pre}>{cycle.improved_prompt_content || <span style={styles.empty}>No improved prompt</span>}</pre>
          </div>
        </div>
      )}
    </div>
  )
}
