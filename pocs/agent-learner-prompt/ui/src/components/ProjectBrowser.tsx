import { useProjects } from '../api/queries'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' },
  header: { padding: '16px', borderBottom: '1px solid #30363d' },
  title: { fontSize: '16px', fontWeight: 600, color: '#f0f6fc' },
  list: { maxHeight: '400px', overflowY: 'auto' as const },
  item: { padding: '12px 16px', borderBottom: '1px solid #21262d', cursor: 'pointer', transition: 'background 0.2s' },
  itemHover: { background: '#21262d' },
  name: { fontSize: '14px', color: '#c9d1d9', marginBottom: '4px' },
  meta: { fontSize: '12px', color: '#8b949e', display: 'flex', gap: '12px' },
  badge: { padding: '2px 6px', borderRadius: '4px', fontSize: '11px' },
  hasMem: { background: '#1f6feb33', color: '#58a6ff' },
  hasMistakes: { background: '#da363333', color: '#f85149' },
  empty: { padding: '24px', textAlign: 'center' as const, color: '#8b949e' },
}

interface Props {
  onSelect: (name: string) => void
}

export function ProjectBrowser({ onSelect }: Props) {
  const { data: projects, isLoading } = useProjects()
  if (isLoading) return <div style={styles.container}><div style={styles.empty}>Loading...</div></div>
  if (!projects?.length) return <div style={styles.container}><div style={styles.empty}>No projects yet</div></div>
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Projects ({projects.length})</h3>
      </div>
      <div style={styles.list}>
        {projects.map((p) => (
          <div key={p.name} style={styles.item} onClick={() => onSelect(p.name)}>
            <div style={styles.name}>{p.name}</div>
            <div style={styles.meta}>
              <span>{p.cycles.length} cycles</span>
              {p.has_memory && <span style={{ ...styles.badge, ...styles.hasMem }}>memory</span>}
              {p.has_mistakes && <span style={{ ...styles.badge, ...styles.hasMistakes }}>mistakes</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
