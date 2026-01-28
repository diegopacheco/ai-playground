import { useProject } from '../api/queries'
import { CycleReport } from './CycleReport'

const styles = {
  container: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' },
  header: { padding: '16px', borderBottom: '1px solid #30363d', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontSize: '18px', fontWeight: 600, color: '#f0f6fc' },
  back: { padding: '6px 12px', background: 'transparent', border: '1px solid #30363d', borderRadius: '6px', color: '#8b949e', cursor: 'pointer', fontSize: '13px' },
  content: { padding: '16px' },
  section: { marginBottom: '20px' },
  sectionTitle: { fontSize: '14px', fontWeight: 600, color: '#c9d1d9', marginBottom: '8px' },
  pre: { background: '#0d1117', padding: '12px', borderRadius: '6px', fontSize: '12px', color: '#8b949e', whiteSpace: 'pre-wrap' as const, maxHeight: '200px', overflowY: 'auto' as const },
  empty: { color: '#6e7681', fontStyle: 'italic' as const },
}

interface Props {
  projectName: string
  onBack: () => void
}

export function ProjectDetail({ projectName, onBack }: Props) {
  const { data: project, isLoading } = useProject(projectName)
  if (isLoading) return <div style={styles.container}><div style={{ ...styles.content, ...styles.empty }}>Loading...</div></div>
  if (!project) return <div style={styles.container}><div style={{ ...styles.content, ...styles.empty }}>Project not found</div></div>
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>{project.name}</h2>
        <button style={styles.back} onClick={onBack}>Back</button>
      </div>
      <div style={styles.content}>
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>memory.txt (learnings)</h3>
          <pre style={styles.pre}>{project.memory || <span style={styles.empty}>No learnings yet</span>}</pre>
        </div>
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>mistakes.txt</h3>
          <pre style={styles.pre}>{project.mistakes || <span style={styles.empty}>No mistakes recorded</span>}</pre>
        </div>
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>prompts.md</h3>
          <pre style={styles.pre}>{project.prompts || <span style={styles.empty}>No prompts</span>}</pre>
        </div>
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>Cycles ({project.cycles.length})</h3>
          {project.cycles.length === 0 ? (
            <span style={styles.empty}>No cycles yet</span>
          ) : (
            project.cycles.map((c) => <CycleReport key={c.cycle_number} cycle={c} />)
          )}
        </div>
      </div>
    </div>
  )
}
