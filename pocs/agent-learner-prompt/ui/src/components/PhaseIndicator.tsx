const styles = {
  container: { display: 'flex', gap: '4px', alignItems: 'center' },
  phase: { width: '12px', height: '12px', borderRadius: '50%', background: '#21262d' },
  active: { background: '#1f6feb', animation: 'pulse 1.5s ease-in-out infinite' },
  completed: { background: '#238636' },
  label: { marginLeft: '8px', fontSize: '12px', color: '#8b949e' },
}

const PHASES = ['generating', 'running', 'reviewing', 'learning', 'improving', 'done']

interface Props {
  currentPhase: string
}

export function PhaseIndicator({ currentPhase }: Props) {
  const currentIdx = PHASES.indexOf(currentPhase.toLowerCase())
  return (
    <div style={styles.container}>
      {PHASES.map((phase, idx) => {
        let style = styles.phase
        if (idx < currentIdx) style = { ...style, ...styles.completed }
        else if (idx === currentIdx) style = { ...style, ...styles.active }
        return <div key={phase} style={style} title={phase} />
      })}
      <span style={styles.label}>{currentPhase}</span>
    </div>
  )
}
