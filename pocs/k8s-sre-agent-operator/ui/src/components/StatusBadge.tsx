interface Props {
  status: string
}

export default function StatusBadge({ status }: Props) {
  const s = status.toLowerCase()
  let bg = '#30363d'
  let color = '#c9d1d9'

  if (s === 'running' || s === 'available' || s === 'ready' || s === 'active') {
    bg = '#238636'; color = '#ffffff'
  } else if (s.includes('error') || s.includes('crash') || s.includes('fail') || s === 'notready') {
    bg = '#da3633'; color = '#ffffff'
  } else if (s.includes('pending') || s.includes('pull') || s.includes('creating') || s === 'progressing') {
    bg = '#d29922'; color = '#000000'
  }

  return (
    <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600, background: bg, color }}>
      {status}
    </span>
  )
}
