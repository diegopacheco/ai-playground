import { useParams, Link } from '@tanstack/react-router'
import SitePreview from '../components/SitePreview'

const barStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '10px 24px',
  background: '#1a1a2e',
  borderBottom: '1px solid #2a2a4a',
}

const linkStyle: React.CSSProperties = {
  padding: '6px 16px',
  fontSize: '13px',
  background: '#2a2a4a',
  color: '#e0e0e0',
  border: '1px solid #3a3a5a',
  borderRadius: '6px',
  textDecoration: 'none',
  cursor: 'pointer',
}

const primaryLinkStyle: React.CSSProperties = {
  ...linkStyle,
  background: '#7c4dff',
  borderColor: '#7c4dff',
  color: '#fff',
}

export default function PreviewPage() {
  const { projectId } = useParams({ from: '/preview/$projectId' })

  return (
    <div>
      <div style={barStyle}>
        <span style={{ color: '#9e9e9e', fontSize: '14px', marginRight: 'auto' }}>
          Preview: {projectId.slice(0, 8)}...
        </span>
        <Link to="/" style={primaryLinkStyle}>New Project</Link>
        <Link to="/history" style={linkStyle}>History</Link>
      </div>
      <div style={{ padding: '16px' }}>
        <SitePreview projectId={projectId} />
      </div>
    </div>
  )
}
