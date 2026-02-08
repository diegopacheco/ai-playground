import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { fetchProjects, deleteProject } from '../api/client'

const containerStyle: React.CSSProperties = {
  padding: '32px',
  maxWidth: '900px',
  margin: '0 auto',
}

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '12px 16px',
  color: '#9e9e9e',
  fontSize: '13px',
  borderBottom: '1px solid #2a2a4a',
  fontWeight: 600,
}

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderBottom: '1px solid #1a1a2e',
  fontSize: '14px',
}

const linkStyle: React.CSSProperties = {
  color: '#7c4dff',
  textDecoration: 'none',
}

const deleteBtnStyle: React.CSSProperties = {
  padding: '4px 12px',
  fontSize: '12px',
  background: '#ff4444',
  color: '#fff',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
}

const statusStyle = (status: string): React.CSSProperties => ({
  padding: '2px 8px',
  borderRadius: '4px',
  fontSize: '12px',
  fontWeight: 600,
  background: status === 'done' ? '#1b5e20' : status === 'error' ? '#b71c1c' : '#e65100',
  color: '#fff',
  display: 'inline-block',
})

export default function HistoryPage() {
  const queryClient = useQueryClient()
  const { data: projects, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: fetchProjects,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteProject,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }),
  })

  if (isLoading) return <div style={containerStyle}><p>Loading...</p></div>

  return (
    <div style={containerStyle}>
      <h1 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '24px' }}>Build History</h1>
      {(!projects || projects.length === 0) ? (
        <p style={{ color: '#9e9e9e' }}>No projects yet. <Link to="/" style={linkStyle}>Create one</Link></p>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Engine</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr key={p.id}>
                <td style={tdStyle}>
                  {p.status === 'done' ? (
                    <Link to="/preview/$projectId" params={{ projectId: p.id }} style={linkStyle}>
                      {p.name}
                    </Link>
                  ) : (
                    p.name
                  )}
                </td>
                <td style={tdStyle}>{p.engine}</td>
                <td style={tdStyle}><span style={statusStyle(p.status)}>{p.status}</span></td>
                <td style={tdStyle}>{new Date(p.created_at).toLocaleString()}</td>
                <td style={tdStyle}>
                  <button
                    style={deleteBtnStyle}
                    onClick={() => deleteMutation.mutate(p.id)}
                  >Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
