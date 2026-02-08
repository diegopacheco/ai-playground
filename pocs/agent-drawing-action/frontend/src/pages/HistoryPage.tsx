import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { fetchGuesses, deleteGuess, drawingUrl } from '../api/client'

const containerStyle: React.CSSProperties = {
  padding: '32px',
  maxWidth: '1000px',
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

const imgStyle: React.CSSProperties = {
  width: 80,
  height: 80,
  objectFit: 'cover',
  borderRadius: '8px',
  border: '1px solid #2a2a4a',
  background: '#1a1a2e',
}

const guessStyle: React.CSSProperties = {
  fontSize: '20px',
  fontWeight: 700,
  color: '#7c4dff',
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

const linkStyle: React.CSSProperties = {
  color: '#7c4dff',
  textDecoration: 'none',
}

export default function HistoryPage() {
  const queryClient = useQueryClient()
  const { data: guesses, isLoading } = useQuery({
    queryKey: ['guesses'],
    queryFn: fetchGuesses,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteGuess,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['guesses'] }),
  })

  if (isLoading) return <div style={containerStyle}><p>Loading...</p></div>

  return (
    <div style={containerStyle}>
      <h1 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '24px' }}>Guess History</h1>
      {(!guesses || guesses.length === 0) ? (
        <p style={{ color: '#9e9e9e' }}>No guesses yet. <Link to="/" style={linkStyle}>Draw something</Link></p>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Drawing</th>
              <th style={thStyle}>Guess</th>
              <th style={thStyle}>Engine</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {guesses.map((g) => (
              <tr key={g.id}>
                <td style={tdStyle}>
                  <img src={drawingUrl(g.id)} alt="drawing" style={imgStyle} />
                </td>
                <td style={tdStyle}>
                  <span style={guessStyle}>{g.guess || '-'}</span>
                </td>
                <td style={tdStyle}>{g.engine}</td>
                <td style={tdStyle}><span style={statusStyle(g.status)}>{g.status}</span></td>
                <td style={tdStyle}>{new Date(g.created_at).toLocaleString()}</td>
                <td style={tdStyle}>
                  <button
                    style={deleteBtnStyle}
                    onClick={() => deleteMutation.mutate(g.id)}
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
