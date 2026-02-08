interface Props {
  guess: string | null
  status: string
  error: string | null
}

const containerStyle: React.CSSProperties = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  background: '#16162a',
  border: '2px solid #7c4dff',
  borderRadius: '16px',
  padding: '32px 48px',
  textAlign: 'center',
  zIndex: 100,
  minWidth: '300px',
}

const guessStyle: React.CSSProperties = {
  fontSize: '48px',
  fontWeight: 700,
  color: '#7c4dff',
  margin: '16px 0',
}

const errorStyle: React.CSSProperties = {
  fontSize: '14px',
  color: '#ff4444',
  margin: '16px 0',
}

const labelStyle: React.CSSProperties = {
  fontSize: '14px',
  color: '#9e9e9e',
}

export default function GuessResult({ guess, status, error }: Props) {
  if (status === 'error') {
    return (
      <div style={containerStyle}>
        <p style={labelStyle}>Error</p>
        <p style={errorStyle}>{error || 'Unknown error'}</p>
      </div>
    )
  }

  return (
    <div style={containerStyle}>
      <p style={labelStyle}>I think this is...</p>
      <p style={guessStyle}>{guess}</p>
    </div>
  )
}
