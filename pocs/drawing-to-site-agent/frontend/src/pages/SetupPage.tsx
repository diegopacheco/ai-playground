import { useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import EngineSelector from '../components/EngineSelector'

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: 'calc(100vh - 49px)',
  gap: '24px',
  padding: '40px',
}

const cardStyle: React.CSSProperties = {
  background: '#16162a',
  border: '1px solid #2a2a4a',
  borderRadius: '16px',
  padding: '40px',
  width: '100%',
  maxWidth: '480px',
  display: 'flex',
  flexDirection: 'column',
  gap: '20px',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '12px 16px',
  fontSize: '16px',
  background: '#1a1a2e',
  color: '#e0e0e0',
  border: '1px solid #2a2a4a',
  borderRadius: '8px',
  outline: 'none',
  boxSizing: 'border-box',
}

const btnStyle: React.CSSProperties = {
  padding: '14px',
  fontSize: '16px',
  fontWeight: 700,
  background: '#7c4dff',
  color: '#fff',
  border: 'none',
  borderRadius: '8px',
  cursor: 'pointer',
  marginTop: '8px',
}

const disabledBtnStyle: React.CSSProperties = {
  ...btnStyle,
  background: '#3a3a5a',
  cursor: 'not-allowed',
}

export default function SetupPage() {
  const [name, setName] = useState('')
  const [engine, setEngine] = useState('')
  const navigate = useNavigate()

  const canProceed = name.trim().length > 0 && engine.length > 0

  const handleNext = () => {
    if (!canProceed) return
    const encoded = encodeURIComponent(JSON.stringify({ name, engine }))
    navigate({ to: '/canvas/$projectId', params: { projectId: encoded } })
  }

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, color: '#e0e0e0', margin: 0, textAlign: 'center' }}>
          New Project
        </h1>
        <label style={{ color: '#9e9e9e', fontSize: '14px' }}>
          Project Name
          <input
            style={inputStyle}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-landing-page"
          />
        </label>
        <label style={{ color: '#9e9e9e', fontSize: '14px' }}>
          Agent Engine
          <EngineSelector value={engine} onChange={setEngine} />
        </label>
        <button
          style={canProceed ? btnStyle : disabledBtnStyle}
          onClick={handleNext}
          disabled={!canProceed}
        >
          Next - Draw Your Site
        </button>
      </div>
    </div>
  )
}
