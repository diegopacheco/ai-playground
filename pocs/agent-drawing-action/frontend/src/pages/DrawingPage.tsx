import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { createGuess, CreateGuessResponse } from '../api/client'
import DrawingCanvas from '../components/DrawingCanvas'
import EngineSelector from '../components/EngineSelector'
import GuessResult from '../components/GuessResult'

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: 'calc(100vh - 49px)',
  position: 'relative',
}

const topBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '10px 16px',
  background: '#12122a',
  borderBottom: '1px solid #2a2a4a',
}

const overlayStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0,0,0,0.6)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 99,
}

const spinnerStyle: React.CSSProperties = {
  width: 60,
  height: 60,
  borderRadius: '50%',
  border: '4px solid #2a2a4a',
  borderTopColor: '#7c4dff',
  animation: 'spin 1s linear infinite',
}

const dismissBtnStyle: React.CSSProperties = {
  marginTop: '16px',
  padding: '8px 24px',
  fontSize: '14px',
  background: '#2a2a4a',
  color: '#e0e0e0',
  border: '1px solid #3a3a5a',
  borderRadius: '6px',
  cursor: 'pointer',
}

export default function DrawingPage() {
  const [engine, setEngine] = useState('')
  const [result, setResult] = useState<CreateGuessResponse | null>(null)

  const mutation = useMutation({
    mutationFn: createGuess,
    onSuccess: (data) => setResult(data),
  })

  const handleCapture = (dataUrl: string) => {
    if (!engine) {
      alert('Select an engine first')
      return
    }
    setResult(null)
    mutation.mutate({ engine, image: dataUrl })
  }

  const dismiss = () => setResult(null)

  return (
    <div style={containerStyle}>
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      <div style={topBarStyle}>
        <span style={{ color: '#9e9e9e', fontSize: '14px' }}>Engine:</span>
        <EngineSelector value={engine} onChange={setEngine} />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <DrawingCanvas onCapture={handleCapture} loading={mutation.isPending} />
      </div>
      {mutation.isPending && (
        <div style={overlayStyle}>
          <div style={{ textAlign: 'center' }}>
            <div style={spinnerStyle} />
            <p style={{ color: '#9e9e9e', marginTop: '16px' }}>Agent is analyzing your drawing...</p>
          </div>
        </div>
      )}
      {result && (
        <div style={overlayStyle} onClick={dismiss}>
          <div onClick={(e) => e.stopPropagation()}>
            <GuessResult guess={result.guess} status={result.status} error={result.error} />
            <div style={{ textAlign: 'center' }}>
              <button style={dismissBtnStyle} onClick={dismiss}>Draw Again</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
