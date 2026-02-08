import { useState, useEffect } from 'react'
import { projectStreamUrl } from '../api/client'

interface Props {
  projectId: string
  onComplete: () => void
  onError: (msg: string) => void
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: 'calc(100vh - 49px)',
  gap: '24px',
  padding: '40px',
}

const barOuterStyle: React.CSSProperties = {
  width: '100%',
  maxWidth: '600px',
  height: '20px',
  background: '#1a1a2e',
  borderRadius: '10px',
  overflow: 'hidden',
  border: '1px solid #2a2a4a',
}

const stepNames: Record<string, string> = {
  analyzing_drawing: 'Analyzing your drawing...',
  generating_code: 'Generating HTML/CSS/JS...',
  saving_files: 'Saving files...',
  done: 'Build complete!',
}

export default function BuildProgress({ projectId, onComplete, onError }: Props) {
  const [step, setStep] = useState('analyzing_drawing')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const url = projectStreamUrl(projectId)
    const es = new EventSource(url)

    es.addEventListener('status', (e) => {
      const data = JSON.parse(e.data)
      setStep(data.step)
      setProgress(data.progress)
    })

    es.addEventListener('build_complete', () => {
      es.close()
      onComplete()
    })

    es.addEventListener('error', (e) => {
      if (e instanceof MessageEvent) {
        const data = JSON.parse(e.data)
        es.close()
        onError(data.message)
      }
    })

    return () => es.close()
  }, [projectId, onComplete, onError])

  return (
    <div style={containerStyle}>
      <h2 style={{ fontSize: '24px', fontWeight: 600, color: '#e0e0e0' }}>
        {stepNames[step] || step}
      </h2>
      <div style={barOuterStyle}>
        <div
          style={{
            width: `${progress}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #7c4dff, #b47cff)',
            borderRadius: '10px',
            transition: 'width 0.5s ease',
          }}
        />
      </div>
      <p style={{ color: '#9e9e9e', fontSize: '14px' }}>{progress}%</p>
    </div>
  )
}
