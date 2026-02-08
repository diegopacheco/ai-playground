import { useState, useEffect, useRef } from 'react'
import { projectStreamUrl } from '../api/client'

interface Props {
  projectId: string
  onComplete: () => void
  onError: (msg: string) => void
}

const steps = [
  { key: 'analyzing_drawing', label: 'Analyzing Drawing', target: 25 },
  { key: 'generating_code', label: 'Generating Code', target: 50 },
  { key: 'saving_files', label: 'Saving Files', target: 75 },
  { key: 'done', label: 'Complete', target: 100 },
]

export default function BuildProgress({ projectId, onComplete, onError }: Props) {
  const [step, setStep] = useState('analyzing_drawing')
  const [serverProgress, setServerProgress] = useState(0)
  const [displayProgress, setDisplayProgress] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const startTime = useRef(Date.now())

  useEffect(() => {
    const url = projectStreamUrl(projectId)
    const es = new EventSource(url)

    es.addEventListener('status', (e) => {
      const data = JSON.parse(e.data)
      setStep(data.step)
      setServerProgress(data.progress)
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

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime.current) / 1000))
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (displayProgress >= serverProgress) return
    const interval = setInterval(() => {
      setDisplayProgress(prev => {
        if (prev >= serverProgress) { clearInterval(interval); return prev }
        return prev + 1
      })
    }, 40)
    return () => clearInterval(interval)
  }, [serverProgress, displayProgress])

  useEffect(() => {
    if (step !== 'done' && serverProgress > 0) {
      const currentStep = steps.find(s => s.key === step)
      if (!currentStep) return
      const nextTarget = currentStep.target
      if (displayProgress >= nextTarget - 2) return
      const drift = setInterval(() => {
        setDisplayProgress(prev => {
          if (prev >= nextTarget - 2) { clearInterval(drift); return prev }
          return prev + 1
        })
      }, 3000)
      return () => clearInterval(drift)
    }
  }, [step, serverProgress, displayProgress])

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  const currentStepIdx = steps.findIndex(s => s.key === step)

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: 'calc(100vh - 49px)', gap: '32px', padding: '40px',
    }}>
      <div style={{ position: 'relative', width: 80, height: 80 }}>
        <div style={{
          width: 80, height: 80, borderRadius: '50%',
          border: '4px solid #2a2a4a', borderTopColor: '#7c4dff',
          animation: 'spin 1s linear infinite',
        }} />
        <div style={{
          position: 'absolute', top: 0, left: 0, width: 80, height: 80,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '18px', fontWeight: 700, color: '#7c4dff',
        }}>
          {displayProgress}%
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg) } }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
@keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }`}</style>

      <h2 style={{ fontSize: '22px', fontWeight: 600, color: '#e0e0e0', margin: 0, textAlign: 'center' }}>
        {steps.find(s => s.key === step)?.label || step}
      </h2>

      <div style={{
        width: '100%', maxWidth: '600px', height: '24px',
        background: '#1a1a2e', borderRadius: '12px', overflow: 'hidden',
        border: '1px solid #2a2a4a', position: 'relative',
      }}>
        <div style={{
          width: `${displayProgress}%`, height: '100%', borderRadius: '12px',
          background: 'linear-gradient(90deg, #7c4dff, #b47cff, #7c4dff)',
          backgroundSize: '200% 100%',
          animation: step !== 'done' ? 'shimmer 2s linear infinite' : 'none',
          transition: 'width 0.3s ease',
        }} />
      </div>

      <div style={{
        display: 'flex', gap: '16px', flexWrap: 'wrap', justifyContent: 'center',
      }}>
        {steps.map((s, i) => {
          const isDone = i < currentStepIdx || step === 'done'
          const isCurrent = i === currentStepIdx && step !== 'done'
          return (
            <div key={s.key} style={{
              display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px',
              color: isDone ? '#4caf50' : isCurrent ? '#b47cff' : '#555',
              animation: isCurrent ? 'pulse 1.5s ease-in-out infinite' : 'none',
            }}>
              <div style={{
                width: 10, height: 10, borderRadius: '50%',
                background: isDone ? '#4caf50' : isCurrent ? '#7c4dff' : '#333',
                border: isCurrent ? '2px solid #b47cff' : '2px solid transparent',
              }} />
              {s.label}
            </div>
          )
        })}
      </div>

      <p style={{ color: '#666', fontSize: '13px', margin: 0 }}>
        Elapsed: {formatTime(elapsed)}
      </p>
    </div>
  )
}
