import { useRef, useState } from 'react'
import { sendQuery } from './api'
import type { Coord, QueryResponse } from './types'

type Status = 'idle' | 'listening' | 'thinking' | 'done' | 'error'

interface Props {
  fallbackCenter: Coord
  onResult: (resp: QueryResponse) => void
}

function createRecognition(): any {
  const w = window as any
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition
  return Ctor ? new Ctor() : null
}

function getLocation(): Promise<Coord | null> {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      resolve(null)
      return
    }
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 8000 },
    )
  })
}

function localISO(d = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  const off = -d.getTimezoneOffset()
  const sign = off >= 0 ? '+' : '-'
  const abs = Math.abs(off)
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`
  )
}

const statusLabel: Record<Status, string> = {
  idle: 'Tap the mic and speak',
  listening: 'Listening...',
  thinking: 'Thinking...',
  done: '',
  error: '',
}

export default function VoiceControl({ fallbackCenter, onResult }: Props) {
  const [status, setStatus] = useState<Status>('idle')
  const [transcript, setTranscript] = useState('')
  const [answer, setAnswer] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [typed, setTyped] = useState('')
  const finalRef = useRef('')
  const speechSupported = typeof window !== 'undefined' && !!(
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
  )

  async function submit(text: string) {
    const cleaned = text.trim()
    if (!cleaned) {
      setStatus('idle')
      return
    }
    setStatus('thinking')
    setAnswer('')
    setError('')
    setNotice('')
    let loc = await getLocation()
    if (!loc) {
      loc = fallbackCenter
      setNotice('Location unavailable — searching around the map center. Allow location access for "near me" accuracy.')
    }
    try {
      const resp = await sendQuery({
        text: cleaned,
        lat: loc.lat,
        lon: loc.lon,
        now: localISO(),
      })
      onResult(resp)
      setAnswer(resp.answer)
      setStatus('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
      setStatus('error')
    }
  }

  function startListening() {
    const rec = createRecognition()
    if (!rec) {
      setError('Speech recognition is not available in this browser. Type your request instead.')
      setStatus('error')
      return
    }
    finalRef.current = ''
    setTranscript('')
    setAnswer('')
    setError('')
    setNotice('')
    rec.lang = 'en-US'
    rec.interimResults = true
    rec.continuous = false
    rec.onresult = (e: any) => {
      let text = ''
      for (let i = 0; i < e.results.length; i++) {
        text += e.results[i][0].transcript
        if (e.results[i].isFinal) finalRef.current = text
      }
      setTranscript(text)
    }
    rec.onerror = (e: any) => {
      setError(e.error === 'not-allowed' ? 'Microphone permission denied' : `Speech error: ${e.error}`)
      setStatus('error')
    }
    rec.onend = () => {
      if (finalRef.current.trim()) {
        submit(finalRef.current)
      } else {
        setStatus((s) => (s === 'error' ? s : 'idle'))
      }
    }
    rec.start()
    setStatus('listening')
  }

  function onTypedSubmit(e: React.FormEvent) {
    e.preventDefault()
    setTranscript(typed)
    submit(typed)
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <button
          className={`mic ${status === 'listening' ? 'mic-on' : ''}`}
          onClick={startListening}
          disabled={status === 'listening' || status === 'thinking'}
          aria-label="Start voice request"
        >
          {status === 'thinking' ? '...' : '\u{1F3A4}'}
        </button>
        <div className="panel-title">
          <h1>Speech to Map</h1>
          <p className="hint">"find a pharmacy near me, open now, walkable"</p>
        </div>
      </div>

      <form className="typed" onSubmit={onTypedSubmit}>
        <input
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={speechSupported ? 'or type a request' : 'type a request'}
          disabled={status === 'thinking'}
        />
        <button type="submit" disabled={status === 'thinking' || !typed.trim()}>
          Go
        </button>
      </form>

      {transcript && <div className="transcript">{transcript}</div>}

      {statusLabel[status] && <div className={`status status-${status}`}>{statusLabel[status]}</div>}

      {notice && <div className="notice">{notice}</div>}

      {answer && <div className="answer">{answer}</div>}

      {error && <div className="error">{error}</div>}
    </div>
  )
}
