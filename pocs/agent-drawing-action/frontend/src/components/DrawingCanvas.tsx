import { useRef, useState, useEffect, useCallback } from 'react'

interface Props {
  onCapture: (dataUrl: string) => void
  loading: boolean
}

type Tool = 'pen' | 'rectangle' | 'eraser'

const colors = ['#ffffff', '#ff4444', '#44ff44', '#4444ff', '#ffff44', '#ff44ff', '#44ffff', '#ff8800', '#000000']
const sizes = [2, 4, 8, 12, 20]

const toolbarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '10px 16px',
  background: '#1a1a2e',
  borderBottom: '1px solid #2a2a4a',
  flexWrap: 'wrap',
}

const btnStyle: React.CSSProperties = {
  padding: '6px 14px',
  fontSize: '13px',
  background: '#2a2a4a',
  color: '#e0e0e0',
  border: '1px solid #3a3a5a',
  borderRadius: '6px',
  cursor: 'pointer',
}

const activeBtnStyle: React.CSSProperties = {
  ...btnStyle,
  background: '#7c4dff',
  borderColor: '#7c4dff',
  color: '#fff',
}

const guessBtnStyle: React.CSSProperties = {
  padding: '8px 24px',
  fontSize: '14px',
  fontWeight: 700,
  background: '#7c4dff',
  color: '#fff',
  border: 'none',
  borderRadius: '8px',
  cursor: 'pointer',
  marginLeft: 'auto',
}

const disabledGuessBtnStyle: React.CSSProperties = {
  ...guessBtnStyle,
  background: '#3a3a5a',
  cursor: 'not-allowed',
}

export default function DrawingCanvas({ onCapture, loading }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [drawing, setDrawing] = useState(false)
  const [tool, setTool] = useState<Tool>('pen')
  const [color, setColor] = useState('#ffffff')
  const [size, setSize] = useState(4)
  const [history, setHistory] = useState<ImageData[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const rectStart = useRef<{ x: number; y: number } | null>(null)
  const snapshotBeforeRect = useRef<ImageData | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    const snap = ctx.getImageData(0, 0, canvas.width, canvas.height)
    setHistory([snap])
    setHistoryIndex(0)
  }, [])

  const saveSnapshot = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const snap = ctx.getImageData(0, 0, canvas.width, canvas.height)
    setHistory(prev => {
      const next = prev.slice(0, historyIndex + 1)
      next.push(snap)
      return next
    })
    setHistoryIndex(prev => prev + 1)
  }, [historyIndex])

  const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const pos = getPos(e)

    if (tool === 'rectangle') {
      rectStart.current = pos
      snapshotBeforeRect.current = ctx.getImageData(0, 0, canvas.width, canvas.height)
      setDrawing(true)
      return
    }

    setDrawing(true)
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
    ctx.strokeStyle = tool === 'eraser' ? '#1a1a2e' : color
    ctx.lineWidth = tool === 'eraser' ? size * 3 : size
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing) return
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const pos = getPos(e)

    if (tool === 'rectangle' && rectStart.current && snapshotBeforeRect.current) {
      ctx.putImageData(snapshotBeforeRect.current, 0, 0)
      ctx.strokeStyle = color
      ctx.lineWidth = size
      ctx.strokeRect(
        rectStart.current.x,
        rectStart.current.y,
        pos.x - rectStart.current.x,
        pos.y - rectStart.current.y,
      )
      return
    }

    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
  }

  const onMouseUp = () => {
    if (!drawing) return
    setDrawing(false)
    rectStart.current = null
    snapshotBeforeRect.current = null
    saveSnapshot()
  }

  const undo = () => {
    if (historyIndex <= 0) return
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const newIndex = historyIndex - 1
    ctx.putImageData(history[newIndex], 0, 0)
    setHistoryIndex(newIndex)
  }

  const redo = () => {
    if (historyIndex >= history.length - 1) return
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const newIndex = historyIndex + 1
    ctx.putImageData(history[newIndex], 0, 0)
    setHistoryIndex(newIndex)
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    saveSnapshot()
  }

  const handleGuess = () => {
    if (loading) return
    const canvas = canvasRef.current!
    const dataUrl = canvas.toDataURL('image/png')
    onCapture(dataUrl)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={toolbarStyle}>
        <button style={tool === 'pen' ? activeBtnStyle : btnStyle} onClick={() => setTool('pen')}>Pen</button>
        <button style={tool === 'rectangle' ? activeBtnStyle : btnStyle} onClick={() => setTool('rectangle')}>Rect</button>
        <button style={tool === 'eraser' ? activeBtnStyle : btnStyle} onClick={() => setTool('eraser')}>Eraser</button>
        <span style={{ color: '#666', margin: '0 4px' }}>|</span>
        {colors.map(c => (
          <div
            key={c}
            onClick={() => setColor(c)}
            style={{
              width: 24, height: 24, borderRadius: '50%', background: c, cursor: 'pointer',
              border: c === color ? '2px solid #7c4dff' : '2px solid transparent',
            }}
          />
        ))}
        <span style={{ color: '#666', margin: '0 4px' }}>|</span>
        {sizes.map(s => (
          <button
            key={s}
            style={s === size ? activeBtnStyle : btnStyle}
            onClick={() => setSize(s)}
          >{s}px</button>
        ))}
        <span style={{ color: '#666', margin: '0 4px' }}>|</span>
        <button style={btnStyle} onClick={undo}>Undo</button>
        <button style={btnStyle} onClick={redo}>Redo</button>
        <button style={btnStyle} onClick={clearCanvas}>Clear</button>
        <button
          style={loading ? disabledGuessBtnStyle : guessBtnStyle}
          onClick={handleGuess}
          disabled={loading}
        >
          {loading ? 'Guessing...' : 'Guess Name'}
        </button>
      </div>
      <canvas
        ref={canvasRef}
        style={{ flex: 1, cursor: 'crosshair', display: 'block' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />
    </div>
  )
}
