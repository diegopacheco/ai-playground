interface TimerProps {
  seconds: number
}

export default function Timer({ seconds }: TimerProps) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  const danger = seconds <= 20

  return (
    <div className={`timer ${danger ? 'timer-danger' : ''}`}>
      {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
    </div>
  )
}
