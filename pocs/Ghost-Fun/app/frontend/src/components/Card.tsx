interface CardProps {
  value: string
  flipped: boolean
  matched: boolean
  onClick: () => void
}

export default function Card({ value, flipped, matched, onClick }: CardProps) {
  return (
    <div
      className={`card ${flipped ? 'flipped' : ''} ${matched ? 'matched' : ''}`}
      onClick={onClick}
    >
      <div className="card-inner">
        <div className="card-back">?</div>
        <div className="card-front">{value}</div>
      </div>
    </div>
  )
}
